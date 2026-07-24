"""Tick backtest for TCN3 signed-excursion regression models.

TCN3 predicts one score in points: the signed dominant excursion over the
training horizon. Positive scores signal long, negative scores signal short.
There is no probability conversion or moving-average smoothing.
"""
from __future__ import annotations

import time
from itertools import product
from pathlib import Path

import numpy as np
import torch

from forex_ml_tick_simulator import (
    build_bid_candles,
    load_torch_model,
    model_files,
    parse_model_name,
    predict_model,
)
from forex_strategy_common import (
    active_session_allowed,
    build_parser,
    day_ids_from_timestamps,
    load_market,
    parse_num_list,
    parse_str_list,
    write_results,
)
from forex_symbolic_return_backtest import (
    append_symbolic_csv,
    default_thresholds_for_pair,
    default_tp_sl_for_pair,
    effective_modes,
    parse_exit_modes,
    parse_pair_num_list,
    simulate_one,
)


DEFAULT_MODEL_DIR = Path("data") / "forex" / "ml_models"


def requested_ints(value: str | None) -> set[int] | None:
    return None if value is None else {int(x) for x in parse_num_list(value, [])}


def audit_model_grid(
    paths: list[Path],
    pairs: set[str],
    timeframes: set[str] | None,
    sessions: set[int] | None,
    windows: set[int] | None,
    horizons: set[int] | None,
    allow_incomplete: bool,
) -> None:
    requested = (timeframes, sessions, windows, horizons)
    if not all(values is not None for values in requested):
        return
    expected = set(product(pairs, timeframes, sessions, windows, horizons))
    actual = set()
    for path in paths:
        meta = parse_model_name(path)
        actual.add((
            str(meta.get("pair", "")).upper(),
            str(meta.get("tf", "")).lower(),
            int(meta.get("label_session", 0)),
            int(meta.get("window", 0)),
            int(meta.get("horizon", 0)),
        ))
    missing = sorted(expected - actual)
    if missing and not allow_incomplete:
        raise SystemExit(
            f"[tcn3] incomplete model grid expected={len(expected):,} "
            f"found={len(actual):,} missing={len(missing):,} "
            f"first_missing={missing[:12]}; pass --allow-incomplete-model-grid to override"
        )


def main() -> None:
    ap = build_parser("TCN3 signed-excursion tick backtest", "forex_tcn3_excursion_results.csv")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--model-glob", default="*excursion*_*tcn3_*.pt")
    ap.add_argument("--thresholds", default="pair", help="'pair' uses separate XAU and FX score grids")
    ap.add_argument("--modes", default="normal,invert")
    ap.add_argument("--tp-modes", default="fixed,fixed_signal,opposite,neutral")
    ap.add_argument("--sl-modes", default="fixed,fixed_signal,opposite,neutral")
    ap.add_argument("--sessions", default=None, help="model/evaluation sessions, or omit to use each label session")
    ap.add_argument("--label-sessions", default=None, help="filter model label sessions without changing evaluation sessions")
    ap.add_argument("--windows", default=None)
    ap.add_argument("--horizons", default=None)
    ap.add_argument("--allow-incomplete-model-grid", action="store_true")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--keep-in-memory", action="store_true", help="retain rows for sorted console sections")
    args = ap.parse_args()

    requested_pairs = {str(pair).upper() for pair in args.pairs}
    requested_tfs = None if not args.timeframes else {str(tf).lower() for tf in parse_str_list(args.timeframes, [])}
    requested_sessions = requested_ints(args.sessions)
    model_sessions = requested_ints(args.label_sessions)
    if model_sessions is None:
        model_sessions = requested_sessions
    requested_windows = requested_ints(args.windows)
    requested_horizons = requested_ints(args.horizons)

    paths = model_files(
        Path(args.model_dir),
        parse_str_list(args.model_glob, ["*.pt"]),
        requested_pairs,
        requested_tfs,
        requested_windows,
        model_sessions,
    )
    paths = [
        path for path in paths
        if str(parse_model_name(path).get("target", "")) == "excursion"
        and str(parse_model_name(path).get("model", "")) == "tcn3"
        and (requested_horizons is None or int(parse_model_name(path).get("horizon", -1)) in requested_horizons)
    ]
    if not paths:
        raise SystemExit("no matching TCN3 excursion models found")

    audit_model_grid(
        paths,
        requested_pairs,
        requested_tfs,
        model_sessions,
        requested_windows,
        requested_horizons,
        args.allow_incomplete_model_grid,
    )

    ticks, _ = load_market(args)
    loaded_pairs = {str(pair).upper() for pair in ticks["pair"].unique()}
    missing_pairs = sorted(requested_pairs - loaded_pairs)
    if missing_pairs:
        raise SystemExit(f"[tcn3] missing requested market data pairs={missing_pairs}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    threshold_override = None
    if str(args.thresholds).strip().lower() not in {"", "auto", "pair"}:
        threshold_override = parse_num_list(args.thresholds, [])
    modes = parse_str_list(args.modes, ["normal", "invert"])
    bad_modes = [mode for mode in modes if mode not in {"normal", "invert"}]
    if bad_modes:
        raise SystemExit(f"bad signal modes={bad_modes}")
    tp_modes = parse_exit_modes(args.tp_modes, ["fixed", "fixed_signal", "opposite", "neutral"])
    sl_modes = parse_exit_modes(args.sl_modes, ["fixed", "fixed_signal", "opposite", "neutral"])

    out_path = Path(args.out)
    if out_path.exists():
        out_path.unlink()
        print(f"[tcn3] overwrite out={out_path}", flush=True)

    kept_results = []
    total_rows = 0
    overall_start = time.time()
    print(
        f"[tcn3] models={len(paths):,} device={device} "
        f"thresholds={'pair-specific' if threshold_override is None else threshold_override} "
        f"modes={modes} tp_modes={tp_modes} sl_modes={sl_modes} ma=none reset=mandatory",
        flush=True,
    )

    for model_number, path in enumerate(paths, 1):
        model_start = time.time()
        meta = parse_model_name(path)
        pair = str(meta["pair"]).upper()
        pair_ticks = ticks[ticks["pair"].str.upper() == pair].sort_values("timestamp").reset_index(drop=True)
        if pair_ticks.empty:
            raise SystemExit(f"[tcn3] no ticks loaded for model pair={pair}")

        bid = pair_ticks["bid"].to_numpy(np.float64)
        ask = pair_ticks["ask"].to_numpy(np.float64)
        ts_ns = pair_ticks["timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64").astype(np.int64)
        model, namespace, point = load_torch_model(path)
        if getattr(namespace, "target", "") != "excursion":
            raise SystemExit(f"[tcn3] wrong target in {path.name}: {getattr(namespace, 'target', '')}")
        candles = build_bid_candles(bid, ask, ts_ns, str(meta["tf"]))
        predictions = predict_model(model, namespace, candles, point, device)
        scores = predictions[:, 0].astype(np.float64)
        finite_scores = scores[np.isfinite(scores)]
        if not len(finite_scores):
            raise SystemExit(f"[tcn3] model produced no finite scores: {path.name}")

        thresholds = default_thresholds_for_pair(pair) if threshold_override is None else threshold_override
        default_tp, default_sl = default_tp_sl_for_pair(pair)
        tp_values = parse_pair_num_list(args.tp_points, default_tp)
        sl_values = parse_pair_num_list(args.sl_points, default_sl)
        eval_sessions = (
            sorted(requested_sessions)
            if requested_sessions is not None
            else [int(meta.get("label_session", 0))]
        )

        combo_total = 0
        for tp in tp_values:
            for sl in sl_values:
                combo_total += len(effective_modes(tp_modes, float(tp))) * len(effective_modes(sl_modes, float(sl)))
        combo_total *= len(thresholds) * len(modes) * len(eval_sessions)

        candle_times = candles.times.astype("int64")
        tick_to_candle = candles.tick_to_candle.astype(np.int64)
        day_ids, max_days = day_ids_from_timestamps(ts_ns)
        rows = []
        combo_done = 0
        print(
            f"[tcn3] model {model_number}/{len(paths)} {path.name} "
            f"candles={len(candles.ohlc):,} score_range={finite_scores.min():.1f}..{finite_scores.max():.1f} "
            f"thresholds={thresholds} combos={combo_total:,}",
            flush=True,
        )

        for session in eval_sessions:
            allowed = active_session_allowed(candle_times, int(session)).astype(np.bool_)
            sim_meta = {
                "pair": pair,
                "session": int(session),
                "timeframe": str(meta["tf"]),
                "window": int(meta["window"]),
                "horizon": int(meta["horizon"]),
                "backend": "tcn3",
                "model_path": str(path),
                "expression": "",
            }
            for threshold in thresholds:
                for mode in modes:
                    for tp in tp_values:
                        for sl in sl_values:
                            for tp_mode in effective_modes(tp_modes, float(tp)):
                                for sl_mode in effective_modes(sl_modes, float(sl)):
                                    result = simulate_one(
                                        pair,
                                        sim_meta,
                                        bid,
                                        ask,
                                        candles.close_tick_idx,
                                        scores,
                                        allowed,
                                        tick_to_candle,
                                        day_ids.astype(np.int64),
                                        int(max_days),
                                        float(point),
                                        float(threshold),
                                        mode,
                                        tp_mode,
                                        sl_mode,
                                        float(tp),
                                        float(sl),
                                        args,
                                    )
                                    result.strategy = "tcn3_excursion"
                                    result.params = (
                                        f"threshold={threshold:g};mode={mode};tp_mode={tp_mode};sl_mode={sl_mode};"
                                        f"session={session};label_session={meta.get('label_session','')};"
                                        f"window={meta['window']};horizon={meta['horizon']};ma=none;"
                                        f"reset=mandatory;file={path.name}"
                                    )
                                    if result.trades >= args.min_trades:
                                        rows.append(result)
                                        total_rows += 1
                                        if args.keep_in_memory:
                                            kept_results.append(result)
                                    combo_done += 1

        append_symbolic_csv(args.out, rows)
        best = max((result.total for result in rows), default=0.0)
        print(
            f"[tcn3] model {model_number}/{len(paths)} done combos={combo_done:,}/{combo_total:,} "
            f"rows={len(rows):,} best=$" + f"{best:+.2f} elapsed={time.time() - model_start:.1f}s",
            flush=True,
        )

    if total_rows == 0:
        raise SystemExit("no TCN3 results survived --min-trades")
    if args.keep_in_memory:
        write_results(args.out, kept_results, args.top, args.sort_by)
    print(
        f"[tcn3] wrote {args.out} rows={total_rows:,} "
        f"models={len(paths):,} elapsed={time.time() - overall_start:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()