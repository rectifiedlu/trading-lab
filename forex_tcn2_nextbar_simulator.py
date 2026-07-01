"""Clean simulator for tcn2/nextbar models.

The model outputs one probability:
    p_up = probability future close is above current close.

Signals use a moving average of p_up:
    p_ma >= upper -> long
    p_ma <= lower -> short
    otherwise neutral

This file intentionally avoids move4 RR/expected-move machinery.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from forex_ml_tick_simulator import (
    build_bid_candles,
    default_sl_grid,
    default_tp_grid,
    load_torch_model,
    model_files,
    parse_model_name,
    predict_model,
    smooth_predictions,
)
from forex_strategy_common import (
    TradeResult,
    active_session_allowed,
    build_parser,
    day_ids_from_timestamps,
    default_point_size,
    load_market,
    parse_num_list,
    parse_str_list,
    write_results,
)

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


DEFAULT_MODEL_DIR = Path("data") / "forex" / "ml_models"
DEFAULT_UPPERS = "0.5,0.55,0.6,0.65,0.7,0.75"
DEFAULT_LOWER_OFFSETS = "auto"  # means lower=1-upper
EXIT_MODES = {"opposite": 0, "neutral": 1, "fixed": 2, "fixed_signal": 3}
REENTRY_MODES = {"immediate": 0, "ma_reset": 1}
SIGNAL_ZONES = {"outer": 0, "midband": 1}


def profit_factor(gross_win: float, gross_loss: float) -> float:
    if gross_loss > 0:
        return gross_win / gross_loss
    return 999.0 if gross_win > 0 else 0.0


def default_contract_size(pair: str) -> float:
    return 100.0 if pair.upper().startswith("XAU") else 100_000.0


def broker_commission_per_lot_side(pair: str) -> float:
    pair = pair.upper()
    if pair == "XAUUSD":
        return 4.0  # $0.04 per side at 0.01 lot.
    if pair == "AUDUSD":
        return 3.6  # $0.18 per side at 0.05 lot.
    return -1.0


def parse_exit_modes(value: str | None) -> list[str]:
    modes = parse_str_list(value, ["opposite", "neutral", "fixed", "fixed_signal"])
    bad = [m for m in modes if m not in EXIT_MODES]
    if bad:
        raise SystemExit(f"bad exit mode(s): {bad}; valid={sorted(EXIT_MODES)}")
    return modes


def effective_exit_modes(modes: list[str], upper: float, lower: float) -> list[str]:
    """At 0.5/0.5 there is no neutral state, so neutral duplicates opposite."""
    if abs(float(upper) - 0.5) < 1e-12 and abs(float(lower) - 0.5) < 1e-12:
        return [m for m in modes if m != "neutral"]
    return modes


def effective_exit_modes_for_zone(
    modes: list[str], upper: float, lower: float, signal_zone: str
) -> list[str]:
    effective = effective_exit_modes(modes, upper, lower)
    if signal_zone == "midband":
        return [mode for mode in effective if mode in {"neutral", "fixed_signal"}]
    return effective


def parse_reentry_mode(value: str | None) -> str:
    mode = (value or "immediate").strip().lower()
    if mode not in REENTRY_MODES:
        raise SystemExit(f"bad reentry mode: {mode}; valid={sorted(REENTRY_MODES)}")
    return mode


def parse_reentry_modes(value: str | None, fallback: str | None = None) -> list[str]:
    raw = value if value is not None else fallback
    modes = parse_str_list(raw, ["immediate"])
    bad = [m for m in modes if m not in REENTRY_MODES]
    if bad:
        raise SystemExit(f"bad reentry mode(s): {bad}; valid={sorted(REENTRY_MODES)}")
    return modes


def lower_thresholds_for(uppers: list[float], lower_arg: str | None) -> list[tuple[float, float]]:
    if not lower_arg or str(lower_arg).lower() == "auto":
        return [(u, 1.0 - u) for u in uppers]
    lowers = parse_num_list(lower_arg, [])
    return [(u, l) for u in uppers for l in lowers]


if njit is not None:
    @njit(cache=True)
    def _signal(p_up: float, upper: float, lower: float, signal_zone: int) -> int:
        if not np.isfinite(p_up):
            return 0
        if signal_zone == 1:
            if 0.5 <= p_up < upper:
                return 1
            if lower < p_up < 0.5:
                return -1
            return 0
        if p_up >= upper:
            return 1
        if p_up <= lower:
            return -1
        return 0


    @njit(cache=True)
    def _simulate_tcn2_core(
        bid, ask, close_tick_idx, tick_to_candle, p_up, allowed, day_id, max_days,
        point_size, upper, lower, tp_mode, sl_mode, fixed_tp, fixed_sl,
        amount, leverage, commission_per_million, lot, contract_size, commission_per_lot_side,
        side_filter_code, start_candle, reentry_mode, signal_zone,
    ):
        daily = np.zeros(max_days, dtype=np.float64)
        daily_active = np.zeros(max_days, dtype=np.bool_)
        trade_pnls = np.zeros(len(close_tick_idx) + 1, dtype=np.float64)
        fixed_units = lot * contract_size if lot > 0.0 else 0.0
        notional = amount * leverage
        cash = amount
        equity_peak = amount
        cum_pnl = 0.0
        cum_peak = 0.0
        max_dd = 0.0
        cum_dd = 0.0
        gross_win = 0.0
        gross_loss = 0.0
        trades = wins = losses = stops = sig_exits = long_trades = short_trades = 0
        worst = 0.0
        max_trade_dd = 0.0
        sum_trade_dd = 0.0
        open_unrealized = 0.0

        candle_i = start_candle
        tick_floor = 0
        pending_side = 0
        block_long = False
        block_short = False
        while candle_i < len(close_tick_idx):
            if pending_side != 0:
                side = pending_side
                pending_side = 0
                entry_tick = tick_floor
                if entry_tick >= len(bid):
                    break
                candle_i = int(tick_to_candle[entry_tick])
                if not allowed[candle_i]:
                    candle_i += 1
                    continue
                if side_filter_code == -side:
                    candle_i += 1
                    continue
                if reentry_mode == 1:
                    if side == 1 and block_long:
                        candle_i += 1
                        continue
                    if side == -1 and block_short:
                        candle_i += 1
                        continue
            else:
                p_now = p_up[candle_i]
                if reentry_mode == 1:
                    current_signal = _signal(p_now, upper, lower, signal_zone)
                    if block_long and current_signal != 1:
                        block_long = False
                    if block_short and current_signal != -1:
                        block_short = False
                side = _signal(p_now, upper, lower, signal_zone)
                if side == 0 or side_filter_code == -side:
                    candle_i += 1
                    continue
                if reentry_mode == 1:
                    if side == 1 and block_long:
                        candle_i += 1
                        continue
                    if side == -1 and block_short:
                        candle_i += 1
                        continue
                entry_tick = int(close_tick_idx[candle_i]) + 1
                if entry_tick < tick_floor:
                    entry_tick = tick_floor
                if entry_tick >= len(bid):
                    break
                entry_candle = int(tick_to_candle[entry_tick])
                if not allowed[entry_candle]:
                    candle_i += 1
                    continue

            entry = ask[entry_tick] if side == 1 else bid[entry_tick]
            exit_tick = len(bid) - 1
            result_points = ((bid[exit_tick] if side == 1 else ask[exit_tick]) - entry) / point_size * side
            trade_max = 0.0
            reason = 4
            reverse_side = 0
            # A candle prediction is only available after that candle closes.
            next_signal_candle = int(tick_to_candle[entry_tick])

            for ti in range(entry_tick + 1, len(bid)):
                live_points = (bid[ti] - entry) / point_size if side == 1 else (entry - ask[ti]) / point_size
                adverse = -live_points if live_points < 0.0 else 0.0
                denom = entry if entry > 1e-12 else 1e-12
                tdd = adverse * point_size * (notional / denom)
                if tdd > trade_max:
                    trade_max = tdd
                live_dd = equity_peak - (cash - tdd)
                if live_dd > max_dd:
                    max_dd = live_dd

                if (tp_mode == 2 or tp_mode == 3) and fixed_tp > 0.0 and live_points >= fixed_tp:
                    result_points = fixed_tp
                    exit_tick = ti
                    reason = 3
                    break
                if (sl_mode == 2 or sl_mode == 3) and fixed_sl > 0.0 and live_points <= -fixed_sl:
                    result_points = -fixed_sl
                    exit_tick = ti
                    reason = 2
                    break

                current_candle = int(tick_to_candle[ti])
                # Do not expose the current candle's final OHLC/prediction at
                # its opening tick. Only fully closed candles are actionable.
                while next_signal_candle < current_candle and next_signal_candle < len(close_tick_idx):
                    sig = _signal(p_up[next_signal_candle], upper, lower, signal_zone)
                    exit_now = False
                    mode = tp_mode if live_points >= 0.0 else sl_mode
                    if mode == 0:  # opposite
                        exit_now = sig == -side
                    elif mode == 1:  # neutral or opposite
                        exit_now = sig != side
                    elif mode == 3:  # fixed plus opposite signal
                        # Midband positions are valid only while probability
                        # remains inside that direction's center range.
                        exit_now = sig != side if signal_zone == 1 else sig == -side
                    else:  # fixed exits are tick-based above
                        exit_now = False
                    if exit_now:
                        result_points = live_points
                        exit_tick = ti
                        reason = 1
                        if sig == -side:
                            reverse_side = sig
                        break
                    next_signal_candle += 1
                if reason == 1:
                    break

            units = fixed_units if fixed_units > 0.0 else notional / (entry if entry > 1e-12 else 1e-12)
            effective_lot = abs(units) / contract_size
            fee = (
                effective_lot * commission_per_lot_side * 2.0
                if commission_per_lot_side >= 0.0
                else notional / 1000000.0 * commission_per_million * 2.0
            )
            pnl = result_points * point_size * units
            if reason == 4:
                # End-of-data mark only. Keep the position open: do not count
                # a completed trade or charge an artificial closing fee.
                open_unrealized = pnl - fee * 0.5
                break
            trade_pnl = pnl - fee
            cash += trade_pnl
            if cash > equity_peak:
                equity_peak = cash
            closed_dd = equity_peak - cash
            if closed_dd > max_dd:
                max_dd = closed_dd
            cum_pnl += trade_pnl
            if cum_pnl > cum_peak:
                cum_peak = cum_pnl
            local_cum_dd = cum_peak - cum_pnl
            if local_cum_dd > cum_dd:
                cum_dd = local_cum_dd
            d = int(day_id[exit_tick])
            if 0 <= d < max_days:
                daily[d] += trade_pnl
                daily_active[d] = True
            trades += 1
            if side == 1:
                long_trades += 1
            else:
                short_trades += 1
            if reason == 2 or reason == 3:
                if reason == 2:
                    stops += 1
                if reentry_mode == 1:
                    if side == 1:
                        block_long = True
                    else:
                        block_short = True
            elif reason == 1:
                sig_exits += 1
            if trade_pnl >= 0.0:
                wins += 1
                gross_win += trade_pnl
            else:
                losses += 1
                gross_loss += -trade_pnl
            if trade_pnl < worst:
                worst = trade_pnl
            trade_pnls[trades - 1] = trade_pnl
            if trade_max > max_trade_dd:
                max_trade_dd = trade_max
            sum_trade_dd += trade_max

            tick_floor = exit_tick + 1
            if tick_floor >= len(bid):
                break
            if reason == 2 or reason == 3:
                # Fixed TP/SL is intrabar. Evaluate that candle's signal at its
                # close; entry_tick below will be the first tick afterward.
                candle_i = int(tick_to_candle[exit_tick])
                continue
            if reason == 1 and reverse_side != 0 and side_filter_code != -reverse_side:
                pending_side = reverse_side
                continue
            next_candle = int(tick_to_candle[tick_floor])
            candle_i = next_candle if next_candle > candle_i else candle_i + 1

        return (
            cash - amount, open_unrealized, trades, wins, losses, gross_win, gross_loss, max_dd, cum_dd,
            long_trades, short_trades, stops, sig_exits, worst, max_trade_dd,
            sum_trade_dd / trades if trades else 0.0, daily, daily_active, trade_pnls[:trades]
        )


def simulate_one(
    pair, meta, bid, ask, ts_ns, candles, p_up, upper, lower, prob_ma,
    signal_mode, signal_zone, tp_mode, sl_mode, fixed_tp, fixed_sl, session,
    amount, leverage, commission, lot, contract_size, commission_per_lot_side,
    side_filter, reentry_mode,
) -> TradeResult:
    point = default_point_size(pair)
    allowed = active_session_allowed(candles.times.astype("int64"), int(session))
    day_id, max_days = day_ids_from_timestamps(ts_ns)
    side_code = 0 if side_filter == "both" else (1 if side_filter == "long" else -1)
    out = _simulate_tcn2_core(
        bid, ask, candles.close_tick_idx, candles.tick_to_candle, p_up, allowed, day_id, max_days,
        point, upper, lower, EXIT_MODES[tp_mode], EXIT_MODES[sl_mode], fixed_tp, fixed_sl,
        amount, leverage, commission, lot, contract_size, commission_per_lot_side,
        side_code, int(meta.get("window", 64)) - 1, REENTRY_MODES[reentry_mode],
        SIGNAL_ZONES[signal_zone],
    )
    realised, open_unrealized, trades, wins, losses, gw, gl, max_dd, cum_dd, longs, shorts, stops, sig, worst, tr_max, tr_avg, daily, daily_active, pnls = out
    total = realised + open_unrealized
    params = (
        f"upper={upper:g};lower={lower:g};prob_ma={prob_ma};signal_mode={signal_mode};signal_zone={signal_zone};"
        f"tp_mode={tp_mode};sl_mode={sl_mode};"
        f"reentry={reentry_mode};"
        f"tp={fixed_tp:g};sl={fixed_sl:g};eval_session={session};label_session={meta.get('label_session','')};"
        f"window={meta.get('window','')};horizon={meta.get('horizon','')};"
        f"lot={lot:g};commission_lot_side={commission_per_lot_side:g};file={meta.get('file','')}"
    )
    r = TradeResult(
        pair=pair, strategy="tcn2_nextbar", params=params, timeframe=str(meta.get("tf", "?")),
        tp_points=float(fixed_tp), sl_points=float(fixed_sl), point_size=point,
        realised=float(realised), open_unrealized=float(open_unrealized), total=float(total), trades=int(trades),
        wins=int(wins), losses=int(losses), win_rate=float(wins / trades * 100.0 if trades else 0.0),
        profit_factor=profit_factor(float(gw), float(gl)), max_drawdown=float(max_dd),
        long_trades=int(longs), short_trades=int(shorts), stop_losses=int(stops),
        signal_exits=int(sig), liquidations=0, account_dead=False, open_side="-", open_bps=0.0,
    )
    for name, val in {
        "upper": upper, "lower": lower, "prob_ma": prob_ma, "tp_mode": tp_mode, "sl_mode": sl_mode,
        "signal_mode": signal_mode, "signal_zone": signal_zone, "reentry_mode": reentry_mode,
        "eval_session": session, "label_session": meta.get("label_session", 0),
        "window": meta.get("window", 0), "horizon": meta.get("horizon", 0),
        "model_file": meta.get("file", ""), "cum_max_drawdown": cum_dd,
        "trade_max_drawdown": tr_max, "trade_avg_drawdown": tr_avg,
        "worst_trade_pnl": worst, "median_loss": float(np.median(pnls[pnls < 0.0]) if np.any(pnls < 0.0) else 0.0),
        "avg_day": float(np.mean(daily) if len(daily) else 0.0),
        "median_day": float(np.median(daily) if len(daily) else 0.0),
        "active_days": int(np.count_nonzero(daily_active)),
    }.items():
        setattr(r, name, val)
    return r


def append_results(path: str, rows: list[TradeResult], header_written: bool) -> bool:
    if not rows:
        return header_written
    import csv
    exists = header_written or os.path.exists(path)
    fields = [
        "pair", "strategy", "timeframe", "upper", "lower", "prob_ma", "tp_mode", "tp_points", "sl_mode", "sl_points",
        "signal_mode", "signal_zone", "reentry_mode",
        "realised", "open_unrealized", "total", "trades", "win_rate", "profit_factor",
        "max_drawdown", "cum_max_drawdown", "trade_max_drawdown", "worst_trade_pnl",
        "avg_day", "median_day", "long_trades", "short_trades", "stop_losses",
        "signal_exits", "label_session", "eval_session", "window", "horizon",
        "model_file", "params",
    ]
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(fields)
        for r in rows:
            w.writerow([getattr(r, k, "") for k in fields])
    return True


def main() -> None:
    ap = build_parser("TCN2 nextbar tick simulator", "forex_tcn2_nextbar_results.csv")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--model-glob", default="*nextbar*.pt")
    ap.add_argument("--upper-thresholds", default=DEFAULT_UPPERS)
    ap.add_argument("--lower-thresholds", default=DEFAULT_LOWER_OFFSETS, help="'auto' means lower=1-upper")
    ap.add_argument("--prob-ma-values", default="1,3,5,8,10,14")
    ap.add_argument("--tp-modes", default="opposite,neutral,fixed_signal")
    ap.add_argument("--sl-modes", default="opposite,neutral,fixed_signal")
    ap.add_argument("--signal-modes", default="normal", help="normal or invert; invert flips p_up before signal generation")
    ap.add_argument(
        "--signal-zones",
        default="outer",
        help="outer uses p>=upper/p<=lower; midband uses 0.5<=p<upper and lower<p<0.5",
    )
    ap.add_argument("--reentry-mode", default=None, help="backward-compatible single reentry mode")
    ap.add_argument("--reentry-modes", default="immediate", help="comma-separated: immediate,ma_reset")
    ap.add_argument("--sessions", default="label")
    ap.add_argument("--label-sessions", default=None)
    ap.add_argument("--windows", default=None)
    ap.add_argument("--horizons", default=None)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--lot", type=float, default=0.0,
                    help="fixed lot size; 0 keeps legacy --amount * --leverage sizing")
    ap.add_argument("--contract-size", type=float, default=0.0,
                    help="units per lot; 0 auto-detects 100000 for FX and 100 for XAU")
    args = ap.parse_args()

    pairs = {p.upper() for p in args.pairs} if args.pairs else None
    tfs = set(parse_str_list(args.timeframes, [])) if args.timeframes else None
    windows = {int(x) for x in parse_num_list(args.windows, [])} if args.windows else None
    horizons = {int(x) for x in parse_num_list(args.horizons, [])} if args.horizons else None
    label_sessions = {int(x) for x in parse_num_list(args.label_sessions, [])} if args.label_sessions else None
    paths = model_files(Path(args.model_dir), parse_str_list(args.model_glob, ["*.pt"]), pairs, tfs, windows, label_sessions)
    if horizons is not None:
        paths = [p for p in paths if int(parse_model_name(p).get("horizon", -1)) in horizons]
    if not paths:
        raise SystemExit("no matching tcn2/nextbar .pt models found")

    ticks, _ = load_market(args)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    upper_pairs = lower_thresholds_for(parse_num_list(args.upper_thresholds, [0.6]), args.lower_thresholds)
    prob_mas = [int(x) for x in parse_num_list(args.prob_ma_values, [1])]
    tp_modes = parse_exit_modes(args.tp_modes)
    sl_modes = parse_exit_modes(args.sl_modes)
    signal_modes = parse_str_list(args.signal_modes, ["normal"])
    bad_signal_modes = [m for m in signal_modes if m not in {"normal", "invert"}]
    if bad_signal_modes:
        raise SystemExit(f"bad signal mode(s): {bad_signal_modes}; valid=['normal', 'invert']")
    signal_zones = parse_str_list(args.signal_zones, ["outer"])
    bad_signal_zones = [z for z in signal_zones if z not in SIGNAL_ZONES]
    if bad_signal_zones:
        raise SystemExit(f"bad signal zone(s): {bad_signal_zones}; valid={sorted(SIGNAL_ZONES)}")
    reentry_modes = parse_reentry_modes(args.reentry_modes, args.reentry_mode)

    out_path = Path(args.out)
    if out_path.exists():
        out_path.unlink()
    results: list[TradeResult] = []
    header_written = False
    print(f"[tcn2] models={len(paths)} device={device} thresholds={upper_pairs} ma={prob_mas} signal_modes={signal_modes} signal_zones={signal_zones} tp_modes={tp_modes} sl_modes={sl_modes} reentry_modes={reentry_modes}", flush=True)

    for idx, path in enumerate(paths, 1):
        t0 = time.time()
        meta = parse_model_name(path)
        pair = str(meta.get("pair", "")).upper()
        g = ticks[ticks["pair"].str.upper() == pair].sort_values("timestamp").reset_index(drop=True)
        if g.empty:
            print(f"[tcn2] skip {path.name} no ticks", flush=True)
            continue
        bid = g["bid"].to_numpy(np.float64)
        ask = g["ask"].to_numpy(np.float64)
        ts_ns = g["timestamp"].astype("int64").to_numpy()
        tf = str(meta.get("tf", "1m"))
        model, ns, point = load_torch_model(path)
        candles = build_bid_candles(bid, ask, ts_ns, tf)
        print(f"[tcn2] {idx}/{len(paths)} {path.name} ticks={len(g):,} candles={len(candles.ohlc):,} predicting...", flush=True)
        preds_raw = predict_model(model, ns, candles, point, device)
        cap_tps = parse_num_list(args.tp_points, default_tp_grid(pair))
        cap_sls = parse_num_list(args.sl_points, default_sl_grid(pair))
        contract_size = args.contract_size if args.contract_size > 0.0 else default_contract_size(pair)
        commission_per_lot_side = broker_commission_per_lot_side(pair)
        sessions = [int(meta.get("label_session", 0))] if str(args.sessions).lower() == "label" else [int(x) for x in parse_num_list(args.sessions, [0])]
        combo_total = 0
        for upper, lower in upper_pairs:
            for signal_zone in signal_zones:
                eff_tp_modes = effective_exit_modes_for_zone(tp_modes, upper, lower, signal_zone)
                eff_sl_modes = effective_exit_modes_for_zone(sl_modes, upper, lower, signal_zone)
                for tm in eff_tp_modes:
                    for sm in eff_sl_modes:
                        values = (len(cap_tps) if tm in ("fixed", "fixed_signal") else 1)
                        values *= (len(cap_sls) if sm in ("fixed", "fixed_signal") else 1)
                        combo_total += values
        combo_total *= len(prob_mas) * len(signal_modes) * len(reentry_modes) * len(sessions)
        print(f"[tcn2] {idx}/{len(paths)} grid={combo_total:,} sessions={sessions}", flush=True)
        model_rows: list[TradeResult] = []
        done = 0
        for ma in prob_mas:
            preds = smooth_predictions(preds_raw, ma)
            p_up_base = preds[:, 0].astype(np.float64)
            for upper, lower in upper_pairs:
                if lower > upper:
                    continue
                for signal_mode in signal_modes:
                    p_up = 1.0 - p_up_base if signal_mode == "invert" else p_up_base
                    for signal_zone in signal_zones:
                        for reentry_mode in reentry_modes:
                            for tm in effective_exit_modes_for_zone(tp_modes, upper, lower, signal_zone):
                                tp_vals = cap_tps if tm in ("fixed", "fixed_signal") else [0.0]
                                for sm in effective_exit_modes_for_zone(sl_modes, upper, lower, signal_zone):
                                    sl_vals = cap_sls if sm in ("fixed", "fixed_signal") else [0.0]
                                    for sess in sessions:
                                        for tp in tp_vals:
                                            for sl in sl_vals:
                                                r = simulate_one(
                                                    pair, meta, bid, ask, ts_ns, candles, p_up,
                                                    float(upper), float(lower), int(ma), signal_mode,
                                                    signal_zone, tm, sm, float(tp), float(sl), int(sess),
                                                    args.amount, args.leverage, args.commission_per_million,
                                                    args.lot, contract_size, commission_per_lot_side,
                                                    args.side, reentry_mode,
                                                )
                                                if r.trades >= args.min_trades:
                                                    model_rows.append(r)
                                                done += 1
                                                if done % 500 == 0 or done == combo_total:
                                                    print(f"[tcn2] {idx}/{len(paths)} sim {done:,}/{combo_total:,} rows={len(model_rows):,}", flush=True)
        results.extend(model_rows)
        header_written = append_results(args.out, model_rows, header_written)
        best = max((r.total for r in model_rows), default=0.0)
        print(f"[tcn2] {idx}/{len(paths)} done rows={len(model_rows):,} best=${best:+.2f} elapsed={time.time()-t0:.1f}s", flush=True)

    if not results:
        raise SystemExit("no results survived --min-trades")
    write_results(args.out, results, args.top, args.sort_by)
    print(f"[tcn2] wrote {args.out} rows={len(results):,}", flush=True)


if __name__ == "__main__":
    main()
