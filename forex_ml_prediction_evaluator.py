"""Evaluate ML prediction CSVs with fixed TP/SL trade simulation.

The training script writes one raw prediction CSV per model combo. This file
turns those raw predictions into normal ranked backtest rows.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from forex_backtest import DEFAULT_COMMISSION_PER_MILLION, FOREX_DIR
from forex_strategy_common import (
    TradeResult,
    active_session_allowed,
    default_point_size,
    parse_num_list,
    parse_str_list,
    write_results,
)


DEFAULT_PRED_DIR = Path("data") / "forex" / "ml_predictions"
DEFAULT_OUT = Path("data") / "forex" / "analysis" / "ml_prediction_eval.csv"


def pair_default_tp(pair: str) -> list[float]:
    return [50, 100, 150, 200, 300, 400] if pair.upper() == "XAUUSD" else [10, 20, 30, 50, 80, 100]


def pair_default_sl(pair: str) -> list[float]:
    return [50, 100, 150, 200, 300, 400] if pair.upper() == "XAUUSD" else [10, 20, 30, 50, 80, 100]


def parse_filename(path: Path) -> dict[str, str]:
    name = path.name
    out: dict[str, str] = {"file": name}
    m = re.match(r"forex_ml_([^_]+)_([^_]+)_([^_]+)_([^_]+)_", name)
    if m:
        out["pair"] = m.group(1)
        out["target"] = m.group(2)
        out["side"] = m.group(3)
        out["model"] = m.group(4)
    for key, pattern in {
        "tf": r"_tf([^_]+)_",
        "label_tp": r"_tp(-?\d+(?:\.\d+)?)_",
        "label_sl": r"_sl(-?\d+(?:\.\d+)?)_",
        "label_session": r"_s(-?\d+)_",
        "window": r"_w(\d+)_",
        "horizon": r"_h(\d+)_",
        "channels": r"_c(\d+)_",
        "kernel": r"_k(\d+)_",
        "layers": r"_l(\d+)",
    }.items():
        mm = re.search(pattern, name)
        if mm:
            out[key] = mm.group(1)
    return out


def moving_average(values: np.ndarray, length: int) -> np.ndarray:
    if length <= 1:
        return values.astype(np.float64, copy=False)
    return pd.Series(values.astype(np.float64)).rolling(length, min_periods=1).mean().to_numpy(np.float64)


def safe_profit_factor(gross_win: float, gross_loss: float) -> float:
    if gross_loss > 0:
        return gross_win / gross_loss
    return 999.0 if gross_win > 0 else 0.0


def simulate_predictions(
    df: pd.DataFrame,
    meta: dict[str, str],
    threshold: float,
    prob_ma: int,
    tp_points: float,
    sl_points: float,
    session: int,
    amount: float,
    leverage: float,
    commission_per_million: float,
    horizon_bars: int,
) -> TradeResult | None:
    pair = meta.get("pair", "UNKNOWN")
    point_size = default_point_size(pair)
    if tp_points <= 0 or sl_points <= 0:
        return None

    required = {
        "time", "open", "high", "low", "close",
        "prob_up", "prob_short", "expected_max_up_points", "expected_max_down_points",
    }
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"{meta.get('file')} missing columns: {missing}")

    ts = pd.to_datetime(df["time"], utc=True).astype("int64").to_numpy(np.int64)
    high = df["high"].to_numpy(np.float64)
    low = df["low"].to_numpy(np.float64)
    close = df["close"].to_numpy(np.float64)
    long_prob = moving_average(df["prob_up"].to_numpy(np.float64), prob_ma)
    short_prob = moving_average(df["prob_short"].to_numpy(np.float64), prob_ma)
    exp_up = moving_average(df["expected_max_up_points"].to_numpy(np.float64), prob_ma)
    exp_down = moving_average(df["expected_max_down_points"].to_numpy(np.float64), prob_ma)
    allowed = active_session_allowed(ts, int(session))
    day = pd.to_datetime(ts, utc=True).floor("D")

    cash = amount
    peak_cash = amount
    max_account_dd = 0.0
    cum_peak = 0.0
    cum_pnl = 0.0
    cum_dd = 0.0
    gross_win = gross_loss = 0.0
    trade_pnls: list[float] = []
    trade_dds: list[float] = []
    daily: dict[pd.Timestamp, float] = {}
    trades = wins = losses = stops = sig = long_trades = short_trades = 0
    worst = 0.0

    k = 0
    n = len(df)
    while k < n - 1:
        if not allowed[k]:
            k += 1
            continue
        side = 0
        if long_prob[k] >= threshold and exp_up[k] >= tp_points and exp_down[k] <= sl_points:
            side = 1
        elif short_prob[k] >= threshold and exp_down[k] >= tp_points and exp_up[k] <= sl_points:
            side = -1
        if side == 0:
            k += 1
            continue

        entry = close[k]
        tp = tp_points * point_size
        sl = sl_points * point_size
        exit_i = min(k + horizon_bars, n - 1)
        result_points = 0.0
        reason = "horizon"
        trade_max_dd = 0.0

        for j in range(k + 1, exit_i + 1):
            if side == 1:
                adverse_points = max(0.0, (entry - low[j]) / point_size)
                win_hit = high[j] >= entry + tp
                loss_hit = low[j] <= entry - sl
            else:
                adverse_points = max(0.0, (high[j] - entry) / point_size)
                win_hit = low[j] <= entry - tp
                loss_hit = high[j] >= entry + sl

            notional = amount * leverage
            adverse_pnl = adverse_points * point_size * (notional / max(entry, 1e-12))
            trade_max_dd = max(trade_max_dd, adverse_pnl)
            max_account_dd = max(max_account_dd, peak_cash - (cash - adverse_pnl))

            # Conservative same-bar ordering: if TP and SL both hit, count SL.
            if win_hit and loss_hit:
                result_points = -sl_points
                reason = "stop"
                exit_i = j
                break
            if loss_hit:
                result_points = -sl_points
                reason = "stop"
                exit_i = j
                break
            if win_hit:
                result_points = tp_points
                reason = "tp"
                exit_i = j
                break
        else:
            result_points = (close[exit_i] - entry) / point_size * side

        notional = amount * leverage
        pnl = result_points * point_size * (notional / max(entry, 1e-12))
        fee = notional / 1_000_000.0 * commission_per_million * 2.0
        trade_pnl = pnl - fee

        cash += trade_pnl
        peak_cash = max(peak_cash, cash)
        max_account_dd = max(max_account_dd, peak_cash - cash)
        cum_pnl += trade_pnl
        cum_peak = max(cum_peak, cum_pnl)
        cum_dd = max(cum_dd, cum_peak - cum_pnl)

        d = day[exit_i]
        daily[d] = daily.get(d, 0.0) + trade_pnl
        trades += 1
        long_trades += int(side == 1)
        short_trades += int(side == -1)
        stops += int(reason == "stop")
        sig += int(reason == "horizon")
        worst = min(worst, trade_pnl)
        trade_pnls.append(trade_pnl)
        trade_dds.append(trade_max_dd)
        if trade_pnl >= 0:
            wins += 1
            gross_win += trade_pnl
        else:
            losses += 1
            gross_loss += -trade_pnl

        while k < n and k <= exit_i:
            k += 1

    result = TradeResult(
        pair=pair,
        strategy="ml_move4",
        params=(
            f"threshold={threshold:g};prob_ma={prob_ma};eval_session={session};"
            f"model={meta.get('model','')};target={meta.get('target','')};"
            f"label_session={meta.get('label_session','')};label_tp={meta.get('label_tp','')};"
            f"label_sl={meta.get('label_sl','')};window={meta.get('window','')};"
            f"horizon={horizon_bars};file={meta.get('file','')}"
        ),
        timeframe=meta.get("tf", "?"),
        tp_points=float(tp_points),
        sl_points=float(sl_points),
        point_size=point_size,
        realised=cash - amount,
        open_unrealized=0.0,
        total=cash - amount,
        trades=trades,
        wins=wins,
        losses=losses,
        win_rate=wins / trades * 100.0 if trades else 0.0,
        profit_factor=safe_profit_factor(gross_win, gross_loss),
        max_drawdown=max_account_dd,
        long_trades=long_trades,
        short_trades=short_trades,
        stop_losses=stops,
        signal_exits=sig,
        liquidations=0,
        account_dead=False,
        open_side="-",
        open_bps=0.0,
    )
    setattr(result, "cum_max_drawdown", float(cum_dd))
    setattr(result, "trade_max_drawdown", float(max(trade_dds) if trade_dds else 0.0))
    setattr(result, "trade_avg_drawdown", float(np.mean(trade_dds) if trade_dds else 0.0))
    setattr(result, "worst_trade_pnl", float(worst))
    loss_values = [x for x in trade_pnls if x < 0]
    setattr(result, "median_loss", float(np.median(loss_values) if loss_values else 0.0))
    day_values = list(daily.values())
    setattr(result, "avg_day", float(np.mean(day_values) if day_values else 0.0))
    setattr(result, "median_day", float(np.median(day_values) if day_values else 0.0))
    setattr(result, "active_days", int(len(day_values)))
    ratio = result.realised / max(getattr(result, "cum_max_drawdown", 0.0), 1.0)
    setattr(result, "realised_cumdd_ratio", float(ratio))
    return result


def iter_prediction_files(pred_dir: Path, patterns: list[str], pairs: set[str] | None) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(pred_dir.glob(pattern))
    unique = sorted(set(files))
    if pairs:
        unique = [p for p in unique if parse_filename(p).get("pair", "").upper() in pairs]
    return unique


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate ML prediction CSVs with TP/SL/session/threshold sweeps")
    ap.add_argument("--pred-dir", default=str(DEFAULT_PRED_DIR))
    ap.add_argument("--glob", default="forex_ml_*move4*_predictions.csv")
    ap.add_argument("--pairs", nargs="+", default=None)
    ap.add_argument("--timeframes", default=None, help="comma list filter, e.g. 1m,3m,5m")
    ap.add_argument("--tp-points", default=None)
    ap.add_argument("--sl-points", default=None)
    ap.add_argument("--thresholds", default="0.55,0.58,0.60,0.62,0.65")
    ap.add_argument("--prob-ma", default="1,3,5")
    ap.add_argument("--sessions", default="label",
                    help="'label' = evaluate only the session the model was trained on; or comma list like -1,0,1,2,3")
    ap.add_argument("--amount", type=float, default=50.0)
    ap.add_argument("--leverage", type=float, default=100.0)
    ap.add_argument("--commission-per-million", type=float, default=DEFAULT_COMMISSION_PER_MILLION)
    ap.add_argument("--horizon-bars", type=int, default=None, help="override horizon parsed from filename")
    ap.add_argument("--min-trades", type=int, default=5)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    patterns = parse_str_list(args.glob, ["forex_ml_*_predictions.csv"])
    pairs = {p.upper() for p in args.pairs} if args.pairs else None
    files = iter_prediction_files(pred_dir, patterns, pairs)
    tf_filter = set(parse_str_list(args.timeframes, [])) if args.timeframes else None
    if tf_filter:
        files = [p for p in files if parse_filename(p).get("tf", "") in tf_filter]
    if not files:
        raise SystemExit(f"no prediction files found in {pred_dir}")

    thresholds = parse_num_list(args.thresholds, [0.55, 0.6, 0.65])
    prob_mas = [int(x) for x in parse_num_list(args.prob_ma, [1])]
    results: list[TradeResult] = []

    print(f"[ml-eval] files={len(files)} pred_dir={pred_dir}", flush=True)
    for file_i, path in enumerate(files, 1):
        meta = parse_filename(path)
        pair = meta.get("pair", "")
        tps = parse_num_list(args.tp_points, pair_default_tp(pair))
        sls = parse_num_list(args.sl_points, pair_default_sl(pair))
        horizon = int(args.horizon_bars or meta.get("horizon", 200))
        if str(args.sessions).strip().lower() == "label":
            sessions = [int(meta.get("label_session", 0))]
        else:
            sessions = [int(x) for x in parse_num_list(args.sessions, [-1, 0, 1, 2])]
        print(
            f"[ml-eval] {file_i}/{len(files)} {path.name} "
            f"tp={','.join(f'{x:g}' for x in tps)} sl={','.join(f'{x:g}' for x in sls)}",
            flush=True,
        )
        df = pd.read_csv(path)
        required = {
            "time", "open", "high", "low", "close",
            "prob_up", "prob_short", "expected_max_up_points", "expected_max_down_points",
        }
        if not required.issubset(df.columns):
            missing = sorted(required - set(df.columns))
            print(f"[ml-eval] skip {path.name} missing={missing}", flush=True)
            continue
        for threshold in thresholds:
            for prob_ma in prob_mas:
                for tp in tps:
                    for sl in sls:
                        for session in sessions:
                            result = simulate_predictions(
                                df, meta, float(threshold), int(prob_ma), float(tp), float(sl), int(session),
                                args.amount, args.leverage, args.commission_per_million, horizon,
                            )
                            if result is not None and result.trades >= args.min_trades:
                                results.append(result)

    if not results:
        raise SystemExit("no results survived --min-trades")
    results.sort(
        key=lambda r: (
            getattr(r, "realised_cumdd_ratio", 0.0),
            r.realised,
            r.profit_factor,
        ),
        reverse=True,
    )
    write_results(args.out, results, args.top, "pnl")
    print(f"[ml-eval] wrote {args.out} rows={len(results):,}", flush=True)


if __name__ == "__main__":
    main()
