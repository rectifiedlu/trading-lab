"""Walk-forward test of pair-specific raw-price impulse persistence.

Each candidate is a recent close-to-close displacement.  The entry cutoff is
the training block's own absolute-return quantile, so pairs and timeframes do
not share arbitrary point thresholds.  Direction and stop are selected on the
training block, then held fixed on the following test block.
"""
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import asdict

import numpy as np
import pandas as pd

from forex_signal_sweep_common import build_bid_ohlc, map_state_to_ticks, simulate_state_strategy
from forex_strategy_common import active_session_allowed, build_parser, default_point_size, load_market


TIMEFRAMES = ["1m", "5m", "15m"]
SESSIONS = [0, 2]
LOOKBACKS = [3, 6, 12, 24]
QUANTILES = [0.70, 0.80, 0.90]
STOP_MULTIPLES = [0.75, 1.0, 1.5]


def make_state(close: np.ndarray, lookback: int, threshold: float, invert: bool) -> np.ndarray:
    move = np.full(len(close), np.nan, dtype=np.float64)
    move[lookback:] = close[lookback:] - close[:-lookback]
    state = np.zeros(len(close), dtype=np.float64)
    state[move >= threshold] = -1.0 if invert else 1.0
    state[move <= -threshold] = 1.0 if invert else -1.0
    return state


def result_score(r, amount: float) -> float:
    dd_pct = r.max_drawdown / max(amount, 1e-9)
    if r.trades < 20 or r.total <= 0 or dd_pct > 0.25 or r.profit_factor < 1.10:
        return -1e18
    return (r.total / max(r.max_drawdown, 1e-9)) * np.log1p(r.trades) + r.median_day / max(amount, 1e-9)


def simulate(pair, tf, bid, ask, ts_ns, close_idx, state_candle, session, stop_points, args):
    state_tick = map_state_to_ticks(len(bid), close_idx, state_candle)
    allowed = active_session_allowed(ts_ns, session)
    return simulate_state_strategy(
        pair, "impulse_walkforward", "", tf, bid, ask, ts_ns, state_tick, state_tick, allowed,
        0.0, 0.0, stop_points, 0.0, default_point_size(pair), args.amount, args.compound,
        args.leverage, args.commission_per_million, "both", signal_exit_always=True,
    )


def split_ticks(ts: pd.Series, train_start, train_end, test_end):
    train = (ts >= train_start) & (ts < train_end)
    test = (ts >= train_end) & (ts < test_end)
    return train.to_numpy(), test.to_numpy()


def main() -> None:
    ap = build_parser("Raw impulse persistence walk-forward backtest", "forex_impulse_walkforward_results.csv")
    ap.add_argument("--sessions", default=",".join(map(str, SESSIONS)))
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--train-days", type=int, default=14)
    ap.add_argument("--test-days", type=int, default=7)
    ap.add_argument("--min-folds", type=int, default=2)
    args = ap.parse_args()
    ticks, _ = load_market(args)
    tfs = [x.strip() for x in (args.timeframes or ",".join(TIMEFRAMES)).split(",") if x.strip()]
    sessions = [int(x) for x in args.sessions.split(",") if x.strip()]
    all_rows = []

    for pair, group in ticks.groupby("pair", sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        bid = group.bid.to_numpy(np.float64)
        ask = group.ask.to_numpy(np.float64)
        ts_ns = group.timestamp.to_numpy(dtype="datetime64[ns]").astype("int64")
        start = pd.Timestamp(args.start, tz="UTC")
        end = pd.Timestamp(args.to, tz="UTC")
        for tf in tfs:
            open_, high, low, close, close_idx = build_bid_ohlc(bid, ts_ns, tf)
            candle_ts = pd.to_datetime(ts_ns[close_idx], utc=True)
            for session in sessions:
                fold_results = {}
                for fold in range(args.folds):
                    test_end = end - pd.Timedelta(days=(args.folds - 1 - fold) * args.test_days)
                    train_end = test_end - pd.Timedelta(days=args.test_days)
                    train_start = train_end - pd.Timedelta(days=args.train_days)
                    train_mask, test_mask = split_ticks(group.timestamp, train_start, train_end, test_end)
                    candle_train = (candle_ts >= train_start) & (candle_ts < train_end)
                    if train_mask.sum() < 1000 or test_mask.sum() < 1000 or candle_train.sum() < 100:
                        continue
                    train_first = np.flatnonzero(train_mask)[0]
                    train_last = np.flatnonzero(train_mask)[-1]
                    train_c0 = np.searchsorted(close_idx, train_first, side="left")
                    train_c1 = np.searchsorted(close_idx, train_last, side="right")
                    train_close_idx = close_idx[train_c0:train_c1] - train_first
                    best = None
                    for lookback in LOOKBACKS:
                        move = np.abs(close[lookback:] - close[:-lookback])
                        eligible = move[candle_train[lookback:]]
                        if len(eligible) < 100:
                            continue
                        for quantile in QUANTILES:
                            threshold = float(np.quantile(eligible, quantile))
                            ranges = (high - low) / default_point_size(pair)
                            base_stop = float(np.nanmedian(ranges[candle_train]))
                            for invert in (False, True):
                                state = make_state(close, lookback, threshold, invert)
                                for stop_mult in STOP_MULTIPLES:
                                    r = simulate(pair, tf, bid[train_mask], ask[train_mask], ts_ns[train_mask],
                                                 np.searchsorted(np.flatnonzero(train_mask), close_idx[(close_idx < np.flatnonzero(train_mask)[-1])]),
                                                 state[:np.searchsorted(close_idx, np.flatnonzero(train_mask)[-1], side="right")],
                                                 session, base_stop * stop_mult, args)
                                    score = result_score(r, args.amount)
                                    if best is None or score > best[0]:
                                        best = (score, lookback, quantile, threshold, invert, base_stop * stop_mult)
                    if best is None or best[0] <= -1e17:
                        continue
                    _, lookback, quantile, threshold, invert, stop_points = best
                    state = make_state(close, lookback, threshold, invert)
                    # Preserve candle state at the test boundary, then map it to test ticks.
                    test_first = np.flatnonzero(test_mask)[0]
                    test_last = np.flatnonzero(test_mask)[-1]
                    c0 = np.searchsorted(close_idx, test_first, side="left")
                    c1 = np.searchsorted(close_idx, test_last, side="right")
                    local_close_idx = close_idx[c0:c1] - test_first
                    r = simulate(pair, tf, bid[test_mask], ask[test_mask], ts_ns[test_mask], local_close_idx,
                                 state[c0:c1], session, stop_points, args)
                    key = (lookback, quantile, invert, round(stop_points, 6))
                    fold_results.setdefault(key, []).append(r)
                for key, rows in fold_results.items():
                    if len(rows) < args.min_folds:
                        continue
                    positives = sum(r.total > 0 for r in rows)
                    total = sum(r.total for r in rows)
                    max_dd = max(r.max_drawdown for r in rows)
                    trades = sum(r.trades for r in rows)
                    if positives < args.min_folds or max_dd / args.amount > 0.25 or trades < 40:
                        continue
                    all_rows.append({
                        "pair": pair, "timeframe": tf, "session": session, "lookback": key[0],
                        "quantile": key[1], "invert": key[2], "stop_points": key[3], "folds": len(rows),
                        "positive_folds": positives, "total": total, "max_fold_dd": max_dd,
                        "dd_pct_amount": max_dd / args.amount * 100.0, "pnl_dd": total / max(max_dd, 1e-9),
                        "trades": trades, "profit_factor_mean": float(np.mean([r.profit_factor for r in rows])),
                        "median_day_mean": float(np.mean([r.median_day for r in rows])),
                    })

    all_rows.sort(key=lambda x: (x["pnl_dd"], x["positive_folds"], x["trades"]), reverse=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]) if all_rows else ["pair"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"[impulse-wf] wrote {args.out} rows={len(all_rows)}", flush=True)
    for row in all_rows[:20]:
        print(row, flush=True)


if __name__ == "__main__":
    main()
