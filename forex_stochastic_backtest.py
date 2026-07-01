"""Stochastic oscillator fixed TP/SL sweep with candle signals and tick execution."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

import numpy as np

from forex_signal_sweep_common import build_bid_ohlc, map_state_to_ticks, simulate_state_strategy
from forex_strategy_common import (
    active_session_allowed,
    build_parser,
    default_point_size,
    load_market,
    parse_num_list,
    parse_str_list,
    write_results,
)


DEFAULT_TIMEFRAMES = ["30s", "1m", "3m"]
DEFAULT_LENGTHS = [7, 10, 14, 21, 34]
DEFAULT_LOW = [15, 20, 25, 30]
DEFAULT_HIGH = [70, 75, 80, 85]
DEFAULT_MODES = ["normal", "invert"]
DEFAULT_TP = [200, 300, 400, 500]
DEFAULT_SL = [200, 300, 400]
DEFAULT_SESSIONS = [-1, 1]


def rolling_min(x: np.ndarray, length: int) -> np.ndarray:
    out = np.full(len(x), np.nan, dtype=np.float64)
    for i in range(length - 1, len(x)):
        out[i] = np.min(x[i - length + 1:i + 1])
    return out


def rolling_max(x: np.ndarray, length: int) -> np.ndarray:
    out = np.full(len(x), np.nan, dtype=np.float64)
    for i in range(length - 1, len(x)):
        out[i] = np.max(x[i - length + 1:i + 1])
    return out


def stochastic_state(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    length: int,
    low_level: float,
    high_level: float,
    mode: str,
) -> np.ndarray:
    lo = rolling_min(lows, length)
    hi = rolling_max(highs, length)
    k = 100.0 * (closes - lo) / np.maximum(hi - lo, 1e-12)
    state = np.zeros(len(closes), dtype=np.float64)

    # Normal: buy strength leaving oversold, short weakness leaving overbought.
    prev = np.empty_like(k)
    prev[0] = np.nan
    prev[1:] = k[:-1]
    state[(prev <= low_level) & (k > low_level)] = 1.0
    state[(prev >= high_level) & (k < high_level)] = -1.0
    state[~np.isfinite(k)] = 0.0
    if mode == "invert":
        state = -state
    return state


def main() -> None:
    ap = build_parser("Stochastic fixed TP/SL sweep", "forex_stochastic_results.csv")
    ap.add_argument("--lengths", default=None)
    ap.add_argument("--low-levels", default=None)
    ap.add_argument("--high-levels", default=None)
    ap.add_argument("--modes", default=None, help="normal,invert")
    ap.add_argument("--sessions", default=None, help="-1=outside sessions, 0=all hours, 1=inside Tokyo/London/New York sessions")
    ap.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    args = ap.parse_args()

    timeframes = parse_str_list(args.timeframes, DEFAULT_TIMEFRAMES)
    lengths = [int(x) for x in parse_num_list(args.lengths, DEFAULT_LENGTHS)]
    lows = parse_num_list(args.low_levels, DEFAULT_LOW)
    highs = parse_num_list(args.high_levels, DEFAULT_HIGH)
    modes = parse_str_list(args.modes, DEFAULT_MODES)
    tps = parse_num_list(args.tp_points, DEFAULT_TP)
    sls = parse_num_list(args.sl_points, DEFAULT_SL)
    sessions = [int(x) for x in parse_num_list(args.sessions, DEFAULT_SESSIONS)]

    ticks, t0 = load_market(args)
    results = []
    combo_count = len(timeframes) * len(lengths) * len(lows) * len(highs) * len(modes) * len(tps) * len(sls) * len(sessions)
    print(f"[stoch] combos_per_pair={combo_count} workers={args.workers}", flush=True)

    for pair, g in ticks.groupby("pair", sort=False):
        g = g.sort_values("timestamp")
        bid = g["bid"].to_numpy(np.float64)
        ask = g["ask"].to_numpy(np.float64)
        ts_ns = g["timestamp"].astype("int64").to_numpy(np.int64)
        point_size = float(args.point_size or default_point_size(pair))
        print(f"[stoch] {pair} ticks={len(g):,}", flush=True)

        candle_cache = {tf: build_bid_ohlc(bid, ts_ns, tf) for tf in timeframes}
        session_cache = {s: active_session_allowed(ts_ns, s) for s in sessions}
        combos = list(product(timeframes, lengths, lows, highs, modes, tps, sls, sessions))

        def run_combo(combo):
            tf, length, low_level, high_level, mode, tp, sl, sess = combo
            _, ch, cl, cc, close_idx = candle_cache[tf]
            state_c = stochastic_state(ch, cl, cc, length, low_level, high_level, mode)
            state = map_state_to_ticks(len(bid), close_idx, state_c)
            params = f"length={length};low={low_level:g};high={high_level:g};mode={mode};session={sess}"
            return simulate_state_strategy(
                pair, "stoch", params, tf, bid, ask, ts_ns, state, state,
                session_cache[int(sess)], tp, 0.0, sl, 0.0, point_size,
                args.amount, args.compound, args.leverage,
                args.commission_per_million, args.side,
            )

        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_combo, c) for c in combos]
            for fut in as_completed(futs):
                results.append(fut.result())
                done += 1
                if done % max(len(combos) // 10, 1) == 0:
                    print(f"[stoch] {pair} progress {done}/{len(combos)}", flush=True)

    filtered = [r for r in results if r.trades >= args.min_trades]
    write_results(args.out, filtered, args.top, args.sort_by)
    print(f"[stoch] wrote {args.out} elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
