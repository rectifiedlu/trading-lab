"""Donchian breakout/fade sweep with candle signals and tick execution."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

import numpy as np

from forex_signal_sweep_common import (
    build_bid_ohlc,
    map_state_to_ticks,
    rolling_high_prev,
    rolling_low_prev,
    simulate_state_strategy,
)
from forex_strategy_common import (
    active_session_allowed,
    build_parser,
    default_point_size,
    load_market,
    parse_num_list,
    parse_str_list,
    write_results,
)


DEFAULT_TIMEFRAMES = ["30s", "1m","3m"]
DEFAULT_LENGTHS = [12, 16, 20, 34]
DEFAULT_MODES = ["normal", "invert"]
DEFAULT_TP = [0,200,300,400]
DEFAULT_CUTS = [200,300,400]
DEFAULT_SESSIONS = [-1, 1]
DEFAULT_HOLDS = [0]
DEFAULT_EXIT_MULTS = [0.5, 0.75, 1.0]
DEFAULT_TRAILS = [0]



def donchian_state(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    length: int,
    mode: str,
    exit_mult: float = 1.0,
) -> np.ndarray:
    upper = rolling_high_prev(highs, length)
    lower = rolling_low_prev(lows, length)
    mid = (upper + lower) / 2.0
    upper = mid + (upper - mid) * exit_mult
    lower = mid - (mid - lower) * exit_mult
    state = np.zeros(len(closes), dtype=np.float64)
    state[closes > upper] = 1.0
    state[closes < lower] = -1.0
    state[~np.isfinite(upper) | ~np.isfinite(lower)] = 0.0
    if mode == "invert":
        state = -state
    return state


def main() -> None:
    ap = build_parser("Donchian breakout/fade sweep", "forex_donchian_signal_results.csv")
    ap.add_argument("--lengths", default=None)
    ap.add_argument("--modes", default=None, help="normal,invert")
    ap.add_argument("--loss-cut-points", default=None, help="hard loss cut in points; 0 disables")
    ap.add_argument("--sessions", default=None, help="-1=outside sessions, 0=all hours, 1=inside Tokyo/London/New York sessions")
    ap.add_argument("--max-hold-minutes", default=None, help="close only negative trades after this age; 0 disables")
    ap.add_argument("--exit-mult", default=None, help="opposite-signal exit threshold multiplier; 1 keeps normal channel")
    ap.add_argument("--trail-points", default=None, help="TP-armed trailing giveback in points; 0 keeps fixed TP")
    ap.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    args = ap.parse_args()

    timeframes = parse_str_list(args.timeframes, DEFAULT_TIMEFRAMES)
    lengths = [int(x) for x in parse_num_list(args.lengths, DEFAULT_LENGTHS)]
    modes = parse_str_list(args.modes, DEFAULT_MODES)
    tp_values = parse_num_list(args.tp_points, DEFAULT_TP)
    cut_values = parse_num_list(args.loss_cut_points or args.sl_points, DEFAULT_CUTS)
    sessions = [int(x) for x in parse_num_list(args.sessions, DEFAULT_SESSIONS)]
    holds = parse_num_list(args.max_hold_minutes, DEFAULT_HOLDS)
    exit_mults = parse_num_list(args.exit_mult, DEFAULT_EXIT_MULTS)
    trails = parse_num_list(args.trail_points, DEFAULT_TRAILS)

    ticks, t0 = load_market(args)
    results = []
    combo_count = len(timeframes) * len(lengths) * len(modes) * len(tp_values) * len(cut_values) * len(sessions) * len(holds) * len(exit_mults) * len(trails)
    print(f"[donchian] combos_per_pair={combo_count} workers={args.workers}", flush=True)

    for pair, g in ticks.groupby("pair", sort=False):
        g = g.sort_values("timestamp")
        bid = g["bid"].to_numpy(np.float64)
        ask = g["ask"].to_numpy(np.float64)
        ts_ns = g["timestamp"].astype("int64").to_numpy(np.int64)
        point_size = float(args.point_size or default_point_size(pair))
        print(f"[donchian] {pair} ticks={len(g):,}", flush=True)

        state_cache = {}
        for tf in timeframes:
            _, highs, lows, closes, close_idx = build_bid_ohlc(bid, ts_ns, tf)
            for length in lengths:
                for mode in modes:
                    key = (tf, length, mode, 1.0)
                    state_cache[key] = map_state_to_ticks(
                        len(bid), close_idx, donchian_state(closes, highs, lows, length, mode, 1.0)
                    )
                    for exit_mult in exit_mults:
                        exit_key = (tf, length, mode, float(exit_mult))
                        if exit_key not in state_cache:
                            state_cache[exit_key] = map_state_to_ticks(
                                len(bid), close_idx,
                                donchian_state(closes, highs, lows, length, mode, float(exit_mult)),
                            )
        session_cache = {s: active_session_allowed(ts_ns, s) for s in sessions}

        combos = list(product(timeframes, lengths, modes, tp_values, cut_values, sessions, holds, exit_mults, trails))

        def run_combo(combo):
            tf, length, mode, tp, cut, sess, hold, exit_mult, trail = combo
            params = f"length={length};mode={mode};session={sess};hold={hold:g};cut={cut:g};exit_mult={exit_mult:g};trail={trail:g}"
            return simulate_state_strategy(
                pair, "donchian", params, tf, bid, ask, ts_ns, state_cache[(tf, length, mode, 1.0)],
                state_cache[(tf, length, mode, float(exit_mult))],
                session_cache[int(sess)], tp, trail, cut, hold, point_size, args.amount, args.compound,
                args.leverage, args.commission_per_million, args.side,
            )

        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_combo, c) for c in combos]
            for fut in as_completed(futs):
                results.append(fut.result())
                done += 1
                if done % max(len(combos) // 10, 1) == 0:
                    print(f"[donchian] {pair} progress {done}/{len(combos)}", flush=True)

    filtered = [r for r in results if r.trades >= args.min_trades]
    write_results(args.out, filtered, args.top, args.sort_by)
    print(f"[donchian] wrote {args.out} elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
