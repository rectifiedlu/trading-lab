"""RSI direction/fade sweep with candle RSI signals and tick execution."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

import numpy as np

from forex_signal_sweep_common import build_bid_ohlc, map_state_to_ticks, rma, simulate_state_strategy
from forex_strategy_common import active_session_allowed, build_parser, default_point_size, load_market, parse_num_list, parse_str_list, write_results


DEFAULT_TIMEFRAMES = ["15s","30s", "1m"]
DEFAULT_PERIODS = [5, 7, 10, 14]
DEFAULT_KINDS = ["rsix"]
DEFAULT_MODES = ["normal"]
DEFAULT_TP = [0,200, 400,600,800]
DEFAULT_CUTS = [0,200,400,600,800]
DEFAULT_SESSIONS = [-1, 1]
DEFAULT_HOLDS = [0, 60]


def rsi_state(closes: np.ndarray, period: int, kind: str, mode: str) -> np.ndarray:
    delta = np.empty(len(closes), dtype=np.float64)
    delta[0] = 0.0
    delta[1:] = closes[1:] - closes[:-1]
    gains = np.maximum(delta, 0.0)
    losses = np.maximum(-delta, 0.0)
    avg_gain = rma(gains, period)
    avg_loss = rma(losses, period)
    rs = avg_gain / np.maximum(avg_loss, 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    state = np.zeros(len(closes), dtype=np.float64)
    if kind == "rsi50":
        state[rsi > 50.0] = 1.0
        state[rsi < 50.0] = -1.0
    elif kind == "rsix":
        state[rsi < 30.0] = 1.0
        state[rsi > 70.0] = -1.0
    else:
        raise ValueError(f"unknown RSI kind: {kind}")
    state[~np.isfinite(rsi)] = 0.0
    if mode == "invert":
        state = -state
    return state


def main() -> None:
    ap = build_parser("RSI signal sweep", "forex_rsi_signal_results.csv")
    ap.add_argument("--periods", default=None)
    ap.add_argument("--rsi-kinds", default=None, help="rsi50,rsix")
    ap.add_argument("--modes", default=None, help="normal,invert")
    ap.add_argument("--loss-cut-points", default=None, help="hard loss cut in points; 0 disables")
    ap.add_argument("--sessions", default=None, help="-1=outside sessions, 0=all hours, 1=inside Tokyo/London/New York sessions")
    ap.add_argument("--max-hold-minutes", default=None, help="close only negative trades after this age; 0 disables")
    ap.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    args = ap.parse_args()

    timeframes = parse_str_list(args.timeframes, DEFAULT_TIMEFRAMES)
    periods = [int(x) for x in parse_num_list(args.periods, DEFAULT_PERIODS)]
    kinds = parse_str_list(args.rsi_kinds, DEFAULT_KINDS)
    modes = parse_str_list(args.modes, DEFAULT_MODES)
    tp_values = parse_num_list(args.tp_points, DEFAULT_TP)
    cut_values = parse_num_list(args.loss_cut_points or args.sl_points, DEFAULT_CUTS)
    sessions = [int(x) for x in parse_num_list(args.sessions, DEFAULT_SESSIONS)]
    holds = parse_num_list(args.max_hold_minutes, DEFAULT_HOLDS)

    ticks, t0 = load_market(args)
    results = []
    combo_count = len(timeframes) * len(periods) * len(kinds) * len(modes) * len(tp_values) * len(cut_values) * len(sessions) * len(holds)
    print(f"[rsi] combos_per_pair={combo_count} workers={args.workers}", flush=True)

    for pair, g in ticks.groupby("pair", sort=False):
        g = g.sort_values("timestamp")
        bid = g["bid"].to_numpy(np.float64)
        ask = g["ask"].to_numpy(np.float64)
        ts_ns = g["timestamp"].astype("int64").to_numpy(np.int64)
        point_size = float(args.point_size or default_point_size(pair))
        print(f"[rsi] {pair} ticks={len(g):,}", flush=True)

        state_cache = {}
        for tf in timeframes:
            _, _, _, closes, close_idx = build_bid_ohlc(bid, ts_ns, tf)
            for period in periods:
                for kind in kinds:
                    for mode in modes:
                        key = (tf, period, kind, mode)
                        state_cache[key] = map_state_to_ticks(
                            len(bid), close_idx, rsi_state(closes, period, kind, mode)
                        )
        session_cache = {s: active_session_allowed(ts_ns, s) for s in sessions}
        combos = list(product(timeframes, periods, kinds, modes, tp_values, cut_values, sessions, holds))

        def run_combo(combo):
            tf, period, kind, mode, tp, cut, sess, hold = combo
            params = f"kind={kind};period={period};mode={mode};session={sess};hold={hold:g};cut={cut:g}"
            return simulate_state_strategy(
                pair, "rsi", params, tf, bid, ask, ts_ns, state_cache[(tf, period, kind, mode)],
                state_cache[(tf, period, kind, mode)],
                session_cache[int(sess)], tp, 0.0, cut, hold, point_size, args.amount, args.compound,
                args.leverage, args.commission_per_million, args.side,
            )

        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_combo, c) for c in combos]
            for fut in as_completed(futs):
                results.append(fut.result())
                done += 1
                if done % max(len(combos) // 10, 1) == 0:
                    print(f"[rsi] {pair} progress {done}/{len(combos)}", flush=True)

    filtered = [r for r in results if r.trades >= args.min_trades]
    write_results(args.out, filtered, args.top, args.sort_by)
    print(f"[rsi] wrote {args.out} elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
