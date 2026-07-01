"""Keltner signal with dynamic ATR RR exits."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

import numpy as np

from forex_keltner_signal_backtest import keltner_raw_and_bands, keltner_state
from forex_signal_sweep_common import build_bid_ohlc, map_state_to_ticks, rma
from forex_strategy_common import (
    build_parser,
    default_point_size,
    load_market,
    parse_num_list,
    parse_str_list,
    write_results,
)
from forex_supertrend_backtest import map_values_to_ticks, simulate_supertrend_rr


DEFAULT_TIMEFRAMES = ["30s", "1m", "3m"]
DEFAULT_LENGTHS = [20, 28, 34, 48, 55, 64]
DEFAULT_MULTS = [1.0, 1.5, 2.0, 2.5]
DEFAULT_RISK_MULTS = [1.0, 1.5, 2.0]
DEFAULT_MODES = ["invert"]
DEFAULT_RR = [1.5, 2.0, 3.0]
DEFAULT_MIN_RISK_POINTS = [100]
DEFAULT_FLIP_EXIT = [0, 1]


def keltner_rr_state(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    length: int,
    mult: float,
    risk_mult: float,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prev_close = np.empty(len(closes), dtype=np.float64)
    prev_close[0] = closes[0]
    prev_close[1:] = closes[:-1]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    atr = rma(tr, length)
    state = keltner_state(closes, highs, lows, length, mult, mode, 1.0)

    risk = np.maximum(atr * risk_mult, 0.0)
    line = closes.copy()
    line[state > 0] = closes[state > 0] - risk[state > 0]
    line[state < 0] = closes[state < 0] + risk[state < 0]
    line[state == 0] = np.nan
    return state, line, atr


def main() -> None:
    ap = build_parser("Keltner dynamic RR sweep", "forex_keltner_rr_results.csv")
    ap.add_argument("--lengths", default=None)
    ap.add_argument("--mults", default=None)
    ap.add_argument("--risk-mult", default=None)
    ap.add_argument("--modes", default=None, help="normal,invert")
    ap.add_argument("--rr", default=None)
    ap.add_argument("--min-risk-points", default=None)
    ap.add_argument("--flip-exit", default=None)
    ap.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    args = ap.parse_args()

    timeframes = parse_str_list(args.timeframes, DEFAULT_TIMEFRAMES)
    lengths = [int(x) for x in parse_num_list(args.lengths, DEFAULT_LENGTHS)]
    mults = parse_num_list(args.mults, DEFAULT_MULTS)
    risk_mults = parse_num_list(args.risk_mult, DEFAULT_RISK_MULTS)
    modes = parse_str_list(args.modes, DEFAULT_MODES)
    rrs = parse_num_list(args.rr, DEFAULT_RR)
    min_risks = parse_num_list(args.min_risk_points, DEFAULT_MIN_RISK_POINTS)
    flip_exits = [int(x) for x in parse_num_list(args.flip_exit, DEFAULT_FLIP_EXIT)]

    ticks, t0 = load_market(args)
    results = []
    combo_count = (
        len(timeframes) * len(lengths) * len(mults) * len(risk_mults)
        * len(modes) * len(rrs) * len(min_risks) * len(flip_exits)
    )
    print(f"[keltner-rr] combos_per_pair={combo_count} workers={args.workers}", flush=True)

    for pair, g in ticks.groupby("pair", sort=False):
        g = g.sort_values("timestamp")
        bid = g["bid"].to_numpy(np.float64)
        ask = g["ask"].to_numpy(np.float64)
        ts_ns = g["timestamp"].astype("int64").to_numpy(np.int64)
        point_size = float(args.point_size or default_point_size(pair))
        print(f"[keltner-rr] {pair} ticks={len(g):,}", flush=True)

        candle_cache = {tf: build_bid_ohlc(bid, ts_ns, tf) for tf in timeframes}
        combos = list(product(timeframes, lengths, mults, risk_mults, modes, rrs, min_risks, flip_exits))

        def run_combo(combo):
            tf, length, mult, risk_mult, mode, rr, min_risk, flip = combo
            _, highs, lows, closes, close_idx = candle_cache[tf]
            state_c, line_c, atr_c = keltner_rr_state(highs, lows, closes, length, mult, risk_mult, mode)
            state = map_state_to_ticks(len(bid), close_idx, state_c)
            line = map_values_to_ticks(len(bid), close_idx, line_c)
            atr = map_values_to_ticks(len(bid), close_idx, atr_c)
            params = (
                f"length={length};mult={mult:g};risk_mult={risk_mult:g};"
                f"mode={mode};rr={rr:g};minrisk={min_risk:g};flip={flip}"
            )
            return simulate_supertrend_rr(
                pair, "keltner_rr", params, tf, bid, ask, state, line, atr, ts_ns,
                rr, min_risk, point_size, args.amount, args.compound, args.leverage,
                args.commission_per_million, args.side, flip,
            )

        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_combo, c) for c in combos]
            for fut in as_completed(futs):
                results.append(fut.result())
                done += 1
                if done % max(len(combos) // 10, 1) == 0:
                    print(f"[keltner-rr] {pair} progress {done}/{len(combos)}", flush=True)

    filtered = [r for r in results if r.trades >= args.min_trades]
    write_results(args.out, filtered, args.top, args.sort_by)
    print(f"[keltner-rr] wrote {args.out} elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
