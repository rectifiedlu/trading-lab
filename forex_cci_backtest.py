"""CCI directional/fade RR sweep with candle signals and tick TP/SL execution."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

import numpy as np

from forex_signal_sweep_common import build_bid_ohlc, map_state_to_ticks, rma, simulate_state_strategy
from forex_strategy_common import (
    active_session_allowed,
    build_parser,
    default_point_size,
    load_market,
    parse_num_list,
    parse_str_list,
    write_results,
)
from forex_supertrend_backtest import map_values_to_ticks, simulate_supertrend_rr


DEFAULT_TIMEFRAMES = ["30s", "1m", "3m"]
DEFAULT_LENGTHS = [14, 20, 34]
DEFAULT_THRESHOLDS = [100, 150, 200]
DEFAULT_RISK_MULTS = [1.0, 1.5, 2.0]
DEFAULT_MODES = ["normal", "invert"]
DEFAULT_RR = [1.5, 2.0, 3.0]
DEFAULT_MIN_RISK_POINTS = [100]
DEFAULT_FLIP_EXIT = [0, 1]
DEFAULT_FIXED_TP = [200, 300, 400, 500]
DEFAULT_FIXED_SL = [200, 300, 400]
DEFAULT_SESSIONS = [-1, 1]


def rolling_sma(x: np.ndarray, length: int) -> np.ndarray:
    out = np.full(len(x), np.nan, dtype=np.float64)
    if len(x) < length:
        return out
    csum = np.cumsum(np.insert(x, 0, 0.0))
    out[length - 1:] = (csum[length:] - csum[:-length]) / length
    return out


def rolling_mean_dev(x: np.ndarray, mean: np.ndarray, length: int) -> np.ndarray:
    out = np.full(len(x), np.nan, dtype=np.float64)
    for i in range(length - 1, len(x)):
        m = mean[i]
        if np.isfinite(m):
            out[i] = np.mean(np.abs(x[i - length + 1:i + 1] - m))
    return out


def cci_state(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    length: int,
    threshold: float,
    risk_mult: float,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    typical = (highs + lows + closes) / 3.0
    ma = rolling_sma(typical, length)
    md = rolling_mean_dev(typical, ma, length)
    cci = (typical - ma) / np.maximum(0.015 * md, 1e-12)

    prev_close = np.empty_like(closes)
    prev_close[0] = closes[0]
    prev_close[1:] = closes[:-1]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    atr = rma(tr, length)

    state = np.zeros(len(closes), dtype=np.float64)
    state[cci >= threshold] = 1.0
    state[cci <= -threshold] = -1.0
    if mode == "invert":
        state = -state

    risk = np.maximum(atr * risk_mult, 0.0)
    line = closes.copy()
    line[state > 0] = closes[state > 0] - risk[state > 0]
    line[state < 0] = closes[state < 0] + risk[state < 0]
    line[state == 0] = np.nan
    return state, line, atr


def main() -> None:
    ap = build_parser("CCI dynamic RR sweep", "forex_cci_results.csv")
    ap.add_argument("--lengths", default=None)
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--risk-mult", default=None)
    ap.add_argument("--modes", default=None, help="normal,invert")
    ap.add_argument("--rr", default=None)
    ap.add_argument("--min-risk-points", default=None)
    ap.add_argument("--flip-exit", default=None)
    ap.add_argument("--exit-style", choices=["rr", "fixed"], default="rr")
    ap.add_argument("--sessions", default=None, help="-1=outside sessions, 0=all hours, 1=inside Tokyo/London/New York sessions")
    ap.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    args = ap.parse_args()

    timeframes = parse_str_list(args.timeframes, DEFAULT_TIMEFRAMES)
    lengths = [int(x) for x in parse_num_list(args.lengths, DEFAULT_LENGTHS)]
    thresholds = parse_num_list(args.thresholds, DEFAULT_THRESHOLDS)
    risk_mults = parse_num_list(args.risk_mult, DEFAULT_RISK_MULTS)
    modes = parse_str_list(args.modes, DEFAULT_MODES)
    rrs = parse_num_list(args.rr, DEFAULT_RR)
    min_risks = parse_num_list(args.min_risk_points, DEFAULT_MIN_RISK_POINTS)
    flip_exits = [int(x) for x in parse_num_list(args.flip_exit, DEFAULT_FLIP_EXIT)]
    fixed_tps = parse_num_list(args.tp_points, DEFAULT_FIXED_TP)
    fixed_sls = parse_num_list(args.sl_points, DEFAULT_FIXED_SL)
    sessions = [int(x) for x in parse_num_list(args.sessions, DEFAULT_SESSIONS)]

    ticks, t0 = load_market(args)
    results = []
    if args.exit_style == "fixed":
        combo_count = (
            len(timeframes) * len(lengths) * len(thresholds) * len(risk_mults)
            * len(modes) * len(fixed_tps) * len(fixed_sls) * len(sessions)
        )
    else:
        combo_count = (
            len(timeframes) * len(lengths) * len(thresholds) * len(risk_mults)
            * len(modes) * len(rrs) * len(min_risks) * len(flip_exits) * len(sessions)
        )
    print(f"[cci] combos_per_pair={combo_count} workers={args.workers}", flush=True)

    for pair, g in ticks.groupby("pair", sort=False):
        g = g.sort_values("timestamp")
        bid = g["bid"].to_numpy(np.float64)
        ask = g["ask"].to_numpy(np.float64)
        ts_ns = g["timestamp"].astype("int64").to_numpy(np.int64)
        point_size = float(args.point_size or default_point_size(pair))
        print(f"[cci] {pair} ticks={len(g):,}", flush=True)
        session_cache = {s: active_session_allowed(ts_ns, s) for s in sessions}

        candle_cache = {tf: build_bid_ohlc(bid, ts_ns, tf) for tf in timeframes}
        if args.exit_style == "fixed":
            combos = list(product(timeframes, lengths, thresholds, risk_mults, modes, fixed_tps, fixed_sls, sessions))
        else:
            combos = list(product(timeframes, lengths, thresholds, risk_mults, modes, rrs, min_risks, flip_exits, sessions))

        def run_combo(combo):
            if args.exit_style == "fixed":
                tf, length, threshold, risk_mult, mode, tp, sl, sess = combo
            else:
                tf, length, threshold, risk_mult, mode, rr, min_risk, flip, sess = combo
            _, highs, lows, closes, close_idx = candle_cache[tf]
            state_c, line_c, atr_c = cci_state(highs, lows, closes, length, threshold, risk_mult, mode)
            state = map_state_to_ticks(len(bid), close_idx, state_c)
            if args.exit_style == "fixed":
                params = f"length={length};threshold={threshold:g};risk_mult={risk_mult:g};mode={mode};exit=fixed;session={sess}"
                return simulate_state_strategy(
                    pair, "cci", params, tf, bid, ask, ts_ns, state, state,
                    session_cache[int(sess)], tp, 0.0, sl, 0.0, point_size,
                    args.amount, args.compound, args.leverage,
                    args.commission_per_million, args.side,
                )
            line = map_values_to_ticks(len(bid), close_idx, line_c)
            atr = map_values_to_ticks(len(bid), close_idx, atr_c)
            params = (
                f"length={length};threshold={threshold:g};risk_mult={risk_mult:g};"
                f"mode={mode};rr={rr:g};minrisk={min_risk:g};flip={flip};session={sess}"
            )
            return simulate_supertrend_rr(
                pair, "cci", params, tf, bid, ask, state, line, atr, ts_ns, session_cache[int(sess)], rr, min_risk,
                point_size, args.amount, args.compound, args.leverage,
                args.commission_per_million, args.side, flip,
            )

        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_combo, c) for c in combos]
            for fut in as_completed(futs):
                results.append(fut.result())
                done += 1
                if done % max(len(combos) // 10, 1) == 0:
                    print(f"[cci] {pair} progress {done}/{len(combos)}", flush=True)

    filtered = [r for r in results if r.trades >= args.min_trades]
    write_results(args.out, filtered, args.top, args.sort_by)
    print(f"[cci] wrote {args.out} elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
