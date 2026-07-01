"""Keltner channel breakout/fade sweep with candle signals and tick execution."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

import numpy as np

from forex_signal_sweep_common import build_bid_ohlc, ema, map_state_to_ticks, rma, simulate_state_strategy
from forex_strategy_common import (
    active_session_allowed,
    build_parser,
    default_point_size,
    load_market,
    parse_num_list,
    parse_str_list,
    timeframe_to_ns,
    day_ids_from_timestamps,
    write_results,
)


DEFAULT_TIMEFRAMES = ["3m", "30s", "1m"]
DEFAULT_LENGTHS = [
    14, 18, 20,
    24, 28, 34, 40, 48, 55, 64, 75, 105
]
DEFAULT_MULTS = [1.0, 1.5, 2.0,2.5]
DEFAULT_MODES = ["normal","invert"]
DEFAULT_TP = [200,300,400]
DEFAULT_CUTS = [150,200,300,400]
DEFAULT_SESSIONS = [-1, 0, 1]
DEFAULT_HOLDS = [0]
DEFAULT_REBREAKS = [-1]
DEFAULT_TRAILS = [0]
DEFAULT_ENTRY_SIGNAL_MODES = ["candle", "tick"]
DEFAULT_EXIT_SIGNAL_MODES = ["tick"]


def keltner_state(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    length: int,
    mult: float,
    mode: str,
    exit_mult: float = 1.0,
) -> np.ndarray:
    prev_close = np.empty(len(closes), dtype=np.float64)
    prev_close[0] = closes[0]
    prev_close[1:] = closes[:-1]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    center = ema(closes, length)
    atr = rma(tr, length)
    upper = center + mult * exit_mult * atr
    lower = center - mult * exit_mult * atr
    state = np.zeros(len(closes), dtype=np.float64)
    state[closes > upper] = 1.0
    state[closes < lower] = -1.0
    state[~np.isfinite(atr)] = 0.0
    if mode == "invert":
        state = -state
    return state


def keltner_raw_and_bands(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    length: int,
    mult: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prev_close = np.empty(len(closes), dtype=np.float64)
    prev_close[0] = closes[0]
    prev_close[1:] = closes[:-1]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    center = ema(closes, length)
    atr = rma(tr, length)
    upper = center + mult * atr
    lower = center - mult * atr
    raw = np.zeros(len(closes), dtype=np.float64)
    raw[closes > upper] = 1.0
    raw[closes < lower] = -1.0
    raw[~np.isfinite(atr)] = 0.0
    return raw, upper, lower


def candle_open_to_ticks(price: np.ndarray, ts_ns: np.ndarray, timeframe: str) -> np.ndarray:
    tf_ns = timeframe_to_ns(timeframe)
    bucket = ts_ns // tf_ns
    out = np.empty(len(price), dtype=np.float64)
    cur_bucket = int(bucket[0])
    cur_open = float(price[0])
    for i, px in enumerate(price):
        b = int(bucket[i])
        if b != cur_bucket:
            cur_bucket = b
            cur_open = float(px)
        out[i] = cur_open
    return out


def main() -> None:
    ap = build_parser("Keltner channel signal sweep", "forex_keltner_signal_results.csv")
    ap.add_argument("--lengths", default=None)
    ap.add_argument("--mults", default=None)
    ap.add_argument("--modes", default=None, help="normal,invert")
    ap.add_argument("--loss-cut-points", default=None, help="hard loss cut in points; 0 disables")
    ap.add_argument("--sessions", default=None, help="0=all hours, 1=Tokyo/London/New York session entries only")
    ap.add_argument("--max-hold-minutes", default=None, help="close only negative trades after this age; 0 disables")
    ap.add_argument("--rebreak-points", default=None,
                    help="-1 disables; otherwise close after return-inside then adverse band rebreak by N points")
    ap.add_argument("--signal-mode", default=None,
                    help="compat shortcut: sets both entry and exit signal modes")
    ap.add_argument("--entry-signal-mode", default=None,
                    help="candle only for now: entries are triggered by closed candle state changes")
    ap.add_argument("--exit-signal-mode", default=None,
                    help="tick,candle; tick exits live from mapped candle signals, candle exits only on closes")
    ap.add_argument("--trail-points", default=None, help="TP-armed trailing giveback in points; 0 keeps fixed TP")
    ap.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    args = ap.parse_args()

    timeframes = parse_str_list(args.timeframes, DEFAULT_TIMEFRAMES)
    lengths = [int(x) for x in parse_num_list(args.lengths, DEFAULT_LENGTHS)]
    mults = parse_num_list(args.mults, DEFAULT_MULTS)
    modes = parse_str_list(args.modes, DEFAULT_MODES)
    tp_values = parse_num_list(args.tp_points, DEFAULT_TP)
    cut_values = parse_num_list(args.loss_cut_points or args.sl_points, DEFAULT_CUTS)
    sessions = [int(x) for x in parse_num_list(args.sessions, DEFAULT_SESSIONS)]
    holds = parse_num_list(args.max_hold_minutes, DEFAULT_HOLDS)
    rebreaks = parse_num_list(args.rebreak_points, DEFAULT_REBREAKS)
    trails = parse_num_list(args.trail_points, DEFAULT_TRAILS)
    entry_signal_modes = parse_str_list(args.entry_signal_mode, DEFAULT_ENTRY_SIGNAL_MODES)
    exit_signal_modes = parse_str_list(
        args.exit_signal_mode or args.signal_mode,
        DEFAULT_EXIT_SIGNAL_MODES,
    )

    ticks, t0 = load_market(args)
    results = []
    combo_count = len(timeframes) * len(lengths) * len(mults) * len(modes) * len(tp_values) * len(cut_values) * len(sessions) * len(holds) * len(rebreaks) * len(trails) * len(entry_signal_modes) * len(exit_signal_modes)
    print(f"[keltner] combos_per_pair={combo_count} workers={args.workers}", flush=True)

    for pair, g in ticks.groupby("pair", sort=False):
        g = g.sort_values("timestamp")
        bid = g["bid"].to_numpy(np.float64)
        ask = g["ask"].to_numpy(np.float64)
        ts_ns = g["timestamp"].astype("int64").to_numpy(np.int64)
        point_size = float(args.point_size or default_point_size(pair))
        print(f"[keltner] {pair} ticks={len(g):,}", flush=True)

        state_cache = {}
        band_cache = {}
        candle_cache = {}
        candle_open_cache = {}
        for tf in timeframes:
            _, highs, lows, closes, close_idx = build_bid_ohlc(bid, ts_ns, tf)
            candle_days, _ = day_ids_from_timestamps(ts_ns[close_idx])
            candle_cache[tf] = (highs, lows, closes, candle_days)
            candle_open_cache[tf] = candle_open_to_ticks(bid, ts_ns, tf)
            for length in lengths:
                for mult in mults:
                    raw, upper, lower = keltner_raw_and_bands(closes, highs, lows, length, mult)
                    band_cache[(tf, length, mult)] = (
                        map_state_to_ticks(len(bid), close_idx, raw),
                        map_state_to_ticks(len(bid), close_idx, upper),
                        map_state_to_ticks(len(bid), close_idx, lower),
                    )
                    for mode in modes:
                        key = (tf, length, mult, mode, 1.0)
                        state_cache[key] = map_state_to_ticks(
                            len(bid), close_idx, keltner_state(closes, highs, lows, length, mult, mode, 1.0)
                        )
        session_cache = {s: active_session_allowed(ts_ns, s) for s in sessions}
        candle_session_cache = {}
        for tf in timeframes:
            _, _, _, _, close_idx = build_bid_ohlc(bid, ts_ns, tf)
            candle_session_cache[tf] = {
                s: active_session_allowed(ts_ns[close_idx], s) for s in sessions
            }
        combos = list(product(timeframes, lengths, mults, modes, tp_values, cut_values, sessions, holds, rebreaks, trails, entry_signal_modes, exit_signal_modes))

        def run_combo(combo):
            tf, length, mult, mode, tp, cut, sess, hold, rebreak, trail, entry_signal_mode, exit_signal_mode = combo
            raw_state, upper_band, lower_band = band_cache[(tf, length, mult)]
            params = f"length={length};mult={mult:g};mode={mode};session={sess};hold={hold:g};cut={cut:g};rebreak={rebreak:g};trail={trail:g};trail_src=candle_open;entry_sig={entry_signal_mode};exit_sig={exit_signal_mode}"
            if entry_signal_mode == "candle" and exit_signal_mode == "candle":
                highs, lows, closes, candle_days = candle_cache[tf]
                raw_c, upper_c, lower_c = keltner_raw_and_bands(closes, highs, lows, length, mult)
                state_c = -raw_c if mode == "invert" else raw_c
                return simulate_state_strategy(
                    pair, "keltner", params, tf, bid, ask, ts_ns, state_c, state_c,
                    candle_session_cache[tf][int(sess)], tp, trail, cut, hold,
                    point_size, args.amount, args.compound, args.leverage,
                    args.commission_per_million, args.side,
                    raw_state=raw_c,
                    upper_band=upper_c,
                    lower_band=lower_c,
                    rebreak_points=rebreak,
                    signal_mode="candle",
                    close_px=closes,
                    candle_day_id=candle_days,
                )
            if entry_signal_mode == "tick" and exit_signal_mode == "candle":
                highs, lows, closes, candle_days = candle_cache[tf]
                raw_c, upper_c, lower_c = keltner_raw_and_bands(closes, highs, lows, length, mult)
                state_c = -raw_c if mode == "invert" else raw_c
                return simulate_state_strategy(
                    pair, "keltner", params, tf, bid, ask, ts_ns, state_c, state_c,
                    candle_session_cache[tf][int(sess)], tp, trail, cut, hold,
                    point_size, args.amount, args.compound, args.leverage,
                    args.commission_per_million, args.side,
                    raw_state=raw_c,
                    upper_band=upper_c,
                    lower_band=lower_c,
                    rebreak_points=rebreak,
                    signal_mode="candle",
                    close_px=closes,
                    candle_day_id=candle_days,
                )
            if exit_signal_mode != "tick":
                raise ValueError("exit_signal_mode must be 'tick' or 'candle'")
            return simulate_state_strategy(
                pair, "keltner", params, tf, bid, ask, ts_ns, state_cache[(tf, length, mult, mode, 1.0)],
                state_cache[(tf, length, mult, mode, 1.0)],
                session_cache[int(sess)], tp, trail, cut, hold, point_size, args.amount, args.compound,
                args.leverage, args.commission_per_million, args.side,
                candle_open=candle_open_cache[tf],
                trail_from_candle_open=True,
                raw_state=raw_state,
                upper_band=upper_band,
                lower_band=lower_band,
                rebreak_points=rebreak,
                signal_mode="tick",
            )

        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_combo, c) for c in combos]
            for fut in as_completed(futs):
                results.append(fut.result())
                done += 1
                if done % max(len(combos) // 10, 1) == 0:
                    print(f"[keltner] {pair} progress {done}/{len(combos)}", flush=True)

    filtered = [r for r in results if r.trades >= args.min_trades]
    write_results(args.out, filtered, args.top, args.sort_by)
    print(f"[keltner] wrote {args.out} elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
