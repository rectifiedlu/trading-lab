"""Session-filtered price/EMA cross backtest with fresh-reset signals."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

import numpy as np

from forex_signal_sweep_common import build_bid_ohlc, ema, map_state_to_ticks, simulate_state_strategy
from forex_strategy_common import (
    active_session_allowed,
    build_parser,
    default_point_size,
    load_market,
    parse_num_list,
    parse_str_list,
    timeframe_to_ns,
    write_results,
)


DEFAULT_TIMEFRAMES = ["30s", "1m"]
DEFAULT_EMA_LENGTHS = [
    21, 34, 50, 89, 100, 144, 150, 200, 233, 300, 325, 377
]
DEFAULT_ENTRY_MODES = ["candle","tick"]
DEFAULT_EXIT_MODES = ["tick"]
DEFAULT_CONFIRMS = [1,2]
DEFAULT_MODES = ["normal","invert"]
DEFAULT_TP = [75,100,150,175]
DEFAULT_SL = [75,100,150,175]
DEFAULT_TP_MODES = ["fixed"]
DEFAULT_TRAILS = [0]
DEFAULT_REVERSE_ON_FLIP = [0]
DEFAULT_SESSIONS = [-1,0, 1]


def fresh_confirm_state(values: np.ndarray, basis: np.ndarray, confirm: int) -> np.ndarray:
    state = np.zeros(len(values), dtype=np.float64)
    above_count = 0
    below_count = 0
    active_side = 0
    for i, (px, ma) in enumerate(zip(values, basis)):
        if not np.isfinite(ma):
            above_count = 0
            below_count = 0
            active_side = 0
            continue
        side = 1 if px > ma else (-1 if px < ma else 0)
        if side == 1:
            above_count += 1
            below_count = 0
        elif side == -1:
            below_count += 1
            above_count = 0
        else:
            above_count = 0
            below_count = 0
            active_side = 0
        if side != active_side and side != 0:
            active_side = 0
        if above_count >= confirm and active_side != 1:
            state[i] = 1.0
            active_side = 1
        elif below_count >= confirm and active_side != -1:
            state[i] = -1.0
            active_side = -1
    return state


def state_from_events(n_ticks: int, event_idx: np.ndarray, event_state: np.ndarray) -> np.ndarray:
    out = np.zeros(n_ticks, dtype=np.float64)
    for idx, st in zip(event_idx, event_state):
        if st != 0.0:
            out[int(idx)] = st
    return out


def candle_open_to_ticks(open_px: np.ndarray, close_idx: np.ndarray, n_ticks: int) -> np.ndarray:
    out = np.zeros(n_ticks, dtype=np.float64)
    prev = 0
    last = np.nan
    for idx, op in zip(close_idx, open_px):
        out[prev:idx + 1] = last
        last = float(op)
        prev = int(idx) + 1
    out[prev:] = last
    return out


def tick_ema(price: np.ndarray, length: int) -> np.ndarray:
    return ema(price, length)


def build_signal_state(
    bid: np.ndarray,
    ts_ns: np.ndarray,
    timeframe: str,
    ema_length: int,
    mode: str,
    confirm: int,
) -> np.ndarray:
    mid = bid
    _, _, _, closes, close_idx = build_bid_ohlc(mid, ts_ns, timeframe)
    basis_c = ema(closes, ema_length)
    basis_by_tick = np.zeros(len(bid), dtype=np.float64)
    prev = 0
    last = np.nan
    for idx, val in zip(close_idx, basis_c):
        basis_by_tick[prev:idx + 1] = last
        last = float(val)
        prev = idx + 1
    basis_by_tick[prev:] = last

    if mode == "candle":
        event_state = fresh_confirm_state(closes, basis_c, confirm)
        return state_from_events(len(bid), close_idx, event_state)
    basis_tick = tick_ema(bid, ema_length)
    raw = np.zeros(len(bid), dtype=np.float64)
    prev_side = 0
    for i in range(len(bid)):
        ma = basis_tick[i]
        side = 1 if bid[i] > ma else (-1 if bid[i] < ma else 0)
        if side != 0 and side != prev_side:
            raw[i] = float(side)
            prev_side = side
    return raw


def main() -> None:
    ap = build_parser("Session-filtered price EMA cross", "forex_ema_session_cross_results.csv")
    ap.add_argument("--ema-lengths", default=None)
    ap.add_argument("--entry-modes", default=None, help="candle,tick")
    ap.add_argument("--exit-modes", default=None, help="candle,tick")
    ap.add_argument("--confirm-candles", default=None)
    ap.add_argument("--mode", default=None, help="normal,invert")
    ap.add_argument("--tp-mode", default=None, help="fixed,trail_arm")
    ap.add_argument("--trail-points", default=None, help="Used only when tp-mode includes trail_arm")
    ap.add_argument("--reverse-on-flip", default=None, help="0,1. If 1, opposite signal closes and opens reverse side.")
    ap.add_argument("--sessions", default=None, help="-1=outside sessions, 0=all hours, 1=inside Tokyo/London/New York sessions")
    ap.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    args = ap.parse_args()

    timeframes = parse_str_list(args.timeframes, DEFAULT_TIMEFRAMES)
    ema_lengths = [int(x) for x in parse_num_list(args.ema_lengths, DEFAULT_EMA_LENGTHS)]
    entry_modes = parse_str_list(args.entry_modes, DEFAULT_ENTRY_MODES)
    exit_modes = parse_str_list(args.exit_modes, DEFAULT_EXIT_MODES)
    confirms = [int(x) for x in parse_num_list(args.confirm_candles, DEFAULT_CONFIRMS)]
    modes = parse_str_list(args.mode, DEFAULT_MODES)
    tps = parse_num_list(args.tp_points, DEFAULT_TP)
    sls = parse_num_list(args.sl_points, DEFAULT_SL)
    tp_modes = parse_str_list(args.tp_mode, DEFAULT_TP_MODES)
    trails = parse_num_list(args.trail_points, DEFAULT_TRAILS)
    reverse_values = [int(x) for x in parse_num_list(args.reverse_on_flip, DEFAULT_REVERSE_ON_FLIP)]
    sessions = [int(x) for x in parse_num_list(args.sessions, DEFAULT_SESSIONS)]

    ticks, t0 = load_market(args)
    results = []

    for pair, g in ticks.groupby("pair", sort=False):
        g = g.sort_values("timestamp")
        bid = g["bid"].to_numpy(np.float64)
        ask = g["ask"].to_numpy(np.float64)
        ts_ns = g["timestamp"].astype("int64").to_numpy(np.int64)
        point_size = float(args.point_size or default_point_size(pair))
        session_cache = {s: active_session_allowed(ts_ns, s) for s in sessions}
        print(f"[ema-session] {pair} ticks={len(g):,}", flush=True)

        signal_cache = {}
        candle_open_cache = {}
        combos = []
        for tf, ema_len, entry_mode, exit_mode, mode, tp, sl, tp_mode, reverse_on_flip, sess in product(
            timeframes, ema_lengths, entry_modes, exit_modes, modes, tps, sls, tp_modes, reverse_values, sessions
        ):
            if mode not in {"normal", "invert"}:
                raise ValueError(f"unsupported mode: {mode}")
            if tp_mode not in {"fixed", "trail_arm"}:
                raise ValueError(f"unsupported tp-mode: {tp_mode}")
            trail_values = [0.0] if tp_mode == "fixed" else trails
            entry_confirms = confirms if entry_mode == "candle" else [1]
            for confirm, trail in product(entry_confirms, trail_values):
                if tp_mode == "trail_arm" and (tp <= 0.0 or trail <= 0.0):
                    continue
                combos.append((
                    tf, ema_len, entry_mode, exit_mode, mode, confirm, tp, sl, tp_mode, trail, reverse_on_flip, sess,
                ))

        print(f"[ema-session] combos_per_pair={len(combos)} workers={args.workers}", flush=True)

        def get_state(tf: str, ema_len: int, mode: str, confirm: int) -> np.ndarray:
            key = (tf, ema_len, mode, confirm if mode == "candle" else 1)
            if key not in signal_cache:
                signal_cache[key] = build_signal_state(bid, ts_ns, tf, ema_len, mode, key[3])
            return signal_cache[key]

        def get_candle_open(tf: str) -> np.ndarray:
            if tf not in candle_open_cache:
                opens, _, _, _, close_idx = build_bid_ohlc(bid, ts_ns, tf)
                candle_open_cache[tf] = candle_open_to_ticks(opens, close_idx, len(bid))
            return candle_open_cache[tf]

        def run_combo(combo):
            (
                tf, ema_len, entry_mode, exit_mode, mode, confirm, tp, sl, tp_mode, trail, reverse_on_flip, sess,
            ) = combo
            entry_state = get_state(tf, ema_len, entry_mode, confirm)
            exit_state = get_state(tf, ema_len, exit_mode, confirm if exit_mode == "candle" else 1)
            if mode == "invert":
                entry_state = -entry_state
                exit_state = -exit_state
            trail_from_candle_open = tp_mode == "trail_arm"
            candle_open = get_candle_open(tf) if trail_from_candle_open else None
            params = (
                f"ema={ema_len};entry={entry_mode};exit={exit_mode};mode={mode};confirm={confirm};"
                f"session={sess};tp_mode={tp_mode};trail={trail:g};reverse={reverse_on_flip}"
            )
            return simulate_state_strategy(
                pair, "ema_session", params, tf, bid, ask, ts_ns,
                entry_state, exit_state, session_cache[int(sess)], tp, trail, sl, 0.0, point_size,
                args.amount, args.compound, args.leverage,
                args.commission_per_million, args.side,
                candle_open=candle_open,
                trail_from_candle_open=trail_from_candle_open,
                reverse_on_signal=bool(reverse_on_flip),
                ignore_signal_exit_when_bracket=True,
            )

        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_combo, c) for c in combos]
            for fut in as_completed(futs):
                results.append(fut.result())
                done += 1
                if done % max(len(combos) // 10, 1) == 0:
                    print(f"[ema-session] progress {done}/{len(combos)}", flush=True)

    filtered = [r for r in results if r.trades >= args.min_trades]
    write_results(args.out, filtered, args.top, args.sort_by)
    print(f"[ema-session] wrote {args.out} elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
