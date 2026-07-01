"""Tick-backed MT5-style MACD zero-cross strategy backtest.

Line used for signals:
    macd_line = SMA(EMA(fast) - EMA(slow), macd_sma)

With fast=12, slow=26, macd_sma=1 this is just EMA(12)-EMA(26),
matching the oscillating line the user described from MT5.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
import csv

import numpy as np
import pandas as pd

from forex_strategy_common import (
    DEFAULT_SL_POINTS,
    DEFAULT_TIMEFRAMES,
    DEFAULT_TP_POINTS,
    active_session_allowed,
    build_parser,
    candle_state_to_ticks,
    closed_candle_series,
    day_ids_from_timestamps,
    default_point_size,
    live_candles,
    load_market,
    parse_num_list,
    parse_str_list,
    simulate_triggers,
    write_results,
)
from forex_ema_cross_backtest import simulate_trailing_signals
from forex_ema_pair_cross_backtest import simulate_pair_switch


def ema(x: np.ndarray, length: int) -> np.ndarray:
    out = np.full(len(x), np.nan, dtype=np.float64)
    if length <= 1:
        return x.astype(np.float64)
    alpha = 2.0 / (length + 1.0)
    val = float(x[0])
    for i, px in enumerate(x):
        val = alpha * float(px) + (1.0 - alpha) * val
        out[i] = val
    return out


def sma(x: np.ndarray, length: int) -> np.ndarray:
    if length <= 1:
        return x.astype(np.float64)
    out = np.full(len(x), np.nan, dtype=np.float64)
    csum = np.cumsum(np.insert(x.astype(np.float64), 0, 0.0))
    out[length - 1:] = (csum[length:] - csum[:-length]) / float(length)
    first = np.where(np.isfinite(out))[0]
    if len(first):
        out[:first[0]] = out[first[0]]
    return out


def main() -> None:
    ap = build_parser("MT5-style MACD line tick backtest", "forex_macd_results.csv")
    ap.set_defaults(
        timeframes="30s,1m,2m,3m",
        tp_points="0,300,400,600",
        sl_points="0,300,400,600",
    )
    ap.add_argument("--fast", default="5,8,12")
    ap.add_argument("--slow", default="13,17,26")
    ap.add_argument("--macd-ema", default="1,3,5",
                    help="SMA smoothing applied to fast EMA - slow EMA (MT5 MACD SMA)")
    ap.add_argument("--deadband", type=float, default=0,
                    help="MACD neutral band; positive >= +deadband, negative <= -deadband")
    ap.add_argument("--signal-mode", default="candle",
                    help="comma list: candle=confirmed candle regions, tick=live/forming candle zero crosses")
    ap.add_argument("--mode", default="normal,invert",
                    help="comma list: normal=MACD positive long/negative short; invert=faded direction")
    ap.add_argument("--reverse-on-flip", default="0,",
                    help="comma list: 0=close only, 1=close and enter opposite side on region flip")
    ap.add_argument("--trail-points", default="0",
                    help="0=off; only active when tp=0 and sl=0")
    ap.add_argument("--sessions", default="-1,1",
                    help="-1=outside sessions, 0=all hours, 1=inside Tokyo/London/New York sessions")
    ap.add_argument("--warmup-mult", type=float, default=1.0,
                    help="warmup candles = ceil((slow + signal) * warmup_mult)")
    ap.add_argument("--confirm-candles", type=int, default=1,
                    help="closed MACD candles required in same region before entry")
    ap.add_argument("--trades-out", default=None,
                    help="optional CSV path for per-trade logs")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    timeframes = parse_str_list(args.timeframes, DEFAULT_TIMEFRAMES)
    signal_modes = parse_str_list(args.signal_mode, ["candle"])
    bad_signal_modes = [m for m in signal_modes if m not in ("candle", "tick")]
    if bad_signal_modes:
        raise SystemExit(f"unsupported --signal-mode values: {bad_signal_modes}")
    modes = parse_str_list(args.mode, ["normal", "invert"])
    valid_modes = {"normal", "invert"}
    bad_modes = [m for m in modes if m not in valid_modes]
    if bad_modes:
        raise SystemExit(f"unsupported --mode values: {bad_modes}")
    fasts = [int(x) for x in parse_num_list(args.fast, [5, 6, 8, 10, 12])]
    slows = [int(x) for x in parse_num_list(args.slow, [10, 13, 17, 21, 26])]
    macd_emas = [int(x) for x in parse_num_list(args.macd_ema, [1])]
    reverse_values = [bool(int(x)) for x in parse_num_list(args.reverse_on_flip, [0, 1])]
    tps = parse_num_list(args.tp_points, [0])
    sls = parse_num_list(args.sl_points, [0])
    trails = parse_num_list(args.trail_points, [0])
    sessions = [int(x) for x in parse_num_list(args.sessions, [-1, 1])]

    ticks, t0 = load_market(args)
    results = []
    all_trades = []
    for pair, g in ticks.groupby("pair", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        bid = g["bid"].to_numpy(np.float64)
        ask = g["ask"].to_numpy(np.float64)
        # MT5 native XAUUSD candles match bid OHLC, not mid/ask.
        mid = bid
        ts = g["timestamp"].astype("int64").to_numpy()
        day_id, max_days = day_ids_from_timestamps(ts)
        session_cache = {s: active_session_allowed(ts, s) for s in sessions}
        point_size = args.point_size or default_point_size(pair)
        print(f"[macd] {pair} ticks={len(g):,}", flush=True)

        combos = [
            c for c in product(signal_modes, modes, timeframes, fasts, slows, macd_emas, reverse_values, tps, sls, trails, sessions)
            if c[3] < c[4]
        ]

        def run_combo(combo):
            signal_mode, mode, tf, fast, slow, macd_ema, rev, tp, sl, trail, sess = combo
            warmup = int(np.ceil((slow + macd_ema) * args.warmup_mult))
            invert = mode == "invert"

            if signal_mode == "candle":
                close, close_tick_idx = closed_candle_series(mid, ts, tf)
                if len(close) < max(slow, fast) + macd_ema + 2:
                    return None
                raw_macd = ema(close, fast) - ema(close, slow)
                if macd_ema <= 1:
                    line = raw_macd
                    if warmup > 0:
                        line[:warmup] = 0.0
                    candle_state = np.where(
                        line >= args.deadband, 1.0,
                        np.where(line <= -args.deadband, -1.0, 0.0),
                    )
                    if args.confirm_candles > 1:
                        confirm = max(1, args.confirm_candles)
                        long_ok = candle_state == 1
                        short_ok = candle_state == -1
                        for k in range(1, confirm):
                            prev_state = np.roll(candle_state, k)
                            prev_state[:k] = 0.0
                            long_ok &= prev_state == 1
                            short_ok &= prev_state == -1
                        confirmed_state = np.where(long_ok, 1.0, np.where(short_ok, -1.0, 0.0))
                    else:
                        confirmed_state = candle_state
                else:
                    signal_line = sma(raw_macd, macd_ema)
                    diff = raw_macd - signal_line
                    prev = np.roll(diff, 1)
                    prev[0] = np.nan
                    candle_state = np.zeros(len(close), dtype=np.float64)
                    candle_state[(prev <= args.deadband) & (diff > args.deadband)] = 1.0
                    candle_state[(prev >= -args.deadband) & (diff < -args.deadband)] = -1.0
                    candle_state[~np.isfinite(signal_line)] = 0.0
                    if warmup > 0:
                        candle_state[:warmup] = 0.0
                    confirmed_state = candle_state
                if invert:
                    candle_state = -candle_state
                    confirmed_state = -confirmed_state
                tick_state = candle_state_to_ticks(len(bid), close_tick_idx, candle_state)
                tick_confirmed_state = candle_state_to_ticks(len(bid), close_tick_idx, confirmed_state)
                prev_tick_confirmed_state = np.roll(tick_confirmed_state, 1)
                prev_tick_confirmed_state[0] = 0.0
                long_trigger = (tick_confirmed_state == 1) & (prev_tick_confirmed_state != 1)
                short_trigger = (tick_confirmed_state == -1) & (prev_tick_confirmed_state != -1)
                if macd_ema <= 1:
                    long_exit = tick_state <= 0
                    short_exit = tick_state >= 0
                else:
                    long_exit = tick_state == -1
                    short_exit = tick_state == 1
            else:
                _, _, _, close, _ = live_candles(mid, ts, tf)
                raw_macd = ema(close, fast) - ema(close, slow)
                if macd_ema <= 1:
                    line = raw_macd
                    raw_state = np.where(
                        line >= args.deadband, 1.0,
                        np.where(line <= -args.deadband, -1.0, 0.0),
                    )
                else:
                    signal_line = sma(raw_macd, macd_ema)
                    diff = raw_macd - signal_line
                    prev = np.roll(diff, 1)
                    prev[0] = np.nan
                    raw_state = np.zeros(len(close), dtype=np.float64)
                    raw_state[(prev <= args.deadband) & (diff > args.deadband)] = 1.0
                    raw_state[(prev >= -args.deadband) & (diff < -args.deadband)] = -1.0
                    raw_state[~np.isfinite(signal_line)] = 0.0
                if invert:
                    raw_state = -raw_state
                prev_state = np.roll(raw_state, 1)
                prev_state[0] = 0.0
                long_trigger = (raw_state == 1.0) & (prev_state != 1.0)
                short_trigger = (raw_state == -1.0) & (prev_state != -1.0)
                if warmup > 0:
                    long_trigger[:warmup] = False
                    short_trigger[:warmup] = False
                if macd_ema <= 1:
                    long_exit = raw_state <= 0.0
                    short_exit = raw_state >= 0.0
                else:
                    long_exit = raw_state == -1.0
                    short_exit = raw_state == 1.0

            entry_allowed = session_cache[int(sess)]
            long_trigger = long_trigger & entry_allowed
            short_trigger = short_trigger & entry_allowed
            long_exit = long_exit & entry_allowed
            short_exit = short_exit & entry_allowed

            params = (
                f"mode={signal_mode};dir={mode};fast={fast};slow={slow};"
                f"macd_ema={macd_ema};warmup={warmup};deadband={args.deadband};"
                f"confirm={args.confirm_candles};reverse={int(rev)};trail={trail:g};session={sess}"
            )
            if trail > 0 and tp == 0 and sl == 0:
                return simulate_trailing_signals(
                    pair, "macd", params, tf, bid, ask, long_trigger, short_trigger,
                    long_exit, short_exit, trail, point_size, args.amount,
                    args.compound, args.leverage, args.commission_per_million,
                    args.side, rev,
                )
            if args.trades_out:
                return simulate_triggers(
                    pair, "macd", params, tf, bid, ask, long_trigger, short_trigger,
                    long_exit, short_exit, tp, sl, point_size, args.amount,
                    args.compound, args.leverage, args.commission_per_million,
                    args.side, reverse_on_flip=rev, day_id=day_id,
                    max_days=max_days, return_trades=True,
                )
            return simulate_pair_switch(
                pair, "macd", params, tf, bid, ask, long_trigger, short_trigger,
                long_exit, short_exit, day_id, max_days, tp, sl, point_size,
                args.amount, args.compound, args.leverage,
                args.commission_per_million, args.side, rev, False,
            )

        if args.workers > 1 and len(combos) > 1 and not args.trades_out:
            print(f"[macd] workers={args.workers} combos={len(combos)}", flush=True)
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(run_combo, c) for c in combos]
                for i, fut in enumerate(as_completed(futs), 1):
                    res = fut.result()
                    if res is not None:
                        results.append(res)
                    if i % max(1, len(combos) // 10) == 0:
                        print(f"[macd] progress {i}/{len(combos)}", flush=True)
        else:
            for combo in combos:
                sim = run_combo(combo)
                if sim is None:
                    continue
                if args.trades_out:
                    res, trades = sim
                    results.append(res)
                    all_trades.extend(trades)
                else:
                    results.append(sim)

    write_results(args.out, [r for r in results if r.trades >= args.min_trades], args.top, args.sort_by)
    if args.trades_out:
        with open(args.trades_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "pair", "strategy", "params", "timeframe", "side", "entry_i",
                "exit_i", "entry_px", "exit_px", "pnl", "reason", "equity",
            ])
            for t in all_trades:
                w.writerow([
                    t.pair, t.strategy, t.params, t.timeframe, t.side,
                    t.entry_i, t.exit_i, round(t.entry_px, 6), round(t.exit_px, 6),
                    round(t.pnl, 6), t.reason, round(t.equity, 6),
                ])
        print(f"[macd] wrote trades {args.trades_out}", flush=True)
    print(f"[macd] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
