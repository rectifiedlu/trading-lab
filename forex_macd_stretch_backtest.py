"""MACD zero-cross backtest with EMA/ATR stretch entry gate.

Signals:
    macd_line = EMA(close, fast) - EMA(close, slow), optionally smoothed.

Entry gate:
    long entries are skipped when close is above EMA + ATR * stretch_mult.
    short entries are skipped when close is below EMA - ATR * stretch_mult.

Signals are candle-confirmed by default. Execution/TP/SL uses bid/ask ticks.
"""

from __future__ import annotations

from itertools import product
import csv

import numpy as np
import pandas as pd

from forex_strategy_common import (
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
    return pd.Series(x).rolling(length, min_periods=length).mean().to_numpy(np.float64)


def candle_ohlc(mid: np.ndarray, close_tick_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    opens = []
    highs = []
    lows = []
    closes = []
    prev = 0
    for idx in close_tick_idx:
        chunk = mid[prev:int(idx) + 1]
        if len(chunk):
            opens.append(float(chunk[0]))
            highs.append(float(np.max(chunk)))
            lows.append(float(np.min(chunk)))
            closes.append(float(chunk[-1]))
        prev = int(idx) + 1
    return (
        np.array(opens, dtype=np.float64),
        np.array(highs, dtype=np.float64),
        np.array(lows, dtype=np.float64),
        np.array(closes, dtype=np.float64),
    )


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    return np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])


def main() -> None:
    ap = build_parser("MACD with EMA/ATR stretch gate tick backtest", "forex_macd_stretch_results.csv")
    ap.set_defaults(timeframes="5m", tp_points="200,400,600,800", sl_points="0")
    ap.add_argument("--fast", default="6,8,10,12")
    ap.add_argument("--slow", default="13,17,21,26")
    ap.add_argument("--macd-ema", default="1",
                    help="EMA smoothing applied to fast EMA - slow EMA")
    ap.add_argument("--deadband", default="0,0.05,0.1",
                    help="comma list of MACD neutral bands")
    ap.add_argument("--signal-mode", choices=["candle", "tick"], default="candle")
    ap.add_argument("--reverse-on-flip", default="0",
                    help="comma list: 0=close only, 1=close and enter opposite side on region flip")
    ap.add_argument("--warmup-mult", type=float, default=1.0)
    ap.add_argument("--stretch-ema", default="10,20,34,50")
    ap.add_argument("--stretch-atr", default="7,14,21")
    ap.add_argument("--stretch-mult", default="0.3,0.5,0.7,1,1.3")
    ap.add_argument("--trades-out", default=None)
    args = ap.parse_args()

    timeframes = parse_str_list(args.timeframes, ["5m"])
    fasts = [int(x) for x in parse_num_list(args.fast, [8])]
    slows = [int(x) for x in parse_num_list(args.slow, [13])]
    macd_emas = [int(x) for x in parse_num_list(args.macd_ema, [1])]
    deadbands = parse_num_list(args.deadband, [0.1])
    reverse_values = [bool(int(x)) for x in parse_num_list(args.reverse_on_flip, [0])]
    tps = parse_num_list(args.tp_points, [400])
    sls = parse_num_list(args.sl_points, [0])
    stretch_emas = [int(x) for x in parse_num_list(args.stretch_ema, [20])]
    stretch_atrs = [int(x) for x in parse_num_list(args.stretch_atr, [14])]
    stretch_mults = parse_num_list(args.stretch_mult, [0.7])

    ticks, _ = load_market(args)
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
        point_size = args.point_size or default_point_size(pair)
        print(f"[macd-stretch] {pair} ticks={len(g):,}", flush=True)

        for tf in timeframes:
            close, close_tick_idx = closed_candle_series(mid, ts, tf)
            if len(close) < 5:
                continue
            _, high, low, ohlc_close = candle_ohlc(mid, close_tick_idx)

            for fast, slow, macd_ema, deadband, rev, tp, sl, se, sa, sm in product(
                fasts, slows, macd_emas, deadbands, reverse_values, tps, sls,
                stretch_emas, stretch_atrs, stretch_mults,
            ):
                if fast >= slow:
                    continue
                needed = max(slow + macd_ema, se, sa) + 2
                if args.signal_mode == "candle" and len(close) < needed:
                    continue
                warmup = int(np.ceil((slow + macd_ema) * args.warmup_mult))

                if args.signal_mode == "candle":
                    raw_macd = ema(close, fast) - ema(close, slow)
                    line = ema(raw_macd, macd_ema)
                    if warmup > 0:
                        line[:warmup] = 0.0

                    basis = ema(ohlc_close, se)
                    atr = sma(true_range(high, low, ohlc_close), sa)
                    upper = basis + atr * sm
                    lower = basis - atr * sm
                    long_allowed = ohlc_close <= upper
                    short_allowed = ohlc_close >= lower

                    candle_state = np.where(
                        line >= deadband, 1.0,
                        np.where(line <= -deadband, -1.0, 0.0),
                    )
                    prev_candle_state = np.roll(candle_state, 1)
                    prev_candle_state[0] = 0.0
                    candle_long_trigger = (
                        (candle_state == 1) & (prev_candle_state != 1) & long_allowed
                    )
                    candle_short_trigger = (
                        (candle_state == -1) & (prev_candle_state != -1) & short_allowed
                    )
                    tick_long_raw = candle_state_to_ticks(len(bid), close_tick_idx, candle_long_trigger.astype(float))
                    tick_short_raw = candle_state_to_ticks(len(bid), close_tick_idx, candle_short_trigger.astype(float))
                    long_trigger = tick_long_raw == 1
                    short_trigger = tick_short_raw == 1
                    tick_state = candle_state_to_ticks(len(bid), close_tick_idx, candle_state)
                    long_exit = tick_state <= 0
                    short_exit = tick_state >= 0
                else:
                    _, high_live, low_live, close_live, _ = live_candles(mid, ts, tf)
                    raw_macd = ema(close_live, fast) - ema(close_live, slow)
                    line = ema(raw_macd, macd_ema)
                    prev = np.roll(line, 1)
                    prev[0] = np.nan
                    basis = ema(close_live, se)
                    atr = sma(true_range(high_live, low_live, close_live), sa)
                    upper = basis + atr * sm
                    lower = basis - atr * sm
                    long_trigger = (prev < deadband) & (line >= deadband) & (close_live <= upper)
                    short_trigger = (prev > -deadband) & (line <= -deadband) & (close_live >= lower)
                    if warmup > 0:
                        long_trigger[:warmup] = False
                        short_trigger[:warmup] = False
                    long_exit = line <= -deadband
                    short_exit = line >= deadband

                params = (
                    f"mode={args.signal_mode};fast={fast};slow={slow};"
                    f"macd_ema={macd_ema};deadband={deadband};"
                    f"stretch_ema={se};stretch_atr={sa};stretch_mult={sm};"
                    f"reverse={int(rev)}"
                )
                sim = simulate_triggers(
                    pair, "macdstr", params, tf, bid, ask, long_trigger,
                    short_trigger, long_exit, short_exit, tp, sl, point_size,
                    args.amount, args.compound, args.leverage,
                    args.commission_per_million, args.side,
                    reverse_on_flip=rev,
                    day_id=day_id,
                    max_days=max_days,
                    return_trades=bool(args.trades_out),
                )
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
        print(f"[macd-stretch] wrote trades {args.trades_out}", flush=True)
    print(f"[macd-stretch] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
