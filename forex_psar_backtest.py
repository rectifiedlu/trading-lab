"""Tick-backed Parabolic SAR strategy backtest."""

from __future__ import annotations

from itertools import product

import numpy as np

from forex_strategy_common import (
    DEFAULT_SL_POINTS,
    DEFAULT_TIMEFRAMES,
    DEFAULT_TP_POINTS,
    build_parser,
    day_ids_from_timestamps,
    default_point_size,
    live_candles,
    load_market,
    parse_num_list,
    parse_str_list,
    simulate_triggers,
    write_results,
)


def psar_signals(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 start: float, inc: float, max_af: float) -> tuple[np.ndarray, np.ndarray]:
    long_trigger = np.zeros(len(close), dtype=np.bool_)
    short_trigger = np.zeros(len(close), dtype=np.bool_)
    if len(close) < 3:
        return long_trigger, short_trigger

    uptrend = close[1] > close[0]
    ep = high[1] if uptrend else low[1]
    sar = low[0] if uptrend else high[0]
    af = start

    for i in range(2, len(close)):
        prev_sar = sar
        sar = prev_sar + af * (ep - prev_sar)
        if uptrend:
            sar = min(sar, low[i - 1], low[i - 2])
            if low[i] < sar:
                uptrend = False
                short_trigger[i] = True
                sar = max(ep, high[i])
                ep = low[i]
                af = start
            elif high[i] > ep:
                ep = high[i]
                af = min(af + inc, max_af)
        else:
            sar = max(sar, high[i - 1], high[i - 2])
            if high[i] > sar:
                uptrend = True
                long_trigger[i] = True
                sar = min(ep, low[i])
                ep = high[i]
                af = start
            elif low[i] < ep:
                ep = low[i]
                af = min(af + inc, max_af)
    return long_trigger, short_trigger


def main() -> None:
    ap = build_parser("Parabolic SAR tick backtest", "forex_psar_results.csv")
    ap.add_argument("--start-values", default="0.02")
    ap.add_argument("--increments", default="0.02")
    ap.add_argument("--maximums", default="0.2")
    args = ap.parse_args()

    timeframes = parse_str_list(args.timeframes, DEFAULT_TIMEFRAMES)
    starts = parse_num_list(args.start_values, [0.02])
    increments = parse_num_list(args.increments, [0.02])
    maximums = parse_num_list(args.maximums, [0.2])
    tps = parse_num_list(args.tp_points, DEFAULT_TP_POINTS)
    sls = parse_num_list(args.sl_points, DEFAULT_SL_POINTS)

    ticks, _ = load_market(args)
    results = []
    for pair, g in ticks.groupby("pair", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        bid = g["bid"].to_numpy(np.float64)
        ask = g["ask"].to_numpy(np.float64)
        # MT5 native XAUUSD candles match bid OHLC, not mid/ask.
        mid = bid
        ts = g["timestamp"].astype("int64").to_numpy()
        day_id, max_days = day_ids_from_timestamps(ts)
        point_size = args.point_size or default_point_size(pair)
        print(f"[psar] {pair} ticks={len(g):,}", flush=True)

        for tf, start, inc, maximum, tp, sl in product(timeframes, starts, increments, maximums, tps, sls):
            _, h, l, c, _ = live_candles(mid, ts, tf)
            long_trigger, short_trigger = psar_signals(h, l, c, start, inc, maximum)
            params = f"start={start};inc={inc};max={maximum}"
            results.append(simulate_triggers(
                pair, "psar", params, tf, bid, ask, long_trigger, short_trigger,
                None, None, tp, sl, point_size, args.amount, args.compound,
                args.leverage, args.commission_per_million, args.side,
                day_id=day_id, max_days=max_days,
            ))

    write_results(args.out, [r for r in results if r.trades >= args.min_trades], args.top, args.sort_by)
    print(f"[psar] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
