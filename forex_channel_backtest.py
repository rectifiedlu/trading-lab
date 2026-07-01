"""Tick-backed Channel Breakout strategy backtest."""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

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


def rolling_high_prev(x: np.ndarray, length: int) -> np.ndarray:
    s = pd.Series(x)
    return s.shift(1).rolling(length, min_periods=length).max().to_numpy(np.float64)


def rolling_low_prev(x: np.ndarray, length: int) -> np.ndarray:
    s = pd.Series(x)
    return s.shift(1).rolling(length, min_periods=length).min().to_numpy(np.float64)


def main() -> None:
    ap = build_parser("Channel breakout tick backtest", "forex_channel_results.csv")
    ap.add_argument("--lengths", default="6")
    args = ap.parse_args()

    timeframes = parse_str_list(args.timeframes, DEFAULT_TIMEFRAMES)
    lengths = [int(x) for x in parse_num_list(args.lengths, [6])]
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
        print(f"[channel] {pair} ticks={len(g):,}", flush=True)

        for tf, length, tp, sl in product(timeframes, lengths, tps, sls):
            _, high, low, _, _ = live_candles(mid, ts, tf)
            up = rolling_high_prev(high, length) + point_size
            dn = rolling_low_prev(low, length) - point_size
            long_trigger = ask >= up
            short_trigger = bid <= dn
            params = f"length={length}"
            results.append(simulate_triggers(
                pair, "channel", params, tf, bid, ask, long_trigger, short_trigger,
                None, None, tp, sl, point_size, args.amount, args.compound,
                args.leverage, args.commission_per_million, args.side,
                day_id=day_id, max_days=max_days,
            ))

    write_results(args.out, [r for r in results if r.trades >= args.min_trades], args.top, args.sort_by)
    print(f"[channel] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
