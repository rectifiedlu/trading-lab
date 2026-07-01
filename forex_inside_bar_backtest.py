"""Tick-backed Inside Bar strategy backtest."""

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


def main() -> None:
    ap = build_parser("Inside bar tick backtest", "forex_inside_bar_results.csv")
    args = ap.parse_args()

    timeframes = parse_str_list(args.timeframes, DEFAULT_TIMEFRAMES)
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
        print(f"[inside] {pair} ticks={len(g):,}", flush=True)

        for tf, tp, sl in product(timeframes, tps, sls):
            o, h, l, c, new_bar = live_candles(mid, ts, tf)
            inside = (h < np.roll(h, 1)) & (l > np.roll(l, 1))
            bullish = c > o
            bearish = c < o
            # Pine enters after the inside bar condition exists. Here we trigger
            # on the first tick of the next candle to avoid same-bar lookahead.
            long_trigger = new_bar & np.roll(inside & bullish, 1)
            short_trigger = new_bar & np.roll(inside & bearish, 1)
            long_trigger[0] = False
            short_trigger[0] = False
            results.append(simulate_triggers(
                pair, "inside", "inside_bar", tf, bid, ask, long_trigger, short_trigger,
                None, None, tp, sl, point_size, args.amount, args.compound,
                args.leverage, args.commission_per_million, args.side,
                day_id=day_id, max_days=max_days,
            ))

    write_results(args.out, [r for r in results if r.trades >= args.min_trades], args.top, args.sort_by)
    print(f"[inside] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
