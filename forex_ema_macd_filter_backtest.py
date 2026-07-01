"""EMA confirmation entry backtest with equity-curve scoring.

Rules:
    - Closed-candle signals only.
    - EMA baseline from the same timeframe.
    - Enter long after N consecutive closes above EMA.
    - Enter short after N consecutive closes below EMA.
    - Once in a trade, ignore signal switches.
    - Exit only by tick TP or tick SL.
"""

from __future__ import annotations

from itertools import product
import csv

import numpy as np

from forex_strategy_common import (
    TradeResult,
    build_parser,
    candle_state_to_ticks,
    closed_candle_series,
    day_ids_from_timestamps,
    default_point_size,
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


def confirmed_side(close: np.ndarray, basis: np.ndarray, confirm: int) -> np.ndarray:
    side = np.zeros(len(close), dtype=np.float64)
    above_count = 0
    below_count = 0
    for i, (px, ma) in enumerate(zip(close, basis)):
        if not np.isfinite(ma):
            above_count = below_count = 0
        elif px > ma:
            above_count += 1
            below_count = 0
        elif px < ma:
            below_count += 1
            above_count = 0
        else:
            above_count = below_count = 0
        if above_count >= confirm:
            side[i] = 1.0
        elif below_count >= confirm:
            side[i] = -1.0
    return side


def add_curve_metrics(result: TradeResult, trades: list, start_balance: float) -> TradeResult:
    equity = [float(start_balance)]
    equity.extend(float(t.equity) for t in trades)
    final_equity = start_balance + result.total
    if not equity or abs(equity[-1] - final_equity) > 1e-9:
        equity.append(float(final_equity))
    y = np.array(equity, dtype=np.float64)
    x = np.arange(len(y), dtype=np.float64)
    if len(y) >= 3 and np.var(y) > 1e-12:
        slope, intercept = np.polyfit(x, y, 1)
        fitted = slope * x + intercept
        resid = y - fitted
        ss_res = float(np.sum(resid * resid))
        ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        resid_std = float(np.std(resid))
    else:
        slope = 0.0
        r2 = 0.0
        resid_std = 0.0
    r2 = max(0.0, min(1.0, float(r2)))
    # Rewards upward, straight curves and punishes drawdown/choppiness.
    curve_score = result.total * r2 - result.max_drawdown - resid_std
    result.curve_r2 = r2
    result.curve_slope = float(slope)
    result.curve_resid_std = resid_std
    result.curve_score = float(curve_score)
    return result


def main() -> None:
    ap = build_parser("EMA confirmation TP/SL backtest", "forex_ema_macd_filter_results.csv")
    ap.set_defaults(timeframes="10s,20s,30s,45s,1m,2m,3m,", tp_points="400,600,800,1000", sl_points="400,600,800,1000")
    ap.add_argument("--ema-lengths", default="21,42,63,84,105,126,147")
    ap.add_argument("--confirm-candles", default="2")
    ap.set_defaults(sort_by="curve")
    ap.add_argument("--trades-out", default=None)
    args = ap.parse_args()

    timeframes = parse_str_list(args.timeframes, ["1m"])
    ema_lengths = [int(x) for x in parse_num_list(args.ema_lengths, [21, 63])]
    confirms = [int(x) for x in parse_num_list(args.confirm_candles, [2])]
    tps = parse_num_list(args.tp_points, [400])
    sls = parse_num_list(args.sl_points, [1000])

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
        print(f"[ema-confirm] {pair} ticks={len(g):,}", flush=True)

        for tf, ema_len, confirm, tp, sl in product(
            timeframes, ema_lengths, confirms, tps, sls,
        ):
            if confirm < 1:
                continue
            close, close_tick_idx = closed_candle_series(mid, ts, tf)
            if len(close) < ema_len + confirm + 2:
                continue

            basis = ema(close, ema_len)
            side = confirmed_side(close, basis, confirm)
            prev_side = np.roll(side, 1)
            prev_side[0] = 0.0

            candle_long = (side == 1) & (prev_side != 1)
            candle_short = (side == -1) & (prev_side != -1)

            long_trigger = candle_state_to_ticks(
                len(bid), close_tick_idx, candle_long.astype(float),
            ) == 1
            short_trigger = candle_state_to_ticks(
                len(bid), close_tick_idx, candle_short.astype(float),
            ) == 1

            params = (
                f"tf={tf};ema={ema_len};confirm={confirm}"
            )
            sim = simulate_triggers(
                pair, "emaconf", params, tf, bid, ask, long_trigger,
                short_trigger, None, None, tp, sl, point_size,
                args.amount, args.compound, args.leverage,
                args.commission_per_million, args.side,
                reverse_on_flip=False,
                day_id=day_id,
                max_days=max_days,
                return_trades=True,
            )
            res, trades = sim
            results.append(add_curve_metrics(res, trades, args.amount))
            if args.trades_out:
                all_trades.extend(trades)

    write_results(
        args.out,
        [r for r in results if r.trades >= args.min_trades],
        args.top,
        args.sort_by,
    )
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
        print(f"[ema-confirm] wrote trades {args.trades_out}", flush=True)
    print(f"[ema-confirm] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
