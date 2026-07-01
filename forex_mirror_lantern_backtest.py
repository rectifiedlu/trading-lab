"""Mid-candle two-sided hedge backtest.

At the temporal midpoint of each selected candle, open one long and one short.
Each leg has independent fixed TP/SL exits. A new hedge is opened only after
both legs from the previous hedge have closed.
"""

from __future__ import annotations

import time

import numpy as np

from forex_ml_tick_simulator import default_sl_grid, default_tp_grid
from forex_strategy_common import (
    TradeResult,
    build_parser,
    day_ids_from_timestamps,
    default_point_size,
    load_market,
    parse_num_list,
    parse_str_list,
    timeframe_to_ns,
    write_results,
)
from forex_tcn2_nextbar_simulator import (
    broker_commission_per_lot_side,
    default_contract_size,
    profit_factor,
)

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


if njit is not None:
    @njit(cache=True)
    def _simulate_hedges(
        bid,
        ask,
        ts_ns,
        entry_ticks,
        day_ids,
        max_days,
        point,
        tp_points,
        sl_points,
        amount,
        leverage,
        lot,
        contract_size,
        commission_per_million,
        commission_per_lot_side,
    ):
        daily = np.zeros(max_days, dtype=np.float64)
        trade_pnls = np.zeros(len(entry_ticks) * 2, dtype=np.float64)
        fixed_units = lot * contract_size if lot > 0.0 else 0.0
        notional = amount * leverage
        cash = amount
        equity_peak = amount
        cum_pnl = 0.0
        cum_peak = 0.0
        max_dd = 0.0
        cum_dd = 0.0
        gross_win = 0.0
        gross_loss = 0.0
        wins = 0
        losses = 0
        stops = 0
        trades = 0
        baskets = 0
        worst = 0.0
        max_trade_dd = 0.0
        sum_trade_dd = 0.0
        open_unrealized = 0.0
        next_free_tick = 0

        for entry_tick in entry_ticks:
            if entry_tick < next_free_tick or entry_tick >= len(bid) - 1:
                continue

            long_entry = ask[entry_tick]
            short_entry = bid[entry_tick]
            units = fixed_units if fixed_units > 0.0 else notional / max((long_entry + short_entry) * 0.5, 1e-12)
            effective_lot = abs(units) / contract_size
            round_trip_fee = (
                effective_lot * commission_per_lot_side * 2.0
                if commission_per_lot_side >= 0.0
                else notional / 1_000_000.0 * commission_per_million * 2.0
            )

            long_open = True
            short_open = True
            long_dd = 0.0
            short_dd = 0.0
            last_tick = entry_tick
            baskets += 1

            for ti in range(entry_tick + 1, len(bid)):
                last_tick = ti
                long_points = (bid[ti] - long_entry) / point
                short_points = (short_entry - ask[ti]) / point

                if long_open:
                    adverse = -long_points if long_points < 0.0 else 0.0
                    live_dd = adverse * point * units
                    if live_dd > long_dd:
                        long_dd = live_dd
                    exit_points = 0.0
                    reason = 0
                    if long_points >= tp_points:
                        exit_points = tp_points
                        reason = 1
                    elif long_points <= -sl_points:
                        exit_points = -sl_points
                        reason = 2
                    if reason:
                        pnl = exit_points * point * units - round_trip_fee
                        cash += pnl
                        cum_pnl += pnl
                        daily[day_ids[ti]] += pnl
                        trade_pnls[trades] = pnl
                        trades += 1
                        if reason == 2:
                            stops += 1
                        if pnl >= 0.0:
                            wins += 1
                            gross_win += pnl
                        else:
                            losses += 1
                            gross_loss += -pnl
                        if pnl < worst:
                            worst = pnl
                        if long_dd > max_trade_dd:
                            max_trade_dd = long_dd
                        sum_trade_dd += long_dd
                        long_open = False

                if short_open:
                    adverse = -short_points if short_points < 0.0 else 0.0
                    live_dd = adverse * point * units
                    if live_dd > short_dd:
                        short_dd = live_dd
                    exit_points = 0.0
                    reason = 0
                    if short_points >= tp_points:
                        exit_points = tp_points
                        reason = 1
                    elif short_points <= -sl_points:
                        exit_points = -sl_points
                        reason = 2
                    if reason:
                        pnl = exit_points * point * units - round_trip_fee
                        cash += pnl
                        cum_pnl += pnl
                        daily[day_ids[ti]] += pnl
                        trade_pnls[trades] = pnl
                        trades += 1
                        if reason == 2:
                            stops += 1
                        if pnl >= 0.0:
                            wins += 1
                            gross_win += pnl
                        else:
                            losses += 1
                            gross_loss += -pnl
                        if pnl < worst:
                            worst = pnl
                        if short_dd > max_trade_dd:
                            max_trade_dd = short_dd
                        sum_trade_dd += short_dd
                        short_open = False

                floating = 0.0
                if long_open:
                    floating += long_points * point * units - round_trip_fee * 0.5
                if short_open:
                    floating += short_points * point * units - round_trip_fee * 0.5
                equity = cash + floating
                if equity > equity_peak:
                    equity_peak = equity
                dd = equity_peak - equity
                if dd > max_dd:
                    max_dd = dd

                if cum_pnl > cum_peak:
                    cum_peak = cum_pnl
                cdd = cum_peak - cum_pnl
                if cdd > cum_dd:
                    cum_dd = cdd

                if not long_open and not short_open:
                    next_free_tick = ti + 1
                    break

            if long_open or short_open:
                open_unrealized = 0.0
                if long_open:
                    open_unrealized += (bid[last_tick] - long_entry) / point * point * units - round_trip_fee * 0.5
                if short_open:
                    open_unrealized += (short_entry - ask[last_tick]) / point * point * units - round_trip_fee * 0.5
                break

        return (
            cash - amount,
            open_unrealized,
            trades,
            wins,
            losses,
            gross_win,
            gross_loss,
            max_dd,
            cum_dd,
            stops,
            worst,
            max_trade_dd,
            sum_trade_dd / trades if trades else 0.0,
            baskets,
            daily,
            trade_pnls[:trades],
        )


def midpoint_entry_ticks(ts_ns: np.ndarray, timeframe: str, fraction: float) -> np.ndarray:
    tf_ns = timeframe_to_ns(timeframe)
    buckets = (ts_ns // tf_ns) * tf_ns
    unique = np.unique(buckets)
    targets = unique + np.int64(tf_ns * fraction)
    idx = np.searchsorted(ts_ns, targets, side="left")
    valid = (idx < len(ts_ns)) & (buckets[np.minimum(idx, len(ts_ns) - 1)] == unique)
    return idx[valid].astype(np.int64)


def main() -> None:
    ap = build_parser("Mid-candle two-sided hedge sweep", "mirror_lantern_results.csv")
    ap.set_defaults(timeframes="1m,3m,5m,10m,15m", tp_points="100")
    ap.add_argument("--entry-fraction", type=float, default=0.5)
    ap.add_argument("--lot", type=float, default=0.0)
    ap.add_argument("--contract-size", type=float, default=0.0)
    args = ap.parse_args()

    if not 0.0 < args.entry_fraction < 1.0:
        raise SystemExit("--entry-fraction must be between 0 and 1")

    ticks, started = load_market(args)
    timeframes = parse_str_list(args.timeframes, ["1m", "3m", "5m"])
    results: list[TradeResult] = []

    for pair, frame in ticks.groupby("pair", sort=False):
        pair = str(pair).upper()
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        bid = frame["bid"].to_numpy(np.float64)
        ask = frame["ask"].to_numpy(np.float64)
        ts_ns = frame["timestamp"].astype("int64").to_numpy()
        point = args.point_size or default_point_size(pair)
        tp_grid = parse_num_list(args.tp_points, default_tp_grid(pair))
        sl_grid = parse_num_list(args.sl_points, default_sl_grid(pair))
        contract_size = args.contract_size if args.contract_size > 0.0 else default_contract_size(pair)
        commission_side = broker_commission_per_lot_side(pair)
        day_ids, max_days = day_ids_from_timestamps(ts_ns)

        for timeframe in timeframes:
            entries = midpoint_entry_ticks(ts_ns, timeframe, args.entry_fraction)
            print(
                f"[mirror-lantern] {pair} tf={timeframe} midpoint_entries={len(entries):,} "
                f"grid={len(tp_grid) * len(sl_grid):,}",
                flush=True,
            )
            for tp in tp_grid:
                for sl in sl_grid:
                    if tp <= 0.0 or sl <= 0.0:
                        continue
                    out = _simulate_hedges(
                        bid,
                        ask,
                        ts_ns,
                        entries,
                        day_ids,
                        max_days,
                        point,
                        float(tp),
                        float(sl),
                        args.amount,
                        args.leverage,
                        args.lot,
                        contract_size,
                        args.commission_per_million,
                        commission_side,
                    )
                    (
                        realised,
                        open_unrealized,
                        trades,
                        wins,
                        losses,
                        gross_win,
                        gross_loss,
                        max_dd,
                        cum_dd,
                        stops,
                        worst,
                        trade_max_dd,
                        trade_avg_dd,
                        baskets,
                        daily,
                        pnls,
                    ) = out
                    if trades < args.min_trades:
                        continue
                    params = (
                        f"entry_fraction={args.entry_fraction:g};basket=one_at_a_time;"
                        f"lot={args.lot:g};commission_lot_side={commission_side:g}"
                    )
                    result = TradeResult(
                        pair=pair,
                        strategy="mirror_lantern",
                        params=params,
                        timeframe=timeframe,
                        tp_points=float(tp),
                        sl_points=float(sl),
                        point_size=point,
                        realised=float(realised),
                        open_unrealized=float(open_unrealized),
                        total=float(realised + open_unrealized),
                        trades=int(trades),
                        wins=int(wins),
                        losses=int(losses),
                        win_rate=float(wins / trades * 100.0 if trades else 0.0),
                        profit_factor=profit_factor(float(gross_win), float(gross_loss)),
                        max_drawdown=float(max_dd),
                        long_trades=int((trades + 1) // 2),
                        short_trades=int(trades // 2),
                        stop_losses=int(stops),
                        signal_exits=0,
                        liquidations=0,
                        account_dead=False,
                        open_side="hedged" if open_unrealized else "-",
                        open_bps=0.0,
                    )
                    for name, value in {
                        "baskets": int(baskets),
                        "cum_max_drawdown": float(cum_dd),
                        "trade_max_drawdown": float(trade_max_dd),
                        "trade_avg_drawdown": float(trade_avg_dd),
                        "worst_trade_pnl": float(worst),
                        "median_loss": float(np.median(pnls[pnls < 0.0]) if np.any(pnls < 0.0) else 0.0),
                        "avg_day": float(np.mean(daily) if len(daily) else 0.0),
                        "median_day": float(np.median(daily) if len(daily) else 0.0),
                    }.items():
                        setattr(result, name, value)
                    results.append(result)

    if not results:
        raise SystemExit("no completed results")
    write_results(args.out, results, args.top, args.sort_by)
    print(f"[mirror-lantern] wrote {args.out} rows={len(results):,} elapsed={time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
