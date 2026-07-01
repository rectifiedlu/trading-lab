"""Pure tick trailing-entry/trailing-exit backtest.

Flat:
    Track latest low/high reference.
    If latest extreme was a low and price rises entry_trail -> long.
    If latest extreme was a high and price falls entry_trail -> short.

Holding:
    Long exits when bid falls exit_trail from best bid since entry.
    Short exits when ask rises exit_trail from best ask since entry.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from forex_strategy_common import (
    TradeResult,
    build_parser,
    commission,
    default_point_size,
    load_market,
    open_unrealized,
    parse_num_list,
    units_for_margin,
    write_results,
)


def simulate_pure_trailing(
    pair: str,
    bid: np.ndarray,
    ask: np.ndarray,
    entry_points: float,
    exit_points: float,
    point_size: float,
    amount: float,
    compound: bool,
    leverage: float,
    commission_per_million: float,
    side: str,
    reverse_on_exit: bool,
) -> TradeResult:
    entry_dist = entry_points * point_size
    exit_dist = exit_points * point_size
    # Use bid as the reference stream; MT5 native XAUUSD candles match bid OHLC.
    mid = bid

    allow_long = side in ("long", "both")
    allow_short = side in ("short", "both")

    start_balance = amount
    cash = start_balance
    equity_peak = start_balance
    max_dd = 0.0
    gross_win = 0.0
    gross_loss = 0.0
    trades = wins = losses = long_trades = short_trades = 0
    signal_exits = 0
    liquidations = 0
    account_dead = False

    pos = 0
    entry = 0.0
    units = 0.0
    best_long = 0.0
    best_short = 0.0
    low_ref = float(mid[0])
    high_ref = float(mid[0])
    last_extreme = 0

    def open_pos(new_pos: int, px: float) -> tuple[float, float, float]:
        margin = cash if compound else amount
        if margin <= 0:
            return 0.0, 0.0, 0.0
        new_units = units_for_margin(margin, leverage, px)
        return px, new_units, commission(px, new_units, commission_per_million)

    for i in range(len(mid)):
        b = float(bid[i])
        a = float(ask[i])
        m = float(mid[i])

        if pos == 1:
            best_long = max(best_long, b)
            unreal = open_unrealized(pos, entry, units, b, a)
            equity = cash + unreal
            equity_peak = max(equity_peak, equity)
            max_dd = max(max_dd, equity_peak - equity)
            if equity <= 0:
                liquidations += 1
                account_dead = True
                cash = 0.0
                pos = 0
                break
            if b <= best_long - exit_dist:
                pnl = (b - entry) * units - commission(b, units, commission_per_million)
                cash += pnl
                trades += 1
                long_trades += 1
                signal_exits += 1
                if pnl >= 0:
                    wins += 1
                    gross_win += pnl
                else:
                    losses += 1
                    gross_loss += -pnl
                pos = 0
                low_ref = high_ref = m
                last_extreme = 0
                if reverse_on_exit and allow_short and cash > 0:
                    new_entry, new_units, fee = open_pos(-1, b)
                    if new_units > 0:
                        cash -= fee
                        pos = -1
                        entry = new_entry
                        units = new_units
                        best_short = a
                continue

        elif pos == -1:
            best_short = min(best_short, a)
            unreal = open_unrealized(pos, entry, units, b, a)
            equity = cash + unreal
            equity_peak = max(equity_peak, equity)
            max_dd = max(max_dd, equity_peak - equity)
            if equity <= 0:
                liquidations += 1
                account_dead = True
                cash = 0.0
                pos = 0
                break
            if a >= best_short + exit_dist:
                pnl = (entry - a) * units - commission(a, units, commission_per_million)
                cash += pnl
                trades += 1
                short_trades += 1
                signal_exits += 1
                if pnl >= 0:
                    wins += 1
                    gross_win += pnl
                else:
                    losses += 1
                    gross_loss += -pnl
                pos = 0
                low_ref = high_ref = m
                last_extreme = 0
                if reverse_on_exit and allow_long and cash > 0:
                    new_entry, new_units, fee = open_pos(1, a)
                    if new_units > 0:
                        cash -= fee
                        pos = 1
                        entry = new_entry
                        units = new_units
                        best_long = b
                continue

        else:
            if m < low_ref:
                low_ref = m
                last_extreme = -1
            if m > high_ref:
                high_ref = m
                last_extreme = 1

            long_trigger = allow_long and last_extreme == -1 and m >= low_ref + entry_dist
            short_trigger = allow_short and last_extreme == 1 and m <= high_ref - entry_dist

            if long_trigger:
                new_entry, new_units, fee = open_pos(1, a)
                if new_units <= 0:
                    break
                cash -= fee
                pos = 1
                entry = new_entry
                units = new_units
                best_long = b
                low_ref = high_ref = m
                last_extreme = 0
            elif short_trigger:
                new_entry, new_units, fee = open_pos(-1, b)
                if new_units <= 0:
                    break
                cash -= fee
                pos = -1
                entry = new_entry
                units = new_units
                best_short = a
                low_ref = high_ref = m
                last_extreme = 0

        if pos == 0:
            equity_peak = max(equity_peak, cash)
            max_dd = max(max_dd, equity_peak - cash)

    open_u = 0.0
    open_side = "-"
    open_bps = 0.0
    if pos == 1:
        open_side = "long"
        open_u = open_unrealized(pos, entry, units, float(bid[-1]), float(ask[-1]))
        open_bps = (float(bid[-1]) / entry - 1.0) * 10000.0
    elif pos == -1:
        open_side = "short"
        open_u = open_unrealized(pos, entry, units, float(bid[-1]), float(ask[-1]))
        open_bps = (entry / float(ask[-1]) - 1.0) * 10000.0

    equity = cash + open_u
    realised = cash - start_balance
    total = equity - start_balance
    win_rate = wins / trades * 100.0 if trades else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
    params = f"entry={entry_points:g};exit={exit_points:g};reverse={int(reverse_on_exit)}"
    return TradeResult(
        pair, "puretrail", params, "tick", 0.0, 0.0, point_size,
        realised, open_u, total, trades, wins, losses, win_rate, pf,
        max_dd, long_trades, short_trades, 0, signal_exits, liquidations,
        account_dead, open_side, open_bps,
    )


def main() -> None:
    ap = build_parser("pure tick trailing entry/exit backtest", "forex_pure_trailing_results.csv")
    ap.add_argument("--entry-points", default="50,75,100,150,200,300")
    ap.add_argument("--exit-points", default="50,75,100,150,200,300")
    ap.add_argument("--reverse-on-exit", default="0,1",
                    help="comma list: 0=wait after exit, 1=immediately open opposite side")
    args = ap.parse_args()

    entries = parse_num_list(args.entry_points, [100])
    exits = parse_num_list(args.exit_points, [150])
    reverses = [bool(int(x)) for x in parse_num_list(args.reverse_on_exit, [0, 1])]

    ticks, _ = load_market(args)
    results = []
    for pair, g in ticks.groupby("pair", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        bid = g["bid"].to_numpy(np.float64)
        ask = g["ask"].to_numpy(np.float64)
        point_size = args.point_size or default_point_size(pair)
        print(f"[puretrail] {pair} ticks={len(g):,}", flush=True)
        for entry_points, exit_points, rev in product(entries, exits, reverses):
            results.append(simulate_pure_trailing(
                pair, bid, ask, entry_points, exit_points, point_size,
                args.amount, args.compound, args.leverage,
                args.commission_per_million, args.side, rev,
            ))

    write_results(args.out, [r for r in results if r.trades >= args.min_trades], args.top, args.sort_by)
    print(f"[puretrail] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
