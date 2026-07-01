"""Bollinger band mean-reversion backtest with candle bands and tick execution.

Default model:
    - Build time-normalized candles, default 1s.
    - Calculate SMA/stdev bands from closed candle closes.
    - During the next candle, tick-mid crossing back above lower band enters long.
    - Tick-mid crossing back below upper band enters short.
    - Opposite signal reverses/closes through simulate_triggers.
"""

from __future__ import annotations

import os
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

import numpy as np

from forex_strategy_common import (
    TradeResult,
    active_session_allowed,
    build_parser,
    commission,
    day_ids_from_timestamps,
    default_point_size,
    live_candles,
    load_market,
    njit,
    open_unrealized,
    parse_num_list,
    parse_str_list,
    simulate_triggers,
    timeframe_to_ns,
    units_for_margin,
    write_results,
)


if njit is not None:
    @njit(cache=True)
    def _simulate_basis_trail_numba(
        bid: np.ndarray,
        ask: np.ndarray,
        long_trigger: np.ndarray,
        short_trigger: np.ndarray,
        long_arm: np.ndarray,
        short_arm: np.ndarray,
        half_width: np.ndarray,
        day_id: np.ndarray,
        max_days: int,
        trail_points: float,
        trail_band_mult: float,
        tp_points: float,
        sl_points: float,
        point_size: float,
        amount: float,
        compound: bool,
        leverage: float,
        commission_per_million: float,
        side_mode: int,
    ):
        allow_long = side_mode == 1 or side_mode == 3
        allow_short = side_mode == 2 or side_mode == 3
        tp_dist = tp_points * point_size if tp_points > 0.0 else 0.0
        sl_dist = sl_points * point_size if sl_points > 0.0 else 0.0
        cash = amount
        equity_peak = amount
        max_dd = 0.0
        gross_win = 0.0
        gross_loss = 0.0
        trades = wins = losses = long_trades = short_trades = signal_exits = liquidations = 0
        account_dead = False
        pos = 0
        entry = 0.0
        units = 0.0
        best_long = 0.0
        best_short = 0.0
        armed = False
        cur_trade_drawdown = 0.0
        max_trade_drawdown = 0.0
        sum_trade_drawdown = 0.0
        daily_pnl = np.zeros(max_days, dtype=np.float64)
        active_days = np.zeros(max_days, dtype=np.int64)

        def add_daily(idx, pnl_value):
            d = day_id[idx]
            if d >= 0 and d < max_days:
                daily_pnl[d] += pnl_value
                active_days[d] = 1

        def close_long(exit_px, idx):
            nonlocal cash, pos, entry, units, trades, wins, losses
            nonlocal long_trades, signal_exits, gross_win, gross_loss
            nonlocal cur_trade_drawdown, max_trade_drawdown, sum_trade_drawdown
            fee = abs(exit_px * units) / 1_000_000.0 * commission_per_million
            pnl = (exit_px - entry) * units - fee
            cash += pnl
            add_daily(idx, pnl)
            trades += 1
            long_trades += 1
            signal_exits += 1
            if cur_trade_drawdown > max_trade_drawdown:
                max_trade_drawdown = cur_trade_drawdown
            sum_trade_drawdown += cur_trade_drawdown
            cur_trade_drawdown = 0.0
            if pnl >= 0.0:
                wins += 1
                gross_win += pnl
            else:
                losses += 1
                gross_loss += -pnl
            pos = 0
            entry = 0.0
            units = 0.0

        def close_short(exit_px, idx):
            nonlocal cash, pos, entry, units, trades, wins, losses
            nonlocal short_trades, signal_exits, gross_win, gross_loss
            nonlocal cur_trade_drawdown, max_trade_drawdown, sum_trade_drawdown
            fee = abs(exit_px * units) / 1_000_000.0 * commission_per_million
            pnl = (entry - exit_px) * units - fee
            cash += pnl
            add_daily(idx, pnl)
            trades += 1
            short_trades += 1
            signal_exits += 1
            if cur_trade_drawdown > max_trade_drawdown:
                max_trade_drawdown = cur_trade_drawdown
            sum_trade_drawdown += cur_trade_drawdown
            cur_trade_drawdown = 0.0
            if pnl >= 0.0:
                wins += 1
                gross_win += pnl
            else:
                losses += 1
                gross_loss += -pnl
            pos = 0
            entry = 0.0
            units = 0.0

        for i in range(len(bid) - 1):
            j = i + 1
            b = bid[j]
            a = ask[j]

            if pos != 0:
                live_u = (b - entry) * units if pos == 1 else (entry - a) * units
                if -live_u > cur_trade_drawdown:
                    cur_trade_drawdown = -live_u
                if cur_trade_drawdown < 0.0:
                    cur_trade_drawdown = 0.0
                equity = cash + live_u
                if equity > equity_peak:
                    equity_peak = equity
                dd = equity_peak - equity
                if dd > max_dd:
                    max_dd = dd
                if equity <= 0.0:
                    liquidations += 1
                    account_dead = True
                    cash = 0.0
                    pos = 0
                    break

            if pos == 1:
                if tp_dist > 0.0 and b >= entry + tp_dist:
                    close_long(b, j)
                    continue
                if sl_dist > 0.0 and b <= entry - sl_dist:
                    close_long(b, j)
                    continue
                if long_arm[i]:
                    armed = True
                if armed:
                    if b > best_long:
                        best_long = b
                    if trail_points > 0.0:
                        dist = trail_points * point_size
                    else:
                        w = half_width[i]
                        dist = w * trail_band_mult if np.isfinite(w) and w > 0.0 else 0.0
                    if dist > 0.0 and b <= best_long - dist:
                        close_long(b, j)
                        continue
            elif pos == -1:
                if tp_dist > 0.0 and a <= entry - tp_dist:
                    close_short(a, j)
                    continue
                if sl_dist > 0.0 and a >= entry + sl_dist:
                    close_short(a, j)
                    continue
                if short_arm[i]:
                    armed = True
                if armed:
                    if a < best_short:
                        best_short = a
                    if trail_points > 0.0:
                        dist = trail_points * point_size
                    else:
                        w = half_width[i]
                        dist = w * trail_band_mult if np.isfinite(w) and w > 0.0 else 0.0
                    if dist > 0.0 and a >= best_short + dist:
                        close_short(a, j)
                        continue

            if pos == 0:
                if allow_long and long_trigger[i]:
                    margin = cash if compound else amount
                    if margin <= 0.0:
                        break
                    entry = a
                    units = (margin * leverage) / entry
                    fee = abs(entry * units) / 1_000_000.0 * commission_per_million
                    cash -= fee
                    add_daily(j, -fee)
                    pos = 1
                    armed = False
                    best_long = b
                    cur_trade_drawdown = 0.0
                elif allow_short and short_trigger[i]:
                    margin = cash if compound else amount
                    if margin <= 0.0:
                        break
                    entry = b
                    units = (margin * leverage) / entry
                    fee = abs(entry * units) / 1_000_000.0 * commission_per_million
                    cash -= fee
                    add_daily(j, -fee)
                    pos = -1
                    armed = False
                    best_short = a
                    cur_trade_drawdown = 0.0

            if cash > equity_peak:
                equity_peak = cash
            dd_cash = equity_peak - cash
            if dd_cash > max_dd:
                max_dd = dd_cash

        open_u = 0.0
        open_side_code = 0
        open_bps = 0.0
        if pos == 1:
            open_side_code = 1
            open_u = (bid[-1] - entry) * units
            open_bps = (bid[-1] / entry - 1.0) * 10000.0
        elif pos == -1:
            open_side_code = -1
            open_u = (entry - ask[-1]) * units
            open_bps = (entry / ask[-1] - 1.0) * 10000.0

        realised = cash - amount
        total = realised + open_u
        pf = gross_win / gross_loss if gross_loss > 0.0 else (999.0 if gross_win > 0.0 else 0.0)
        trade_avg_drawdown = sum_trade_drawdown / trades if trades > 0 else 0.0

        day_sum = 0.0
        for d in range(max_days):
            day_sum += daily_pnl[d]
        avg_day = day_sum / max(max_days, 1)
        daily_sorted = daily_pnl.copy()
        daily_sorted.sort()
        if max_days == 0:
            median_day = 0.0
        elif max_days % 2 == 1:
            median_day = daily_sorted[max_days // 2]
        else:
            median_day = 0.5 * (daily_sorted[max_days // 2 - 1] + daily_sorted[max_days // 2])
        active_day_count = 0
        for d in range(max_days):
            active_day_count += active_days[d]

        return (
            realised, open_u, total, trades, wins, losses, pf, max_dd,
            long_trades, short_trades, signal_exits, liquidations,
            account_dead, open_side_code, open_bps, max_trade_drawdown,
            trade_avg_drawdown, avg_day, median_day, active_day_count,
        )


def rolling_sma_std(x: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    sma = np.full(len(x), np.nan, dtype=np.float64)
    std = np.full(len(x), np.nan, dtype=np.float64)
    if len(x) < length:
        return sma, std
    csum = np.cumsum(np.insert(x.astype(np.float64), 0, 0.0))
    csum2 = np.cumsum(np.insert((x.astype(np.float64) ** 2), 0, 0.0))
    sums = csum[length:] - csum[:-length]
    sums2 = csum2[length:] - csum2[:-length]
    mean = sums / float(length)
    var = np.maximum((sums2 / float(length)) - mean ** 2, 0.0)
    sma[length - 1:] = mean
    std[length - 1:] = np.sqrt(var)
    return sma, std


def closed_band_state_to_ticks(
    mid: np.ndarray,
    ts_ns: np.ndarray,
    timeframe: str,
    length: int,
    mult: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return previous-closed-candle basis/upper/lower mapped to every tick."""
    _, _, _, live_close, closed = live_candles(mid, ts_ns, timeframe)
    close_idx = np.flatnonzero(closed)
    if len(close_idx) == 0:
        nan = np.full(len(mid), np.nan, dtype=np.float64)
        return nan, nan, nan, nan

    closed_closes = live_close[close_idx]
    basis_c, std_c = rolling_sma_std(closed_closes, length)
    upper_c = basis_c + mult * std_c
    lower_c = basis_c - mult * std_c

    basis = np.full(len(mid), np.nan, dtype=np.float64)
    upper = np.full(len(mid), np.nan, dtype=np.float64)
    lower = np.full(len(mid), np.nan, dtype=np.float64)

    prev = 0
    last_basis = last_upper = last_lower = np.nan
    for j, idx in enumerate(close_idx):
        basis[prev:idx + 1] = last_basis
        upper[prev:idx + 1] = last_upper
        lower[prev:idx + 1] = last_lower
        last_basis = basis_c[j]
        last_upper = upper_c[j]
        last_lower = lower_c[j]
        prev = idx + 1
    basis[prev:] = last_basis
    upper[prev:] = last_upper
    lower[prev:] = last_lower
    return basis, upper, lower, live_close


def simulate_basis_trail(
    pair: str,
    params: str,
    tf: str,
    bid: np.ndarray,
    ask: np.ndarray,
    long_trigger: np.ndarray,
    short_trigger: np.ndarray,
    long_arm: np.ndarray,
    short_arm: np.ndarray,
    half_width: np.ndarray,
    trail_points: float,
    trail_band_mult: float,
    tp_points: float,
    sl_points: float,
    point_size: float,
    amount: float,
    compound: bool,
    leverage: float,
    commission_per_million: float,
    side: str,
    return_trades: bool = False,
    ts_ns: np.ndarray | None = None,
) -> TradeResult | tuple[TradeResult, list[dict]]:
    if njit is not None and not return_trades:
        if ts_ns is not None and len(ts_ns):
            day_id, max_days = day_ids_from_timestamps(ts_ns)
        else:
            day_id = np.zeros(len(bid), dtype=np.int64)
            max_days = 1
        side_mode = 3
        if side == "long":
            side_mode = 1
        elif side == "short":
            side_mode = 2
        out = _simulate_basis_trail_numba(
            bid, ask, long_trigger, short_trigger, long_arm, short_arm,
            half_width, day_id, max_days, trail_points, trail_band_mult,
            tp_points, sl_points,
            point_size, amount, compound, leverage, commission_per_million,
            side_mode,
        )
        (
            realised, open_u, total, trades, wins, losses, pf, max_dd,
            long_trades, short_trades, signal_exits, liquidations,
            account_dead, open_side_code, open_bps, trade_max_drawdown,
            trade_avg_drawdown, avg_day, median_day, active_days,
        ) = out
        open_side = "long" if int(open_side_code) == 1 else ("short" if int(open_side_code) == -1 else "-")
        win_rate = wins / trades * 100.0 if trades else 0.0
        result = TradeResult(
            pair, "bbtick", params, tf, tp_points, sl_points, point_size,
            float(realised), float(open_u), float(total), int(trades), int(wins),
            int(losses), float(win_rate), float(pf), float(max_dd),
            int(long_trades), int(short_trades), 0, int(signal_exits),
            int(liquidations), bool(account_dead), open_side, float(open_bps),
        )
        result.trade_max_drawdown = float(trade_max_drawdown)
        result.trade_avg_drawdown = float(trade_avg_drawdown)
        result.avg_day = float(avg_day)
        result.median_day = float(median_day)
        result.active_days = int(active_days)
        return result

    allow_long = side in ("long", "both")
    allow_short = side in ("short", "both")
    cash = amount
    equity_peak = amount
    max_dd = 0.0
    gross_win = gross_loss = 0.0
    trades = wins = losses = long_trades = short_trades = signal_exits = liquidations = 0
    account_dead = False
    pos = 0
    entry = units = 0.0
    best_long = best_short = 0.0
    armed = False
    trade_logs: list[dict] = []
    entry_i = -1
    if ts_ns is not None and len(ts_ns):
        day_id, max_days = day_ids_from_timestamps(ts_ns)
    else:
        day_id = np.zeros(len(bid), dtype=np.int64)
        max_days = 1
    daily_pnl = np.zeros(max_days, dtype=np.float64)
    daily_active = np.zeros(max_days, dtype=np.bool_)

    def add_daily(i: int, pnl: float) -> None:
        d = int(day_id[i]) if 0 <= i < len(day_id) else -1
        if 0 <= d < max_days:
            daily_pnl[d] += pnl
            daily_active[d] = True

    def trail_dist(i: int) -> float:
        if trail_points > 0:
            return trail_points * point_size
        w = float(half_width[i])
        return w * trail_band_mult if np.isfinite(w) and w > 0 else 0.0

    def open_pos(new_pos: int, px: float, i: int) -> None:
        nonlocal cash, pos, entry, units, best_long, best_short, armed, entry_i
        margin = cash if compound else amount
        if margin <= 0:
            return
        units = units_for_margin(margin, leverage, px)
        fee = commission(px, units, commission_per_million)
        cash -= fee
        add_daily(i, -fee)
        entry = px
        entry_i = i
        pos = new_pos
        armed = False
        if new_pos == 1:
            best_long = float(bid[i])
        else:
            best_short = float(ask[i])

    def close_pos(exit_px: float, exit_i: int) -> None:
        nonlocal cash, pos, entry, units, trades, wins, losses
        nonlocal long_trades, short_trades, signal_exits, gross_win, gross_loss
        side_name = "long" if pos == 1 else "short"
        pnl = ((exit_px - entry) if pos == 1 else (entry - exit_px)) * units
        pnl -= commission(exit_px, units, commission_per_million)
        cash += pnl
        add_daily(exit_i, pnl)
        trades += 1
        signal_exits += 1
        if pos == 1:
            long_trades += 1
        else:
            short_trades += 1
        if pnl >= 0:
            wins += 1
            gross_win += pnl
        else:
            losses += 1
            gross_loss += -pnl
        if return_trades:
            entry_time = ""
            exit_time = ""
            if ts_ns is not None and entry_i >= 0:
                entry_time = str(np.datetime64(int(ts_ns[entry_i]), "ns"))
            if ts_ns is not None:
                exit_time = str(np.datetime64(int(ts_ns[exit_i]), "ns"))
            trade_logs.append({
                "pair": pair,
                "strategy": "bbtick",
                "params": params,
                "timeframe": tf,
                "side": side_name,
                "entry_i": entry_i,
                "exit_i": exit_i,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_px": entry,
                "exit_px": exit_px,
                "pnl": pnl,
                "reason": "basis_trail",
                "equity": cash,
            })
        pos = 0
        entry = units = 0.0

    for i in range(len(bid) - 1):
        j = i + 1
        b = float(bid[j])
        a = float(ask[j])
        if pos != 0:
            live_u = open_unrealized(pos, entry, units, b, a)
            equity = cash + live_u
            equity_peak = max(equity_peak, equity)
            max_dd = max(max_dd, equity_peak - equity)
            if equity <= 0:
                liquidations += 1
                account_dead = True
                cash = 0.0
                pos = 0
                break

        if pos == 1:
            if tp_points > 0 and b >= entry + tp_points * point_size:
                close_pos(b, j)
                continue
            if sl_points > 0 and b <= entry - sl_points * point_size:
                close_pos(b, j)
                continue
            if long_arm[i]:
                armed = True
            if armed:
                best_long = max(best_long, b)
                dist = trail_dist(i)
                if dist > 0 and b <= best_long - dist:
                    close_pos(b, j)
                    continue
        elif pos == -1:
            if tp_points > 0 and a <= entry - tp_points * point_size:
                close_pos(a, j)
                continue
            if sl_points > 0 and a >= entry + sl_points * point_size:
                close_pos(a, j)
                continue
            if short_arm[i]:
                armed = True
            if armed:
                best_short = min(best_short, a)
                dist = trail_dist(i)
                if dist > 0 and a >= best_short + dist:
                    close_pos(a, j)
                    continue

        if pos == 0:
            if allow_long and long_trigger[i]:
                open_pos(1, a, j)
            elif allow_short and short_trigger[i]:
                open_pos(-1, b, j)

        equity_peak = max(equity_peak, cash)
        max_dd = max(max_dd, equity_peak - cash)

    open_u = open_unrealized(pos, entry, units, float(bid[-1]), float(ask[-1]))
    realised = cash - amount
    total = realised + open_u
    win_rate = wins / trades * 100.0 if trades else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
    open_side = "long" if pos == 1 else ("short" if pos == -1 else "-")
    open_bps = 0.0
    if pos == 1:
        open_bps = (float(bid[-1]) / entry - 1.0) * 10000.0
    elif pos == -1:
        open_bps = (entry / float(ask[-1]) - 1.0) * 10000.0
    result = TradeResult(
        pair, "bbtick", params, tf, tp_points, sl_points, point_size,
        realised, open_u, total, trades, wins, losses, win_rate, pf,
        max_dd, long_trades, short_trades, 0, signal_exits, liquidations,
        account_dead, open_side, open_bps,
    )
    result.avg_day = float(np.mean(daily_pnl)) if len(daily_pnl) else 0.0
    result.median_day = float(np.median(daily_pnl)) if len(daily_pnl) else 0.0
    result.active_days = int(np.count_nonzero(daily_active))
    if return_trades:
        return result, trade_logs
    return result


def main() -> None:
    ap = build_parser("Bollinger candle-band/tick-execution backtest", "forex_bollinger_tick_results.csv")
    ap.set_defaults(timeframes="30s,1m", tp_points="0,200,300,400", sl_points="0,200,300,400")
    ap.add_argument("--length", default="8,13,20,34")
    ap.add_argument("--mult", default="1.5,2,2.5")

    
    ap.add_argument("--entry-mode", default="tick_reclaim,close_reclaim",
                    help="comma list: tick_reclaim=live tick reclaim/reject; close_reclaim=closed candle reclaim/reject")
    ap.add_argument("--exit-mode", default="opposite",
                    help="comma list: opposite=reversal signal exits; basis=exit at middle band; basis_trail=arm trail after basis touch")
    ap.add_argument("--trail-points", default="0",
                    help="fixed trail override for basis_trail; 0 uses band-width multiplier")
    ap.add_argument("--trail-band-mult", default="0.15,0.2,0.25,0.3",
                    help="basis_trail distance = (basis-lower) * multiplier when trail-points=0")
    ap.add_argument("--reverse-on-flip", default="1",
                    help="comma list: 0=close only on signal, 1=reverse on signal")
    ap.add_argument("--sessions", default="-1,1",
                    help="-1=outside sessions, 0=all hours, 1=inside Tokyo/London/New York sessions")
    ap.add_argument("--invert-signals", action="store_true",
                    help="invert Bollinger entries: lower-band recovery -> short, upper-band rejection -> long")
    ap.add_argument("--trades-out", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    timeframes = parse_str_list(args.timeframes, ["1s"])
    lengths = [int(x) for x in parse_num_list(args.length, [13])]
    mults = parse_num_list(args.mult, [2.0])
    entry_modes = parse_str_list(args.entry_mode, ["tick_reclaim"])
    bad_entry_modes = [m for m in entry_modes if m not in ("tick_reclaim", "close_reclaim")]
    if bad_entry_modes:
        raise SystemExit(f"unsupported --entry-mode values: {bad_entry_modes}")
    exit_modes = parse_str_list(args.exit_mode, ["opposite", "basis"])
    bad_modes = [m for m in exit_modes if m not in ("opposite", "basis", "basis_trail")]
    if bad_modes:
        raise SystemExit(f"unsupported --exit-mode values: {bad_modes}")
    reverses = [bool(int(x)) for x in parse_num_list(args.reverse_on_flip, [1])]
    tps = parse_num_list(args.tp_points, [0.0])
    sls = parse_num_list(args.sl_points, [0.0])
    trails = parse_num_list(args.trail_points, [100.0])
    trail_mults = parse_num_list(args.trail_band_mult, [1.0])
    sessions = [int(x) for x in parse_num_list(args.sessions, [-1, 1])]

    ticks, _ = load_market(args)
    results = []
    all_trades = []
    for pair, g in ticks.groupby("pair", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        bid = g["bid"].to_numpy(np.float64)
        ask = g["ask"].to_numpy(np.float64)
        # MT5 native XAUUSD candles match bid OHLC, not mid/ask.
        mid = bid
        ts_ns = g["timestamp"].astype("int64").to_numpy()
        day_id, max_days = day_ids_from_timestamps(ts_ns)
        point_size = args.point_size or default_point_size(pair)
        session_cache = {s: active_session_allowed(ts_ns, s) for s in sessions}
        print(f"[bbtick] {pair} ticks={len(g):,}", flush=True)

        combos = [
            c for c in product(timeframes, lengths, mults, entry_modes, exit_modes, reverses, tps, sls, trails, trail_mults, sessions)
            if c[4] == "basis_trail" or c[9] == trail_mults[0]
        ]

        def run_combo(combo):
            tf, length, mult, entry_mode, exit_mode, rev, tp, sl, trail, trail_mult, sess = combo
            # Validate early so bad CLI values fail before a long loop.
            timeframe_to_ns(tf)
            basis, upper, lower, live_close = closed_band_state_to_ticks(mid, ts_ns, tf, length, mult)
            valid = np.isfinite(upper) & np.isfinite(lower)
            if entry_mode == "tick_reclaim":
                prev_mid = np.roll(mid, 1)
                prev_mid[0] = mid[0]
                # Levels are previous-closed-candle bands, execution is live tick.
                long_trigger = valid & (prev_mid <= lower) & (mid > lower)
                short_trigger = valid & (prev_mid >= upper) & (mid < upper)
            else:
                # Closed-candle confirmation: close below previous band, then a
                # later closed candle back above lower = long. Opposite for short.
                prev_close = np.roll(live_close, 1)
                prev_close[0] = np.nan
                candle_closed = np.r_[False, ts_ns[1:] // timeframe_to_ns(tf) != ts_ns[:-1] // timeframe_to_ns(tf)]
                long_trigger = valid & candle_closed & (prev_close <= lower) & (live_close > lower)
                short_trigger = valid & candle_closed & (prev_close >= upper) & (live_close < upper)

            if exit_mode in ("basis", "basis_trail"):
                long_exit = valid & (mid >= basis)
                short_exit = valid & (mid <= basis)
            else:
                long_exit = short_trigger
                short_exit = long_trigger

            if args.invert_signals:
                long_trigger, short_trigger = short_trigger, long_trigger
                long_exit, short_exit = short_exit, long_exit
            entry_allowed = session_cache[int(sess)]
            long_trigger = long_trigger & entry_allowed
            short_trigger = short_trigger & entry_allowed

            params = (
                f"tf={tf};length={length};mult={mult:g};entry={entry_mode};exit={exit_mode};"
                f"reverse={int(rev)};invert={int(args.invert_signals)};"
                f"trail={trail:g};trail_mult={trail_mult:g};session={sess}"
            )
            if exit_mode == "basis_trail":
                half_width = np.maximum(np.abs(basis - lower), np.abs(upper - basis))
                return simulate_basis_trail(
                    pair, params, tf, bid, ask, long_trigger, short_trigger,
                    long_exit, short_exit, half_width, trail, trail_mult,
                    tp, sl,
                    point_size, args.amount, args.compound, args.leverage,
                    args.commission_per_million, args.side,
                    return_trades=bool(args.trades_out),
                    ts_ns=ts_ns,
                )
            return simulate_triggers(
                pair, "bbtick", params, tf, bid, ask, long_trigger, short_trigger,
                long_exit, short_exit, tp, sl, point_size, args.amount, args.compound,
                args.leverage, args.commission_per_million, args.side,
                reverse_on_flip=rev,
                day_id=day_id,
                max_days=max_days,
                return_trades=bool(args.trades_out),
            )

        if args.workers > 1 and len(combos) > 1 and not args.trades_out:
            print(f"[bbtick] workers={args.workers} combos={len(combos)}", flush=True)
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(run_combo, c) for c in combos]
                for i, fut in enumerate(as_completed(futs), 1):
                    results.append(fut.result())
                    if i % max(1, len(combos) // 10) == 0:
                        print(f"[bbtick] progress {i}/{len(combos)}", flush=True)
        else:
            for combo in combos:
                out = run_combo(combo)
                if args.trades_out:
                    res, trades = out
                    results.append(res)
                    for t in trades:
                        if not isinstance(t, dict):
                            entry_i = int(t.entry_i)
                            exit_i = int(t.exit_i)
                            t = {
                                "pair": t.pair,
                                "strategy": t.strategy,
                                "params": t.params,
                                "timeframe": t.timeframe,
                                "side": t.side,
                                "entry_i": entry_i,
                                "exit_i": exit_i,
                                "entry_time": str(np.datetime64(int(ts_ns[entry_i]), "ns")),
                                "exit_time": str(np.datetime64(int(ts_ns[exit_i]), "ns")),
                                "entry_px": t.entry_px,
                                "exit_px": t.exit_px,
                                "pnl": t.pnl,
                                "reason": t.reason,
                                "equity": t.equity,
                            }
                        all_trades.append(t)
                else:
                    results.append(out)

    write_results(args.out, [r for r in results if r.trades >= args.min_trades], args.top, args.sort_by)
    if args.trades_out:
        os.makedirs(os.path.dirname(args.trades_out) or ".", exist_ok=True)
        with open(args.trades_out, "w", newline="", encoding="utf-8") as f:
            fields = [
                "pair", "strategy", "params", "timeframe", "side",
                "entry_i", "exit_i", "entry_time", "exit_time",
                "entry_px", "exit_px", "pnl", "reason", "equity",
            ]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for t in all_trades:
                w.writerow({
                    k: (round(v, 6) if isinstance(v, float) else v)
                    for k, v in t.items()
                    if k in fields
                })
        print(f"[bbtick] wrote trades {args.trades_out}", flush=True)
    print(f"[bbtick] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
