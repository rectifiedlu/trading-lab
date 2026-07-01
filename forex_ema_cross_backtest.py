"""Single-EMA confirmed side-change backtest.

Logic:
    - Compute EMA on closed candle closes.
    - Track consecutive closes above/below EMA at all times.
    - Enter long when confirmed above EMA.
    - Enter short when confirmed below EMA.
    - Long exits only when confirmed below EMA.
    - Short exits only when confirmed above EMA.
    - TP/SL default to 0, so trades switch only on confirmed side changes.

Execution uses bid/ask ticks through the shared simulator.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
import csv

import numpy as np

from forex_strategy_common import (
    TradeResult,
    build_parser,
    candle_state_to_ticks,
    commission,
    closed_candle_series,
    day_ids_from_timestamps,
    default_point_size,
    load_market,
    njit,
    open_unrealized,
    parse_num_list,
    parse_str_list,
    simulate_triggers,
    units_for_margin,
    write_results,
)


if njit is not None:
    @njit(cache=True)
    def _ema_numba(x: np.ndarray, length: int) -> np.ndarray:
        out = np.empty(len(x), dtype=np.float64)
        if length <= 1:
            for i in range(len(x)):
                out[i] = x[i]
            return out
        alpha = 2.0 / (length + 1.0)
        val = x[0]
        for i in range(len(x)):
            val = alpha * x[i] + (1.0 - alpha) * val
            out[i] = val
        return out


    @njit(cache=True)
    def _confirmed_states_numba(
        close: np.ndarray,
        basis: np.ndarray,
        confirm: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        side = np.zeros(len(close), dtype=np.float64)
        above = np.zeros(len(close), dtype=np.bool_)
        below = np.zeros(len(close), dtype=np.bool_)
        above_count = 0
        below_count = 0
        for i in range(len(close)):
            px = close[i]
            ma = basis[i]
            if not np.isfinite(ma):
                above_count = 0
                below_count = 0
            elif px > ma:
                above[i] = True
                above_count += 1
                below_count = 0
            elif px < ma:
                below[i] = True
                below_count += 1
                above_count = 0
            else:
                above_count = 0
                below_count = 0
            if above_count >= confirm:
                side[i] = 1.0
            elif below_count >= confirm:
                side[i] = -1.0
        return side, above, below


    @njit(cache=True)
    def _dual_confirmed_states_numba(
        signal_close: np.ndarray,
        signal_close_idx: np.ndarray,
        ema_basis_by_tick: np.ndarray,
        confirm: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        side = np.zeros(len(signal_close), dtype=np.float64)
        above = np.zeros(len(signal_close), dtype=np.bool_)
        below = np.zeros(len(signal_close), dtype=np.bool_)
        above_count = 0
        below_count = 0
        for i in range(len(signal_close)):
            px = signal_close[i]
            ma = ema_basis_by_tick[signal_close_idx[i]]
            if not np.isfinite(ma):
                above_count = 0
                below_count = 0
            elif px > ma:
                above[i] = True
                above_count += 1
                below_count = 0
            elif px < ma:
                below[i] = True
                below_count += 1
                above_count = 0
            else:
                above_count = 0
                below_count = 0
            if above_count >= confirm:
                side[i] = 1.0
            elif below_count >= confirm:
                side[i] = -1.0
        return side, above, below


    @njit(cache=True)
    def _simulate_trailing_signals_numba(
        bid: np.ndarray,
        ask: np.ndarray,
        long_trigger: np.ndarray,
        short_trigger: np.ndarray,
        long_exit: np.ndarray,
        short_exit: np.ndarray,
        trail_points: float,
        point_size: float,
        amount: float,
        compound: bool,
        leverage: float,
        commission_per_million: float,
        side_mode: int,
        reverse_on_flip: bool,
    ):
        trail_dist = trail_points * point_size
        allow_long = side_mode == 1 or side_mode == 3
        allow_short = side_mode == 2 or side_mode == 3
        cash = amount
        equity_peak = amount
        max_dd = 0.0
        gross_win = 0.0
        gross_loss = 0.0
        trades = wins = losses = long_trades = short_trades = signal_exits = 0
        liquidations = 0
        account_dead = False
        pos = 0
        entry = 0.0
        units = 0.0
        best_long = 0.0
        best_short = 0.0

        for i in range(len(bid) - 1):
            b = bid[i + 1]
            a = ask[i + 1]
            reverse_to = 0

            if pos != 0:
                live_u = (b - entry) * units if pos == 1 else (entry - a) * units
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
                if b > best_long:
                    best_long = b
                trail_hit = trail_dist > 0.0 and b <= best_long - trail_dist
                flip_hit = long_exit[i]
                if trail_hit or flip_hit:
                    fee = abs(b * units) / 1_000_000.0 * commission_per_million
                    pnl = (b - entry) * units - fee
                    cash += pnl
                    trades += 1
                    long_trades += 1
                    signal_exits += 1
                    if pnl >= 0.0:
                        wins += 1
                        gross_win += pnl
                    else:
                        losses += 1
                        gross_loss += -pnl
                    pos = 0
                    entry = 0.0
                    units = 0.0
                    if flip_hit and reverse_on_flip and allow_short:
                        reverse_to = -1
                    else:
                        continue
            elif pos == -1:
                if a < best_short:
                    best_short = a
                trail_hit = trail_dist > 0.0 and a >= best_short + trail_dist
                flip_hit = short_exit[i]
                if trail_hit or flip_hit:
                    fee = abs(a * units) / 1_000_000.0 * commission_per_million
                    pnl = (entry - a) * units - fee
                    cash += pnl
                    trades += 1
                    short_trades += 1
                    signal_exits += 1
                    if pnl >= 0.0:
                        wins += 1
                        gross_win += pnl
                    else:
                        losses += 1
                        gross_loss += -pnl
                    pos = 0
                    entry = 0.0
                    units = 0.0
                    if flip_hit and reverse_on_flip and allow_long:
                        reverse_to = 1
                    else:
                        continue

            if pos == 0:
                if reverse_to == 1:
                    margin = cash if compound else amount
                    if margin <= 0.0:
                        break
                    entry = a
                    units = (margin * leverage) / entry
                    cash -= abs(entry * units) / 1_000_000.0 * commission_per_million
                    pos = 1
                    best_long = entry
                elif reverse_to == -1:
                    margin = cash if compound else amount
                    if margin <= 0.0:
                        break
                    entry = b
                    units = (margin * leverage) / entry
                    cash -= abs(entry * units) / 1_000_000.0 * commission_per_million
                    pos = -1
                    best_short = entry
                elif allow_long and long_trigger[i]:
                    margin = cash if compound else amount
                    if margin <= 0.0:
                        break
                    entry = a
                    units = (margin * leverage) / entry
                    cash -= abs(entry * units) / 1_000_000.0 * commission_per_million
                    pos = 1
                    best_long = entry
                elif allow_short and short_trigger[i]:
                    margin = cash if compound else amount
                    if margin <= 0.0:
                        break
                    entry = b
                    units = (margin * leverage) / entry
                    cash -= abs(entry * units) / 1_000_000.0 * commission_per_million
                    pos = -1
                    best_short = entry

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
        return (
            realised, open_u, total, trades, wins, losses, pf, max_dd,
            long_trades, short_trades, signal_exits, liquidations,
            account_dead, open_side_code, open_bps,
        )


def ema(x: np.ndarray, length: int) -> np.ndarray:
    if njit is not None:
        return _ema_numba(x.astype(np.float64), int(length))
    out = np.full(len(x), np.nan, dtype=np.float64)
    if length <= 1:
        return x.astype(np.float64)
    alpha = 2.0 / (length + 1.0)
    val = float(x[0])
    for i, px in enumerate(x):
        val = alpha * float(px) + (1.0 - alpha) * val
        out[i] = val
    return out


def confirmed_states(close: np.ndarray, basis: np.ndarray, confirm: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if njit is not None:
        return _confirmed_states_numba(
            close.astype(np.float64),
            basis.astype(np.float64),
            int(confirm),
        )
    side = np.zeros(len(close), dtype=np.float64)
    above_count = 0
    below_count = 0
    for i, (px, ma) in enumerate(zip(close, basis)):
        if not np.isfinite(ma):
            above_count = 0
            below_count = 0
        elif px > ma:
            above_count += 1
            below_count = 0
        elif px < ma:
            below_count += 1
            above_count = 0
        else:
            above_count = 0
            below_count = 0
        if above_count >= confirm:
            side[i] = 1.0
        elif below_count >= confirm:
            side[i] = -1.0
    above = close > basis
    below = close < basis
    return side, above, below


def dual_confirmed_states(
    signal_close: np.ndarray,
    signal_close_idx: np.ndarray,
    ema_basis_by_tick: np.ndarray,
    confirm: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if njit is not None:
        return _dual_confirmed_states_numba(
            signal_close.astype(np.float64),
            signal_close_idx.astype(np.int64),
            ema_basis_by_tick.astype(np.float64),
            int(confirm),
        )
    basis_at_signal = ema_basis_by_tick[signal_close_idx]
    side = np.zeros(len(signal_close), dtype=np.float64)
    above_count = 0
    below_count = 0
    for i, (px, ma) in enumerate(zip(signal_close, basis_at_signal)):
        if not np.isfinite(ma):
            above_count = 0
            below_count = 0
        elif px > ma:
            above_count += 1
            below_count = 0
        elif px < ma:
            below_count += 1
            above_count = 0
        else:
            above_count = 0
            below_count = 0
        if above_count >= confirm:
            side[i] = 1.0
        elif below_count >= confirm:
            side[i] = -1.0
    above = signal_close > basis_at_signal
    below = signal_close < basis_at_signal
    return side, above, below


def deadband_cross_triggers(diff: np.ndarray, deadband: float) -> tuple[np.ndarray, np.ndarray]:
    prev = np.roll(diff, 1)
    prev[0] = np.nan
    long_trigger = (prev < deadband) & (diff >= deadband)
    short_trigger = (prev > -deadband) & (diff <= -deadband)
    return long_trigger, short_trigger


def simulate_trailing_signals(
    pair: str,
    strategy: str,
    params: str,
    timeframe: str,
    bid: np.ndarray,
    ask: np.ndarray,
    long_trigger: np.ndarray,
    short_trigger: np.ndarray,
    long_exit: np.ndarray,
    short_exit: np.ndarray,
    trail_points: float,
    point_size: float,
    amount: float,
    compound: bool,
    leverage: float,
    commission_per_million: float,
    side: str,
    reverse_on_flip: bool,
) -> TradeResult:
    if njit is not None:
        side_mode = 3
        if side == "long":
            side_mode = 1
        elif side == "short":
            side_mode = 2
        out = _simulate_trailing_signals_numba(
            bid, ask, long_trigger, short_trigger, long_exit, short_exit,
            trail_points, point_size, amount, compound, leverage,
            commission_per_million, side_mode, reverse_on_flip,
        )
        (
            realised, open_u, total, trades, wins, losses, pf, max_dd,
            long_trades, short_trades, signal_exits, liquidations,
            account_dead, open_side_code, open_bps,
        ) = out
        open_side = "long" if int(open_side_code) == 1 else ("short" if int(open_side_code) == -1 else "-")
        win_rate = wins / trades * 100.0 if trades else 0.0
        return TradeResult(
            pair, strategy, params, timeframe, 0.0, 0.0, point_size,
            float(realised), float(open_u), float(total), int(trades),
            int(wins), int(losses), float(win_rate), float(pf),
            float(max_dd), int(long_trades), int(short_trades), 0,
            int(signal_exits), int(liquidations), bool(account_dead),
            open_side, float(open_bps),
        )

    trail_dist = trail_points * point_size
    allow_long = side in ("long", "both")
    allow_short = side in ("short", "both")
    cash = amount
    start_balance = amount
    equity_peak = amount
    max_dd = 0.0
    gross_win = gross_loss = 0.0
    trades = wins = losses = long_trades = short_trades = signal_exits = 0
    liquidations = 0
    account_dead = False
    pos = 0
    entry = 0.0
    units = 0.0
    best_long = 0.0
    best_short = 0.0

    def open_pos(new_pos: int, px: float) -> bool:
        nonlocal cash, pos, entry, units, best_long, best_short
        margin = cash if compound else amount
        if margin <= 0:
            return False
        units = units_for_margin(margin, leverage, px)
        cash -= commission(px, units, commission_per_million)
        entry = px
        pos = new_pos
        if new_pos == 1:
            best_long = px
        else:
            best_short = px
        return True

    def close_pos(exit_px: float) -> float:
        nonlocal cash, pos, entry, units, trades, wins, losses
        nonlocal long_trades, short_trades, signal_exits, gross_win, gross_loss
        pnl = ((exit_px - entry) if pos == 1 else (entry - exit_px)) * units
        pnl -= commission(exit_px, units, commission_per_million)
        cash += pnl
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
        pos = 0
        entry = units = 0.0
        return pnl

    for i in range(len(bid) - 1):
        b = float(bid[i + 1])
        a = float(ask[i + 1])
        reverse_to = 0

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
            best_long = max(best_long, b)
            trail_hit = b <= best_long - trail_dist
            flip_hit = bool(long_exit[i])
            if trail_hit or flip_hit:
                close_pos(b)
                if flip_hit and reverse_on_flip and allow_short:
                    reverse_to = -1
                else:
                    continue
        elif pos == -1:
            best_short = min(best_short, a)
            trail_hit = a >= best_short + trail_dist
            flip_hit = bool(short_exit[i])
            if trail_hit or flip_hit:
                close_pos(a)
                if flip_hit and reverse_on_flip and allow_long:
                    reverse_to = 1
                else:
                    continue

        if pos == 0:
            if reverse_to == 1:
                open_pos(1, a)
            elif reverse_to == -1:
                open_pos(-1, b)
            elif allow_long and long_trigger[i]:
                open_pos(1, a)
            elif allow_short and short_trigger[i]:
                open_pos(-1, b)

        equity_peak = max(equity_peak, cash)
        max_dd = max(max_dd, equity_peak - cash)

    open_u = open_unrealized(pos, entry, units, float(bid[-1]), float(ask[-1]))
    realised = cash - start_balance
    total = realised + open_u
    win_rate = wins / trades * 100.0 if trades else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
    open_side = "long" if pos == 1 else ("short" if pos == -1 else "-")
    open_bps = 0.0
    if pos == 1:
        open_bps = (float(bid[-1]) / entry - 1.0) * 10000.0
    elif pos == -1:
        open_bps = (entry / float(ask[-1]) - 1.0) * 10000.0
    return TradeResult(
        pair, strategy, params, timeframe, 0.0, 0.0, point_size,
        realised, open_u, total, trades, wins, losses, win_rate, pf,
        max_dd, long_trades, short_trades, 0, signal_exits, liquidations,
        account_dead, open_side, open_bps,
    )


def main() -> None:
    ap = build_parser("EMA candle-confirmation cross tick backtest", "forex_ema_cross_results.csv")
    ap.set_defaults(timeframes="1s,5s,10s,15s,20s,30s,1m", tp_points="0", sl_points="0")
    ap.add_argument("--ema-timeframes", default=None,
                    help="optional EMA candle timeframes; default uses the same timeframe as the signal")
    ap.add_argument(
        "--ema-lengths",
        default="3,4,5,6,8,10,12,14,16,18,21,25,30,34,42,50,63,75,84,100,125,150,200,250,300,400,500",
    )
    ap.add_argument("--confirm-candles", default="1,2")
    ap.add_argument("--signal-mode", default="candle",
                    help="comma list from: candle,tick,sample")
    ap.add_argument("--reverse-on-flip", default="0",
                    help="comma list: 0=close only, 1=close and open opposite")
    ap.add_argument("--sample-sec", default="5,10,20,30",
                    help="comma list of sample intervals for sample mode")
    ap.add_argument("--deadband-points", default="0,25,50,100",
                    help="shortcut deadband in XAU points when entry/exit deadband are omitted")
    ap.add_argument("--entry-deadband-points", default=None)
    ap.add_argument("--exit-deadband-points", default=None)
    ap.add_argument("--trail-points", default="0,100,150,200,300,500,800",
                    help="0=off; otherwise exit longs/shorts by pure trailing points from best price")
    ap.add_argument("--trades-out", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    modes = parse_str_list(args.signal_mode, ["sample", "tick", "candle"])
    bad_modes = [m for m in modes if m not in ("sample", "tick", "candle")]
    if bad_modes:
        raise SystemExit(f"unsupported signal mode(s): {','.join(bad_modes)}")
    reverse_values = [bool(int(x)) for x in parse_num_list(args.reverse_on_flip, [0, 1])]

    timeframes = parse_str_list(args.timeframes, ["1m", "3m", "5m"])
    ema_timeframes = parse_str_list(args.ema_timeframes, [])
    ema_lengths = [int(x) for x in parse_num_list(args.ema_lengths, [21])]
    confirms = [int(x) for x in parse_num_list(args.confirm_candles, [2])]
    sample_secs = [int(x) for x in parse_num_list(args.sample_sec, [10])]
    entry_deadbands = parse_num_list(args.entry_deadband_points, parse_num_list(args.deadband_points, [0]))
    exit_deadbands = parse_num_list(args.exit_deadband_points, parse_num_list(args.deadband_points, [0]))
    tps = parse_num_list(args.tp_points, [0])
    sls = parse_num_list(args.sl_points, [0])
    trails = parse_num_list(args.trail_points, [0])

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
        print(f"[ema-cross] {pair} ticks={len(g):,}", flush=True)
        candle_cache = {
            tf: closed_candle_series(mid, ts, tf)
            for tf in sorted(set(timeframes + ema_timeframes))
        }
        ema_tick_cache = {}
        for ema_tf in sorted(set(ema_timeframes + timeframes)):
            ema_close, ema_close_tick_idx = candle_cache[ema_tf]
            if len(ema_close) < 2:
                continue
            for ema_len in ema_lengths:
                if len(ema_close) < ema_len + 2:
                    continue
                ema_basis = ema(ema_close, ema_len)
                ema_tick_cache[(ema_tf, ema_len)] = candle_state_to_ticks(
                    len(bid), ema_close_tick_idx, ema_basis,
                )

        mode_grid = []
        for mode in modes:
            if mode == "candle":
                mode_grid.extend(
                    (mode, signal_tf, ema_tf, ema_len, confirm, 0, 0.0, 0.0, rev, tp, sl, trail)
                    for signal_tf, ema_len, confirm, rev, tp, sl, trail in product(
                        timeframes, ema_lengths, confirms, reverse_values, tps, sls, trails,
                    )
                    for ema_tf in ([signal_tf] if not ema_timeframes else ema_timeframes)
                )
            elif mode == "tick":
                mode_grid.extend(
                    (mode, tf, tf, ema_len, 1, 0, entry_db, exit_db, rev, tp, sl, trail)
                    for tf, ema_len, entry_db, exit_db, rev, tp, sl, trail in product(
                        timeframes, ema_lengths, entry_deadbands, exit_deadbands,
                        reverse_values, tps, sls, trails,
                    )
                )
            else:
                mode_grid.extend(
                    (mode, tf, tf, ema_len, 1, sample_sec, entry_db, exit_db, rev, tp, sl, trail)
                    for tf, ema_len, sample_sec, entry_db, exit_db, rev, tp, sl, trail in product(
                        timeframes, ema_lengths, sample_secs, entry_deadbands,
                        exit_deadbands, reverse_values, tps, sls, trails,
                    )
                )

        print(f"[ema-cross] combos={len(mode_grid):,} workers={args.workers}", flush=True)

        def run_combo(combo):
            mode, tf, ema_tf, ema_len, confirm, sample_sec, entry_db_points, exit_db_points, rev, tp, sl, trail = combo
            if confirm < 1:
                return None
            entry_deadband = entry_db_points * point_size
            exit_deadband = exit_db_points * point_size

            if mode == "candle":
                signal_close, signal_close_tick_idx = candle_cache[tf]
                basis_by_tick = ema_tick_cache.get((ema_tf, ema_len))
                if len(signal_close) < confirm + 2 or basis_by_tick is None:
                    return None
                confirmed_side, above, below = dual_confirmed_states(
                    signal_close, signal_close_tick_idx, basis_by_tick, confirm,
                )

                prev_confirmed = np.roll(confirmed_side, 1)
                prev_confirmed[0] = 0.0
                candle_long_trigger = (confirmed_side == 1) & (prev_confirmed != 1)
                candle_short_trigger = (confirmed_side == -1) & (prev_confirmed != -1)

                # Symmetric exits: same confirmation requirement as entries.
                candle_long_exit = confirmed_side == -1
                candle_short_exit = confirmed_side == 1

                long_trigger = candle_state_to_ticks(
                    len(bid), signal_close_tick_idx, candle_long_trigger.astype(float),
                ) == 1
                short_trigger = candle_state_to_ticks(
                    len(bid), signal_close_tick_idx, candle_short_trigger.astype(float),
                ) == 1
                long_exit = candle_state_to_ticks(
                    len(bid), signal_close_tick_idx, candle_long_exit.astype(float),
                ) == 1
                short_exit = candle_state_to_ticks(
                    len(bid), signal_close_tick_idx, candle_short_exit.astype(float),
                ) == 1
            elif mode == "tick":
                close, close_tick_idx = candle_cache[tf]
                basis = ema_tick_cache.get((tf, ema_len))
                if len(close) < ema_len + 2 or basis is None:
                    return None
                entry_upper = basis + entry_deadband
                entry_lower = basis - entry_deadband
                exit_upper = basis + exit_deadband
                exit_lower = basis - exit_deadband
                live_price = mid
                prev_close = np.roll(live_price, 1)
                prev_close[0] = np.nan
                prev_entry_upper = np.roll(entry_upper, 1)
                prev_entry_upper[0] = np.nan
                prev_entry_lower = np.roll(entry_lower, 1)
                prev_entry_lower[0] = np.nan
                prev_exit_upper = np.roll(exit_upper, 1)
                prev_exit_upper[0] = np.nan
                prev_exit_lower = np.roll(exit_lower, 1)
                prev_exit_lower[0] = np.nan
                # Explicit band-cross logic:
                # entry can use a different band than exit/flip.
                long_trigger = (prev_close < prev_entry_upper) & (live_price >= entry_upper)
                short_trigger = (prev_close > prev_entry_lower) & (live_price <= entry_lower)
                long_exit = (prev_close > prev_exit_lower) & (live_price <= exit_lower)
                short_exit = (prev_close < prev_exit_upper) & (live_price >= exit_upper)
            else:
                close, close_tick_idx = candle_cache[tf]
                basis = ema_tick_cache.get((tf, ema_len))
                if len(close) < ema_len + 2 or basis is None:
                    return None
                entry_upper = basis + entry_deadband
                entry_lower = basis - entry_deadband
                exit_upper = basis + exit_deadband
                exit_lower = basis - exit_deadband
                sample_ns = int(sample_sec * 1_000_000_000)
                long_trigger = np.zeros(len(bid), dtype=np.bool_)
                short_trigger = np.zeros(len(bid), dtype=np.bool_)
                last_region = 0
                next_sample = int(ts[0])
                for i, (t_ns, px, ent_up, ent_lo, ex_up, ex_lo) in enumerate(
                    zip(ts, mid, entry_upper, entry_lower, exit_upper, exit_lower)
                ):
                    if not all(np.isfinite(v) for v in (ent_up, ent_lo, ex_up, ex_lo)):
                        continue
                    if int(t_ns) < next_sample:
                        continue
                    while next_sample <= int(t_ns):
                        next_sample += sample_ns
                    if px >= ent_up:
                        region = 1
                    elif px <= ent_lo:
                        region = -1
                    elif last_region == 1 and px <= ex_lo:
                        region = -1
                    elif last_region == -1 and px >= ex_up:
                        region = 1
                    else:
                        region = 0
                    if region == 1 and last_region != 1:
                        long_trigger[i] = True
                    elif region == -1 and last_region != -1:
                        short_trigger[i] = True
                    last_region = region
                long_exit = short_trigger
                short_exit = long_trigger

            params = (
                f"mode={mode};tf={tf};ema_tf={ema_tf};ema={ema_len};confirm={confirm};"
                f"sample={sample_sec};entry_db={entry_db_points:g};"
                f"exit_db={exit_db_points:g};reverse={int(rev)};trail={trail:g}"
            )
            if trail > 0 and tp == 0 and sl == 0:
                return simulate_trailing_signals(
                    pair, "emacross", params, tf, bid, ask, long_trigger, short_trigger,
                    long_exit, short_exit, trail, point_size, args.amount,
                    args.compound, args.leverage, args.commission_per_million,
                    args.side, rev,
                )
            return simulate_triggers(
                pair, "emacross", params, tf, bid, ask, long_trigger,
                short_trigger, long_exit, short_exit, tp, sl, point_size,
                args.amount, args.compound, args.leverage,
                args.commission_per_million, args.side,
                reverse_on_flip=rev,
                day_id=day_id,
                max_days=max_days,
                return_trades=bool(args.trades_out),
            )

        if args.workers > 1 and len(mode_grid) > 1 and not args.trades_out:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(run_combo, combo) for combo in mode_grid]
                for i, fut in enumerate(as_completed(futs), 1):
                    sim = fut.result()
                    if sim is not None:
                        results.append(sim)
                    if i % max(1, len(mode_grid) // 20) == 0 or i == len(mode_grid):
                        print(f"[ema-cross] progress {i}/{len(mode_grid)}", flush=True)
        else:
            for i, combo in enumerate(mode_grid, 1):
                sim = run_combo(combo)
                if sim is None:
                    continue
                if args.trades_out:
                    res, trades = sim
                    results.append(res)
                    all_trades.extend(trades)
                else:
                    results.append(sim)
                if i % max(1, len(mode_grid) // 20) == 0 or i == len(mode_grid):
                    print(f"[ema-cross] progress {i}/{len(mode_grid)}", flush=True)

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
        print(f"[ema-cross] wrote trades {args.trades_out}", flush=True)
    print(f"[ema-cross] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
