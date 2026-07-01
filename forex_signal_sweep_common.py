"""Shared helpers for simple signal/state forex strategy sweeps."""

from __future__ import annotations

import numpy as np
import pandas as pd

from forex_strategy_common import (
    TradeResult,
    commission,
    day_ids_from_timestamps,
    default_point_size,
    njit,
    timeframe_to_ns,
    units_for_margin,
)


def build_bid_ohlc(mid: np.ndarray, ts_ns: np.ndarray, timeframe: str):
    tf_ns = timeframe_to_ns(timeframe)
    bucket = ts_ns // tf_ns
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    close_idx: list[int] = []
    cur_bucket = int(bucket[0])
    o = h = l = c = float(mid[0])
    last_i = 0
    for i, px0 in enumerate(mid):
        b = int(bucket[i])
        px = float(px0)
        if b != cur_bucket:
            opens.append(o)
            highs.append(h)
            lows.append(l)
            closes.append(c)
            close_idx.append(last_i)
            cur_bucket = b
            o = h = l = c = px
        else:
            h = max(h, px)
            l = min(l, px)
            c = px
        last_i = i
    return (
        np.array(opens, dtype=np.float64),
        np.array(highs, dtype=np.float64),
        np.array(lows, dtype=np.float64),
        np.array(closes, dtype=np.float64),
        np.array(close_idx, dtype=np.int64),
    )


def map_state_to_ticks(n_ticks: int, close_idx: np.ndarray, state: np.ndarray) -> np.ndarray:
    out = np.zeros(n_ticks, dtype=np.float64)
    prev = 0
    last = 0.0
    for idx, st in zip(close_idx, state):
        out[prev:idx + 1] = last
        last = float(st)
        prev = idx + 1
    out[prev:] = last
    return out


def rma(x: np.ndarray, length: int) -> np.ndarray:
    out = np.full(len(x), np.nan, dtype=np.float64)
    if len(x) < length:
        return out
    seed_i = -1
    for i in range(length - 1, len(x)):
        window = x[i - length + 1:i + 1]
        if np.all(np.isfinite(window)):
            seed_i = i
            break
    if seed_i < 0:
        return out
    val = float(np.mean(x[seed_i - length + 1:seed_i + 1]))
    out[seed_i] = val
    for i in range(seed_i + 1, len(x)):
        if not np.isfinite(x[i]):
            continue
        val = (val * (length - 1) + float(x[i])) / length
        out[i] = val
    return out


def ema(x: np.ndarray, length: int) -> np.ndarray:
    out = np.empty(len(x), dtype=np.float64)
    alpha = 2.0 / (length + 1.0)
    val = float(x[0])
    for i, v in enumerate(x):
        val = alpha * float(v) + (1.0 - alpha) * val
        out[i] = val
    return out


def rolling_high_prev(x: np.ndarray, length: int) -> np.ndarray:
    return pd.Series(x).rolling(length, min_periods=length).max().shift(1).to_numpy(np.float64)


def rolling_low_prev(x: np.ndarray, length: int) -> np.ndarray:
    return pd.Series(x).rolling(length, min_periods=length).min().shift(1).to_numpy(np.float64)


def side_mode_value(side: str) -> int:
    if side == "long":
        return 1
    if side == "short":
        return 2
    return 3


if njit is not None:
    @njit(cache=True)
    def _simulate_state_candle_numba(
        close_px: np.ndarray,
        day_id: np.ndarray,
        max_days: int,
        state: np.ndarray,
        raw_state: np.ndarray,
        upper_band: np.ndarray,
        lower_band: np.ndarray,
        entry_allowed: np.ndarray,
        tp_points: float,
        trail_points: float,
        loss_cut_points: float,
        point_size: float,
        amount: float,
        compound: bool,
        leverage: float,
        commission_per_million: float,
        rebreak_points: float,
        side_mode: int,
    ):
        tp_dist = tp_points * point_size if tp_points > 0.0 else 0.0
        trail_dist = trail_points * point_size if trail_points > 0.0 else 0.0
        loss_dist = loss_cut_points * point_size if loss_cut_points > 0.0 else 0.0
        rebreak_dist = rebreak_points * point_size if rebreak_points >= 0.0 else -1.0
        allow_long = side_mode == 1 or side_mode == 3
        allow_short = side_mode == 2 or side_mode == 3

        cash = amount
        equity_peak = amount
        cash_peak = amount
        max_dd = 0.0
        cum_max_dd = 0.0
        gross_win = 0.0
        gross_loss = 0.0
        trades = wins = losses = long_trades = short_trades = 0
        stop_losses = signal_exits = liquidations = 0
        pos = 0
        entry = 0.0
        units = 0.0
        best_px = 0.0
        returned_inside = False
        cur_trade_drawdown = 0.0
        max_trade_drawdown = 0.0
        worst_trade_pnl = 0.0
        loss_values = np.empty(100000, dtype=np.float64)
        loss_count = 0
        daily_pnl = np.zeros(max_days, dtype=np.float64)

        def add_daily(idx, pnl_value):
            d = day_id[idx]
            if d >= 0 and d < max_days:
                daily_pnl[d] += pnl_value

        prev_state = np.empty(len(state), dtype=np.float64)
        prev_state[0] = 0.0
        for k in range(1, len(state)):
            prev_state[k] = state[k - 1]

        for i in range(len(close_px)):
            px_live = close_px[i]
            if pos != 0:
                live_u = (px_live - entry) * units if pos == 1 else (entry - px_live) * units
                if -live_u > cur_trade_drawdown:
                    cur_trade_drawdown = -live_u
                if cur_trade_drawdown < 0.0:
                    cur_trade_drawdown = 0.0
                eq = cash + live_u
                if eq > equity_peak:
                    equity_peak = eq
                dd = equity_peak - eq
                if dd > max_dd:
                    max_dd = dd

            close = False
            exit_px = 0.0
            is_stop = False
            if pos == 1:
                if raw_state[i] == 0.0:
                    returned_inside = True
                open_pnl = (px_live - entry) * units
                open_points = (px_live - entry) / point_size
                if rebreak_dist >= 0.0 and returned_inside and px_live <= lower_band[i] - rebreak_dist:
                    close = True
                    exit_px = px_live
                elif tp_dist > 0.0 and trail_dist > 0.0:
                    if px_live >= entry + tp_dist and px_live > best_px:
                        best_px = px_live
                    if best_px >= entry + tp_dist and px_live <= best_px - trail_dist:
                        close = True
                        exit_px = px_live
                elif tp_dist > 0.0 and trail_dist <= 0.0 and px_live >= entry + tp_dist:
                    close = True
                    exit_px = px_live
                if (not close) and loss_dist > 0.0 and open_points <= -loss_cut_points:
                    close = True
                    exit_px = px_live
                    is_stop = True
                elif (not close) and state[i] == -1.0 and (tp_points <= 0.0 or open_pnl < 0.0):
                    close = True
                    exit_px = px_live
                if close:
                    pnl = (exit_px - entry) * units
                    pnl -= abs(exit_px * units) / 1_000_000.0 * commission_per_million
                    cash += pnl
                    add_daily(i, pnl)
                    trades += 1
                    long_trades += 1
                    if is_stop:
                        stop_losses += 1
                    else:
                        signal_exits += 1
                    if cur_trade_drawdown > max_trade_drawdown:
                        max_trade_drawdown = cur_trade_drawdown
                    cur_trade_drawdown = 0.0
                    if pnl >= 0.0:
                        wins += 1
                        gross_win += pnl
                    else:
                        losses += 1
                        gross_loss += -pnl
                        if pnl < worst_trade_pnl:
                            worst_trade_pnl = pnl
                        if loss_count < len(loss_values):
                            loss_values[loss_count] = pnl
                            loss_count += 1
                    pos = 0
                    returned_inside = False
                    continue
            elif pos == -1:
                if raw_state[i] == 0.0:
                    returned_inside = True
                open_pnl = (entry - px_live) * units
                open_points = (entry - px_live) / point_size
                if rebreak_dist >= 0.0 and returned_inside and px_live >= upper_band[i] + rebreak_dist:
                    close = True
                    exit_px = px_live
                elif tp_dist > 0.0 and trail_dist > 0.0:
                    if px_live <= entry - tp_dist and (best_px == 0.0 or px_live < best_px):
                        best_px = px_live
                    if best_px <= entry - tp_dist and px_live >= best_px + trail_dist:
                        close = True
                        exit_px = px_live
                elif tp_dist > 0.0 and trail_dist <= 0.0 and px_live <= entry - tp_dist:
                    close = True
                    exit_px = px_live
                if (not close) and loss_dist > 0.0 and open_points <= -loss_cut_points:
                    close = True
                    exit_px = px_live
                    is_stop = True
                elif (not close) and state[i] == 1.0 and (tp_points <= 0.0 or open_pnl < 0.0):
                    close = True
                    exit_px = px_live
                if close:
                    pnl = (entry - exit_px) * units
                    pnl -= abs(exit_px * units) / 1_000_000.0 * commission_per_million
                    cash += pnl
                    add_daily(i, pnl)
                    trades += 1
                    short_trades += 1
                    if is_stop:
                        stop_losses += 1
                    else:
                        signal_exits += 1
                    if cur_trade_drawdown > max_trade_drawdown:
                        max_trade_drawdown = cur_trade_drawdown
                    cur_trade_drawdown = 0.0
                    if pnl >= 0.0:
                        wins += 1
                        gross_win += pnl
                    else:
                        losses += 1
                        gross_loss += -pnl
                        if pnl < worst_trade_pnl:
                            worst_trade_pnl = pnl
                        if loss_count < len(loss_values):
                            loss_values[loss_count] = pnl
                            loss_count += 1
                    pos = 0
                    returned_inside = False
                    continue

            if pos == 0 and entry_allowed[i]:
                margin = cash if compound else amount
                if margin > 0.0:
                    if allow_long and state[i] == 1.0 and prev_state[i] != 1.0:
                        entry = px_live
                        units = (margin * leverage) / entry
                        fee = abs(entry * units) / 1_000_000.0 * commission_per_million
                        cash -= fee
                        add_daily(i, -fee)
                        pos = 1
                        best_px = entry
                        returned_inside = False
                    elif allow_short and state[i] == -1.0 and prev_state[i] != -1.0:
                        entry = px_live
                        units = (margin * leverage) / entry
                        fee = abs(entry * units) / 1_000_000.0 * commission_per_million
                        cash -= fee
                        add_daily(i, -fee)
                        pos = -1
                        best_px = entry
                        returned_inside = False

            if cash > equity_peak:
                equity_peak = cash
            dd_cash = equity_peak - cash
            if dd_cash > max_dd:
                max_dd = dd_cash
            if cash > cash_peak:
                cash_peak = cash
            cum_dd = cash_peak - cash
            if cum_dd > cum_max_dd:
                cum_max_dd = cum_dd

        open_u = 0.0
        open_side_code = 0
        open_bps = 0.0
        if pos == 1:
            open_side_code = 1
            open_u = (close_px[-1] - entry) * units
            open_bps = (close_px[-1] / entry - 1.0) * 10000.0
        elif pos == -1:
            open_side_code = -1
            open_u = (entry - close_px[-1]) * units
            open_bps = (entry / close_px[-1] - 1.0) * 10000.0

        realised = cash - amount
        total = realised + open_u
        pf = gross_win / gross_loss if gross_loss > 0.0 else (999.0 if gross_win > 0.0 else 0.0)
        daily_sorted = daily_pnl.copy()
        daily_sorted.sort()
        avg_day = 0.0
        for d in range(max_days):
            avg_day += daily_pnl[d]
        avg_day /= max(max_days, 1)
        if max_days == 0:
            median_day = 0.0
        elif max_days % 2 == 1:
            median_day = daily_sorted[max_days // 2]
        else:
            median_day = 0.5 * (daily_sorted[max_days // 2 - 1] + daily_sorted[max_days // 2])
        median_loss = 0.0
        if loss_count > 0:
            loss_sorted = loss_values[:loss_count].copy()
            loss_sorted.sort()
            if loss_count % 2 == 1:
                median_loss = loss_sorted[loss_count // 2]
            else:
                median_loss = 0.5 * (loss_sorted[loss_count // 2 - 1] + loss_sorted[loss_count // 2])

        return (
            realised, open_u, total, trades, wins, losses, pf, max_dd,
            long_trades, short_trades, stop_losses, signal_exits,
            liquidations, open_side_code, open_bps, max_trade_drawdown,
            avg_day, median_day, worst_trade_pnl, median_loss, cum_max_dd,
        )


if njit is not None:
    @njit(cache=True)
    def _simulate_state_numba(
        bid: np.ndarray,
        ask: np.ndarray,
        ts_ns: np.ndarray,
        day_id: np.ndarray,
        max_days: int,
        state: np.ndarray,
        profit_exit_state: np.ndarray,
        loss_exit_state: np.ndarray,
        entry_allowed: np.ndarray,
        tp_points: float,
        trail_points: float,
        loss_cut_points: float,
        max_hold_minutes: float,
        point_size: float,
        amount: float,
        compound: bool,
        leverage: float,
        commission_per_million: float,
        candle_open: np.ndarray,
        trail_from_candle_open: bool,
        raw_state: np.ndarray,
        upper_band: np.ndarray,
        lower_band: np.ndarray,
        rebreak_points: float,
        side_mode: int,
        reverse_on_signal: bool,
        ignore_signal_exit_when_bracket: bool,
        signal_exit_always: bool,
        signal_exit_profit: bool,
        signal_exit_loss: bool,
    ):
        tp_dist = tp_points * point_size if tp_points > 0.0 else 0.0
        trail_dist = trail_points * point_size if trail_points > 0.0 else 0.0
        loss_dist = loss_cut_points * point_size if loss_cut_points > 0.0 else 0.0
        rebreak_dist = rebreak_points * point_size if rebreak_points >= 0.0 else -1.0
        hold_ns = max_hold_minutes * 60.0 * 1_000_000_000.0
        allow_long = side_mode == 1 or side_mode == 3
        allow_short = side_mode == 2 or side_mode == 3

        cash = amount
        equity_peak = amount
        cash_peak = amount
        max_dd = 0.0
        cum_max_dd = 0.0
        gross_win = 0.0
        gross_loss = 0.0
        trades = wins = losses = long_trades = short_trades = 0
        stop_losses = signal_exits = liquidations = 0
        pos = 0
        entry = 0.0
        units = 0.0
        entry_ts = 0
        best_px = 0.0
        returned_inside = False
        entry_upper_band = 0.0
        entry_lower_band = 0.0
        cur_trade_drawdown = 0.0
        max_trade_drawdown = 0.0
        worst_trade_pnl = 0.0
        loss_values = np.empty(100000, dtype=np.float64)
        loss_count = 0
        daily_pnl = np.zeros(max_days, dtype=np.float64)
        active_days = np.zeros(max_days, dtype=np.int64)

        def add_daily(idx, pnl_value):
            d = day_id[idx]
            if d >= 0 and d < max_days:
                daily_pnl[d] += pnl_value
                active_days[d] = 1

        prev_state = np.empty(len(state), dtype=np.float64)
        prev_state[0] = 0.0
        for k in range(1, len(state)):
            prev_state[k] = state[k - 1]

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
                eq = cash + live_u
                if eq > equity_peak:
                    equity_peak = eq
                dd = equity_peak - eq
                if dd > max_dd:
                    max_dd = dd
                if eq <= 0.0:
                    liquidations += 1
                    cash = 0.0
                    pos = 0
                    break

            close = False
            exit_px = 0.0
            is_stop = False
            if pos == 1:
                if raw_state[i] == 0.0:
                    returned_inside = True
                open_pnl = (b - entry) * units
                open_points = (b - entry) / point_size
                exit_signal = (
                    profit_exit_state[i] == -1.0
                    if open_pnl >= 0.0
                    else loss_exit_state[i] == -1.0
                )
                if rebreak_dist >= 0.0 and returned_inside and b <= lower_band[i] - rebreak_dist:
                    close = True
                    exit_px = b
                elif trail_from_candle_open and tp_dist > 0.0 and trail_dist > 0.0:
                    open_ref = candle_open[j]
                    if open_ref >= entry + tp_dist and open_ref > best_px:
                        best_px = open_ref
                    if best_px >= entry + tp_dist and b <= best_px - trail_dist:
                        close = True
                        exit_px = b
                else:
                    if b > best_px:
                        best_px = b
                    if tp_dist > 0.0 and trail_dist <= 0.0 and b >= entry + tp_dist:
                        close = True
                        exit_px = b
                    elif tp_dist > 0.0 and trail_dist > 0.0 and best_px >= entry + tp_dist and b <= best_px - trail_dist:
                        close = True
                        exit_px = b
                if (not close) and loss_dist > 0.0 and open_points <= -loss_cut_points:
                    close = True
                    exit_px = b
                    is_stop = True
                elif (not close) and max_hold_minutes > 0.0 and open_pnl < 0.0 and ts_ns[j] - entry_ts >= hold_ns:
                    close = True
                    exit_px = b
                    is_stop = True
                elif (
                    (not close)
                    and (not (ignore_signal_exit_when_bracket and tp_dist > 0.0 and loss_dist > 0.0))
                    and exit_signal
                    and (
                        signal_exit_always
                        or reverse_on_signal
                        or (signal_exit_profit and open_pnl >= 0.0)
                        or (signal_exit_loss and open_pnl < 0.0)
                    )
                ):
                    close = True
                    exit_px = b
                if close:
                    signal_reverse = reverse_on_signal and (not is_stop) and exit_signal and allow_short
                    pnl = (exit_px - entry) * units
                    pnl -= abs(exit_px * units) / 1_000_000.0 * commission_per_million
                    cash += pnl
                    add_daily(j, pnl)
                    trades += 1
                    long_trades += 1
                    if is_stop:
                        stop_losses += 1
                    else:
                        signal_exits += 1
                    if cur_trade_drawdown > max_trade_drawdown:
                        max_trade_drawdown = cur_trade_drawdown
                    cur_trade_drawdown = 0.0
                    if pnl >= 0.0:
                        wins += 1
                        gross_win += pnl
                    else:
                        losses += 1
                        gross_loss += -pnl
                        if pnl < worst_trade_pnl:
                            worst_trade_pnl = pnl
                        if loss_count < len(loss_values):
                            loss_values[loss_count] = pnl
                            loss_count += 1
                    pos = 0
                    returned_inside = False
                    if signal_reverse and entry_allowed[j] and cash > 0.0:
                        margin = cash if compound else amount
                        if margin > 0.0:
                            entry = b
                            units = (margin * leverage) / entry
                            fee = abs(entry * units) / 1_000_000.0 * commission_per_million
                            cash -= fee
                            add_daily(j, -fee)
                            pos = -1
                            entry_ts = ts_ns[j]
                            best_px = entry
                            returned_inside = False
                    continue
            elif pos == -1:
                if raw_state[i] == 0.0:
                    returned_inside = True
                open_pnl = (entry - a) * units
                open_points = (entry - a) / point_size
                exit_signal = (
                    profit_exit_state[i] == 1.0
                    if open_pnl >= 0.0
                    else loss_exit_state[i] == 1.0
                )
                if rebreak_dist >= 0.0 and returned_inside and a >= upper_band[i] + rebreak_dist:
                    close = True
                    exit_px = a
                elif trail_from_candle_open and tp_dist > 0.0 and trail_dist > 0.0:
                    open_ref = candle_open[j]
                    if open_ref <= entry - tp_dist and (best_px == 0.0 or open_ref < best_px):
                        best_px = open_ref
                    if best_px <= entry - tp_dist and a >= best_px + trail_dist:
                        close = True
                        exit_px = a
                else:
                    if best_px == 0.0 or a < best_px:
                        best_px = a
                    if tp_dist > 0.0 and trail_dist <= 0.0 and a <= entry - tp_dist:
                        close = True
                        exit_px = a
                    elif tp_dist > 0.0 and trail_dist > 0.0 and best_px <= entry - tp_dist and a >= best_px + trail_dist:
                        close = True
                        exit_px = a
                if (not close) and loss_dist > 0.0 and open_points <= -loss_cut_points:
                    close = True
                    exit_px = a
                    is_stop = True
                elif (not close) and max_hold_minutes > 0.0 and open_pnl < 0.0 and ts_ns[j] - entry_ts >= hold_ns:
                    close = True
                    exit_px = a
                    is_stop = True
                elif (
                    (not close)
                    and (not (ignore_signal_exit_when_bracket and tp_dist > 0.0 and loss_dist > 0.0))
                    and exit_signal
                    and (
                        signal_exit_always
                        or reverse_on_signal
                        or (signal_exit_profit and open_pnl >= 0.0)
                        or (signal_exit_loss and open_pnl < 0.0)
                    )
                ):
                    close = True
                    exit_px = a
                if close:
                    signal_reverse = reverse_on_signal and (not is_stop) and exit_signal and allow_long
                    pnl = (entry - exit_px) * units
                    pnl -= abs(exit_px * units) / 1_000_000.0 * commission_per_million
                    cash += pnl
                    add_daily(j, pnl)
                    trades += 1
                    short_trades += 1
                    if is_stop:
                        stop_losses += 1
                    else:
                        signal_exits += 1
                    if cur_trade_drawdown > max_trade_drawdown:
                        max_trade_drawdown = cur_trade_drawdown
                    cur_trade_drawdown = 0.0
                    if pnl >= 0.0:
                        wins += 1
                        gross_win += pnl
                    else:
                        losses += 1
                        gross_loss += -pnl
                        if pnl < worst_trade_pnl:
                            worst_trade_pnl = pnl
                        if loss_count < len(loss_values):
                            loss_values[loss_count] = pnl
                            loss_count += 1
                    pos = 0
                    returned_inside = False
                    if signal_reverse and entry_allowed[j] and cash > 0.0:
                        margin = cash if compound else amount
                        if margin > 0.0:
                            entry = a
                            units = (margin * leverage) / entry
                            fee = abs(entry * units) / 1_000_000.0 * commission_per_million
                            cash -= fee
                            add_daily(j, -fee)
                            pos = 1
                            entry_ts = ts_ns[j]
                            best_px = entry
                            returned_inside = False
                    continue

            if pos == 0 and entry_allowed[j]:
                margin = cash if compound else amount
                if margin > 0.0:
                    if allow_long and state[i] == 1.0 and prev_state[i] != 1.0:
                        entry = a
                        units = (margin * leverage) / entry
                        fee = abs(entry * units) / 1_000_000.0 * commission_per_million
                        cash -= fee
                        add_daily(j, -fee)
                        pos = 1
                        entry_ts = ts_ns[j]
                        best_px = entry
                        returned_inside = False
                    elif allow_short and state[i] == -1.0 and prev_state[i] != -1.0:
                        entry = b
                        units = (margin * leverage) / entry
                        fee = abs(entry * units) / 1_000_000.0 * commission_per_million
                        cash -= fee
                        add_daily(j, -fee)
                        pos = -1
                        entry_ts = ts_ns[j]
                        best_px = entry
                        returned_inside = False

            if cash > equity_peak:
                equity_peak = cash
            dd_cash = equity_peak - cash
            if dd_cash > max_dd:
                max_dd = dd_cash
            if cash > cash_peak:
                cash_peak = cash
            cum_dd = cash_peak - cash
            if cum_dd > cum_max_dd:
                cum_max_dd = cum_dd

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
        daily_sorted = daily_pnl.copy()
        daily_sorted.sort()
        avg_day = 0.0
        for d in range(max_days):
            avg_day += daily_pnl[d]
        avg_day /= max(max_days, 1)
        if max_days == 0:
            median_day = 0.0
        elif max_days % 2 == 1:
            median_day = daily_sorted[max_days // 2]
        else:
            median_day = 0.5 * (daily_sorted[max_days // 2 - 1] + daily_sorted[max_days // 2])
        median_loss = 0.0
        if loss_count > 0:
            loss_sorted = loss_values[:loss_count].copy()
            loss_sorted.sort()
            if loss_count % 2 == 1:
                median_loss = loss_sorted[loss_count // 2]
            else:
                median_loss = 0.5 * (loss_sorted[loss_count // 2 - 1] + loss_sorted[loss_count // 2])

        return (
            realised, open_u, total, trades, wins, losses, pf, max_dd,
            long_trades, short_trades, stop_losses, signal_exits,
            liquidations, open_side_code, open_bps, max_trade_drawdown,
            avg_day, median_day, worst_trade_pnl, median_loss, cum_max_dd,
        )


def simulate_state_strategy(
    pair: str,
    strategy: str,
    params: str,
    timeframe: str,
    bid: np.ndarray,
    ask: np.ndarray,
    ts_ns: np.ndarray,
    state: np.ndarray,
    exit_state: np.ndarray,
    entry_allowed: np.ndarray,
    tp_points: float,
    trail_points: float,
    loss_cut_points: float,
    max_hold_minutes: float,
    point_size: float,
    amount: float,
    compound: bool,
    leverage: float,
    commission_per_million: float,
    side: str,
    candle_open: np.ndarray | None = None,
    trail_from_candle_open: bool = False,
    raw_state: np.ndarray | None = None,
    upper_band: np.ndarray | None = None,
    lower_band: np.ndarray | None = None,
    rebreak_points: float = -1.0,
    signal_mode: str = "tick",
    close_px: np.ndarray | None = None,
    candle_day_id: np.ndarray | None = None,
    reverse_on_signal: bool = False,
    ignore_signal_exit_when_bracket: bool = False,
    signal_exit_always: bool = False,
    signal_exit_profit: bool = False,
    signal_exit_loss: bool = False,
    profit_exit_state: np.ndarray | None = None,
    loss_exit_state: np.ndarray | None = None,
) -> TradeResult:
    if njit is None:
        raise RuntimeError("numba is required for signal sweep files")
    day_id, max_days = day_ids_from_timestamps(ts_ns)
    if candle_open is None:
        candle_open = np.zeros(len(bid), dtype=np.float64)
    if raw_state is None:
        raw_state = np.zeros(len(bid), dtype=np.float64)
    if upper_band is None:
        upper_band = np.zeros(len(bid), dtype=np.float64)
    if lower_band is None:
        lower_band = np.zeros(len(bid), dtype=np.float64)
    if profit_exit_state is None:
        profit_exit_state = exit_state
    if loss_exit_state is None:
        loss_exit_state = exit_state
    if signal_mode == "candle":
        if close_px is None or candle_day_id is None:
            raise ValueError("close_px and candle_day_id are required for signal_mode='candle'")
        max_candle_days = int(np.max(candle_day_id)) + 1 if len(candle_day_id) else 1
        out = _simulate_state_candle_numba(
            close_px.astype(np.float64, copy=False),
            candle_day_id.astype(np.int64, copy=False),
            int(max_candle_days),
            state.astype(np.float64, copy=False),
            raw_state.astype(np.float64, copy=False),
            upper_band.astype(np.float64, copy=False),
            lower_band.astype(np.float64, copy=False),
            entry_allowed.astype(np.bool_, copy=False),
            float(tp_points),
            float(trail_points),
            float(loss_cut_points),
            float(point_size),
            float(amount),
            bool(compound),
            float(leverage),
            float(commission_per_million),
            float(rebreak_points),
            side_mode_value(side),
        )
        (
            realised, open_u, total, trades, wins, losses, pf, max_dd,
            long_trades, short_trades, stop_losses, signal_exits,
            liquidations, open_side_code, open_bps, trade_max_drawdown,
            avg_day, median_day, worst_trade_pnl, median_loss, cum_max_dd,
        ) = out
        open_side = "long" if int(open_side_code) == 1 else ("short" if int(open_side_code) == -1 else "-")
        win_rate = wins / trades * 100.0 if trades else 0.0
        result = TradeResult(
            pair, strategy, params, timeframe, tp_points, loss_cut_points, point_size,
            float(realised), float(open_u), float(total), int(trades), int(wins),
            int(losses), float(win_rate), float(pf), float(max_dd),
            int(long_trades), int(short_trades), int(stop_losses),
            int(signal_exits), int(liquidations), bool(liquidations),
            open_side, float(open_bps),
        )
        result.trade_max_drawdown = float(trade_max_drawdown)
        result.avg_day = float(avg_day)
        result.median_day = float(median_day)
        result.worst_trade_pnl = float(worst_trade_pnl)
        result.median_loss = float(median_loss)
        result.cum_max_drawdown = float(cum_max_dd)
        return result
    out = _simulate_state_numba(
        bid.astype(np.float64, copy=False),
        ask.astype(np.float64, copy=False),
        ts_ns.astype(np.int64, copy=False),
        day_id.astype(np.int64, copy=False),
        int(max_days),
        state.astype(np.float64, copy=False),
        profit_exit_state.astype(np.float64, copy=False),
        loss_exit_state.astype(np.float64, copy=False),
        entry_allowed.astype(np.bool_, copy=False),
        float(tp_points),
        float(trail_points),
        float(loss_cut_points),
        float(max_hold_minutes),
        float(point_size),
        float(amount),
        bool(compound),
        float(leverage),
        float(commission_per_million),
        candle_open.astype(np.float64, copy=False),
        bool(trail_from_candle_open),
        raw_state.astype(np.float64, copy=False),
        upper_band.astype(np.float64, copy=False),
        lower_band.astype(np.float64, copy=False),
        float(rebreak_points),
        side_mode_value(side),
        bool(reverse_on_signal),
        bool(ignore_signal_exit_when_bracket),
        bool(signal_exit_always),
        bool(signal_exit_profit or tp_points <= 0.0),
        bool(signal_exit_loss or loss_cut_points <= 0.0),
    )
    (
        realised, open_u, total, trades, wins, losses, pf, max_dd,
        long_trades, short_trades, stop_losses, signal_exits,
        liquidations, open_side_code, open_bps, trade_max_drawdown,
        avg_day, median_day, worst_trade_pnl, median_loss, cum_max_dd,
    ) = out
    open_side = "long" if int(open_side_code) == 1 else ("short" if int(open_side_code) == -1 else "-")
    win_rate = wins / trades * 100.0 if trades else 0.0
    result = TradeResult(
        pair, strategy, params, timeframe, tp_points, loss_cut_points, point_size,
        float(realised), float(open_u), float(total), int(trades), int(wins),
        int(losses), float(win_rate), float(pf), float(max_dd),
        int(long_trades), int(short_trades), int(stop_losses),
        int(signal_exits), int(liquidations), bool(liquidations),
        open_side, float(open_bps),
    )
    result.trade_max_drawdown = float(trade_max_drawdown)
    result.avg_day = float(avg_day)
    result.median_day = float(median_day)
    result.worst_trade_pnl = float(worst_trade_pnl)
    result.median_loss = float(median_loss)
    result.cum_max_drawdown = float(cum_max_dd)
    return result
