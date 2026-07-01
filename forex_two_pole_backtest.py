"""Two-Pole Oscillator signal backtest.

Mirrors the BigBeluga-style indicator mechanics:
    - buy when two-pole oscillator crosses above its 4-bar lag while below 0
    - sell when it crosses below its 4-bar lag while above 0
    - buy/sell levels are low - SMA(range, 100) and high + SMA(range, 100)

Execution uses bid-built candles for confirmed signals and tick bid/ask for fills.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

import numpy as np
import pandas as pd

from forex_signal_sweep_common import build_bid_ohlc, map_state_to_ticks
from forex_strategy_common import (
    TradeResult,
    active_session_allowed,
    build_parser,
    day_ids_from_timestamps,
    default_point_size,
    load_market,
    njit,
    parse_num_list,
    parse_str_list,
)
from forex_unified_signal_backtest import write_unified_results

DEFAULT_TIMEFRAMES = ["1m", "3m", "5m", "10m", "15m", "30m"]
GOLD_TP = [-1,-2,-3,0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
GOLD_SL = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
FX_TP = [-1, -2, -3, 0, 20, 30, 50, 65, 80, 95, 110, 125, 140, 155, 170]
FX_SL = [0, 20, 30, 50, 65, 80, 95, 110, 125, 140, 155, 170]
DEFAULT_SESSIONS = [0, 1, 2, -1]


def default_tp_sl_for_pair(pair: str) -> tuple[list[float], list[float]]:
    if pair.upper() == "XAUUSD":
        return GOLD_TP, GOLD_SL
    return FX_TP, FX_SL


def rolling_sma(x: np.ndarray, length: int) -> np.ndarray:
    return pd.Series(x).rolling(length, min_periods=length).mean().to_numpy(np.float64)


def rolling_stdev(x: np.ndarray, length: int) -> np.ndarray:
    # Pine ta.stdev uses population-style stdev for indicator use.
    return pd.Series(x).rolling(length, min_periods=length).std(ddof=0).to_numpy(np.float64)


def two_pole_values(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, length: int):
    sma1 = rolling_sma(closes, 25)
    dev = closes - sma1
    dev_sma = rolling_sma(dev, 25)
    dev_std = rolling_stdev(dev, 25)
    sma_n1 = (dev - dev_sma) / np.maximum(dev_std, 1e-12)

    alpha = 2.0 / (length + 1.0)
    smooth1 = np.full(len(closes), np.nan, dtype=np.float64)
    smooth2 = np.full(len(closes), np.nan, dtype=np.float64)
    s1 = np.nan
    s2 = np.nan
    for i, v in enumerate(sma_n1):
        if not np.isfinite(v):
            continue
        if not np.isfinite(s1):
            s1 = float(v)
        else:
            s1 = (1.0 - alpha) * s1 + alpha * float(v)
        if not np.isfinite(s2):
            s2 = s1
        else:
            s2 = (1.0 - alpha) * s2 + alpha * s1
        smooth1[i] = s1
        smooth2[i] = s2

    lag = np.full(len(closes), np.nan, dtype=np.float64)
    if len(closes) > 4:
        lag[4:] = smooth2[:-4]
    area = rolling_sma(highs - lows, 100)
    return smooth2, lag, area


def two_pole_signals(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, length: int):
    two_p, two_pp, area = two_pole_values(highs, lows, closes, length)
    state = np.zeros(len(closes), dtype=np.float64)
    level = np.full(len(closes), np.nan, dtype=np.float64)
    for i in range(1, len(closes)):
        if not (np.isfinite(two_p[i]) and np.isfinite(two_pp[i]) and np.isfinite(two_p[i - 1]) and np.isfinite(two_pp[i - 1])):
            continue
        if not np.isfinite(area[i]):
            continue
        buy = two_p[i - 1] <= two_pp[i - 1] and two_p[i] > two_pp[i] and two_p[i] < 0.0
        sell = two_p[i - 1] >= two_pp[i - 1] and two_p[i] < two_pp[i] and two_p[i] > 0.0
        if buy:
            state[i] = 1.0
            level[i] = lows[i] - area[i]
        elif sell:
            state[i] = -1.0
            level[i] = highs[i] + area[i]
    return state, level, two_p, two_pp


def map_level_to_ticks(n_ticks: int, close_idx: np.ndarray, level: np.ndarray) -> np.ndarray:
    out = np.full(n_ticks, np.nan, dtype=np.float64)
    prev = 0
    last = np.nan
    for idx, lv in zip(close_idx, level):
        out[prev:idx + 1] = last
        if np.isfinite(lv):
            last = float(lv)
        prev = idx + 1
    out[prev:] = last
    return out


if njit is not None:
    @njit(cache=True)
    def _simulate_two_pole_numba(
        bid: np.ndarray,
        ask: np.ndarray,
        day_id: np.ndarray,
        max_days: int,
        state: np.ndarray,
        level: np.ndarray,
        entry_allowed: np.ndarray,
        tp_points: float,
        sl_points: float,
        point_size: float,
        amount: float,
        compound: bool,
        leverage: float,
        commission_per_million: float,
        side_mode: int,
    ):
        daily_pnl = np.zeros(max_days, dtype=np.float64)
        prev = np.empty(len(state), dtype=np.float64)
        prev[0] = 0.0
        for k in range(1, len(state)):
            prev[k] = state[k - 1]

        allow_long = side_mode == 1 or side_mode == 3
        allow_short = side_mode == 2 or side_mode == 3
        cash = amount
        equity_peak = amount
        cash_peak = amount
        max_dd = 0.0
        cum_dd = 0.0
        trade_max_dd = 0.0
        cur_trade_dd = 0.0
        gross_win = 0.0
        gross_loss = 0.0
        trades = 0
        wins = 0
        losses = 0
        long_trades = 0
        short_trades = 0
        stop_losses = 0
        signal_exits = 0
        liquidations = 0
        worst_trade = 0.0
        loss_values = np.empty(100000, dtype=np.float64)
        loss_count = 0

        pos = 0
        entry = 0.0
        units = 0.0
        sl_level = np.nan
        tp_level = np.nan

        for i in range(len(bid) - 1):
            j = i + 1
            b = bid[j]
            a = ask[j]
            if pos != 0:
                live_u = (b - entry) * units if pos == 1 else (entry - a) * units
                if -live_u > cur_trade_dd:
                    cur_trade_dd = -live_u
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
                open_pnl = (b - entry) * units
                if tp_points > 0.0 and b >= entry + tp_points * point_size:
                    close = True
                    exit_px = b
                elif tp_points < 0.0 and np.isfinite(tp_level) and b >= tp_level:
                    close = True
                    exit_px = b
                elif sl_points > 0.0 and b <= entry - sl_points * point_size:
                    close = True
                    exit_px = b
                    is_stop = True
                elif sl_points == 0.0 and open_pnl < 0.0 and np.isfinite(sl_level) and b <= sl_level:
                    close = True
                    exit_px = b
                    is_stop = True
                elif tp_points == 0.0 and open_pnl > 0.0 and state[i] == -1.0:
                    close = True
                    exit_px = b
            elif pos == -1:
                open_pnl = (entry - a) * units
                if tp_points > 0.0 and a <= entry - tp_points * point_size:
                    close = True
                    exit_px = a
                elif tp_points < 0.0 and np.isfinite(tp_level) and a <= tp_level:
                    close = True
                    exit_px = a
                elif sl_points > 0.0 and a >= entry + sl_points * point_size:
                    close = True
                    exit_px = a
                    is_stop = True
                elif sl_points == 0.0 and open_pnl < 0.0 and np.isfinite(sl_level) and a >= sl_level:
                    close = True
                    exit_px = a
                    is_stop = True
                elif tp_points == 0.0 and open_pnl > 0.0 and state[i] == 1.0:
                    close = True
                    exit_px = a

            if close:
                pnl = (exit_px - entry) * units if pos == 1 else (entry - exit_px) * units
                pnl -= abs(exit_px * units) / 1_000_000.0 * commission_per_million
                cash += pnl
                d = day_id[j]
                if d >= 0 and d < max_days:
                    daily_pnl[d] += pnl
                trades += 1
                if pos == 1:
                    long_trades += 1
                else:
                    short_trades += 1
                if is_stop:
                    stop_losses += 1
                else:
                    signal_exits += 1
                if cur_trade_dd > trade_max_dd:
                    trade_max_dd = cur_trade_dd
                cur_trade_dd = 0.0
                if pnl >= 0.0:
                    wins += 1
                    gross_win += pnl
                else:
                    losses += 1
                    gross_loss += -pnl
                    if pnl < worst_trade:
                        worst_trade = pnl
                    if loss_count < len(loss_values):
                        loss_values[loss_count] = pnl
                        loss_count += 1
                pos = 0
                entry = 0.0
                units = 0.0
                sl_level = np.nan
                tp_level = np.nan
                continue

            if pos == 0 and entry_allowed[i]:
                margin = cash if compound else amount
                if margin > 0.0 and allow_long and state[i] == 1.0 and prev[i] != 1.0:
                    lv = level[i]
                    if np.isfinite(lv):
                        entry = a
                        units = (margin * leverage) / entry
                        fee = abs(entry * units) / 1_000_000.0 * commission_per_million
                        cash -= fee
                        d = day_id[j]
                        if d >= 0 and d < max_days:
                            daily_pnl[d] -= fee
                        pos = 1
                        sl_level = lv
                        risk = entry - sl_level
                        if risk < point_size:
                            risk = point_size
                        tp_level = entry + abs(tp_points) * risk if tp_points < 0.0 else np.nan
                elif margin > 0.0 and allow_short and state[i] == -1.0 and prev[i] != -1.0:
                    lv = level[i]
                    if np.isfinite(lv):
                        entry = b
                        units = (margin * leverage) / entry
                        fee = abs(entry * units) / 1_000_000.0 * commission_per_million
                        cash -= fee
                        d = day_id[j]
                        if d >= 0 and d < max_days:
                            daily_pnl[d] -= fee
                        pos = -1
                        sl_level = lv
                        risk = sl_level - entry
                        if risk < point_size:
                            risk = point_size
                        tp_level = entry - abs(tp_points) * risk if tp_points < 0.0 else np.nan

            if cash > equity_peak:
                equity_peak = cash
            dd_cash = equity_peak - cash
            if dd_cash > max_dd:
                max_dd = dd_cash
            if cash > cash_peak:
                cash_peak = cash
            cash_dd = cash_peak - cash
            if cash_dd > cum_dd:
                cum_dd = cash_dd

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
        avg_day = 0.0
        for d in range(max_days):
            avg_day += daily_pnl[d]
        avg_day /= max(max_days, 1)
        daily_sorted = daily_pnl.copy()
        daily_sorted.sort()
        if max_days == 0:
            med_day = 0.0
        elif max_days % 2 == 1:
            med_day = daily_sorted[max_days // 2]
        else:
            med_day = 0.5 * (daily_sorted[max_days // 2 - 1] + daily_sorted[max_days // 2])
        med_loss = 0.0
        if loss_count > 0:
            loss_sorted = loss_values[:loss_count].copy()
            loss_sorted.sort()
            if loss_count % 2 == 1:
                med_loss = loss_sorted[loss_count // 2]
            else:
                med_loss = 0.5 * (loss_sorted[loss_count // 2 - 1] + loss_sorted[loss_count // 2])

        return (
            realised, open_u, total, trades, wins, losses, pf, max_dd,
            long_trades, short_trades, stop_losses, signal_exits,
            liquidations, open_side_code, open_bps, trade_max_dd,
            avg_day, med_day, worst_trade, med_loss, cum_dd,
        )


def side_mode_value(side: str) -> int:
    if side == "long":
        return 1
    if side == "short":
        return 2
    return 3


def simulate_two_pole(
    pair: str,
    timeframe: str,
    params: str,
    bid: np.ndarray,
    ask: np.ndarray,
    ts_ns: np.ndarray,
    state: np.ndarray,
    level: np.ndarray,
    entry_allowed: np.ndarray,
    tp_points: float,
    sl_points: float,
    point_size: float,
    amount: float,
    compound: bool,
    leverage: float,
    commission_per_million: float,
    side: str,
) -> TradeResult:
    if njit is not None:
        day_id, max_days = day_ids_from_timestamps(ts_ns)
        out = _simulate_two_pole_numba(
            bid.astype(np.float64, copy=False),
            ask.astype(np.float64, copy=False),
            day_id.astype(np.int64, copy=False),
            int(max_days),
            state.astype(np.float64, copy=False),
            level.astype(np.float64, copy=False),
            entry_allowed.astype(np.bool_, copy=False),
            float(tp_points),
            float(sl_points),
            float(point_size),
            float(amount),
            bool(compound),
            float(leverage),
            float(commission_per_million),
            side_mode_value(side),
        )
        (
            realised, open_u, total, trades, wins, losses, pf, max_dd,
            long_trades, short_trades, stop_losses, signal_exits,
            liquidations, open_side_code, open_bps, trade_max_dd,
            avg_day, med_day, worst_trade, med_loss, cum_dd,
        ) = out
        open_side = "long" if int(open_side_code) == 1 else ("short" if int(open_side_code) == -1 else "-")
        win_rate = wins / trades * 100.0 if trades else 0.0
        r = TradeResult(
            pair, "two_pole", params, timeframe, tp_points, sl_points, point_size,
            float(realised), float(open_u), float(total), int(trades), int(wins),
            int(losses), float(win_rate), float(pf), float(max_dd),
            int(long_trades), int(short_trades), int(stop_losses),
            int(signal_exits), int(liquidations), bool(liquidations),
            open_side, float(open_bps),
        )
        r.avg_day = float(avg_day)
        r.median_day = float(med_day)
        r.trade_max_drawdown = float(trade_max_dd)
        r.cum_max_drawdown = float(cum_dd)
        r.worst_trade_pnl = float(worst_trade)
        r.median_loss = float(med_loss)
        return r

    day_id, max_days = day_ids_from_timestamps(ts_ns)
    daily_pnl = np.zeros(max_days, dtype=np.float64)

    def add_daily(i: int, pnl: float) -> None:
        d = int(day_id[i])
        if 0 <= d < max_days:
            daily_pnl[d] += pnl

    prev = np.empty_like(state)
    prev[0] = 0.0
    prev[1:] = state[:-1]

    allow_long = side in {"long", "both"}
    allow_short = side in {"short", "both"}
    cash = amount
    equity_peak = amount
    cash_peak = amount
    max_dd = 0.0
    cum_dd = 0.0
    trade_max_dd = 0.0
    cur_trade_dd = 0.0
    gross_win = 0.0
    gross_loss = 0.0
    trades = wins = losses = long_trades = short_trades = stop_losses = signal_exits = liquidations = 0
    worst_trade = 0.0
    loss_values: list[float] = []

    pos = 0
    entry = 0.0
    units = 0.0
    sl_level = np.nan
    tp_level = np.nan

    for i in range(len(bid) - 1):
        j = i + 1
        b = float(bid[j])
        a = float(ask[j])
        if pos:
            live_u = (b - entry) * units if pos == 1 else (entry - a) * units
            cur_trade_dd = max(cur_trade_dd, -live_u)
            eq = cash + live_u
            equity_peak = max(equity_peak, eq)
            max_dd = max(max_dd, equity_peak - eq)
            if eq <= 0.0:
                liquidations += 1
                cash = 0.0
                pos = 0
                break

        close = False
        exit_px = 0.0
        is_stop = False
        if pos == 1:
            open_pnl = (b - entry) * units
            if tp_points > 0.0 and b >= entry + tp_points * point_size:
                close = True
                exit_px = b
            elif tp_points < 0.0 and np.isfinite(tp_level) and b >= tp_level:
                close = True
                exit_px = b
            elif sl_points > 0.0 and b <= entry - sl_points * point_size:
                close = True
                exit_px = b
                is_stop = True
            elif sl_points == 0.0 and open_pnl < 0.0 and np.isfinite(sl_level) and b <= sl_level:
                close = True
                exit_px = b
                is_stop = True
            elif tp_points == 0.0 and open_pnl > 0.0 and state[i] == -1.0:
                close = True
                exit_px = b
        elif pos == -1:
            open_pnl = (entry - a) * units
            if tp_points > 0.0 and a <= entry - tp_points * point_size:
                close = True
                exit_px = a
            elif tp_points < 0.0 and np.isfinite(tp_level) and a <= tp_level:
                close = True
                exit_px = a
            elif sl_points > 0.0 and a >= entry + sl_points * point_size:
                close = True
                exit_px = a
                is_stop = True
            elif sl_points == 0.0 and open_pnl < 0.0 and np.isfinite(sl_level) and a >= sl_level:
                close = True
                exit_px = a
                is_stop = True
            elif tp_points == 0.0 and open_pnl > 0.0 and state[i] == 1.0:
                close = True
                exit_px = a

        if close:
            pnl = (exit_px - entry) * units if pos == 1 else (entry - exit_px) * units
            pnl -= abs(exit_px * units) / 1_000_000.0 * commission_per_million
            cash += pnl
            add_daily(j, pnl)
            trades += 1
            if pos == 1:
                long_trades += 1
            else:
                short_trades += 1
            if is_stop:
                stop_losses += 1
            else:
                signal_exits += 1
            trade_max_dd = max(trade_max_dd, cur_trade_dd)
            cur_trade_dd = 0.0
            if pnl >= 0.0:
                wins += 1
                gross_win += pnl
            else:
                losses += 1
                gross_loss += -pnl
                worst_trade = min(worst_trade, pnl)
                loss_values.append(pnl)
            pos = 0
            entry = 0.0
            units = 0.0
            sl_level = np.nan
            tp_level = np.nan
            continue

        if pos == 0 and entry_allowed[i]:
            margin = cash if compound else amount
            if margin > 0.0 and allow_long and state[i] == 1.0 and prev[i] != 1.0:
                lv = float(level[i])
                if np.isfinite(lv):
                    entry = a
                    units = (margin * leverage) / entry
                    fee = abs(entry * units) / 1_000_000.0 * commission_per_million
                    cash -= fee
                    add_daily(j, -fee)
                    pos = 1
                    sl_level = lv
                    risk = max(entry - sl_level, point_size)
                    tp_level = entry + abs(tp_points) * risk if tp_points < 0.0 else np.nan
            elif margin > 0.0 and allow_short and state[i] == -1.0 and prev[i] != -1.0:
                lv = float(level[i])
                if np.isfinite(lv):
                    entry = b
                    units = (margin * leverage) / entry
                    fee = abs(entry * units) / 1_000_000.0 * commission_per_million
                    cash -= fee
                    add_daily(j, -fee)
                    pos = -1
                    sl_level = lv
                    risk = max(sl_level - entry, point_size)
                    tp_level = entry - abs(tp_points) * risk if tp_points < 0.0 else np.nan

        equity_peak = max(equity_peak, cash)
        max_dd = max(max_dd, equity_peak - cash)
        cash_peak = max(cash_peak, cash)
        cum_dd = max(cum_dd, cash_peak - cash)

    open_u = 0.0
    open_side = "-"
    open_bps = 0.0
    if pos == 1:
        open_side = "long"
        open_u = (bid[-1] - entry) * units
        open_bps = (bid[-1] / entry - 1.0) * 10000.0
    elif pos == -1:
        open_side = "short"
        open_u = (entry - ask[-1]) * units
        open_bps = (entry / ask[-1] - 1.0) * 10000.0

    realised = cash - amount
    total = realised + open_u
    pf = gross_win / gross_loss if gross_loss > 0.0 else (999.0 if gross_win > 0.0 else 0.0)
    win_rate = wins / trades * 100.0 if trades else 0.0
    med_day = float(np.median(daily_pnl)) if len(daily_pnl) else 0.0
    avg_day = float(np.mean(daily_pnl)) if len(daily_pnl) else 0.0
    med_loss = float(np.median(loss_values)) if loss_values else 0.0

    r = TradeResult(
        pair, "two_pole", params, timeframe, tp_points, sl_points, point_size,
        float(realised), float(open_u), float(total), trades, wins, losses,
        float(win_rate), float(pf), float(max_dd), long_trades, short_trades,
        stop_losses, signal_exits, liquidations, bool(liquidations),
        open_side, float(open_bps),
    )
    r.avg_day = avg_day
    r.median_day = med_day
    r.trade_max_drawdown = float(trade_max_dd)
    r.cum_max_drawdown = float(cum_dd)
    r.worst_trade_pnl = float(worst_trade)
    r.median_loss = med_loss
    return r


def main() -> None:
    ap = build_parser("Two-Pole Oscillator backtest", "forex_two_pole_results.csv")
    ap.add_argument("--length", default="15", help="two-pole filter length list; default 15")
    ap.add_argument("--sessions", default=",".join(str(x) for x in DEFAULT_SESSIONS))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    timeframes = parse_str_list(args.timeframes, DEFAULT_TIMEFRAMES)
    lengths = [int(x) for x in parse_num_list(args.length, [15])]
    sessions = [int(x) for x in parse_num_list(args.sessions, DEFAULT_SESSIONS)]
    ticks, t0 = load_market(args)
    del t0
    results: list[TradeResult] = []

    for pair, g in ticks.groupby("pair", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        bid = g["bid"].to_numpy(np.float64)
        ask = g["ask"].to_numpy(np.float64)
        ts_ns = g["timestamp"].astype("int64").to_numpy(np.int64)
        point_size = float(args.point_size or default_point_size(pair))
        default_tp, default_sl = default_tp_sl_for_pair(pair)
        tp_values = parse_num_list(args.tp_points, default_tp)
        sl_values = parse_num_list(args.sl_points, default_sl)
        session_cache = {s: active_session_allowed(ts_ns, s) for s in sessions}
        print(
            f"[two-pole] {pair} ticks={len(g):,} point={point_size:g} "
            f"tp={','.join(f'{x:g}' for x in tp_values)} "
            f"sl={','.join(f'{x:g}' for x in sl_values)}",
            flush=True,
        )

        jobs = []
        for tf in timeframes:
            _, highs, lows, closes, close_idx = build_bid_ohlc(bid, ts_ns, tf)
            if len(closes) < 110:
                continue
            for length in lengths:
                st_bar, lv_bar, _, _ = two_pole_signals(highs, lows, closes, length)
                st = map_state_to_ticks(len(bid), close_idx, st_bar)
                lv = map_level_to_ticks(len(bid), close_idx, lv_bar)
                params = f"length={length};sl_mode=level;tp_mode=fixed/rr"
                for tp, sl, sess in product(tp_values, sl_values, sessions):
                    jobs.append((tf, params, st, lv, tp, sl, sess))
        print(f"[two-pole] {pair} combos={len(jobs):,} workers={args.workers}", flush=True)

        def run_job(job):
            tf, params, st, lv, tp, sl, sess = job
            return simulate_two_pole(
                pair, tf, f"{params};session={sess}", bid, ask, ts_ns, st, lv,
                session_cache[int(sess)], float(tp), float(sl), point_size,
                args.amount, args.compound, args.leverage,
                args.commission_per_million, args.side,
            )

        done = 0
        if args.workers > 1 and len(jobs) > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(run_job, j) for j in jobs]
                for fut in as_completed(futs):
                    results.append(fut.result())
                    done += 1
                    if done % max(1, len(jobs) // 10) == 0:
                        print(f"[two-pole] {pair} progress {done:,}/{len(jobs):,}", flush=True)
        else:
            for j in jobs:
                results.append(run_job(j))
                done += 1

    filtered = [r for r in results if r.trades >= args.min_trades]
    write_unified_results(args.out, filtered, args.top)
    print(f"[two-pole] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
