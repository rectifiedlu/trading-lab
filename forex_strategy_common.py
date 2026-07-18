"""Shared forex tick backtest helpers for Pine-style strategies."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from forex_backtest import (
    DEFAULT_COMMISSION_PER_MILLION,
    DEFAULT_MT5_TICK_CSV,
    FOREX_DIR,
    _default_date_window,
    _default_hour_window,
    load_ticks,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from numba import njit
except Exception:  # pragma: no cover - optional speed dependency
    njit = None

DEFAULT_PAIRS = ["XAUUSD"]
DEFAULT_TIMEFRAMES = ["3m"]
DEFAULT_TP_POINTS = [100, 200, 400, 600, 800, 950, 1200]
DEFAULT_SL_POINTS = [0.0]
DEFAULT_AMOUNT = 50.0
BACKTEST_WINDOW_DAYS = 1.0


@dataclass
class TradeResult:
    pair: str
    strategy: str
    params: str
    timeframe: str
    tp_points: float
    sl_points: float
    point_size: float
    realised: float
    open_unrealized: float
    total: float
    trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    long_trades: int
    short_trades: int
    stop_losses: int
    signal_exits: int
    liquidations: int
    account_dead: bool
    open_side: str
    open_bps: float


@dataclass
class TradeLog:
    pair: str
    strategy: str
    params: str
    timeframe: str
    side: str
    entry_i: int
    exit_i: int
    entry_px: float
    exit_px: float
    pnl: float
    reason: str
    equity: float


def parse_num_list(s: str | None, default: Iterable[float]) -> list[float]:
    if not s:
        return [float(x) for x in default]
    return [float(p.strip()) for p in s.split(",") if p.strip()]


def parse_str_list(s: str | None, default: Iterable[str]) -> list[str]:
    if not s:
        return [str(x) for x in default]
    return [p.strip() for p in s.split(",") if p.strip()]


def timeframe_to_ns(timeframe: str) -> int:
    tf = timeframe.strip().lower()
    if tf.endswith("m"):
        return int(float(tf[:-1]) * 60 * 1_000_000_000)
    if tf.endswith("h"):
        return int(float(tf[:-1]) * 3600 * 1_000_000_000)
    if tf.endswith("s"):
        return int(float(tf[:-1]) * 1_000_000_000)
    raise ValueError(f"unsupported timeframe: {timeframe}")


def default_point_size(pair: str) -> float:
    p = pair.upper()
    if p == "XAUUSD":
        return 0.01
    if "JPY" in p:
        return 0.001
    return 0.00001


def active_session_allowed(ts_ns: np.ndarray, enabled: int) -> np.ndarray:
    """TradingView-style Tokyo/London/New York session gate for entries.

    enabled:
        0  = all hours
        1  = inside Tokyo/London/New York sessions
        -1 = outside Tokyo/London/New York sessions
        2  = all hours except UTC 20:30-01:00 rollover block
        3  = all hours except UTC 18:30-21:00 block
    """
    if enabled == 0:
        return np.ones(len(ts_ns), dtype=np.bool_)
    dt = pd.to_datetime(ts_ns, utc=True)
    if enabled == 2:
        utc_min = dt.hour.to_numpy(np.int64) * 60 + dt.minute.to_numpy(np.int64)
        blocked = (utc_min >= 20 * 60 + 30) | (utc_min < 1 * 60)
        return (~blocked).astype(np.bool_)
    if enabled == 3:
        utc_min = dt.hour.to_numpy(np.int64) * 60 + dt.minute.to_numpy(np.int64)
        blocked = (utc_min >= 18 * 60 + 30) & (utc_min < 21 * 60)
        return (~blocked).astype(np.bool_)
    tokyo = dt.tz_convert("Asia/Tokyo")
    london = dt.tz_convert("Europe/London")
    new_york = dt.tz_convert("America/New_York")

    tokyo_min = tokyo.hour.to_numpy(np.int64) * 60 + tokyo.minute.to_numpy(np.int64)
    london_min = london.hour.to_numpy(np.int64) * 60 + london.minute.to_numpy(np.int64)
    new_york_min = new_york.hour.to_numpy(np.int64) * 60 + new_york.minute.to_numpy(np.int64)

    in_tokyo = (tokyo_min >= 9 * 60) & (tokyo_min < 15 * 60)
    in_london = (london_min >= 8 * 60 + 30) & (london_min < 16 * 60 + 30)
    in_new_york = (new_york_min >= 9 * 60 + 30) & (new_york_min < 16 * 60)
    inside = in_tokyo | in_london | in_new_york
    if enabled < 0:
        inside = ~inside
    return inside.astype(np.bool_)


def commission(px: float, units: float, commission_per_million: float) -> float:
    return abs(px * units) / 1_000_000.0 * commission_per_million


def units_for_margin(margin: float, leverage: float, px: float) -> float:
    return (margin * leverage) / px


def open_unrealized(pos: int, entry: float, units: float, bid: float, ask: float) -> float:
    if pos == 1:
        return (bid - entry) * units
    if pos == -1:
        return (entry - ask) * units
    return 0.0


if njit is not None:
    @njit(cache=True)
    def _simulate_triggers_numba(
        bid: np.ndarray,
        ask: np.ndarray,
        long_trigger: np.ndarray,
        short_trigger: np.ndarray,
        long_exit: np.ndarray,
        short_exit: np.ndarray,
        day_id: np.ndarray,
        max_days: int,
        has_long_exit: bool,
        has_short_exit: bool,
        tp_points: float,
        sl_points: float,
        point_size: float,
        amount: float,
        compound: bool,
        leverage: float,
        commission_per_million: float,
        side_mode: int,
        reverse_on_flip: bool,
    ):
        tp_dist = tp_points * point_size
        sl_dist = sl_points * point_size if sl_points > 0 else 0.0
        cash = amount
        equity_peak = amount
        max_dd = 0.0
        gross_win = 0.0
        gross_loss = 0.0
        trades = wins = losses = long_trades = short_trades = 0
        stop_losses = signal_exits = liquidations = 0
        account_dead = False
        liquidation_bps = 10000.0 / max(leverage, 1e-9)
        max_trade_drawdown = 0.0
        sum_trade_drawdown = 0.0
        cur_trade_drawdown = 0.0
        pos = 0
        entry = 0.0
        units = 0.0
        allow_long = side_mode == 1 or side_mode == 3
        allow_short = side_mode == 2 or side_mode == 3
        daily_pnl = np.zeros(max_days, dtype=np.float64)
        active_days = np.zeros(max_days, dtype=np.int64)

        def add_daily(exit_i, pnl_value):
            d = day_id[exit_i]
            if d >= 0 and d < max_days:
                daily_pnl[d] += pnl_value
                active_days[d] = 1

        for i in range(len(bid) - 1):
            next_bid = bid[i + 1]
            next_ask = ask[i + 1]
            reverse_to = 0

            if pos != 0:
                live_u = 0.0
                if pos == 1:
                    live_u = (next_bid - entry) * units
                elif pos == -1:
                    live_u = (entry - next_ask) * units
                live_eq = cash + live_u
                if -live_u > cur_trade_drawdown:
                    cur_trade_drawdown = -live_u
                if cur_trade_drawdown < 0:
                    cur_trade_drawdown = 0.0
                if live_eq > equity_peak:
                    equity_peak = live_eq
                dd = equity_peak - live_eq
                if dd > max_dd:
                    max_dd = dd

            if pos == 1:
                adverse_bps = (entry / next_bid - 1.0) * 10000.0 if next_bid > 0 else 1e18
                if adverse_bps >= liquidation_bps:
                    pnl = -max(0.0, cash)
                    cash = 0.0
                    add_daily(i + 1, pnl)
                    trades += 1; losses += 1; long_trades += 1; liquidations += 1
                    loss = -pnl
                    if loss > max_trade_drawdown:
                        max_trade_drawdown = loss
                    if cur_trade_drawdown > max_trade_drawdown:
                        max_trade_drawdown = cur_trade_drawdown
                    sum_trade_drawdown += max(loss, cur_trade_drawdown)
                    account_dead = True
                    gross_loss += loss
                    pos = 0
                    break
                if sl_dist > 0 and next_bid <= entry - sl_dist:
                    fee = abs(next_bid * units) / 1_000_000.0 * commission_per_million
                    pnl = (next_bid - entry) * units - fee
                    add_daily(i + 1, pnl)
                    cash += pnl; trades += 1; long_trades += 1; stop_losses += 1
                    max_trade_drawdown = max(max_trade_drawdown, cur_trade_drawdown)
                    sum_trade_drawdown += cur_trade_drawdown; cur_trade_drawdown = 0.0
                    if pnl >= 0:
                        wins += 1; gross_win += pnl
                    else:
                        losses += 1; gross_loss += -pnl
                    pos = 0; entry = 0.0; units = 0.0
                    continue
                if tp_dist > 0 and next_bid >= entry + tp_dist:
                    fee = abs(next_bid * units) / 1_000_000.0 * commission_per_million
                    pnl = (next_bid - entry) * units - fee
                    add_daily(i + 1, pnl)
                    cash += pnl; trades += 1; long_trades += 1
                    max_trade_drawdown = max(max_trade_drawdown, cur_trade_drawdown)
                    sum_trade_drawdown += cur_trade_drawdown; cur_trade_drawdown = 0.0
                    if pnl >= 0:
                        wins += 1; gross_win += pnl
                    else:
                        losses += 1; gross_loss += -pnl
                    pos = 0; entry = 0.0; units = 0.0
                    continue
                if has_long_exit and long_exit[i]:
                    fee = abs(next_bid * units) / 1_000_000.0 * commission_per_million
                    pnl = (next_bid - entry) * units - fee
                    add_daily(i + 1, pnl)
                    cash += pnl; trades += 1; long_trades += 1; signal_exits += 1
                    max_trade_drawdown = max(max_trade_drawdown, cur_trade_drawdown)
                    sum_trade_drawdown += cur_trade_drawdown; cur_trade_drawdown = 0.0
                    if pnl >= 0:
                        wins += 1; gross_win += pnl
                    else:
                        losses += 1; gross_loss += -pnl
                    pos = 0; entry = 0.0; units = 0.0
                    if reverse_on_flip and allow_short:
                        reverse_to = -1
                    else:
                        continue

            if pos == -1:
                adverse_bps = (next_ask / entry - 1.0) * 10000.0 if entry > 0 else 1e18
                if adverse_bps >= liquidation_bps:
                    pnl = -max(0.0, cash)
                    cash = 0.0
                    add_daily(i + 1, pnl)
                    trades += 1; losses += 1; short_trades += 1; liquidations += 1
                    loss = -pnl
                    if loss > max_trade_drawdown:
                        max_trade_drawdown = loss
                    if cur_trade_drawdown > max_trade_drawdown:
                        max_trade_drawdown = cur_trade_drawdown
                    sum_trade_drawdown += max(loss, cur_trade_drawdown)
                    account_dead = True
                    gross_loss += loss
                    pos = 0
                    break
                if sl_dist > 0 and next_ask >= entry + sl_dist:
                    fee = abs(next_ask * units) / 1_000_000.0 * commission_per_million
                    pnl = (entry - next_ask) * units - fee
                    add_daily(i + 1, pnl)
                    cash += pnl; trades += 1; short_trades += 1; stop_losses += 1
                    max_trade_drawdown = max(max_trade_drawdown, cur_trade_drawdown)
                    sum_trade_drawdown += cur_trade_drawdown; cur_trade_drawdown = 0.0
                    if pnl >= 0:
                        wins += 1; gross_win += pnl
                    else:
                        losses += 1; gross_loss += -pnl
                    pos = 0; entry = 0.0; units = 0.0
                    continue
                if tp_dist > 0 and next_ask <= entry - tp_dist:
                    fee = abs(next_ask * units) / 1_000_000.0 * commission_per_million
                    pnl = (entry - next_ask) * units - fee
                    add_daily(i + 1, pnl)
                    cash += pnl; trades += 1; short_trades += 1
                    max_trade_drawdown = max(max_trade_drawdown, cur_trade_drawdown)
                    sum_trade_drawdown += cur_trade_drawdown; cur_trade_drawdown = 0.0
                    if pnl >= 0:
                        wins += 1; gross_win += pnl
                    else:
                        losses += 1; gross_loss += -pnl
                    pos = 0; entry = 0.0; units = 0.0
                    continue
                if has_short_exit and short_exit[i]:
                    fee = abs(next_ask * units) / 1_000_000.0 * commission_per_million
                    pnl = (entry - next_ask) * units - fee
                    add_daily(i + 1, pnl)
                    cash += pnl; trades += 1; short_trades += 1; signal_exits += 1
                    max_trade_drawdown = max(max_trade_drawdown, cur_trade_drawdown)
                    sum_trade_drawdown += cur_trade_drawdown; cur_trade_drawdown = 0.0
                    if pnl >= 0:
                        wins += 1; gross_win += pnl
                    else:
                        losses += 1; gross_loss += -pnl
                    pos = 0; entry = 0.0; units = 0.0
                    if reverse_on_flip and allow_long:
                        reverse_to = 1
                    else:
                        continue

            if pos == 0:
                if reverse_to == 1:
                    margin = cash if compound else amount
                    if margin <= 0:
                        break
                    entry = next_ask
                    units = (margin * leverage) / entry
                    fee = abs(entry * units) / 1_000_000.0 * commission_per_million
                    cash -= fee
                    add_daily(i + 1, -fee)
                    pos = 1
                    cur_trade_drawdown = 0.0
                elif reverse_to == -1:
                    margin = cash if compound else amount
                    if margin <= 0:
                        break
                    entry = next_bid
                    units = (margin * leverage) / entry
                    fee = abs(entry * units) / 1_000_000.0 * commission_per_million
                    cash -= fee
                    add_daily(i + 1, -fee)
                    pos = -1
                    cur_trade_drawdown = 0.0
                elif allow_long and long_trigger[i]:
                    margin = cash if compound else amount
                    if margin <= 0:
                        break
                    entry = next_ask
                    units = (margin * leverage) / entry
                    fee = abs(entry * units) / 1_000_000.0 * commission_per_million
                    cash -= fee
                    add_daily(i + 1, -fee)
                    pos = 1
                    cur_trade_drawdown = 0.0
                elif allow_short and short_trigger[i]:
                    margin = cash if compound else amount
                    if margin <= 0:
                        break
                    entry = next_bid
                    units = (margin * leverage) / entry
                    fee = abs(entry * units) / 1_000_000.0 * commission_per_million
                    cash -= fee
                    add_daily(i + 1, -fee)
                    pos = -1
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

        equity = cash + open_u
        realised = cash - amount
        total = equity - amount
        pf = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
        trade_avg_drawdown = sum_trade_drawdown / trades if trades > 0 else 0.0
        day_count = 0
        day_sum = 0.0
        for d in range(max_days):
            day_sum += daily_pnl[d]
            day_count += 1
        avg_day = day_sum / max(day_count, 1)
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
            long_trades, short_trades, stop_losses, signal_exits,
            liquidations, account_dead, open_side_code, open_bps,
            max_trade_drawdown, trade_avg_drawdown, avg_day, median_day,
            active_day_count,
        )


def live_candles(mid: np.ndarray, ts_ns: np.ndarray, timeframe: str) -> tuple[np.ndarray, ...]:
    tf_ns = timeframe_to_ns(timeframe)
    bucket = ts_ns // tf_ns
    o = np.empty(len(mid), dtype=np.float64)
    h = np.empty(len(mid), dtype=np.float64)
    l = np.empty(len(mid), dtype=np.float64)
    c = np.empty(len(mid), dtype=np.float64)
    closed = np.zeros(len(mid), dtype=np.bool_)

    cur_bucket = int(bucket[0])
    cur_o = cur_h = cur_l = cur_c = float(mid[0])
    for i, px in enumerate(mid):
        b = int(bucket[i])
        if b != cur_bucket:
            cur_bucket = b
            cur_o = cur_h = cur_l = cur_c = float(px)
            closed[i] = True
        else:
            px = float(px)
            cur_h = max(cur_h, px)
            cur_l = min(cur_l, px)
            cur_c = px
        o[i] = cur_o
        h[i] = cur_h
        l[i] = cur_l
        c[i] = cur_c
    return o, h, l, c, closed


def closed_candle_series(mid: np.ndarray, ts_ns: np.ndarray, timeframe: str) -> tuple[np.ndarray, np.ndarray]:
    tf_ns = timeframe_to_ns(timeframe)
    bucket = ts_ns // tf_ns
    close_by_bucket: list[float] = []
    close_tick_idx: list[int] = []
    cur_bucket = int(bucket[0])
    last_mid = float(mid[0])
    last_idx = 0
    for i, px in enumerate(mid):
        b = int(bucket[i])
        if b != cur_bucket:
            close_by_bucket.append(last_mid)
            close_tick_idx.append(last_idx)
            cur_bucket = b
        last_mid = float(px)
        last_idx = i
    return np.array(close_by_bucket, dtype=np.float64), np.array(close_tick_idx, dtype=np.int64)


def candle_state_to_ticks(
    n_ticks: int,
    close_tick_idx: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    out = np.zeros(n_ticks, dtype=np.float64)
    if len(close_tick_idx) == 0:
        return out
    prev = 0
    last_state = 0.0
    for idx, st in zip(close_tick_idx, state):
        out[prev:idx + 1] = last_state
        last_state = float(st)
        prev = idx + 1
    out[prev:] = last_state
    return out


def build_parser(description: str, default_out_name: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--source", choices=["mt5", "local", "dukascopy"], default="mt5")
    ap.add_argument("--csv", default=DEFAULT_MT5_TICK_CSV,
                    help="local tick CSV with timestamp,bid,ask[,pair]")
    ap.add_argument("--pair", help="pair for local CSV if no pair column exists")
    ap.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--to", default=None)
    ap.add_argument("--side", choices=["long", "short", "both"], default="both")
    ap.add_argument("--timeframes", default=None)
    ap.add_argument("--tp-points", default=None)
    ap.add_argument("--sl-points", default=None)
    ap.add_argument("--point-size", type=float, default=None)
    ap.add_argument("--amount", type=float, default=DEFAULT_AMOUNT,
                    help="fixed margin per trade by default; starting balance with --compound")
    ap.add_argument("--compound", action="store_true",
                    help="use current equity as trade margin")
    ap.add_argument("--leverage", type=float, default=100.0)
    ap.add_argument("--commission-per-million", type=float, default=DEFAULT_COMMISSION_PER_MILLION)
    ap.add_argument("--min-trades", type=int, default=0)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--sort-by", choices=["pnl", "avg", "drawdown", "curve"], default="pnl",
                    help="CSV ordering only; console always prints PnL and avg/trade sections")
    ap.add_argument("--out", default=os.path.join(FOREX_DIR, default_out_name))
    return ap


def prepare_args(args) -> None:
    global BACKTEST_WINDOW_DAYS
    if hasattr(args, "pairs"):
        pairs: list[str] = []
        for item in args.pairs or []:
            pairs.extend(p.strip().upper() for p in str(item).split(",") if p.strip())
        args.pairs = pairs or list(DEFAULT_PAIRS)
    if args.source == "local" and not args.csv:
        raise SystemExit("--csv is required for --source local")
    if args.start is None or args.to is None:
        if args.days is not None:
            args.start, args.to = _default_date_window(args.days)
        else:
            args.start, args.to = _default_hour_window(args.hours)
    start_ts = pd.to_datetime(args.start, utc=True, format="mixed")
    end_ts = pd.to_datetime(args.to, utc=True, format="mixed")
    seconds = max((end_ts - start_ts).total_seconds(), 1.0)
    BACKTEST_WINDOW_DAYS = seconds / 86400.0
    os.makedirs(FOREX_DIR, exist_ok=True)


def day_ids_from_timestamps(ts_ns: np.ndarray) -> tuple[np.ndarray, int]:
    if len(ts_ns) == 0:
        return np.zeros(0, dtype=np.int64), 1
    day_ns = np.int64(86_400_000_000_000)
    days = (ts_ns.astype(np.int64, copy=False) // day_ns).astype(np.int64)
    first = int(days[0])
    ids = (days - first).astype(np.int64)
    return ids, int(ids[-1]) + 1


def simulate_triggers(
    pair: str,
    strategy: str,
    params: str,
    timeframe: str,
    bid: np.ndarray,
    ask: np.ndarray,
    long_trigger: np.ndarray,
    short_trigger: np.ndarray,
    long_exit: np.ndarray | None,
    short_exit: np.ndarray | None,
    tp_points: float,
    sl_points: float,
    point_size: float,
    amount: float,
    compound: bool,
    leverage: float,
    commission_per_million: float,
    side: str,
    reverse_on_flip: bool = False,
    entry_allowed: np.ndarray | None = None,
    day_id: np.ndarray | None = None,
    max_days: int | None = None,
    return_trades: bool = False,
) -> TradeResult | tuple[TradeResult, list[TradeLog]]:
    if day_id is None:
        day_id = np.zeros(len(bid), dtype=np.int64)
        max_days = 1
    elif max_days is None:
        max_days = int(np.max(day_id)) + 1 if len(day_id) else 1
    if njit is not None and not return_trades and entry_allowed is None:
        lx = long_exit if long_exit is not None else np.zeros(len(bid), dtype=np.bool_)
        sx = short_exit if short_exit is not None else np.zeros(len(bid), dtype=np.bool_)
        side_mode = 3 if side == "both" else (1 if side == "long" else 2)
        out = _simulate_triggers_numba(
            bid.astype(np.float64, copy=False),
            ask.astype(np.float64, copy=False),
            long_trigger.astype(np.bool_, copy=False),
            short_trigger.astype(np.bool_, copy=False),
            lx.astype(np.bool_, copy=False),
            sx.astype(np.bool_, copy=False),
            day_id.astype(np.int64, copy=False),
            int(max_days),
            long_exit is not None,
            short_exit is not None,
            tp_points,
            sl_points,
            point_size,
            amount,
            bool(compound),
            leverage,
            commission_per_million,
            side_mode,
            bool(reverse_on_flip),
        )
        (
            realised, open_u, total, trades, wins, losses, pf, max_dd,
            long_trades, short_trades, stop_losses, signal_exits,
            liquidations, account_dead, open_side_code, open_bps,
            max_trade_drawdown, trade_avg_drawdown, avg_day, median_day,
            active_days,
        ) = out
        open_side = "long" if open_side_code == 1 else ("short" if open_side_code == -1 else "-")
        win_rate = wins / trades * 100.0 if trades else 0.0
        result = TradeResult(
            pair, strategy, params, timeframe, tp_points, sl_points, point_size,
            realised, open_u, total, int(trades), int(wins), int(losses), win_rate, pf, max_dd,
            int(long_trades), int(short_trades), int(stop_losses), int(signal_exits),
            int(liquidations), bool(account_dead), open_side, open_bps,
        )
        result.trade_max_drawdown = max_trade_drawdown
        result.trade_avg_drawdown = trade_avg_drawdown
        result.avg_day = avg_day
        result.median_day = median_day
        result.active_days = int(active_days)
        return result

    tp_dist = tp_points * point_size
    sl_dist = sl_points * point_size if sl_points > 0 else 0.0
    start_balance = amount
    cash = start_balance
    equity_peak = start_balance
    max_dd = 0.0
    gross_win = gross_loss = 0.0
    trades = wins = losses = long_trades = short_trades = 0
    stop_losses = signal_exits = liquidations = 0
    account_dead = False
    liquidation_bps = 10000.0 / max(leverage, 1e-9)
    max_trade_drawdown = 0.0
    sum_trade_drawdown = 0.0

    pos = 0
    entry = units = 0.0
    entry_i = -1
    cur_trade_drawdown = 0.0
    trade_logs: list[TradeLog] = []
    daily_pnl = np.zeros(int(max_days), dtype=np.float64)
    daily_active = np.zeros(int(max_days), dtype=np.bool_)

    def add_daily(exit_i: int, pnl: float) -> None:
        d = int(day_id[exit_i]) if 0 <= exit_i < len(day_id) else -1
        if 0 <= d < len(daily_pnl):
            daily_pnl[d] += pnl
            daily_active[d] = True

    def close_trade(exit_i: int, exit_px: float, pnl: float, reason: str, side_name: str) -> None:
        nonlocal cash, trades, wins, losses, gross_win, gross_loss
        nonlocal long_trades, short_trades, stop_losses, signal_exits
        nonlocal max_trade_drawdown, sum_trade_drawdown, cur_trade_drawdown
        cash += pnl
        add_daily(exit_i, pnl)
        trades += 1
        max_trade_drawdown = max(max_trade_drawdown, cur_trade_drawdown)
        sum_trade_drawdown += cur_trade_drawdown
        cur_trade_drawdown = 0.0
        if side_name == "long":
            long_trades += 1
        else:
            short_trades += 1
        if reason == "stop_loss":
            stop_losses += 1
        elif reason == "signal_exit":
            signal_exits += 1
        if pnl >= 0:
            wins += 1
            gross_win += pnl
        else:
            losses += 1
            gross_loss += -pnl
        trade_logs.append(TradeLog(
            pair, strategy, params, timeframe, side_name, entry_i, exit_i,
            entry, exit_px, pnl, reason, cash,
        ))

    for i in range(len(bid) - 1):
        next_bid = float(bid[i + 1])
        next_ask = float(ask[i + 1])
        reverse_to = 0

        if pos != 0:
            live_u = open_unrealized(pos, entry, units, next_bid, next_ask)
            live_eq = cash + live_u
            cur_trade_drawdown = max(cur_trade_drawdown, max(0.0, -live_u))
            equity_peak = max(equity_peak, live_eq)
            max_dd = max(max_dd, equity_peak - live_eq)

        if pos == 1:
            adverse_bps = (entry / next_bid - 1.0) * 10000.0 if next_bid > 0 else np.inf
            if adverse_bps >= liquidation_bps:
                pnl = -max(0.0, cash)
                add_daily(i + 1, pnl)
                cash = 0.0
                trades += 1; losses += 1; long_trades += 1; liquidations += 1
                max_trade_drawdown = max(max_trade_drawdown, max(0.0, -pnl), cur_trade_drawdown)
                sum_trade_drawdown += max(max(0.0, -pnl), cur_trade_drawdown)
                cur_trade_drawdown = 0.0
                account_dead = True; gross_loss += -pnl
                trade_logs.append(TradeLog(
                    pair, strategy, params, timeframe, "long", entry_i, i + 1,
                    entry, next_bid, pnl, "liquidation", cash,
                ))
                pos = 0
                break
            if sl_dist > 0 and next_bid <= entry - sl_dist:
                pnl = (next_bid - entry) * units - commission(next_bid, units, commission_per_million)
                close_trade(i + 1, next_bid, pnl, "stop_loss", "long")
                pos = 0; entry = units = 0.0
                continue
            if tp_dist > 0 and next_bid >= entry + tp_dist:
                pnl = (next_bid - entry) * units - commission(next_bid, units, commission_per_million)
                close_trade(i + 1, next_bid, pnl, "take_profit", "long")
                pos = 0; entry = units = 0.0
                continue
            if long_exit is not None and long_exit[i]:
                pnl = (next_bid - entry) * units - commission(next_bid, units, commission_per_million)
                close_trade(i + 1, next_bid, pnl, "signal_exit", "long")
                pos = 0; entry = units = 0.0
                if reverse_on_flip and side in ("short", "both"):
                    reverse_to = -1
                else:
                    continue

        if pos == -1:
            adverse_bps = (next_ask / entry - 1.0) * 10000.0 if entry > 0 else np.inf
            if adverse_bps >= liquidation_bps:
                pnl = -max(0.0, cash)
                add_daily(i + 1, pnl)
                cash = 0.0
                trades += 1; losses += 1; short_trades += 1; liquidations += 1
                max_trade_drawdown = max(max_trade_drawdown, max(0.0, -pnl), cur_trade_drawdown)
                sum_trade_drawdown += max(max(0.0, -pnl), cur_trade_drawdown)
                cur_trade_drawdown = 0.0
                account_dead = True; gross_loss += -pnl
                trade_logs.append(TradeLog(
                    pair, strategy, params, timeframe, "short", entry_i, i + 1,
                    entry, next_ask, pnl, "liquidation", cash,
                ))
                pos = 0
                break
            if sl_dist > 0 and next_ask >= entry + sl_dist:
                pnl = (entry - next_ask) * units - commission(next_ask, units, commission_per_million)
                close_trade(i + 1, next_ask, pnl, "stop_loss", "short")
                pos = 0; entry = units = 0.0
                continue
            if tp_dist > 0 and next_ask <= entry - tp_dist:
                pnl = (entry - next_ask) * units - commission(next_ask, units, commission_per_million)
                close_trade(i + 1, next_ask, pnl, "take_profit", "short")
                pos = 0; entry = units = 0.0
                continue
            if short_exit is not None and short_exit[i]:
                pnl = (entry - next_ask) * units - commission(next_ask, units, commission_per_million)
                close_trade(i + 1, next_ask, pnl, "signal_exit", "short")
                pos = 0; entry = units = 0.0
                if reverse_on_flip and side in ("long", "both"):
                    reverse_to = 1
                else:
                    continue

        if pos == 0:
            allow_long = side in ("long", "both")
            allow_short = side in ("short", "both")
            can_enter = True
            if entry_allowed is not None:
                can_enter = bool(entry_allowed[i + 1])
            if not can_enter:
                continue
            if reverse_to == 1:
                margin = cash if compound else amount
                if margin <= 0:
                    break
                entry = next_ask
                units = units_for_margin(margin, leverage, entry)
                fee = commission(entry, units, commission_per_million)
                cash -= fee
                add_daily(i + 1, -fee)
                pos = 1
                entry_i = i + 1
                cur_trade_drawdown = 0.0
            elif reverse_to == -1:
                margin = cash if compound else amount
                if margin <= 0:
                    break
                entry = next_bid
                units = units_for_margin(margin, leverage, entry)
                fee = commission(entry, units, commission_per_million)
                cash -= fee
                add_daily(i + 1, -fee)
                pos = -1
                entry_i = i + 1
                cur_trade_drawdown = 0.0
            elif allow_long and long_trigger[i]:
                margin = cash if compound else amount
                if margin <= 0:
                    break
                entry = next_ask
                units = units_for_margin(margin, leverage, entry)
                fee = commission(entry, units, commission_per_million)
                cash -= fee
                add_daily(i + 1, -fee)
                pos = 1
                entry_i = i + 1
                cur_trade_drawdown = 0.0
            elif allow_short and short_trigger[i]:
                margin = cash if compound else amount
                if margin <= 0:
                    break
                entry = next_bid
                units = units_for_margin(margin, leverage, entry)
                fee = commission(entry, units, commission_per_million)
                cash -= fee
                add_daily(i + 1, -fee)
                pos = -1
                entry_i = i + 1
                cur_trade_drawdown = 0.0

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
    result = TradeResult(
        pair, strategy, params, timeframe, tp_points, sl_points, point_size,
        realised, open_u, total, trades, wins, losses, win_rate, pf, max_dd,
        long_trades, short_trades, stop_losses, signal_exits, liquidations, account_dead,
        open_side, open_bps,
    )
    if return_trades:
        result.trade_max_drawdown = max_trade_drawdown
        result.trade_avg_drawdown = sum_trade_drawdown / trades if trades else 0.0
        result.avg_day = float(np.mean(daily_pnl)) if len(daily_pnl) else 0.0
        result.median_day = float(np.median(daily_pnl)) if len(daily_pnl) else 0.0
        result.active_days = int(np.count_nonzero(daily_active))
        return result, trade_logs
    result.trade_max_drawdown = max_trade_drawdown
    result.trade_avg_drawdown = sum_trade_drawdown / trades if trades else 0.0
    result.avg_day = float(np.mean(daily_pnl)) if len(daily_pnl) else 0.0
    result.median_day = float(np.median(daily_pnl)) if len(daily_pnl) else 0.0
    result.active_days = int(np.count_nonzero(daily_active))
    return result


def _result_extra_fields(results: list[TradeResult]) -> list[str]:
    preferred = [
        "threshold",
        "upper",
        "lower",
        "risk_mode",
        "risk_value",
        "tp_mode",
        "sl_mode",
        "reentry_mode",
        "rr_floor",
        "prob_ma",
        "eval_session",
        "label_session",
        "window",
        "horizon",
        "model_file",
        "cum_max_drawdown",
        "cum_avg_drawdown",
        "trade_max_drawdown",
        "trade_avg_drawdown",
        "worst_trade_pnl",
        "median_loss",
        "curve_score",
        "curve_r2",
        "curve_slope",
        "curve_resid_std",
        "avg_day",
        "median_day",
        "active_days",
    ]
    out = []
    for name in preferred:
        if any(hasattr(r, name) for r in results):
            out.append(name)
    return out


def _result_csv_row(r: TradeResult, extra_fields: list[str]) -> list:
    row = [
        r.pair, r.strategy, r.params, r.timeframe, r.tp_points, r.sl_points,
        r.point_size, round(r.realised, 6), round(r.open_unrealized, 6),
        round(r.total, 6), r.trades, r.wins, r.losses, round(r.win_rate, 2),
        round(r.profit_factor, 4), round(r.max_drawdown, 6),
    ]
    for name in extra_fields:
        value = getattr(r, name, 0.0)
        if isinstance(value, (str, bool)):
            row.append(value)
        else:
            row.append(round(float(value), 6))
    row.extend([
        r.long_trades, r.short_trades, r.stop_losses, r.signal_exits,
        r.liquidations, int(r.account_dead), r.open_side, round(r.open_bps, 4),
    ])
    return row


def _parse_param_fields(params: str) -> dict[str, str]:
    out: dict[str, str] = {}
    extras = []
    for part in str(params or "").split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            extras.append(part)
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            out[key] = value
        else:
            extras.append(part)
    if extras:
        out["extra"] = ",".join(extras)
    return out


def _fmt_result_value(name: str, value) -> str:
    if value == "" or value is None:
        return ""
    money = {
        "total", "realised", "open", "acct_dd", "cum_dd", "avg_cdd",
        "tr_mdd", "tr_add", "tr_max", "med_loss", "avg_day", "med_day",
    }
    pct = {"wr"}
    if name in money:
        return f"${float(value):+.4f}" if name in {"total", "realised", "open", "avg_day", "med_day"} else f"${float(value):.2f}"
    if name in pct:
        return f"{float(value):.1f}"
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:.1f}"
        return f"{value:.4g}"
    return str(value)


def _result_print_rows(results: list[TradeResult]) -> list[dict]:
    rows = []
    for r in results:
        avg_day = getattr(r, "avg_day", r.total / max(BACKTEST_WINDOW_DAYS, 1e-9))
        median_day = getattr(r, "median_day", avg_day)
        row = {
            "#": 0,
            "pair": r.pair,
            "strat": r.strategy,
            "tf": r.timeframe,
            "tp": f"{r.tp_points:g}",
            "sl": f"{r.sl_points:g}",
            "total": r.total,
            "realised": r.realised,
            "open": r.open_unrealized,
            "tr": r.trades,
            "wr": r.win_rate,
            "pf": r.profit_factor,
            "avg_day": avg_day,
            "med_day": median_day,
            "acct_dd": r.max_drawdown,
        }
        reserved = set(row) | {
            "tr_max", "med_loss", "curve", "r2",
            "stops", "sig", "liq", "dead",
        }
        for key, value in _parse_param_fields(getattr(r, "params", "")).items():
            field = key if key not in reserved else f"p_{key}"
            row[field] = value
        optional = {
            "tr_max": getattr(r, "trade_max_drawdown", None),
            "med_loss": getattr(r, "median_loss", None),
            "curve": getattr(r, "curve_score", None),
            "r2": getattr(r, "curve_r2", None),
        }
        for key, value in optional.items():
            if value is not None:
                row[key] = float(value)
        row.update({
            "stops": getattr(r, "stop_losses", 0),
            "sig": getattr(r, "signal_exits", 0),
            "liq": getattr(r, "liquidations", 0),
            "dead": int(getattr(r, "account_dead", False)),
        })
        rows.append(row)
    return rows


def _print_result_table(title: str, ranked: list[TradeResult], top: int) -> None:
    shown = ranked[:max(1, top)]
    rows = _result_print_rows(shown)
    if not rows:
        print("", flush=True)
        print(f"  {title}", flush=True)
        print("  no results", flush=True)
        return
    for i, row in enumerate(rows, 1):
        row["#"] = i
    base_fields = [
        "#", "pair", "strat", "tf", "tp", "sl", "total", "realised", "open",
        "tr", "wr", "pf", "avg_day", "med_day",
    ]
    metric_fields = [
        "acct_dd", "tr_max", "med_loss", "curve", "r2",
        "stops", "sig", "liq", "dead",
    ]
    known = set(base_fields) | set(metric_fields)
    param_fields = []
    for row in rows:
        for key in row:
            if key not in known and key not in param_fields:
                param_fields.append(key)
    field_order = [
        *base_fields, *param_fields, *metric_fields,
    ]
    fields = [f for f in field_order if any(f in row for row in rows)]
    labels = {
        "wr": "wr%",
        "acct_dd": "acct_dd",
        "tr_max": "tr_max",
        "med_loss": "med_loss",
        "avg_day": "avg/day",
        "med_day": "med/day",
    }
    rendered = []
    widths = {}
    for row in rows:
        rendered_row = {}
        for field in fields:
            value = row.get(field, "")
            text = _fmt_result_value(field, value)
            if field == "file" and len(text) > 38:
                text = text[:17] + "..." + text[-18:]
            rendered_row[field] = text
            widths[field] = max(widths.get(field, 0), len(labels.get(field, field)), len(text))
        rendered.append(rendered_row)

    print("", flush=True)
    print(f"  {title}", flush=True)
    sep = "  "
    header = "  " + sep.join(labels.get(f, f).rjust(widths[f]) for f in fields)
    print(header, flush=True)
    print("  " + "-" * (len(header) - 2), flush=True)
    for row in rendered:
        print("  " + sep.join(row[f].rjust(widths[f]) for f in fields), flush=True)
        print("", flush=True)


def print_ranked_sections(results: list[TradeResult], top: int) -> None:
    by_pnl = sorted(results, key=lambda r: (r.total, r.realised, r.profit_factor), reverse=True)
    by_daily = sorted(
        results,
        key=lambda r: (
            getattr(r, "median_day", r.total / max(BACKTEST_WINDOW_DAYS, 1e-9)),
            getattr(r, "avg_day", r.total / max(BACKTEST_WINDOW_DAYS, 1e-9)),
            r.profit_factor,
            -r.max_drawdown,
            r.realised,
        ),
        reverse=True,
    )
    traded = [r for r in results if r.trades > 0]
    by_low_drawdown = sorted(
        traded or results,
        key=lambda r: (
            r.max_drawdown,
            -r.total,
            -getattr(r, "median_day", r.total / max(BACKTEST_WINDOW_DAYS, 1e-9)),
            -r.win_rate,
            -r.trades,
            -r.profit_factor,
        ),
    )
    _print_result_table(f"top {top} by total PnL", by_pnl, top)
    _print_result_table(f"top {top} by median daily PnL", by_daily, top)
    _print_result_table(f"top {top} by lowest account drawdown", by_low_drawdown, top)


def _sort_results(results: list[TradeResult], sort_by: str) -> None:
    if sort_by == "avg":
        results.sort(
            key=lambda r: (
                getattr(r, "median_day", r.total / max(BACKTEST_WINDOW_DAYS, 1e-9)),
                getattr(r, "avg_day", r.total / max(BACKTEST_WINDOW_DAYS, 1e-9)),
                r.profit_factor,
                -r.max_drawdown,
                r.realised,
            ),
            reverse=True,
        )
    elif sort_by == "drawdown":
        results.sort(
            key=lambda r: (
                getattr(r, "cum_avg_drawdown", r.max_drawdown),
                getattr(r, "cum_max_drawdown", r.max_drawdown),
                -r.total,
            ),
            )
    elif sort_by == "curve":
        results.sort(
            key=lambda r: (
                getattr(r, "curve_score", -1e18),
                getattr(r, "curve_r2", 0.0),
                r.total,
            ),
            reverse=True,
        )
    else:
        results.sort(key=lambda r: (r.total, r.realised, r.profit_factor), reverse=True)


def write_results(path: str, results: list[TradeResult], top: int, sort_by: str = "pnl") -> None:
    _sort_results(results, sort_by)
    extra_fields = _result_extra_fields(results)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = [
            "pair", "strategy", "params", "timeframe", "tp_points", "sl_points",
            "point_size", "realised", "open_unrealized", "total", "trades",
            "wins", "losses", "win_rate", "profit_factor", "max_drawdown",
        ]
        header.extend(extra_fields)
        header.extend([
            "long_trades", "short_trades", "stop_losses", "signal_exits",
            "liquidations", "account_dead", "open_side", "open_bps",
        ])
        w.writerow(header)
        for r in results:
            w.writerow(_result_csv_row(r, extra_fields))
    print_ranked_sections(results, top)


def load_market(args):
    prepare_args(args)
    print(
        f"[pine] source={args.source} from={args.start} to={args.to} "
        f"pairs={len(args.pairs)} amount=${args.amount:g} "
        f"compound={int(args.compound)}",
        flush=True,
    )
    t0 = time.time()
    ticks = load_ticks(args)
    start_ts = pd.to_datetime(args.start, utc=True, format="mixed")
    end_ts = pd.to_datetime(args.to, utc=True, format="mixed")
    ticks = ticks[(ticks["timestamp"] >= start_ts) & (ticks["timestamp"] < end_ts)]
    if ticks.empty:
        raise SystemExit("no ticks loaded")
    return ticks, t0
