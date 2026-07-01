"""Unified base-signal forex strategy sweep.

This intentionally strips the old experiments down to one execution model:
    - indicator produces long/short/flat state
    - mode=normal uses that state, mode=invert flips it
    - fixed TP/SL exits if either is non-zero
    - tp=0 and sl=0 exits only on opposite strategy signal

No trails, rebreaks, hold timers, exit multipliers, or strategy-specific exits.
"""

from __future__ import annotations

import os
import time
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

import numpy as np

from forex_signal_sweep_common import (
    build_bid_ohlc,
    ema,
    map_state_to_ticks,
    rma,
    rolling_high_prev,
    rolling_low_prev,
    simulate_state_strategy,
)
from forex_strategy_common import (
    active_session_allowed,
    build_parser,
    day_ids_from_timestamps,
    default_point_size,
    load_market,
    parse_num_list,
    parse_str_list,
)
from forex_parabolic_sar_tick_backtest import parabolic_sar
from forex_supertrend_backtest import supertrend_state


DEFAULT_STRATEGIES = [
    "keltner", "donchian", "bollinger", "rsi", "stoch", "macd",
    "ema", "ema_pair", "cci", "dmi", "supertrend", "volty", "psar",
    "bb_rsi"
]
DEFAULT_TIMEFRAMES = ["1m", "3m", "5m", "10m", "15m","30m"]
DEFAULT_MODES = ["normal", "invert"]
DEFAULT_SESSIONS = [-1, 0, 1, 2]

GOLD_TP = [0,50, 100, 200, 300, 400]
GOLD_SL = [0,50, 100, 200, 300, 400]
FX_TP = [0,15, 30, 45, 60, 75, 90]
FX_SL = [0,15, 30, 45, 60, 75, 90]

GRID_SMALL = {
    "keltner_lengths": [20, 34, 48, 55, 64, 89, 144],
    "keltner_mults": [1.0, 1.5, 2.0, 2.5],
    "donchian_lengths": [8, 12, 16, 20, 34],
    "bollinger_lengths": [13, 20, 34, 55],
    "bollinger_mults": [1.5, 2.0, 2.5],
    "rsi_periods": [7, 10, 14],
    "rsi_kinds": ["rsix_reentry"],
    "stoch_lengths": [7, 14, 21],
    "stoch_lows": [20, 30],
    "stoch_highs": [70, 80],
    "macd_fasts": [5, 8, 12],
    "macd_slows": [13, 17, 26],
    "macd_signals": [1, 3, 5],
    "ema_lengths": [9, 21, 34, 55, 89, 150, 200, 377],
    "ema_pair_fasts": [4, 6, 9, 12, 21],
    "ema_pair_slows": [63, 105, 150, 200, 377],
    "cci_lengths": [14, 20, 34],
    "cci_thresholds": [100, 150, 200],
    "dmi_di_lengths": [7, 14, 21],
    "dmi_adx_mins": [15, 20, 25],
    "supertrend_lengths": [7, 10, 14],
    "supertrend_mults": [1.5, 2.0, 3.0],
    "psar_starts": [0.01, 0.02, 0.03],
    "psar_incs": [0.01, 0.02, 0.03],
    "psar_maxs": [0.1, 0.2],
    "volty_lengths": [4, 5, 8, 10, 14, 20],
    "volty_mults": [0.5, 0.75, 1.0, 1.5, 2.0],
    "bb_rsi_lengths": [34, 55, 89],
    "bb_rsi_mults": [1.5, 1.8, 2.0, 2.2],
    "bb_rsi_periods": [10, 14],
    "bb_rsi_oversold": [20, 25],
    "bb_rsi_overbought": [60, 65],
}

GRID_WIDE = {
    "keltner_lengths": [8, 14, 20, 28, 34, 48, 64, 89, 144],
    "keltner_mults": [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
    "donchian_lengths": [4, 6, 8, 12, 16, 20, 28, 34, 48, 64],
    "bollinger_lengths": [8, 13, 20, 34, 55, 89],
    "bollinger_mults": [1.2, 1.5, 1.8, 2.0, 2.2, 2.5],
    "rsi_periods": [5, 7, 10, 14, 21],
    "rsi_kinds": ["rsix_reentry"],
    "stoch_lengths": [5, 7, 10, 14, 21, 34],
    "stoch_lows": [15, 20, 25, 30],
    "stoch_highs": [70, 75, 80, 85],
    "macd_fasts": [3, 5, 8, 12,15,21],
    "macd_slows": [10, 13, 17, 21, 26, 34, 55],
    "macd_signals": [1, 3, 5, 9],
    "ema_lengths": [9, 14, 21, 34, 55, 89, 144, 200, 377],
    "ema_pair_fasts": [3, 4, 6, 9, 12, 21],
    "ema_pair_slows": [34, 55, 89, 150, 200, 377],
    "cci_lengths": [10, 14, 20, 34, 55],
    "cci_thresholds": [75, 100, 150, 200],
    "dmi_di_lengths": [7, 10, 14, 21],
    "dmi_adx_mins": [10, 15, 20, 25, 30],
    "supertrend_lengths": [5, 7, 10, 14, 21],
    "supertrend_mults": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
    "psar_starts": [0.01, 0.02, 0.03, 0.04],
    "psar_incs": [0.01, 0.02, 0.03, 0.04],
    "psar_maxs": [0.1, 0.2, 0.3],
    "volty_lengths": [3, 4, 5, 6, 8, 10, 14, 20],
    "volty_mults": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5],
    "bb_rsi_lengths": [5, 8, 13, 20, 28, 34, 48, 55, 89],
    "bb_rsi_mults": [1.5, 1.8, 2.0, 2.2, 2.5],
    "bb_rsi_periods": [7, 10, 14],
    "bb_rsi_oversold": [20, 25, 30, 35],
    "bb_rsi_overbought": [60, 65, 70, 75],
}

GRID_MEDIUM = {
    key: value if len(value) <= 4 else value[: max(4, (len(value) + 1) // 2)]
    for key, value in GRID_WIDE.items()
}

GRID_FOCUSED = {
    **GRID_SMALL,
    "bollinger_lengths": [20, 34, 55, 89],
    "bollinger_mults": [1.5, 1.8, 2.0, 2.2],
    "ema_pair_fasts": [1, 6, 9, 12, 21],
    "ema_pair_slows": [50, 100, 150, 200, 377],
    "cci_lengths": [10, 14, 20, 34],
    "cci_thresholds": [100, 150, 200],
    "stoch_lengths": [5, 7, 10, 14],
    "stoch_lows": [15, 20, 25, 30],
    "stoch_highs": [85, 80, 75, 70],
    "macd_signals": [1, 3, 5, 9],
    "supertrend_lengths": [7, 10, 14, 21],
    "supertrend_mults": [1.5, 2.0, 2.5, 3.0],
    "psar_starts": [0.01, 0.02, 0.03],
    "psar_incs": [0.01, 0.02, 0.03],
    "psar_maxs": [0.1, 0.2],
    "bb_rsi_lengths": [20, 34, 55, 89],
    "bb_rsi_mults": [1.25,1.5, 1.75, 2.0],
    "bb_rsi_periods": [14],
    "bb_rsi_oversold": [30,35,40],
    "bb_rsi_overbought": [70,65,60],
}


def get_grid(name: str) -> dict:
    if name == "small":
        return GRID_SMALL
    if name == "medium":
        return GRID_MEDIUM
    if name == "wide":
        return GRID_WIDE
    if name == "focused":
        return GRID_FOCUSED
    raise SystemExit("--grid must be small, medium, wide, or focused")


def fmt_money_signed(value: float) -> str:
    return f"${value:+.4f}"


def fmt_money_dd(value: float) -> str:
    return f"${value:.2f}"


def print_unified_sections(results, top: int) -> None:
    def daily(r):
        return getattr(r, "median_day", r.total)

    def pnl_dd_ratio(r):
        if r.max_drawdown <= 0:
            return 0.0
        return r.total / r.max_drawdown

    sections = [
        ("top by total PnL", sorted(results, key=lambda r: (r.total, r.realised, r.profit_factor), reverse=True)),
        ("top by median daily PnL", sorted(results, key=lambda r: (daily(r), getattr(r, "avg_day", r.total), r.profit_factor, -r.max_drawdown), reverse=True)),
        ("top by total/account DD", sorted([r for r in results if r.trades > 0 and r.total > 0 and r.max_drawdown > 0] or results, key=lambda r: (pnl_dd_ratio(r), r.total, daily(r), r.profit_factor), reverse=True)),
    ]
    headers = [
        "#", "pair", "strat", "tf", "tp", "sl", "total", "realised", "open",
        "tr", "wr%", "pf", "avg/day", "med/day", "acct_dd", "tr_max",
        "cum_dd", "worst_loss", "pnl/dd", "med_loss", "stops", "sig", "liq", "dead", "params",
    ]
    for title, ranked in sections:
        rows = []
        for i, r in enumerate(ranked[:max(1, top)], 1):
            rows.append([
                str(i),
                r.pair,
                r.strategy,
                r.timeframe,
                f"{r.tp_points:g}",
                f"{r.sl_points:g}",
                fmt_money_signed(r.total),
                fmt_money_signed(r.realised),
                fmt_money_signed(r.open_unrealized),
                str(r.trades),
                f"{r.win_rate:.1f}",
                f"{r.profit_factor:.4g}",
                fmt_money_signed(getattr(r, "avg_day", r.total)),
                fmt_money_signed(getattr(r, "median_day", r.total)),
                fmt_money_dd(r.max_drawdown),
                fmt_money_dd(getattr(r, "trade_max_drawdown", 0.0)),
                fmt_money_dd(getattr(r, "cum_max_drawdown", 0.0)),
                fmt_money_signed(getattr(r, "worst_trade_pnl", 0.0)),
                f"{pnl_dd_ratio(r):.2f}",
                fmt_money_signed(getattr(r, "median_loss", 0.0)),
                str(getattr(r, "stop_losses", 0)),
                str(getattr(r, "signal_exits", 0)),
                str(getattr(r, "liquidations", 0)),
                str(int(getattr(r, "account_dead", False))),
                str(getattr(r, "params", "")),
            ])
        print("", flush=True)
        print(f"  {title}", flush=True)
        if not rows:
            print("  no results", flush=True)
            continue
        widths = [len(h) for h in headers]
        for row in rows:
            for j, cell in enumerate(row):
                widths[j] = max(widths[j], len(cell))
        print("  " + " ".join(h.rjust(widths[i]) for i, h in enumerate(headers)), flush=True)
        print("  " + "-" * (sum(widths) + len(widths) - 1), flush=True)
        for row in rows:
            print("  " + " ".join(cell.rjust(widths[i]) for i, cell in enumerate(row)), flush=True)


def write_unified_csv(path: str, results) -> None:
    results.sort(key=lambda r: (r.total, r.realised, r.profit_factor), reverse=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = [
        "pair", "strategy", "timeframe", "tp_points", "sl_points", "point_size",
        "realised", "open_unrealized", "total", "trades", "wins", "losses",
        "win_rate", "profit_factor", "max_drawdown", "avg_day", "median_day",
        "trade_max_drawdown", "cum_max_drawdown", "worst_trade_pnl",
        "median_loss", "long_trades", "short_trades", "stop_losses",
        "signal_exits", "liquidations", "account_dead", "open_side",
        "open_bps", "params",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in results:
            w.writerow([
                r.pair, r.strategy, r.timeframe, r.tp_points, r.sl_points,
                r.point_size, round(r.realised, 6), round(r.open_unrealized, 6),
                round(r.total, 6), r.trades, r.wins, r.losses,
                round(r.win_rate, 2), round(r.profit_factor, 4),
                round(r.max_drawdown, 6), round(getattr(r, "avg_day", 0.0), 6),
                round(getattr(r, "median_day", 0.0), 6),
                round(getattr(r, "trade_max_drawdown", 0.0), 6),
                round(getattr(r, "cum_max_drawdown", 0.0), 6),
                round(getattr(r, "worst_trade_pnl", 0.0), 6),
                round(getattr(r, "median_loss", 0.0), 6),
                r.long_trades, r.short_trades, r.stop_losses, r.signal_exits,
                r.liquidations, int(r.account_dead), r.open_side,
                round(r.open_bps, 4), r.params,
            ])


def write_unified_results(path: str, results, top: int) -> None:
    write_unified_csv(path, results)
    print_unified_sections(results, top)


def rolling_sma(x: np.ndarray, length: int) -> np.ndarray:
    out = np.full(len(x), np.nan, dtype=np.float64)
    if length <= 1:
        return x.astype(np.float64)
    if len(x) < length:
        return out
    csum = np.cumsum(np.insert(x.astype(np.float64), 0, 0.0))
    out[length - 1:] = (csum[length:] - csum[:-length]) / float(length)
    return out


def rolling_min(x: np.ndarray, length: int) -> np.ndarray:
    out = np.full(len(x), np.nan, dtype=np.float64)
    for i in range(length - 1, len(x)):
        out[i] = np.min(x[i - length + 1:i + 1])
    return out


def rolling_max(x: np.ndarray, length: int) -> np.ndarray:
    out = np.full(len(x), np.nan, dtype=np.float64)
    for i in range(length - 1, len(x)):
        out[i] = np.max(x[i - length + 1:i + 1])
    return out


def true_range(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    prev = np.empty_like(closes)
    prev[0] = closes[0]
    prev[1:] = closes[:-1]
    return np.maximum(highs - lows, np.maximum(np.abs(highs - prev), np.abs(lows - prev)))


def apply_mode(state: np.ndarray, mode: str) -> np.ndarray:
    if mode == "invert":
        return -state
    return state


def keltner_state(highs, lows, closes, length: int, mult: float, mode: str) -> tuple[np.ndarray, str]:
    tr = true_range(highs, lows, closes)
    center = ema(closes, length)
    atr = rma(tr, length)
    upper = center + mult * atr
    lower = center - mult * atr
    state = np.zeros(len(closes), dtype=np.float64)
    if mode == "invert":
        armed_long = False
        armed_short = False
        for i in range(len(closes)):
            if not np.isfinite(atr[i]):
                continue
            if closes[i] < lower[i]:
                armed_long = True
                armed_short = False
            elif armed_long and closes[i] >= lower[i]:
                state[i] = 1.0
                armed_long = False
            if closes[i] > upper[i]:
                armed_short = True
                armed_long = False
            elif armed_short and closes[i] <= upper[i]:
                state[i] = -1.0
                armed_short = False
        state[~np.isfinite(atr)] = 0.0
        return state, f"length={length};mult={mult:g};mode={mode}"
    state[closes > upper] = 1.0
    state[closes < lower] = -1.0
    state[~np.isfinite(atr)] = 0.0
    return state, f"length={length};mult={mult:g};mode={mode}"


def keltner_inside_exit_state(highs, lows, closes, length: int, mult: float, mode: str) -> np.ndarray:
    tr = true_range(highs, lows, closes)
    center = ema(closes, length)
    atr = rma(tr, length)
    upper = center + mult * atr
    lower = center - mult * atr
    state = np.zeros(len(closes), dtype=np.float64)
    if mode == "invert":
        # Failed re-entry exit: after fading back inside, close if price breaks outside again.
        state[closes < lower] = -1.0
        state[closes > upper] = 1.0
    else:
        # Continuation exit: close when price returns inside from the breakout side.
        for i in range(1, len(closes)):
            if not (np.isfinite(upper[i]) and np.isfinite(lower[i]) and np.isfinite(upper[i - 1]) and np.isfinite(lower[i - 1])):
                continue
            if closes[i - 1] > upper[i - 1] and closes[i] <= upper[i]:
                state[i] = -1.0
            elif closes[i - 1] < lower[i - 1] and closes[i] >= lower[i]:
                state[i] = 1.0
    state[~np.isfinite(atr)] = 0.0
    return state


def keltner_neutral_exit_state(highs, lows, closes, length: int, mult: float, mode: str) -> np.ndarray:
    del highs, lows, mult
    center = ema(closes, length)
    state = np.zeros(len(closes), dtype=np.float64)
    if mode == "invert":
        # Mean-reversion long targets an upward center cross; short targets a downward cross.
        for i in range(1, len(closes)):
            if not (np.isfinite(center[i]) and np.isfinite(center[i - 1])):
                continue
            if closes[i - 1] < center[i - 1] and closes[i] >= center[i]:
                state[i] = -1.0
            elif closes[i - 1] > center[i - 1] and closes[i] <= center[i]:
                state[i] = 1.0
    else:
        # Continuation long invalidates on a downward center cross; short on an upward cross.
        for i in range(1, len(closes)):
            if not (np.isfinite(center[i]) and np.isfinite(center[i - 1])):
                continue
            if closes[i - 1] > center[i - 1] and closes[i] <= center[i]:
                state[i] = -1.0
            elif closes[i - 1] < center[i - 1] and closes[i] >= center[i]:
                state[i] = 1.0
    state[~np.isfinite(center)] = 0.0
    return state


def donchian_state(highs, lows, closes, length: int, mode: str) -> tuple[np.ndarray, str]:
    upper = rolling_high_prev(highs, length)
    lower = rolling_low_prev(lows, length)
    state = np.zeros(len(closes), dtype=np.float64)
    state[closes > upper] = 1.0
    state[closes < lower] = -1.0
    state[~np.isfinite(upper) | ~np.isfinite(lower)] = 0.0
    return apply_mode(state, mode), f"length={length};mode={mode}"


def bollinger_state(highs, lows, closes, length: int, mult: float, mode: str) -> tuple[np.ndarray, str]:
    basis = rolling_sma(closes, length)
    dev = np.full(len(closes), np.nan, dtype=np.float64)
    for i in range(length - 1, len(closes)):
        dev[i] = np.std(closes[i - length + 1:i + 1])
    upper = basis + mult * dev
    lower = basis - mult * dev
    state = np.zeros(len(closes), dtype=np.float64)
    if mode == "invert":
        state[closes > upper] = 1.0
        state[closes < lower] = -1.0
        state[~np.isfinite(upper) | ~np.isfinite(lower)] = 0.0
        return state, f"length={length};mult={mult:g};mode={mode}"
    prev_close = np.empty_like(closes)
    prev_close[0] = np.nan
    prev_close[1:] = closes[:-1]
    prev_upper = np.empty_like(upper)
    prev_lower = np.empty_like(lower)
    prev_upper[0] = np.nan
    prev_lower[0] = np.nan
    prev_upper[1:] = upper[:-1]
    prev_lower[1:] = lower[:-1]
    state[(prev_close <= prev_lower) & (closes > lower)] = 1.0
    state[(prev_close >= prev_upper) & (closes < upper)] = -1.0
    return state, f"length={length};mult={mult:g};mode={mode}"


def bollinger_level_exit_state(highs, lows, closes, length: int, mult: float, mode: str) -> np.ndarray:
    basis = rolling_sma(closes, length)
    dev = np.full(len(closes), np.nan, dtype=np.float64)
    for i in range(length - 1, len(closes)):
        dev[i] = np.std(closes[i - length + 1:i + 1])
    upper = basis + mult * dev
    lower = basis - mult * dev
    state = np.zeros(len(closes), dtype=np.float64)
    if mode == "invert":
        # Continuation invalidation: close if price falls back inside from the breakout side.
        for i in range(1, len(closes)):
            if not (
                np.isfinite(upper[i])
                and np.isfinite(lower[i])
                and np.isfinite(upper[i - 1])
                and np.isfinite(lower[i - 1])
            ):
                continue
            if closes[i - 1] > upper[i - 1] and closes[i] <= upper[i]:
                state[i] = -1.0
            elif closes[i - 1] < lower[i - 1] and closes[i] >= lower[i]:
                state[i] = 1.0
    else:
        # Mean-reversion invalidation: close if re-entry fails back outside.
        state[closes < lower] = -1.0
        state[closes > upper] = 1.0
    state[~np.isfinite(upper) | ~np.isfinite(lower)] = 0.0
    return state


def bollinger_target_exit_state(highs, lows, closes, length: int, mult: float, mode: str, target: str) -> np.ndarray:
    basis = rolling_sma(closes, length)
    dev = np.full(len(closes), np.nan, dtype=np.float64)
    for i in range(length - 1, len(closes)):
        dev[i] = np.std(closes[i - length + 1:i + 1])
    upper = basis + mult * dev
    lower = basis - mult * dev
    state = np.zeros(len(closes), dtype=np.float64)
    if target == "neutral":
        long_take = closes >= basis if mode == "normal" else closes <= basis
        short_take = closes <= basis if mode == "normal" else closes >= basis
    elif target == "opposite":
        long_take = closes >= upper if mode == "normal" else closes <= lower
        short_take = closes <= lower if mode == "normal" else closes >= upper
    else:
        raise ValueError(f"bad Bollinger target: {target}")
    state[long_take] = -1.0
    state[short_take] = 1.0
    state[~np.isfinite(upper) | ~np.isfinite(lower) | ~np.isfinite(basis)] = 0.0
    return state


def bb_rsi_state(highs, lows, closes, bb_length: int, bb_mult: float, rsi_period: int,
                 oversold: float, overbought: float, mode: str) -> tuple[np.ndarray, str]:
    basis = rolling_sma(closes, bb_length)
    dev = np.full(len(closes), np.nan, dtype=np.float64)
    for i in range(bb_length - 1, len(closes)):
        dev[i] = np.std(closes[i - bb_length + 1:i + 1])
    upper = basis + bb_mult * dev
    lower = basis - bb_mult * dev
    delta = np.empty(len(closes), dtype=np.float64)
    delta[0] = 0.0
    delta[1:] = closes[1:] - closes[:-1]
    gains = np.maximum(delta, 0.0)
    losses = np.maximum(-delta, 0.0)
    avg_gain = rma(gains, rsi_period)
    avg_loss = rma(losses, rsi_period)
    rs = avg_gain / np.maximum(avg_loss, 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    state = np.zeros(len(closes), dtype=np.float64)
    long_armed = False
    short_armed = False
    for i in range(len(closes)):
        if not np.isfinite(lower[i]) or not np.isfinite(upper[i]) or not np.isfinite(rsi[i]):
            continue
        if closes[i] < lower[i] and rsi[i] <= oversold:
            long_armed = True
            short_armed = False
        elif long_armed and closes[i] > lower[i]:
            state[i] = 1.0
            long_armed = False
        if closes[i] > upper[i] and rsi[i] >= overbought:
            short_armed = True
            long_armed = False
        elif short_armed and closes[i] < upper[i]:
            state[i] = -1.0
            short_armed = False
    return apply_mode(state, mode), (
        f"bb={bb_length};mult={bb_mult:g};rsi={rsi_period};"
        f"os={oversold:g};ob={overbought:g};mode={mode}"
    )


def bb_rsi_level_exit_state(highs, lows, closes, bb_length: int, bb_mult: float, rsi_period: int,
                            oversold: float, overbought: float, mode: str) -> np.ndarray:
    basis = rolling_sma(closes, bb_length)
    dev = np.full(len(closes), np.nan, dtype=np.float64)
    for i in range(bb_length - 1, len(closes)):
        dev[i] = np.std(closes[i - bb_length + 1:i + 1])
    upper = basis + bb_mult * dev
    lower = basis - bb_mult * dev
    rsi = rsi_values(closes, rsi_period)
    state = np.zeros(len(closes), dtype=np.float64)
    # Normal exits long if re-entry fails back below lower/OS, exits short if it fails above upper/OB.
    # Invert flips both entries and exits, matching apply_mode() semantics.
    state[(closes < lower) | (rsi <= oversold)] = -1.0
    state[(closes > upper) | (rsi >= overbought)] = 1.0
    state[~np.isfinite(lower) | ~np.isfinite(upper) | ~np.isfinite(rsi)] = 0.0
    return apply_mode(state, mode)


def bb_rsi_legacy_state(highs, lows, closes, bb_length: int, bb_mult: float, rsi_period: int,
                        oversold: float, overbought: float, mode: str) -> tuple[np.ndarray, str]:
    """Original BB_RSI arm logic, preserved as an explicit strategy."""
    basis = rolling_sma(closes, bb_length)
    dev = np.full(len(closes), np.nan, dtype=np.float64)
    for i in range(bb_length - 1, len(closes)):
        dev[i] = np.std(closes[i - bb_length + 1:i + 1])
    upper = basis + bb_mult * dev
    lower = basis - bb_mult * dev
    delta = np.empty(len(closes), dtype=np.float64)
    delta[0] = 0.0
    delta[1:] = closes[1:] - closes[:-1]
    gains = np.maximum(delta, 0.0)
    losses = np.maximum(-delta, 0.0)
    avg_gain = rma(gains, rsi_period)
    avg_loss = rma(losses, rsi_period)
    rs = avg_gain / np.maximum(avg_loss, 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    state = np.zeros(len(closes), dtype=np.float64)
    long_armed = False
    short_armed = False
    for i in range(len(closes)):
        if not np.isfinite(lower[i]) or not np.isfinite(upper[i]) or not np.isfinite(rsi[i]):
            continue
        if closes[i] < lower[i] and rsi[i] <= oversold:
            long_armed = True
        elif long_armed and closes[i] > lower[i]:
            state[i] = 1.0
            long_armed = False
        if closes[i] > upper[i] and rsi[i] >= overbought:
            short_armed = True
        elif short_armed and closes[i] < upper[i]:
            state[i] = -1.0
            short_armed = False
    return apply_mode(state, mode), (
        f"bb={bb_length};mult={bb_mult:g};rsi={rsi_period};"
        f"os={oversold:g};ob={overbought:g};mode={mode};legacy=1"
    )


def rsi_values(closes, period: int) -> np.ndarray:
    delta = np.empty(len(closes), dtype=np.float64)
    delta[0] = 0.0
    delta[1:] = closes[1:] - closes[:-1]
    gains = np.maximum(delta, 0.0)
    losses = np.maximum(-delta, 0.0)
    avg_gain = rma(gains, period)
    avg_loss = rma(losses, period)
    rs = avg_gain / np.maximum(avg_loss, 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def rsi_state(closes, period: int, kind: str, mode: str) -> tuple[np.ndarray, str]:
    rsi = rsi_values(closes, period)
    state = np.zeros(len(closes), dtype=np.float64)
    if kind == "rsi50":
        state[rsi > 50.0] = 1.0
        state[rsi < 50.0] = -1.0
    elif kind == "rsix_reentry":
        long_armed = False
        short_armed = False
        for i in range(len(closes)):
            if not np.isfinite(rsi[i]):
                continue
            if rsi[i] <= 30.0:
                long_armed = True
                short_armed = False
            elif long_armed and rsi[i] > 30.0:
                state[i] = 1.0
                long_armed = False
            if rsi[i] >= 70.0:
                short_armed = True
                long_armed = False
            elif short_armed and rsi[i] < 70.0:
                state[i] = -1.0
                short_armed = False
    else:
        state[rsi < 30.0] = 1.0
        state[rsi > 70.0] = -1.0
    state[~np.isfinite(rsi)] = 0.0
    return apply_mode(state, mode), f"period={period};kind={kind};mode={mode}"


def rsi_level_exit_state(closes, period: int, kind: str, mode: str) -> np.ndarray:
    rsi = rsi_values(closes, period)
    state = np.zeros(len(closes), dtype=np.float64)
    if kind == "rsi50":
        state[rsi < 50.0] = -1.0
        state[rsi > 50.0] = 1.0
    else:
        state[rsi <= 30.0] = -1.0
        state[rsi >= 70.0] = 1.0
    state[~np.isfinite(rsi)] = 0.0
    return apply_mode(state, mode)


def stochastic_state(highs, lows, closes, length: int, low_level: float, high_level: float, mode: str) -> tuple[np.ndarray, str]:
    lo = rolling_min(lows, length)
    hi = rolling_max(highs, length)
    k = 100.0 * (closes - lo) / np.maximum(hi - lo, 1e-12)
    prev = np.empty_like(k)
    prev[0] = np.nan
    prev[1:] = k[:-1]
    state = np.zeros(len(closes), dtype=np.float64)
    if mode == "invert":
        # Explicit continuation: enter with the stretch, not by blindly flipping re-entry.
        state[(prev < high_level) & (k >= high_level)] = 1.0
        state[(prev > low_level) & (k <= low_level)] = -1.0
    else:
        state[(prev <= low_level) & (k > low_level)] = 1.0
        state[(prev >= high_level) & (k < high_level)] = -1.0
    state[~np.isfinite(k)] = 0.0
    return state, f"length={length};low={low_level:g};high={high_level:g};mode={mode}"


def stochastic_level_exit_state(highs, lows, closes, length: int, low_level: float, high_level: float, mode: str) -> np.ndarray:
    lo = rolling_min(lows, length)
    hi = rolling_max(highs, length)
    k = 100.0 * (closes - lo) / np.maximum(hi - lo, 1e-12)
    state = np.zeros(len(closes), dtype=np.float64)
    if mode == "invert":
        # Continuation invalidation: close long when it falls back below high,
        # close short when it recovers back above low.
        prev = np.empty_like(k)
        prev[0] = np.nan
        prev[1:] = k[:-1]
        state[(prev >= high_level) & (k < high_level)] = -1.0
        state[(prev <= low_level) & (k > low_level)] = 1.0
    else:
        state[k <= low_level] = -1.0
        state[k >= high_level] = 1.0
    state[~np.isfinite(k)] = 0.0
    return state


def macd_state(closes, fast: int, slow: int, signal: int, deadband: float, mode: str) -> tuple[np.ndarray, str]:
    line = ema(closes, fast) - ema(closes, slow)
    state = np.zeros(len(closes), dtype=np.float64)
    if signal > 1:
        sig = rolling_sma(line, signal)
        prev_line = np.empty_like(line)
        prev_sig = np.empty_like(sig)
        prev_line[0] = np.nan
        prev_sig[0] = np.nan
        prev_line[1:] = line[:-1]
        prev_sig[1:] = sig[:-1]
        state[(prev_line <= prev_sig) & (line > sig)] = 1.0
        state[(prev_line >= prev_sig) & (line < sig)] = -1.0
        state[~np.isfinite(line) | ~np.isfinite(sig)] = 0.0
        return apply_mode(state, mode), f"fast={fast};slow={slow};signal={signal};deadband={deadband:g};mode={mode}"
    state[line > deadband] = 1.0
    state[line < -deadband] = -1.0
    state[~np.isfinite(line)] = 0.0
    return apply_mode(state, mode), f"fast={fast};slow={slow};signal={signal};deadband={deadband:g};mode={mode}"


def ema_price_state(closes, length: int, mode: str) -> tuple[np.ndarray, str]:
    line = ema(closes, length)
    state = np.zeros(len(closes), dtype=np.float64)
    state[closes > line] = 1.0
    state[closes < line] = -1.0
    state[~np.isfinite(line)] = 0.0
    return apply_mode(state, mode), f"length={length};mode={mode}"


def ema_pair_state(closes, fast: int, slow: int, mode: str) -> tuple[np.ndarray, str]:
    f = ema(closes, fast)
    s = ema(closes, slow)
    state = np.zeros(len(closes), dtype=np.float64)
    state[f > s] = 1.0
    state[f < s] = -1.0
    state[~np.isfinite(f) | ~np.isfinite(s)] = 0.0
    return apply_mode(state, mode), f"fast={fast};slow={slow};mode={mode}"


def cci_state(highs, lows, closes, length: int, threshold: float, mode: str) -> tuple[np.ndarray, str]:
    cci = cci_values(highs, lows, closes, length)
    state = np.zeros(len(closes), dtype=np.float64)
    if mode == "invert":
        # Explicit continuation for inverted mode.
        state[cci >= threshold] = 1.0
        state[cci <= -threshold] = -1.0
    else:
        # Mean reversion at extremes.
        state[cci <= -threshold] = 1.0
        state[cci >= threshold] = -1.0
    state[~np.isfinite(cci)] = 0.0
    return state, f"length={length};threshold={threshold:g};mode={mode}"


def cci_values(highs, lows, closes, length: int) -> np.ndarray:
    typical = (highs + lows + closes) / 3.0
    ma = rolling_sma(typical, length)
    md = np.full(len(closes), np.nan, dtype=np.float64)
    for i in range(length - 1, len(closes)):
        if np.isfinite(ma[i]):
            md[i] = np.mean(np.abs(typical[i - length + 1:i + 1] - ma[i]))
    return (typical - ma) / np.maximum(0.015 * md, 1e-12)


def cci_level_exit_state(highs, lows, closes, length: int, threshold: float, mode: str) -> np.ndarray:
    cci = cci_values(highs, lows, closes, length)
    state = np.zeros(len(closes), dtype=np.float64)
    if mode == "invert":
        # Continuation invalidation: close when CCI falls back inside the threshold.
        prev = np.empty_like(cci)
        prev[0] = np.nan
        prev[1:] = cci[:-1]
        state[(prev >= threshold) & (cci < threshold)] = -1.0
        state[(prev <= -threshold) & (cci > -threshold)] = 1.0
    else:
        # Mean-reversion invalidation: close if it stretches back outside.
        state[cci <= -threshold] = -1.0
        state[cci >= threshold] = 1.0
    state[~np.isfinite(cci)] = 0.0
    return state


def dmi_state(highs, lows, closes, di_length: int, adx_length: int, adx_min: float, mode: str) -> tuple[np.ndarray, str]:
    prev_high = np.empty_like(highs)
    prev_low = np.empty_like(lows)
    prev_close = np.empty_like(closes)
    prev_high[0] = highs[0]
    prev_low[0] = lows[0]
    prev_close[0] = closes[0]
    prev_high[1:] = highs[:-1]
    prev_low[1:] = lows[:-1]
    prev_close[1:] = closes[:-1]
    up_move = highs - prev_high
    down_move = prev_low - lows
    plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)
    atr = rma(true_range(highs, lows, closes), di_length)
    plus_di = 100.0 * rma(plus_dm, di_length) / np.maximum(atr, 1e-12)
    minus_di = 100.0 * rma(minus_dm, di_length) / np.maximum(atr, 1e-12)
    dx = 100.0 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-12)
    adx = rma(dx, adx_length)
    state = np.zeros(len(closes), dtype=np.float64)
    strong = np.isfinite(adx) & (adx >= adx_min)
    state[strong & (plus_di > minus_di)] = 1.0
    state[strong & (minus_di > plus_di)] = -1.0
    return apply_mode(state, mode), f"di={di_length};adx={adx_length};adx_min={adx_min:g};mode={mode}"


def psar_state(highs, lows, closes, start: float, inc: float, max_af: float, mode: str) -> tuple[np.ndarray, str]:
    uptrend, _, _ = parabolic_sar(highs, lows, closes, start, inc, max_af)
    state = np.where(uptrend, 1.0, -1.0).astype(np.float64)
    state[:2] = 0.0
    return apply_mode(state, mode), f"start={start:g};inc={inc:g};max={max_af:g};mode={mode}"


def volty_tick_state(
    n_ticks: int,
    close_idx: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    length: int,
    mult: float,
    mode: str,
) -> tuple[np.ndarray, str]:
    atrs = rolling_sma(true_range(highs, lows, closes), length) * mult
    upper = closes + atrs
    lower = closes - atrs
    state = np.zeros(n_ticks, dtype=np.float64)
    for bar_i in range(length, len(close_idx) - 1):
        up = upper[bar_i]
        lo = lower[bar_i]
        if not np.isfinite(up) or not np.isfinite(lo):
            continue
        start = int(close_idx[bar_i]) + 1
        end = int(close_idx[bar_i + 1]) + 1
        for j in range(start, min(end, n_ticks)):
            hit_long = ask[j] >= up
            hit_short = bid[j] <= lo
            if hit_long or hit_short:
                if hit_long and hit_short:
                    mid_prev = 0.5 * (bid[j - 1] + ask[j - 1]) if j > 0 else 0.5 * (bid[j] + ask[j])
                    hit_long = abs(up - mid_prev) <= abs(mid_prev - lo)
                    hit_short = not hit_long
                state[j] = 1.0 if hit_long else -1.0
                break
    return apply_mode(state, mode), f"length={length};mult={mult:g};mode={mode}"


def default_tp_sl_for_pair(pair: str) -> tuple[list[float], list[float]]:
    if pair.upper() == "XAUUSD":
        return GOLD_TP, GOLD_SL
    return FX_TP, FX_SL


def paired_thresholds(lows: list[float], highs: list[float]) -> list[tuple[float, float]]:
    if len(lows) == len(highs):
        return list(zip(lows, highs))
    if len(lows) == 1:
        return [(lows[0], high) for high in highs]
    if len(highs) == 1:
        return [(low, highs[0]) for low in lows]
    raise ValueError("threshold lists must have the same length, or one list must contain one value")


def iter_strategy_states(
    strategies: list[str],
    modes: list[str],
    sl0_exit_modes: list[str],
    grid: dict,
    tf: str,
    bid: np.ndarray,
    ask: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    close_idx: np.ndarray,
):
    strategy_set = set(strategies)
    for mode in modes:
        if "keltner" in strategy_set:
            for length, mult in product(grid["keltner_lengths"], grid["keltner_mults"]):
                st, p = keltner_state(highs, lows, closes, length, mult, mode)
                st_tick = map_state_to_ticks(len(bid), close_idx, st)
                if "signal" in sl0_exit_modes:
                    yield "keltner", tf, f"{p};sl0_exit=signal", st_tick, st_tick
                if "inside" in sl0_exit_modes:
                    ex = keltner_inside_exit_state(highs, lows, closes, length, mult, mode)
                    yield "keltner", tf, f"{p};sl0_exit=inside", st_tick, map_state_to_ticks(len(bid), close_idx, ex)
                if "neutral" in sl0_exit_modes:
                    ex = keltner_neutral_exit_state(highs, lows, closes, length, mult, mode)
                    yield "keltner", tf, f"{p};sl0_exit=neutral", st_tick, map_state_to_ticks(len(bid), close_idx, ex)
        if "donchian" in strategy_set:
            for length in grid["donchian_lengths"]:
                st, p = donchian_state(highs, lows, closes, length, mode)
                yield "donchian", tf, p, map_state_to_ticks(len(bid), close_idx, st)
        if "bollinger" in strategy_set:
            for length, mult in product(grid["bollinger_lengths"], grid["bollinger_mults"]):
                st, p = bollinger_state(highs, lows, closes, length, mult, mode)
                st_tick = map_state_to_ticks(len(bid), close_idx, st)
                if "level" in sl0_exit_modes:
                    ex = bollinger_level_exit_state(highs, lows, closes, length, mult, mode)
                    yield "bollinger", tf, f"{p};sl0_exit=level", st_tick, map_state_to_ticks(len(bid), close_idx, ex)
                if "neutral" in sl0_exit_modes:
                    target_ex = bollinger_target_exit_state(highs, lows, closes, length, mult, mode, "neutral")
                    level_ex = bollinger_level_exit_state(highs, lows, closes, length, mult, mode)
                    yield (
                        "bollinger",
                        tf,
                        f"{p};sl0_exit=neutral",
                        st_tick,
                        map_state_to_ticks(len(bid), close_idx, target_ex),
                        map_state_to_ticks(len(bid), close_idx, level_ex),
                    )
                if "opposite" in sl0_exit_modes:
                    target_ex = bollinger_target_exit_state(highs, lows, closes, length, mult, mode, "opposite")
                    level_ex = bollinger_level_exit_state(highs, lows, closes, length, mult, mode)
                    yield (
                        "bollinger",
                        tf,
                        f"{p};sl0_exit=opposite",
                        st_tick,
                        map_state_to_ticks(len(bid), close_idx, target_ex),
                        map_state_to_ticks(len(bid), close_idx, level_ex),
                    )
                if "signal" in sl0_exit_modes:
                    yield "bollinger", tf, f"{p};sl0_exit=signal", st_tick, st_tick
        if "bb_rsi" in strategy_set:
            rsi_thresholds = paired_thresholds(grid["bb_rsi_oversold"], grid["bb_rsi_overbought"])
            for length, mult, period, (os_, ob) in product(
                grid["bb_rsi_lengths"], grid["bb_rsi_mults"], grid["bb_rsi_periods"], rsi_thresholds,
            ):
                if mode == "invert":
                    continue
                st, p = bb_rsi_state(highs, lows, closes, length, mult, period, os_, ob, mode)
                st_tick = map_state_to_ticks(len(bid), close_idx, st)
                ex = bb_rsi_level_exit_state(highs, lows, closes, length, mult, period, os_, ob, mode)
                yield "bb_rsi", tf, f"{p};sl0_exit=level", st_tick, map_state_to_ticks(len(bid), close_idx, ex)
        if "bb_rsi_legacy" in strategy_set:
            rsi_thresholds = paired_thresholds(grid["bb_rsi_oversold"], grid["bb_rsi_overbought"])
            for length, mult, period, (os_, ob) in product(
                grid["bb_rsi_lengths"], grid["bb_rsi_mults"], grid["bb_rsi_periods"], rsi_thresholds,
            ):
                if mode == "invert":
                    continue
                st, p = bb_rsi_legacy_state(highs, lows, closes, length, mult, period, os_, ob, mode)
                yield "bb_rsi_legacy", tf, p, map_state_to_ticks(len(bid), close_idx, st)
        if "rsi" in strategy_set:
            for period, kind in product(grid["rsi_periods"], grid["rsi_kinds"]):
                st, p = rsi_state(closes, period, kind, mode)
                st_tick = map_state_to_ticks(len(bid), close_idx, st)
                ex = rsi_level_exit_state(closes, period, kind, mode)
                yield (
                    "rsi",
                    tf,
                    f"{p};sl0_exit=level",
                    st_tick,
                    st_tick,
                    map_state_to_ticks(len(bid), close_idx, ex),
                )
        if "stoch" in strategy_set:
            for length, lo, hi in product(grid["stoch_lengths"], grid["stoch_lows"], grid["stoch_highs"]):
                st, p = stochastic_state(highs, lows, closes, length, lo, hi, mode)
                st_tick = map_state_to_ticks(len(bid), close_idx, st)
                ex = stochastic_level_exit_state(highs, lows, closes, length, lo, hi, mode)
                yield "stoch", tf, f"{p};sl0_exit=level", st_tick, map_state_to_ticks(len(bid), close_idx, ex)
        if "macd" in strategy_set:
            for fast, slow, signal in product(grid["macd_fasts"], grid["macd_slows"], grid["macd_signals"]):
                if fast >= slow:
                    continue
                st, p = macd_state(closes, fast, slow, signal, 0.0, mode)
                st_tick = map_state_to_ticks(len(bid), close_idx, st)
                for reverse_signal in (0, 1):
                    yield "macd", tf, f"{p};reverse_signal={reverse_signal}", st_tick
        if "ema" in strategy_set:
            for length in grid["ema_lengths"]:
                st, p = ema_price_state(closes, length, mode)
                yield "ema", tf, p, map_state_to_ticks(len(bid), close_idx, st)
        if "ema_pair" in strategy_set:
            for fast, slow in product(grid["ema_pair_fasts"], grid["ema_pair_slows"]):
                if fast >= slow:
                    continue
                st, p = ema_pair_state(closes, fast, slow, mode)
                yield "ema_pair", tf, p, map_state_to_ticks(len(bid), close_idx, st)
        if "cci" in strategy_set:
            for length, threshold in product(grid["cci_lengths"], grid["cci_thresholds"]):
                st, p = cci_state(highs, lows, closes, length, threshold, mode)
                st_tick = map_state_to_ticks(len(bid), close_idx, st)
                ex = cci_level_exit_state(highs, lows, closes, length, threshold, mode)
                yield "cci", tf, f"{p};sl0_exit=level", st_tick, map_state_to_ticks(len(bid), close_idx, ex)
        if "dmi" in strategy_set:
            for di_len, adx_min in product(grid["dmi_di_lengths"], grid["dmi_adx_mins"]):
                st, p = dmi_state(highs, lows, closes, di_len, 14, adx_min, mode)
                yield "dmi", tf, p, map_state_to_ticks(len(bid), close_idx, st)
        if "supertrend" in strategy_set:
            for length, mult in product(grid["supertrend_lengths"], grid["supertrend_mults"]):
                st, _, _ = supertrend_state(highs, lows, closes, length, mult, mode)
                p = f"length={length};mult={mult:g};mode={mode}"
                yield "supertrend", tf, p, map_state_to_ticks(len(bid), close_idx, st)
        if "psar" in strategy_set:
            for start, inc, max_af in product(grid["psar_starts"], grid["psar_incs"], grid["psar_maxs"]):
                st, p = psar_state(highs, lows, closes, start, inc, max_af, mode)
                yield "psar", tf, p, map_state_to_ticks(len(bid), close_idx, st)
        if "volty" in strategy_set:
            for length, mult in product(grid["volty_lengths"], grid["volty_mults"]):
                st, p = volty_tick_state(len(bid), close_idx, bid, ask, highs, lows, closes, length, mult, mode)
                yield "volty", tf, p, st


def count_states_for_grid(strategies: list[str], modes: list[str], sl0_exit_modes: list[str], grid: dict, timeframes: list[str]) -> int:
    per_mode = 0
    total = 0
    if "keltner" in strategies:
        base = len(grid["keltner_lengths"]) * len(grid["keltner_mults"])
        k_exit_count = (
            int("signal" in sl0_exit_modes)
            + int("inside" in sl0_exit_modes)
            + int("neutral" in sl0_exit_modes)
        )
        total += base * len(modes) * max(1, k_exit_count) * len(timeframes)
    if "donchian" in strategies:
        per_mode += len(grid["donchian_lengths"])
    if "bollinger" in strategies:
        boll_exit_count = (
            int("signal" in sl0_exit_modes)
            + int("level" in sl0_exit_modes)
            + int("neutral" in sl0_exit_modes)
            + int("opposite" in sl0_exit_modes)
        )
        per_mode += len(grid["bollinger_lengths"]) * len(grid["bollinger_mults"]) * max(1, boll_exit_count)
    if "rsi" in strategies:
        per_mode += len(grid["rsi_periods"]) * len(grid["rsi_kinds"])
    if "stoch" in strategies:
        per_mode += len(grid["stoch_lengths"]) * len(grid["stoch_lows"]) * len(grid["stoch_highs"])
    if "macd" in strategies:
        per_mode += (
            sum(1 for f in grid["macd_fasts"] for s in grid["macd_slows"] if f < s)
            * len(grid["macd_signals"])
            * 2
        )
    if "ema" in strategies:
        per_mode += len(grid["ema_lengths"])
    if "ema_pair" in strategies:
        per_mode += sum(1 for f in grid["ema_pair_fasts"] for s in grid["ema_pair_slows"] if f < s)
    if "cci" in strategies:
        per_mode += len(grid["cci_lengths"]) * len(grid["cci_thresholds"])
    if "dmi" in strategies:
        per_mode += len(grid["dmi_di_lengths"]) * len(grid["dmi_adx_mins"])
    if "supertrend" in strategies:
        per_mode += len(grid["supertrend_lengths"]) * len(grid["supertrend_mults"])
    if "psar" in strategies:
        per_mode += len(grid["psar_starts"]) * len(grid["psar_incs"]) * len(grid["psar_maxs"])
    if "volty" in strategies:
        per_mode += len(grid["volty_lengths"]) * len(grid["volty_mults"])
    if "bb_rsi" in strategies:
        per_mode += (
            len(grid["bb_rsi_lengths"]) * len(grid["bb_rsi_mults"])
            * len(grid["bb_rsi_periods"])
            * len(paired_thresholds(grid["bb_rsi_oversold"], grid["bb_rsi_overbought"]))
        )
    if "bb_rsi_legacy" in strategies:
        per_mode_normal_only = (
            len(grid["bb_rsi_lengths"]) * len(grid["bb_rsi_mults"])
            * len(grid["bb_rsi_periods"])
            * len(paired_thresholds(grid["bb_rsi_oversold"], grid["bb_rsi_overbought"]))
        )
        total += per_mode_normal_only * len(timeframes)
    bb_rsi_normal_only = 0
    if "bb_rsi" in strategies:
        bb_rsi_normal_only = (
            len(grid["bb_rsi_lengths"]) * len(grid["bb_rsi_mults"])
            * len(grid["bb_rsi_periods"])
            * len(paired_thresholds(grid["bb_rsi_oversold"], grid["bb_rsi_overbought"]))
        )
        per_mode -= bb_rsi_normal_only
        total += bb_rsi_normal_only * len(timeframes)
    return total + per_mode * len(modes) * len(timeframes)


def main() -> None:
    ap = build_parser("Unified base-signal strategy sweep", "forex_unified_signal_results.csv")
    ap.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES))
    ap.add_argument("--modes", default=",".join(DEFAULT_MODES))
    ap.add_argument("--sl0-exit-modes", default="signal,inside,neutral,level",
                    help="sl=0 exit variants: signal, inside/neutral(Keltner), level/neutral/opposite(Bollinger), level(RSI/Stoch/CCI/BB_RSI)")
    ap.add_argument("--tp-exit-modes", default="fixed_signal",
                    help="fixed or fixed_signal; fixed_signal also permits strategy exits while PnL is non-negative")
    ap.add_argument("--sl-exit-modes", default="fixed_signal",
                    help="fixed or fixed_signal; fixed_signal also permits strategy exits while PnL is negative")
    ap.add_argument("--sessions", default=",".join(str(x) for x in DEFAULT_SESSIONS))
    ap.add_argument("--grid", choices=["small", "medium", "wide", "focused"], default="small")
    ap.add_argument("--state-batch-size", type=int, default=64, help="number of generated signal states to simulate at once")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--progress-every", type=int, default=1000,
                    help="Print progress every N completed simulations, plus timed updates.")
    args = ap.parse_args()

    strategies = parse_str_list(args.strategies, DEFAULT_STRATEGIES)
    timeframes = parse_str_list(args.timeframes, DEFAULT_TIMEFRAMES)
    modes = parse_str_list(args.modes, DEFAULT_MODES)
    sl0_exit_modes = parse_str_list(args.sl0_exit_modes, ["signal", "inside", "neutral", "level"])
    bad_exit_modes = [m for m in sl0_exit_modes if m not in {"signal", "inside", "neutral", "level", "opposite"}]
    if bad_exit_modes:
        raise SystemExit(f"unknown --sl0-exit-modes values: {','.join(bad_exit_modes)}")
    tp_exit_modes = parse_str_list(args.tp_exit_modes, ["fixed_signal"])
    sl_exit_modes = parse_str_list(args.sl_exit_modes, ["fixed_signal"])
    bad_bracket_modes = [
        m for m in tp_exit_modes + sl_exit_modes if m not in {"fixed", "fixed_signal"}
    ]
    if bad_bracket_modes:
        raise SystemExit(f"unknown TP/SL exit mode values: {','.join(bad_bracket_modes)}")
    sessions = [int(x) for x in parse_num_list(args.sessions, DEFAULT_SESSIONS)]
    grid = get_grid(args.grid)

    ticks, t0 = load_market(args)
    results = []

    for pair, g in ticks.groupby("pair", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        bid = g["bid"].to_numpy(np.float64)
        ask = g["ask"].to_numpy(np.float64)
        ts_ns = g["timestamp"].astype("int64").to_numpy(np.int64)
        point_size = float(args.point_size or default_point_size(pair))
        tp_values = parse_num_list(args.tp_points, default_tp_sl_for_pair(pair)[0])
        sl_values = parse_num_list(args.sl_points, default_tp_sl_for_pair(pair)[1])
        session_cache = {s: active_session_allowed(ts_ns, s) for s in sessions}
        print(f"[unified] {pair} ticks={len(g):,} point={point_size:g}", flush=True)

        total_states = count_states_for_grid(strategies, modes, sl0_exit_modes, grid, timeframes)
        total_combos = (
            total_states * len(tp_values) * len(sl_values) * len(sessions)
            * len(tp_exit_modes) * len(sl_exit_modes)
        )
        print(f"[unified] {pair} states~={total_states:,} combos~={total_combos:,}", flush=True)

        def run_combo(item):
            idx, tp, sl, sess, tp_exit_mode, sl_exit_mode, state_items = item
            state_item = state_items[int(idx)]
            if len(state_item) == 6:
                strat, tf, params, state, profit_exit_state, loss_exit_state = state_item
                exit_state = loss_exit_state
            elif len(state_item) == 5:
                strat, tf, params, state, exit_state = state_item
                profit_exit_state = exit_state
                loss_exit_state = exit_state
            else:
                strat, tf, params, state = state_item
                exit_state = state
                profit_exit_state = state
                loss_exit_state = state
            full_params = (
                f"{params};tp_exit={tp_exit_mode};sl_exit={sl_exit_mode};session={sess}"
            )
            tp_fixed_signal = tp_exit_mode == "fixed_signal"
            sl_fixed_signal = sl_exit_mode == "fixed_signal"
            reverse_on_signal = strat == "macd" and "reverse_signal=1" in params
            return simulate_state_strategy(
                pair, strat, full_params, tf, bid, ask, ts_ns, state, exit_state,
                session_cache[int(sess)], tp, 0.0, sl, 0.0, point_size,
                args.amount, args.compound, args.leverage, args.commission_per_million,
                args.side, reverse_on_signal=reverse_on_signal,
                ignore_signal_exit_when_bracket=(
                    tp > 0 and sl > 0 and not tp_fixed_signal and not sl_fixed_signal
                ),
                signal_exit_profit=tp_fixed_signal,
                signal_exit_loss=sl_fixed_signal,
                profit_exit_state=profit_exit_state,
                loss_exit_state=loss_exit_state,
            )

        done = 0
        last_progress_t = time.time()

        def run_state_batch(strategy: str, state_items: list[tuple], batch_no: int) -> None:
            nonlocal done, last_progress_t
            if not state_items:
                return
            strategy_combos = []
            for idx in range(len(state_items)):
                params = state_items[idx][2]
                for tp, sl, sess, tp_exit_mode, sl_exit_mode in product(
                    tp_values, sl_values, sessions, tp_exit_modes, sl_exit_modes
                ):
                    if (
                        "sl0_exit=inside" in params
                        and float(sl) > 0.0
                        and sl_exit_mode == "fixed"
                    ):
                        continue
                    strategy_combos.append((idx, tp, sl, sess, tp_exit_mode, sl_exit_mode))
            print(
                f"[unified] {pair} {strategy} batch={batch_no} states={len(state_items):,} "
                f"combos={len(strategy_combos):,}",
                flush=True,
            )
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [
                    ex.submit(
                        run_combo,
                        (idx, tp, sl, sess, tp_exit_mode, sl_exit_mode, state_items),
                    )
                    for idx, tp, sl, sess, tp_exit_mode, sl_exit_mode in strategy_combos
                ]
                for fut in as_completed(futs):
                    results.append(fut.result())
                    done += 1
                    now = time.time()
                    if done % max(1, args.progress_every) == 0 or now - last_progress_t >= 30.0:
                        last_progress_t = now
                        pct = done / max(1, total_combos) * 100.0
                        elapsed = now - t0
                        rate = done / max(elapsed, 1e-9)
                        remaining = (total_combos - done) / max(rate, 1e-9)
                        print(
                            f"[unified] {pair} progress {done:,}/{total_combos:,} "
                            f"({pct:.1f}%) rate={rate:.1f}/s eta={remaining/60.0:.1f}m",
                            flush=True,
                        )

        for strategy in strategies:
            state_items: list[tuple] = []
            batch_no = 1
            for tf in timeframes:
                _, highs, lows, closes, close_idx = build_bid_ohlc(bid, ts_ns, tf)
                if len(closes) < 5:
                    continue
                for state_item in iter_strategy_states([strategy], modes, sl0_exit_modes, grid, tf, bid, ask, highs, lows, closes, close_idx):
                    state_items.append(state_item)
                    if len(state_items) >= args.state_batch_size:
                        run_state_batch(strategy, state_items, batch_no)
                        batch_no += 1
                        state_items = []
            run_state_batch(strategy, state_items, batch_no)
            del state_items

        partial = [r for r in results if r.trades >= args.min_trades]
        write_unified_csv(args.out, partial)
        print(
            f"[unified] {pair} flushed partial results rows={len(partial):,} to {args.out}",
            flush=True,
        )

    filtered = [r for r in results if r.trades >= args.min_trades]
    write_unified_results(args.out, filtered, args.top)
    print(f"[unified] wrote {args.out} elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
