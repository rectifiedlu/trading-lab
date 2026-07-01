"""Candle Rider backtest.

Signal:
    - Build candles from MT5 bid ticks.
    - On each closed candle, green -> long, red -> short.
    - mode=invert flips the candle direction.

Execution:
    - Entries execute on the next tick after the candle close.
    - Fixed TP/SL are tick-based.
    - If tp=0, a positive trade exits when an opposite candle closes.
    - If sl=0, a negative trade exits when an opposite candle closes.
    - reverse_on_sl can immediately reverse after a fixed SL or sl=0 loss exit.
"""

from __future__ import annotations

import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

import numpy as np

from forex_signal_sweep_common import build_bid_ohlc
from forex_strategy_common import (
    TradeResult,
    active_session_allowed,
    build_parser,
    commission,
    day_ids_from_timestamps,
    default_point_size,
    load_market,
    njit,
    parse_num_list,
    parse_str_list,
    units_for_margin,
)
from forex_unified_signal_backtest import write_unified_results


DEFAULT_TIMEFRAMES = ["1h","2h","3h","4h","6h"]
DEFAULT_MODES = ["normal","invert"]
DEFAULT_SESSIONS = [-1, 0, 1, 2]
DEFAULT_REVERSE_ON_SL = [0]
DEFAULT_CONFIRM = [0]
GOLD_TP = [0]
GOLD_SL = [0]
FX_TP = [0]
FX_SL = [0]


def default_tp_sl_for_pair(pair: str) -> tuple[list[int], list[int]]:
    return (GOLD_TP, GOLD_SL) if pair.upper() == "XAUUSD" else (FX_TP, FX_SL)


def candle_event_state(
    opens: np.ndarray,
    closes: np.ndarray,
    close_idx: np.ndarray,
    n_ticks: int,
    mode: str,
    confirm: int,
) -> np.ndarray:
    events = np.zeros(n_ticks, dtype=np.float64)
    streak_dir = 0.0
    streak_count = 0
    for o, c, idx in zip(opens, closes, close_idx):
        if c > o:
            st = 1.0
        elif c < o:
            st = -1.0
        else:
            st = 0.0
        if mode == "invert":
            st = -st
        if st == 0.0:
            streak_dir = 0.0
            streak_count = 0
            continue
        if st == streak_dir:
            streak_count += 1
        else:
            streak_dir = st
            streak_count = 1
        if streak_count >= confirm + 1:
            events[int(idx)] = st
    return events


if njit is not None:
    @njit(cache=True)
    def _simulate_candle_rider_numba(
        bid: np.ndarray,
        ask: np.ndarray,
        event: np.ndarray,
        entry_allowed: np.ndarray,
        day_id: np.ndarray,
        max_days: int,
        tp_points: float,
        sl_points: float,
        point_size: float,
        amount: float,
        compound: bool,
        leverage: float,
        commission_per_million: float,
        side_mode: int,
        reverse_on_sl: bool,
    ):
        tp_dist = tp_points * point_size if tp_points > 0.0 else 0.0
        sl_dist = sl_points * point_size if sl_points > 0.0 else 0.0
        allow_long = side_mode == 1 or side_mode == 3
        allow_short = side_mode == 2 or side_mode == 3

        cash = amount
        cash_peak = amount
        cum_max_dd = 0.0
        equity_peak = amount
        max_dd = 0.0
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
        account_dead = False
        liquidation_bps = 10000.0 / max(leverage, 1e-9)

        pos = 0
        entry = 0.0
        units = 0.0
        cur_trade_dd = 0.0
        max_trade_dd = 0.0
        sum_trade_dd = 0.0
        worst_trade = 0.0
        loss_values = np.zeros(len(bid), dtype=np.float64)
        loss_count = 0
        daily_pnl = np.zeros(max_days, dtype=np.float64)

        def add_daily(idx, pnl_value):
            d = day_id[idx]
            if d >= 0 and d < max_days:
                daily_pnl[d] += pnl_value

        def record_trade(pnl_value, side_code, reason_code, idx):
            nonlocal cash, trades, wins, losses, long_trades, short_trades
            nonlocal stop_losses, signal_exits, gross_win, gross_loss
            nonlocal cur_trade_dd, max_trade_dd, sum_trade_dd, worst_trade, loss_count
            cash += pnl_value
            add_daily(idx, pnl_value)
            trades += 1
            if side_code == 1:
                long_trades += 1
            else:
                short_trades += 1
            if reason_code == 1:
                stop_losses += 1
            elif reason_code == 2:
                signal_exits += 1
            if cur_trade_dd > max_trade_dd:
                max_trade_dd = cur_trade_dd
            sum_trade_dd += cur_trade_dd
            cur_trade_dd = 0.0
            if pnl_value >= 0.0:
                wins += 1
                gross_win += pnl_value
            else:
                losses += 1
                gross_loss += -pnl_value
                loss_values[loss_count] = pnl_value
                loss_count += 1
                if pnl_value < worst_trade:
                    worst_trade = pnl_value

        for i in range(len(bid) - 1):
            px_bid = bid[i + 1]
            px_ask = ask[i + 1]
            reverse_to = 0

            if pos != 0:
                live_u = 0.0
                if pos == 1:
                    live_u = (px_bid - entry) * units
                else:
                    live_u = (entry - px_ask) * units
                live_eq = cash + live_u
                if -live_u > cur_trade_dd:
                    cur_trade_dd = -live_u
                if cur_trade_dd < 0.0:
                    cur_trade_dd = 0.0
                if live_eq > equity_peak:
                    equity_peak = live_eq
                dd = equity_peak - live_eq
                if dd > max_dd:
                    max_dd = dd

            if pos == 1:
                adverse_bps = (entry / px_bid - 1.0) * 10000.0 if px_bid > 0.0 else 1e18
                if adverse_bps >= liquidation_bps:
                    pnl = -max(0.0, cash)
                    cash = 0.0
                    add_daily(i + 1, pnl)
                    trades += 1
                    losses += 1
                    long_trades += 1
                    liquidations += 1
                    gross_loss += -pnl
                    account_dead = True
                    pos = 0
                    break
                if sl_dist > 0.0 and px_bid <= entry - sl_dist:
                    fee = abs(px_bid * units) / 1_000_000.0 * commission_per_million
                    pnl = (px_bid - entry) * units - fee
                    record_trade(pnl, 1, 1, i + 1)
                    pos = 0
                    entry = 0.0
                    units = 0.0
                    if reverse_on_sl and allow_short:
                        reverse_to = -1
                    else:
                        continue
                elif tp_dist > 0.0 and px_bid >= entry + tp_dist:
                    fee = abs(px_bid * units) / 1_000_000.0 * commission_per_million
                    pnl = (px_bid - entry) * units - fee
                    record_trade(pnl, 1, 0, i + 1)
                    pos = 0
                    entry = 0.0
                    units = 0.0
                    continue
                elif event[i] < 0.0:
                    live_u = (px_bid - entry) * units
                    should_exit = False
                    is_loss_exit = False
                    if tp_points <= 0.0 and live_u > 0.0:
                        should_exit = True
                    if sl_points <= 0.0 and live_u < 0.0:
                        should_exit = True
                        is_loss_exit = True
                    if should_exit:
                        fee = abs(px_bid * units) / 1_000_000.0 * commission_per_million
                        pnl = live_u - fee
                        record_trade(pnl, 1, 1 if is_loss_exit else 2, i + 1)
                        pos = 0
                        entry = 0.0
                        units = 0.0
                        if is_loss_exit and reverse_on_sl and allow_short:
                            reverse_to = -1
                        else:
                            continue

            if pos == -1:
                adverse_bps = (px_ask / entry - 1.0) * 10000.0 if entry > 0.0 else 1e18
                if adverse_bps >= liquidation_bps:
                    pnl = -max(0.0, cash)
                    cash = 0.0
                    add_daily(i + 1, pnl)
                    trades += 1
                    losses += 1
                    short_trades += 1
                    liquidations += 1
                    gross_loss += -pnl
                    account_dead = True
                    pos = 0
                    break
                if sl_dist > 0.0 and px_ask >= entry + sl_dist:
                    fee = abs(px_ask * units) / 1_000_000.0 * commission_per_million
                    pnl = (entry - px_ask) * units - fee
                    record_trade(pnl, -1, 1, i + 1)
                    pos = 0
                    entry = 0.0
                    units = 0.0
                    if reverse_on_sl and allow_long:
                        reverse_to = 1
                    else:
                        continue
                elif tp_dist > 0.0 and px_ask <= entry - tp_dist:
                    fee = abs(px_ask * units) / 1_000_000.0 * commission_per_million
                    pnl = (entry - px_ask) * units - fee
                    record_trade(pnl, -1, 0, i + 1)
                    pos = 0
                    entry = 0.0
                    units = 0.0
                    continue
                elif event[i] > 0.0:
                    live_u = (entry - px_ask) * units
                    should_exit = False
                    is_loss_exit = False
                    if tp_points <= 0.0 and live_u > 0.0:
                        should_exit = True
                    if sl_points <= 0.0 and live_u < 0.0:
                        should_exit = True
                        is_loss_exit = True
                    if should_exit:
                        fee = abs(px_ask * units) / 1_000_000.0 * commission_per_million
                        pnl = live_u - fee
                        record_trade(pnl, -1, 1 if is_loss_exit else 2, i + 1)
                        pos = 0
                        entry = 0.0
                        units = 0.0
                        if is_loss_exit and reverse_on_sl and allow_long:
                            reverse_to = 1
                        else:
                            continue

            if pos == 0:
                if not entry_allowed[i + 1]:
                    if cash > cash_peak:
                        cash_peak = cash
                    cum_dd = cash_peak - cash
                    if cum_dd > cum_max_dd:
                        cum_max_dd = cum_dd
                    continue
                sig = event[i]
                if reverse_to != 0:
                    sig = reverse_to
                if sig > 0.0 and allow_long:
                    margin = cash if compound else amount
                    if margin <= 0.0:
                        break
                    entry = px_ask
                    units = (margin * leverage) / entry
                    fee = abs(entry * units) / 1_000_000.0 * commission_per_million
                    cash -= fee
                    add_daily(i + 1, -fee)
                    pos = 1
                    cur_trade_dd = 0.0
                elif sig < 0.0 and allow_short:
                    margin = cash if compound else amount
                    if margin <= 0.0:
                        break
                    entry = px_bid
                    units = (margin * leverage) / entry
                    fee = abs(entry * units) / 1_000_000.0 * commission_per_million
                    cash -= fee
                    add_daily(i + 1, -fee)
                    pos = -1
                    cur_trade_dd = 0.0

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
        pf = 0.0
        if gross_loss > 0.0:
            pf = gross_win / gross_loss
        elif gross_win > 0.0:
            pf = 999.0

        avg_day = 0.0
        median_day = 0.0
        active_days = 0
        if max_days > 0:
            avg_day = np.mean(daily_pnl)
            median_day = np.median(daily_pnl)
            for d in range(max_days):
                if daily_pnl[d] != 0.0:
                    active_days += 1

        median_loss = 0.0
        if loss_count > 0:
            tmp = np.sort(loss_values[:loss_count])
            mid = loss_count // 2
            if loss_count % 2 == 1:
                median_loss = tmp[mid]
            else:
                median_loss = 0.5 * (tmp[mid - 1] + tmp[mid])

        return (
            realised, open_u, total, trades, wins, losses, pf, max_dd,
            long_trades, short_trades, stop_losses, signal_exits,
            liquidations, account_dead, open_side_code, open_bps,
            max_trade_dd, sum_trade_dd / trades if trades > 0 else 0.0,
            avg_day, median_day, worst_trade, median_loss,
            cum_max_dd, active_days,
        )


def simulate_candle_rider(
    pair: str,
    timeframe: str,
    params: str,
    bid: np.ndarray,
    ask: np.ndarray,
    event: np.ndarray,
    entry_allowed: np.ndarray,
    day_id: np.ndarray,
    max_days: int,
    tp_points: float,
    sl_points: float,
    point_size: float,
    amount: float,
    compound: bool,
    leverage: float,
    commission_per_million: float,
    side: str,
    reverse_on_sl: bool,
) -> TradeResult:
    side_mode = 3 if side == "both" else (1 if side == "long" else 2)
    if njit is None:
        raise RuntimeError("Candle Rider requires numba for now")
    out = _simulate_candle_rider_numba(
        bid.astype(np.float64, copy=False),
        ask.astype(np.float64, copy=False),
        event.astype(np.float64, copy=False),
        entry_allowed.astype(np.bool_, copy=False),
        day_id.astype(np.int64, copy=False),
        int(max_days),
        float(tp_points),
        float(sl_points),
        float(point_size),
        float(amount),
        bool(compound),
        float(leverage),
        float(commission_per_million),
        int(side_mode),
        bool(reverse_on_sl),
    )
    (
        realised, open_u, total, trades, wins, losses, pf, max_dd,
        long_trades, short_trades, stop_losses, signal_exits,
        liquidations, account_dead, open_side_code, open_bps,
        trade_max_dd, trade_avg_dd, avg_day, median_day, worst_trade, median_loss,
        cum_max_dd, active_days,
    ) = out
    open_side = "long" if open_side_code == 1 else ("short" if open_side_code == -1 else "-")
    win_rate = float(wins) / float(trades) * 100.0 if trades else 0.0
    r = TradeResult(
        pair, "candle_rider", params, timeframe, tp_points, sl_points, point_size,
        realised, open_u, total, int(trades), int(wins), int(losses), win_rate, pf, max_dd,
        int(long_trades), int(short_trades), int(stop_losses), int(signal_exits),
        int(liquidations), bool(account_dead), open_side, open_bps,
    )
    r.trade_max_drawdown = trade_max_dd
    r.trade_avg_drawdown = trade_avg_dd
    r.avg_day = avg_day
    r.median_day = median_day
    r.worst_trade_pnl = worst_trade
    r.median_loss = median_loss
    r.cum_max_drawdown = cum_max_dd
    r.active_days = int(active_days)
    return r


def trace_candle_rider_trades(
    pair: str,
    timeframe: str,
    params: str,
    bid: np.ndarray,
    ask: np.ndarray,
    ts_ns: np.ndarray,
    event: np.ndarray,
    entry_allowed: np.ndarray,
    tp_points: float,
    sl_points: float,
    point_size: float,
    amount: float,
    compound: bool,
    leverage: float,
    commission_per_million: float,
    side: str,
    reverse_on_sl: bool,
) -> list[dict[str, object]]:
    tp_dist = tp_points * point_size if tp_points > 0.0 else 0.0
    sl_dist = sl_points * point_size if sl_points > 0.0 else 0.0
    allow_long = side in ("long", "both")
    allow_short = side in ("short", "both")

    cash = amount
    pos = 0
    entry = 0.0
    units = 0.0
    entry_fee = 0.0
    entry_i = -1
    cur_trade_dd = 0.0
    rows: list[dict[str, object]] = []

    def ts_str(idx: int) -> str:
        return str(np.datetime64(int(ts_ns[idx]), "ns")) if 0 <= idx < len(ts_ns) else ""

    def record_trade(pnl_value: float, side_name: str, reason: str, exit_i: int, exit_px: float) -> None:
        nonlocal cash, cur_trade_dd, entry_fee
        cash += pnl_value
        net_pnl = pnl_value - entry_fee
        if side_name == "long":
            move_points = (exit_px - entry) / point_size
        else:
            move_points = (entry - exit_px) / point_size
        hold_seconds = (int(ts_ns[exit_i]) - int(ts_ns[entry_i])) / 1_000_000_000.0 if entry_i >= 0 else 0.0
        rows.append({
            "pair": pair,
            "strategy": "candle_rider",
            "params": params,
            "timeframe": timeframe,
            "side": side_name,
            "entry_time": ts_str(entry_i),
            "exit_time": ts_str(exit_i),
            "entry_px": entry,
            "exit_px": exit_px,
            "move_points": move_points,
            "exit_pnl": pnl_value,
            "entry_fee": entry_fee,
            "pnl": net_pnl,
            "reason": reason,
            "equity": cash,
            "max_adverse_usd": cur_trade_dd,
            "hold_seconds": hold_seconds,
        })
        entry_fee = 0.0
        cur_trade_dd = 0.0

    for i in range(len(bid) - 1):
        px_bid = bid[i + 1]
        px_ask = ask[i + 1]
        reverse_to = 0

        if pos != 0:
            live_u = (px_bid - entry) * units if pos == 1 else (entry - px_ask) * units
            cur_trade_dd = max(cur_trade_dd, max(0.0, -live_u))

        if pos == 1:
            if sl_dist > 0.0 and px_bid <= entry - sl_dist:
                fee = abs(px_bid * units) / 1_000_000.0 * commission_per_million
                pnl = (px_bid - entry) * units - fee
                record_trade(pnl, "long", "stop_loss", i + 1, px_bid)
                pos = 0
                entry = units = 0.0
                if reverse_on_sl and allow_short:
                    reverse_to = -1
                else:
                    continue
            elif tp_dist > 0.0 and px_bid >= entry + tp_dist:
                fee = abs(px_bid * units) / 1_000_000.0 * commission_per_million
                pnl = (px_bid - entry) * units - fee
                record_trade(pnl, "long", "take_profit", i + 1, px_bid)
                pos = 0
                entry = units = 0.0
                continue
            elif event[i] < 0.0:
                live_u = (px_bid - entry) * units
                should_exit = False
                is_loss_exit = False
                if tp_points <= 0.0 and live_u > 0.0:
                    should_exit = True
                if sl_points <= 0.0 and live_u < 0.0:
                    should_exit = True
                    is_loss_exit = True
                if should_exit:
                    fee = abs(px_bid * units) / 1_000_000.0 * commission_per_million
                    pnl = live_u - fee
                    record_trade(pnl, "long", "signal_loss" if is_loss_exit else "signal_profit", i + 1, px_bid)
                    pos = 0
                    entry = units = 0.0
                    if is_loss_exit and reverse_on_sl and allow_short:
                        reverse_to = -1
                    else:
                        continue

        if pos == -1:
            if sl_dist > 0.0 and px_ask >= entry + sl_dist:
                fee = abs(px_ask * units) / 1_000_000.0 * commission_per_million
                pnl = (entry - px_ask) * units - fee
                record_trade(pnl, "short", "stop_loss", i + 1, px_ask)
                pos = 0
                entry = units = 0.0
                if reverse_on_sl and allow_long:
                    reverse_to = 1
                else:
                    continue
            elif tp_dist > 0.0 and px_ask <= entry - tp_dist:
                fee = abs(px_ask * units) / 1_000_000.0 * commission_per_million
                pnl = (entry - px_ask) * units - fee
                record_trade(pnl, "short", "take_profit", i + 1, px_ask)
                pos = 0
                entry = units = 0.0
                continue
            elif event[i] > 0.0:
                live_u = (entry - px_ask) * units
                should_exit = False
                is_loss_exit = False
                if tp_points <= 0.0 and live_u > 0.0:
                    should_exit = True
                if sl_points <= 0.0 and live_u < 0.0:
                    should_exit = True
                    is_loss_exit = True
                if should_exit:
                    fee = abs(px_ask * units) / 1_000_000.0 * commission_per_million
                    pnl = live_u - fee
                    record_trade(pnl, "short", "signal_loss" if is_loss_exit else "signal_profit", i + 1, px_ask)
                    pos = 0
                    entry = units = 0.0
                    if is_loss_exit and reverse_on_sl and allow_long:
                        reverse_to = 1
                    else:
                        continue

        if pos == 0:
            if not entry_allowed[i + 1]:
                continue
            sig = reverse_to if reverse_to != 0 else event[i]
            if sig > 0.0 and allow_long:
                margin = cash if compound else amount
                if margin <= 0.0:
                    break
                entry = px_ask
                units = (margin * leverage) / entry
                fee = abs(entry * units) / 1_000_000.0 * commission_per_million
                cash -= fee
                entry_fee = fee
                pos = 1
                entry_i = i + 1
                cur_trade_dd = 0.0
            elif sig < 0.0 and allow_short:
                margin = cash if compound else amount
                if margin <= 0.0:
                    break
                entry = px_bid
                units = (margin * leverage) / entry
                fee = abs(entry * units) / 1_000_000.0 * commission_per_million
                cash -= fee
                entry_fee = fee
                pos = -1
                entry_i = i + 1
                cur_trade_dd = 0.0

    return rows


def main() -> None:
    ap = build_parser("Candle Rider candle-color strategy sweep", "forex_candle_rider_results.csv")
    ap.set_defaults(timeframes=None)
    for action in ap._actions:
        if action.dest == "timeframes":
            action.help = "comma list of candle sizes, supports seconds/minutes/hours like 10s,30s,1m,5m,1h"
    ap.add_argument("--modes", default=None, help="normal,invert")
    ap.add_argument("--sessions", default=None, help="-1,0,1,2")
    ap.add_argument("--reverse-on-sl", default=None, help="0,1")
    ap.add_argument("--confirm", default=None, help="0=act on first closed candle, 1=need two same-color closes")
    ap.add_argument("--trades-out", default=None, help="write per-trade CSV; intended for exact single/few combo inspection")
    ap.add_argument("--workers", type=int, default=max(1, min(15, (os.cpu_count() or 4) - 1)))
    args = ap.parse_args()

    ticks, t0 = load_market(args)
    timeframes = parse_str_list(args.timeframes, DEFAULT_TIMEFRAMES)
    modes = parse_str_list(args.modes, DEFAULT_MODES)
    sessions = [int(x) for x in parse_num_list(args.sessions, DEFAULT_SESSIONS)]
    reverse_values = [int(x) for x in parse_num_list(args.reverse_on_sl, DEFAULT_REVERSE_ON_SL)]
    confirm_values = [int(x) for x in parse_num_list(args.confirm, DEFAULT_CONFIRM)]

    all_results: list[TradeResult] = []
    for pair in args.pairs:
        df = ticks[ticks["pair"].astype(str).str.upper() == pair.upper()].copy()
        if df.empty:
            print(f"[candle-rider] {pair} no ticks", flush=True)
            continue
        bid = df["bid"].to_numpy(np.float64)
        ask = df["ask"].to_numpy(np.float64)
        ts_ns = df["timestamp"].astype("int64").to_numpy(np.int64)
        point_size = float(args.point_size or default_point_size(pair))
        default_tp, default_sl = default_tp_sl_for_pair(pair)
        tp_values = parse_num_list(args.tp_points, default_tp)
        sl_values = parse_num_list(args.sl_points, default_sl)
        day_id, max_days = day_ids_from_timestamps(ts_ns)
        session_cache = {s: active_session_allowed(ts_ns, s) for s in sessions}
        print(
            f"[candle-rider] {pair} ticks={len(bid):,} point={point_size:g} "
            f"tf={','.join(timeframes)} combos_pending",
            flush=True,
        )

        jobs = []
        event_cache: dict[tuple[str, str, int], np.ndarray] = {}
        for tf in timeframes:
            opens, _, _, closes, close_idx = build_bid_ohlc(bid, ts_ns, tf)
            for mode in modes:
                for confirm in confirm_values:
                    event_cache[(tf, mode, confirm)] = candle_event_state(opens, closes, close_idx, len(bid), mode, confirm)
            print(f"[candle-rider] {pair} tf={tf} bars={len(closes):,}", flush=True)

        for tf, mode, session, tp, sl, rev, confirm in product(timeframes, modes, sessions, tp_values, sl_values, reverse_values, confirm_values):
            params = f"mode={mode};session={session};reverse_sl={rev};confirm={confirm}"
            jobs.append((tf, params, event_cache[(tf, mode, confirm)], session_cache[session], tp, sl, rev))
        print(f"[candle-rider] {pair} combos={len(jobs):,} workers={args.workers}", flush=True)

        def run_job(job):
            tf, params, event, allowed, tp, sl, rev = job
            return simulate_candle_rider(
                pair, tf, params, bid, ask, event, allowed, day_id, max_days,
                tp, sl, point_size, args.amount, args.compound, args.leverage,
                args.commission_per_million, args.side, bool(rev),
            )

        trace_rows: list[dict[str, object]] = []
        if args.trades_out:
            for done, job in enumerate(jobs, start=1):
                tf, params, event, allowed, tp, sl, rev = job
                all_results.append(run_job(job))
                trace_rows.extend(trace_candle_rider_trades(
                    pair, tf, params, bid, ask, ts_ns, event, allowed,
                    tp, sl, point_size, args.amount, args.compound, args.leverage,
                    args.commission_per_million, args.side, bool(rev),
                ))
                print(f"[candle-rider] {pair} trace progress {done:,}/{len(jobs):,}", flush=True)
        else:
            done = 0
            step = max(1, len(jobs) // 10)
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
                futs = [ex.submit(run_job, job) for job in jobs]
                for fut in as_completed(futs):
                    all_results.append(fut.result())
                    done += 1
                    if done % step == 0 or done == len(jobs):
                        print(f"[candle-rider] {pair} progress {done:,}/{len(jobs):,}", flush=True)

        if args.trades_out and trace_rows:
            os.makedirs(os.path.dirname(args.trades_out) or ".", exist_ok=True)
            fields = list(trace_rows[0].keys())
            with open(args.trades_out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(trace_rows)

            daily: dict[str, dict[str, float]] = {}
            for row in trace_rows:
                day = str(row["exit_time"])[:10]
                bucket = daily.setdefault(day, {"trades": 0.0, "pnl": 0.0, "wins": 0.0, "losses": 0.0, "worst": 0.0})
                pnl = float(row["pnl"])
                bucket["trades"] += 1.0
                bucket["pnl"] += pnl
                if pnl >= 0:
                    bucket["wins"] += 1.0
                else:
                    bucket["losses"] += 1.0
                    bucket["worst"] = min(bucket["worst"], pnl)
            daily_out = os.path.splitext(args.trades_out)[0] + "_daily.csv"
            with open(daily_out, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["day", "trades", "pnl", "wins", "losses", "worst_trade"])
                for day in sorted(daily):
                    b = daily[day]
                    w.writerow([day, int(b["trades"]), b["pnl"], int(b["wins"]), int(b["losses"]), b["worst"]])
            print(f"[candle-rider] wrote trades {os.path.abspath(args.trades_out)}", flush=True)
            print(f"[candle-rider] wrote daily {os.path.abspath(daily_out)}", flush=True)

    filtered = [r for r in all_results if r.trades >= args.min_trades]
    write_unified_results(args.out, filtered, args.top)
    print(f"[candle-rider] wrote {os.path.abspath(args.out)} elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
