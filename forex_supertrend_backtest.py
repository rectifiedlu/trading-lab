"""Supertrend RR sweep with candle signals and tick TP/SL execution."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

import numpy as np

from forex_signal_sweep_common import build_bid_ohlc, map_state_to_ticks, rma, simulate_state_strategy
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
    write_results,
)


DEFAULT_TIMEFRAMES = ["30s", "1m", "3m",]
DEFAULT_LENGTHS = [7, 10, 14, 21]
DEFAULT_MULTS = [1.5, 2.0, 2.5, 3.0, 3.5]
DEFAULT_MODES = ["normal", "invert"]
DEFAULT_RR = [1.5, 2.0, 3.0]
DEFAULT_FLIP_EXIT = [0, 1]
DEFAULT_MIN_RISK_POINTS = [100]
DEFAULT_FIXED_TP = [200, 300, 400, 500]
DEFAULT_FIXED_SL = [200, 300, 400]
DEFAULT_SESSIONS = [-1, 1]


def map_values_to_ticks(n_ticks: int, close_idx: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.full(n_ticks, np.nan, dtype=np.float64)
    prev = 0
    last = np.nan
    for idx, value in zip(close_idx, values):
        out[prev:idx + 1] = last
        last = float(value)
        prev = idx + 1
    out[prev:] = last
    return out


def supertrend_state(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    length: int,
    mult: float,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prev_close = np.empty_like(closes)
    prev_close[0] = closes[0]
    prev_close[1:] = closes[:-1]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    atr = rma(tr, length)

    hl2 = (highs + lows) / 2.0
    basic_upper = hl2 + mult * atr
    basic_lower = hl2 - mult * atr
    final_upper = np.full(len(closes), np.nan, dtype=np.float64)
    final_lower = np.full(len(closes), np.nan, dtype=np.float64)
    trend = np.zeros(len(closes), dtype=np.float64)
    line = np.full(len(closes), np.nan, dtype=np.float64)

    for i in range(len(closes)):
        if not np.isfinite(atr[i]):
            continue
        if i == 0 or not np.isfinite(final_upper[i - 1]):
            final_upper[i] = basic_upper[i]
            final_lower[i] = basic_lower[i]
            trend[i] = 1.0 if closes[i] >= hl2[i] else -1.0
        else:
            final_upper[i] = (
                basic_upper[i]
                if basic_upper[i] < final_upper[i - 1] or closes[i - 1] > final_upper[i - 1]
                else final_upper[i - 1]
            )
            final_lower[i] = (
                basic_lower[i]
                if basic_lower[i] > final_lower[i - 1] or closes[i - 1] < final_lower[i - 1]
                else final_lower[i - 1]
            )
            if trend[i - 1] <= 0 and closes[i] > final_upper[i - 1]:
                trend[i] = 1.0
            elif trend[i - 1] >= 0 and closes[i] < final_lower[i - 1]:
                trend[i] = -1.0
            else:
                trend[i] = trend[i - 1] if trend[i - 1] != 0 else 1.0
        line[i] = final_lower[i] if trend[i] > 0 else final_upper[i]

    state = trend.copy()
    if mode == "invert":
        state = -state
    state[~np.isfinite(line)] = 0.0
    return state, line, atr


if njit is not None:
    @njit(cache=True)
    def _simulate_supertrend_rr_numba(
        bid,
        ask,
        state,
        line,
        atr,
        entry_allowed,
        day_id,
        max_days,
        rr,
        min_risk_points,
        point_size,
        amount,
        compound,
        leverage,
        commission_per_million,
        side_mode,
        flip_exit,
    ):
        cash = amount
        realised = 0.0
        peak_equity = amount
        max_dd = 0.0
        pos = 0
        entry = 0.0
        units = 0.0
        stop_px = 0.0
        take_px = 0.0
        entry_fee = 0.0
        trades = wins = losses = long_trades = short_trades = stop_losses = signal_exits = 0
        gross_profit = 0.0
        gross_loss = 0.0
        open_worst = 0.0
        max_trade_dd = 0.0
        total_trade_dd = 0.0
        loss_values = np.empty(len(bid), dtype=np.float64)
        loss_count = 0
        daily = np.zeros(max_days, dtype=np.float64)
        prev_state = state[0] if len(state) else 0.0

        for i in range(1, len(bid)):
            px_mid = 0.5 * (bid[i] + ask[i])
            unreal = 0.0
            if pos == 1:
                unreal = (bid[i] - entry) * units
            elif pos == -1:
                unreal = (entry - ask[i]) * units
            equity = cash + unreal
            if equity > peak_equity:
                peak_equity = equity
            dd = peak_equity - equity
            if dd > max_dd:
                max_dd = dd
            if pos != 0 and unreal < open_worst:
                open_worst = unreal

            cur_state = state[i]
            changed = cur_state != prev_state

            if pos != 0:
                exit_reason = 0
                exit_px = 0.0
                if pos == 1:
                    if bid[i] >= take_px:
                        exit_reason = 1
                        exit_px = bid[i]
                    elif bid[i] <= stop_px:
                        exit_reason = 2
                        exit_px = bid[i]
                    elif flip_exit and changed and cur_state < 0:
                        exit_reason = 3
                        exit_px = bid[i]
                else:
                    if ask[i] <= take_px:
                        exit_reason = 1
                        exit_px = ask[i]
                    elif ask[i] >= stop_px:
                        exit_reason = 2
                        exit_px = ask[i]
                    elif flip_exit and changed and cur_state > 0:
                        exit_reason = 3
                        exit_px = ask[i]

                if exit_reason:
                    exit_fee = abs(exit_px * units) / 1_000_000.0 * commission_per_million
                    pnl = ((exit_px - entry) * units if pos == 1 else (entry - exit_px) * units) - entry_fee - exit_fee
                    cash += pnl
                    realised += pnl
                    daily[day_id[i]] += pnl
                    trades += 1
                    if pnl >= 0.0:
                        wins += 1
                        gross_profit += pnl
                    else:
                        losses += 1
                        gross_loss += -pnl
                        loss_values[loss_count] = pnl
                        loss_count += 1
                    if exit_reason == 2:
                        stop_losses += 1
                    elif exit_reason == 3:
                        signal_exits += 1
                    if -open_worst > max_trade_dd:
                        max_trade_dd = -open_worst
                    total_trade_dd += -open_worst
                    pos = 0
                    entry = 0.0
                    units = 0.0
                    stop_px = 0.0
                    take_px = 0.0
                    entry_fee = 0.0
                    open_worst = 0.0

            if pos == 0 and entry_allowed[i] and changed and cur_state != 0.0 and np.isfinite(line[i]) and np.isfinite(atr[i]):
                want_long = cur_state > 0.0
                if (want_long and side_mode == 2) or ((not want_long) and side_mode == 1):
                    prev_state = cur_state
                    continue
                margin = cash if compound else amount
                if margin <= 0.0:
                    prev_state = cur_state
                    continue
                if want_long:
                    entry = ask[i]
                    pos = 1
                    long_trades += 1
                else:
                    entry = bid[i]
                    pos = -1
                    short_trades += 1
                risk = abs(entry - line[i])
                atr_risk = abs(atr[i])
                if atr_risk > risk:
                    risk = atr_risk
                min_risk = min_risk_points * point_size
                if risk < min_risk:
                    risk = min_risk
                if pos == 1:
                    stop_px = entry - risk
                    take_px = entry + risk * rr
                else:
                    stop_px = entry + risk
                    take_px = entry - risk * rr
                units = (margin * leverage) / entry
                entry_fee = abs(entry * units) / 1_000_000.0 * commission_per_million
                cash -= entry_fee
                realised -= entry_fee
                daily[day_id[i]] -= entry_fee
                open_worst = 0.0

            prev_state = cur_state

        open_unreal = 0.0
        open_bps = 0.0
        if pos == 1:
            open_unreal = (bid[-1] - entry) * units
            open_bps = (bid[-1] / entry - 1.0) * 10000.0
        elif pos == -1:
            open_unreal = (entry - ask[-1]) * units
            open_bps = (entry / ask[-1] - 1.0) * 10000.0
        total = realised + open_unreal
        win_rate = wins / trades * 100.0 if trades else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0.0 else (999.0 if gross_profit > 0.0 else 0.0)
        avg_trade_dd = total_trade_dd / trades if trades else 0.0
        avg_day = 0.0
        if max_days > 0:
            for j in range(max_days):
                avg_day += daily[j]
            avg_day /= max_days
        daily_sorted = np.sort(daily.copy())
        if max_days <= 0:
            median_day = 0.0
        elif max_days % 2 == 1:
            median_day = daily_sorted[max_days // 2]
        else:
            median_day = 0.5 * (daily_sorted[max_days // 2 - 1] + daily_sorted[max_days // 2])
        median_loss = 0.0
        if loss_count:
            loss_sorted = np.sort(loss_values[:loss_count].copy())
            if loss_count % 2 == 1:
                median_loss = loss_sorted[loss_count // 2]
            else:
                median_loss = 0.5 * (loss_sorted[loss_count // 2 - 1] + loss_sorted[loss_count // 2])
        return (
            realised, open_unreal, total, trades, wins, losses, win_rate, profit_factor,
            max_dd, long_trades, short_trades, stop_losses, signal_exits, pos, open_bps,
            max_trade_dd, avg_trade_dd, avg_day, median_day, median_loss,
        )


def simulate_supertrend_rr(
    pair: str,
    strategy: str,
    params: str,
    timeframe: str,
    bid: np.ndarray,
    ask: np.ndarray,
    state: np.ndarray,
    line: np.ndarray,
    atr: np.ndarray,
    ts_ns: np.ndarray,
    entry_allowed: np.ndarray,
    rr: float,
    min_risk_points: float,
    point_size: float,
    amount: float,
    compound: bool,
    leverage: float,
    commission_per_million: float,
    side: str,
    flip_exit: int,
) -> TradeResult:
    day_id, max_days = day_ids_from_timestamps(ts_ns)
    side_mode = 3 if side == "both" else (1 if side == "long" else 2)
    if njit is None:
        raise RuntimeError("numba is required for forex_supertrend_backtest.py")
    out = _simulate_supertrend_rr_numba(
        bid.astype(np.float64, copy=False),
        ask.astype(np.float64, copy=False),
        state.astype(np.float64, copy=False),
        line.astype(np.float64, copy=False),
        atr.astype(np.float64, copy=False),
        entry_allowed.astype(np.bool_, copy=False),
        day_id.astype(np.int64, copy=False),
        int(max_days),
        float(rr),
        float(min_risk_points),
        float(point_size),
        float(amount),
        bool(compound),
        float(leverage),
        float(commission_per_million),
        int(side_mode),
        int(flip_exit),
    )
    (
        realised, open_unreal, total, trades, wins, losses, win_rate, profit_factor,
        max_dd, long_trades, short_trades, stop_losses, signal_exits, open_pos, open_bps,
        trade_max_dd, trade_avg_dd, avg_day, median_day, median_loss,
    ) = out
    result = TradeResult(
        pair, strategy, params, timeframe, rr, min_risk_points, point_size,
        realised, open_unreal, total, int(trades), int(wins), int(losses),
        win_rate, profit_factor, max_dd, int(long_trades), int(short_trades),
        int(stop_losses), int(signal_exits), 0, False,
        "L" if open_pos == 1 else ("S" if open_pos == -1 else "-"),
        open_bps,
    )
    result.trade_max_drawdown = float(trade_max_dd)
    result.trade_avg_drawdown = float(trade_avg_dd)
    result.avg_day = float(avg_day)
    result.median_day = float(median_day)
    result.median_loss = float(median_loss)
    return result


def main() -> None:
    ap = build_parser("Supertrend dynamic RR sweep", "forex_supertrend_results.csv")
    ap.add_argument("--lengths", default=None)
    ap.add_argument("--mults", default=None)
    ap.add_argument("--modes", default=None, help="normal,invert")
    ap.add_argument("--rr", default=None, help="reward/risk multiples")
    ap.add_argument("--min-risk-points", default=None)
    ap.add_argument("--flip-exit", default=None, help="0=only TP/SL, 1=close on Supertrend flip too")
    ap.add_argument("--exit-style", choices=["rr", "fixed"], default="rr")
    ap.add_argument("--sessions", default=None, help="-1=outside sessions, 0=all hours, 1=inside Tokyo/London/New York sessions")
    ap.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    args = ap.parse_args()

    timeframes = parse_str_list(args.timeframes, DEFAULT_TIMEFRAMES)
    lengths = [int(x) for x in parse_num_list(args.lengths, DEFAULT_LENGTHS)]
    mults = parse_num_list(args.mults, DEFAULT_MULTS)
    modes = parse_str_list(args.modes, DEFAULT_MODES)
    rrs = parse_num_list(args.rr, DEFAULT_RR)
    min_risks = parse_num_list(args.min_risk_points, DEFAULT_MIN_RISK_POINTS)
    flip_exits = [int(x) for x in parse_num_list(args.flip_exit, DEFAULT_FLIP_EXIT)]
    fixed_tps = parse_num_list(args.tp_points, DEFAULT_FIXED_TP)
    fixed_sls = parse_num_list(args.sl_points, DEFAULT_FIXED_SL)
    sessions = [int(x) for x in parse_num_list(args.sessions, DEFAULT_SESSIONS)]

    ticks, t0 = load_market(args)
    results: list[TradeResult] = []
    if args.exit_style == "fixed":
        combo_count = len(timeframes) * len(lengths) * len(mults) * len(modes) * len(fixed_tps) * len(fixed_sls) * len(sessions)
    else:
        combo_count = len(timeframes) * len(lengths) * len(mults) * len(modes) * len(rrs) * len(min_risks) * len(flip_exits) * len(sessions)
    print(f"[supertrend] combos_per_pair={combo_count} workers={args.workers}", flush=True)

    for pair, g in ticks.groupby("pair", sort=False):
        g = g.sort_values("timestamp")
        bid = g["bid"].to_numpy(np.float64)
        ask = g["ask"].to_numpy(np.float64)
        ts_ns = g["timestamp"].astype("int64").to_numpy(np.int64)
        point_size = float(args.point_size or default_point_size(pair))
        print(f"[supertrend] {pair} ticks={len(g):,}", flush=True)
        session_cache = {s: active_session_allowed(ts_ns, s) for s in sessions}

        cache = {}
        for tf in timeframes:
            _, highs, lows, closes, close_idx = build_bid_ohlc(bid, ts_ns, tf)
            for length in lengths:
                for mult in mults:
                    for mode in modes:
                        state_c, line_c, atr_c = supertrend_state(highs, lows, closes, length, mult, mode)
                        cache[(tf, length, float(mult), mode)] = (
                            map_state_to_ticks(len(bid), close_idx, state_c),
                            map_values_to_ticks(len(bid), close_idx, line_c),
                            map_values_to_ticks(len(bid), close_idx, atr_c),
                        )

        if args.exit_style == "fixed":
            combos = list(product(timeframes, lengths, mults, modes, fixed_tps, fixed_sls, sessions))
        else:
            combos = list(product(timeframes, lengths, mults, modes, rrs, min_risks, flip_exits, sessions))

        def run_combo(combo):
            if args.exit_style == "fixed":
                tf, length, mult, mode, tp, sl, sess = combo
                state, _, _ = cache[(tf, length, float(mult), mode)]
                params = f"length={length};mult={mult:g};mode={mode};exit=fixed;session={sess}"
                return simulate_state_strategy(
                    pair, "supertrend", params, tf, bid, ask, ts_ns, state, state,
                    session_cache[int(sess)], tp, 0.0, sl, 0.0, point_size,
                    args.amount, args.compound, args.leverage, args.commission_per_million, args.side,
                )
            tf, length, mult, mode, rr, min_risk, flip, sess = combo
            state, line, atr = cache[(tf, length, float(mult), mode)]
            params = f"length={length};mult={mult:g};mode={mode};rr={rr:g};minrisk={min_risk:g};flip={flip};session={sess}"
            return simulate_supertrend_rr(
                pair, "supertrend", params, tf, bid, ask, state, line, atr, ts_ns, session_cache[int(sess)], rr, min_risk,
                point_size, args.amount, args.compound, args.leverage,
                args.commission_per_million, args.side, flip,
            )

        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_combo, c) for c in combos]
            for fut in as_completed(futs):
                results.append(fut.result())
                done += 1
                if done % max(len(combos) // 10, 1) == 0:
                    print(f"[supertrend] {pair} progress {done}/{len(combos)}", flush=True)

    filtered = [r for r in results if r.trades >= args.min_trades]
    write_results(args.out, filtered, args.top, args.sort_by)
    print(f"[supertrend] wrote {args.out} elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
