"""Parabolic SAR stop-order backtest with candle SAR and tick execution.

This mirrors the TradingView sample structure:
    - Calculate SAR/nextBarSAR on confirmed candles.
    - If current SAR state is uptrend, place a short stop at nextBarSAR.
    - If current SAR state is downtrend, place a long stop at nextBarSAR.
    - Stop orders are checked on ticks during the next candle.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

import numpy as np

from forex_strategy_common import (
    TradeResult,
    build_parser,
    candle_state_to_ticks,
    commission,
    day_ids_from_timestamps,
    default_point_size,
    live_candles,
    load_market,
    open_unrealized,
    parse_num_list,
    parse_str_list,
    timeframe_to_ns,
    units_for_margin,
    write_results,
    simulate_triggers,
)

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


def closed_ohlc(mid: np.ndarray, ts_ns: np.ndarray, timeframe: str) -> tuple[np.ndarray, ...]:
    """Return closed candle OHLC and the tick index where each candle closed."""
    o, h, l, c, closed = live_candles(mid, ts_ns, timeframe)
    close_idx = np.flatnonzero(closed)
    if len(close_idx) == 0:
        empty_f = np.array([], dtype=np.float64)
        empty_i = np.array([], dtype=np.int64)
        return empty_f, empty_f, empty_f, empty_f, empty_i
    return o[close_idx], h[close_idx], l[close_idx], c[close_idx], close_idx


def parabolic_sar(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    start: float,
    increment: float,
    maximum: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    uptrend = np.zeros(n, dtype=np.bool_)
    sar = np.full(n, np.nan, dtype=np.float64)
    next_sar = np.full(n, np.nan, dtype=np.float64)

    if n < 2:
        return uptrend, sar, next_sar

    af = start
    ep = np.nan
    ns = np.nan
    is_up = False

    for i in range(1, n):
        first_trend_bar = False
        cur_sar = ns

        if i == 1:
            low_prev = low[i - 1]
            high_prev = high[i - 1]
            if close[i] > close[i - 1]:
                is_up = True
                ep = high[i]
                prev_sar = low_prev
                prev_ep = high[i]
            else:
                is_up = False
                ep = low[i]
                prev_sar = high_prev
                prev_ep = low[i]
            first_trend_bar = True
            cur_sar = prev_sar + start * (prev_ep - prev_sar)

        if is_up:
            if cur_sar > low[i]:
                first_trend_bar = True
                is_up = False
                cur_sar = max(ep, high[i])
                ep = low[i]
                af = start
        else:
            if cur_sar < high[i]:
                first_trend_bar = True
                is_up = True
                cur_sar = min(ep, low[i])
                ep = high[i]
                af = start

        if not first_trend_bar:
            if is_up:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + increment, maximum)
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + increment, maximum)

        if is_up:
            cur_sar = min(cur_sar, low[i - 1])
            if i > 1:
                cur_sar = min(cur_sar, low[i - 2])
        else:
            cur_sar = max(cur_sar, high[i - 1])
            if i > 1:
                cur_sar = max(cur_sar, high[i - 2])

        ns = cur_sar + af * (ep - cur_sar)
        uptrend[i] = is_up
        sar[i] = cur_sar
        next_sar[i] = ns

    return uptrend, sar, next_sar


def candle_only_signals(
    n_ticks: int,
    close_idx: np.ndarray,
    signal: np.ndarray,
) -> np.ndarray:
    out = np.zeros(n_ticks, dtype=np.bool_)
    if len(close_idx) == 0:
        return out
    n = min(len(close_idx), len(signal))
    for i in range(n):
        idx = int(close_idx[i])
        if 0 <= idx < n_ticks and bool(signal[i]):
            out[idx] = True
    return out


if njit is not None:
    @njit(cache=True)
    def _simulate_sar_stops_numba(
        bid, ask, close_idx, uptrend, next_sar, amount, compound,
        leverage, commission_per_million, side_mode,
    ):
        allow_long = side_mode == 1 or side_mode == 3
        allow_short = side_mode == 2 or side_mode == 3
        cash = amount
        equity_peak = amount
        max_dd = 0.0
        gross_win = 0.0
        gross_loss = 0.0
        trades = wins = losses = long_trades = short_trades = 0
        signal_exits = liquidations = 0
        account_dead = False
        pos = 0
        entry = 0.0
        units = 0.0
        for bar_i in range(1, len(close_idx) - 1):
            level = next_sar[bar_i]
            if not np.isfinite(level):
                continue
            start_i = int(close_idx[bar_i]) + 1
            end_i = int(close_idx[bar_i + 1]) + 1
            if start_i >= len(bid):
                break
            if end_i > len(bid):
                end_i = len(bid)
            wants_short = uptrend[bar_i]
            wants_long = not wants_short
            for i in range(start_i, end_i):
                b = bid[i]
                a = ask[i]
                if pos != 0:
                    live_u = (b - entry) * units if pos == 1 else (entry - a) * units
                    equity = cash + live_u
                    if equity > equity_peak:
                        equity_peak = equity
                    dd = equity_peak - equity
                    if dd > max_dd:
                        max_dd = dd
                    if equity <= 0:
                        liquidations += 1
                        account_dead = True
                        cash = 0.0
                        pos = 0
                        break
                hit_long = wants_long and allow_long and a >= level
                hit_short = wants_short and allow_short and b <= level
                if hit_long:
                    if pos == -1:
                        exit_px = a
                        pnl = (entry - exit_px) * units
                        pnl -= abs(exit_px * units) / 1_000_000.0 * commission_per_million
                        cash += pnl
                        trades += 1; short_trades += 1; signal_exits += 1
                        if pnl >= 0:
                            wins += 1; gross_win += pnl
                        else:
                            losses += 1; gross_loss += -pnl
                        pos = 0; entry = 0.0; units = 0.0
                    if pos == 0:
                        margin = cash if compound else amount
                        if margin > 0:
                            entry = a
                            units = (margin * leverage) / entry
                            cash -= abs(entry * units) / 1_000_000.0 * commission_per_million
                            pos = 1
                    break
                if hit_short:
                    if pos == 1:
                        exit_px = b
                        pnl = (exit_px - entry) * units
                        pnl -= abs(exit_px * units) / 1_000_000.0 * commission_per_million
                        cash += pnl
                        trades += 1; long_trades += 1; signal_exits += 1
                        if pnl >= 0:
                            wins += 1; gross_win += pnl
                        else:
                            losses += 1; gross_loss += -pnl
                        pos = 0; entry = 0.0; units = 0.0
                    if pos == 0:
                        margin = cash if compound else amount
                        if margin > 0:
                            entry = b
                            units = (margin * leverage) / entry
                            cash -= abs(entry * units) / 1_000_000.0 * commission_per_million
                            pos = -1
                    break
            if account_dead:
                break
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
        pf = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
        return (
            realised, open_u, total, trades, wins, losses, pf, max_dd,
            long_trades, short_trades, signal_exits, liquidations,
            account_dead, open_side_code, open_bps,
        )


def simulate_sar_stops(
    pair: str,
    bid: np.ndarray,
    ask: np.ndarray,
    close_idx: np.ndarray,
    uptrend: np.ndarray,
    next_sar: np.ndarray,
    params: str,
    timeframe: str,
    amount: float,
    compound: bool,
    leverage: float,
    commission_per_million: float,
    side: str,
) -> TradeResult:
    point_size = default_point_size(pair)
    if njit is not None:
        side_num = 3 if side == "both" else (1 if side == "long" else 2)
        out = _simulate_sar_stops_numba(
            bid.astype(np.float64, copy=False),
            ask.astype(np.float64, copy=False),
            close_idx.astype(np.int64, copy=False),
            uptrend.astype(np.bool_, copy=False),
            next_sar.astype(np.float64, copy=False),
            amount,
            bool(compound),
            leverage,
            commission_per_million,
            side_num,
        )
        (
            realised, open_u, total, trades, wins, losses, pf, max_dd,
            long_trades, short_trades, signal_exits, liquidations,
            account_dead, open_side_code, open_bps,
        ) = out
        open_side_name = "long" if open_side_code == 1 else ("short" if open_side_code == -1 else "-")
        win_rate = wins / trades * 100.0 if trades else 0.0
        return TradeResult(
            pair, "psar", params, timeframe, 0.0, 0.0, point_size,
            realised, open_u, total, int(trades), int(wins), int(losses), win_rate, pf,
            max_dd, int(long_trades), int(short_trades), 0, int(signal_exits),
            int(liquidations), bool(account_dead), open_side_name, open_bps,
        )

    allow_long = side in ("long", "both")
    allow_short = side in ("short", "both")

    start_balance = amount
    cash = start_balance
    equity_peak = start_balance
    max_dd = 0.0
    gross_win = gross_loss = 0.0
    trades = wins = losses = long_trades = short_trades = 0
    signal_exits = liquidations = 0
    account_dead = False

    pos = 0
    entry = 0.0
    units = 0.0

    def close_current(exit_px: float) -> None:
        nonlocal cash, trades, wins, losses, gross_win, gross_loss
        nonlocal long_trades, short_trades, signal_exits, pos, entry, units
        if pos == 1:
            pnl = (exit_px - entry) * units - commission(exit_px, units, commission_per_million)
            long_trades += 1
        else:
            pnl = (entry - exit_px) * units - commission(exit_px, units, commission_per_million)
            short_trades += 1
        cash += pnl
        trades += 1
        signal_exits += 1
        if pnl >= 0:
            wins += 1
            gross_win += pnl
        else:
            losses += 1
            gross_loss += -pnl
        pos = 0
        entry = units = 0.0

    def open_side(new_pos: int, px: float) -> None:
        nonlocal cash, pos, entry, units
        margin = cash if compound else amount
        if margin <= 0:
            return
        new_units = units_for_margin(margin, leverage, px)
        cash -= commission(px, new_units, commission_per_million)
        pos = new_pos
        entry = px
        units = new_units

    for bar_i in range(1, len(close_idx) - 1):
        level = float(next_sar[bar_i])
        if not np.isfinite(level):
            continue

        start_i = int(close_idx[bar_i]) + 1
        end_i = int(close_idx[bar_i + 1]) + 1
        if start_i >= len(bid):
            break
        end_i = min(end_i, len(bid))

        wants_short = bool(uptrend[bar_i])
        wants_long = not wants_short

        for i in range(start_i, end_i):
            b = float(bid[i])
            a = float(ask[i])
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

            hit_long = wants_long and allow_long and a >= level
            hit_short = wants_short and allow_short and b <= level
            if hit_long:
                if pos == -1:
                    close_current(a)
                if pos == 0:
                    open_side(1, a)
                break
            if hit_short:
                if pos == 1:
                    close_current(b)
                if pos == 0:
                    open_side(-1, b)
                break
        if account_dead:
            break

    open_u = 0.0
    open_side_name = "-"
    open_bps = 0.0
    if pos == 1:
        open_side_name = "long"
        open_u = open_unrealized(pos, entry, units, float(bid[-1]), float(ask[-1]))
        open_bps = (float(bid[-1]) / entry - 1.0) * 10000.0
    elif pos == -1:
        open_side_name = "short"
        open_u = open_unrealized(pos, entry, units, float(bid[-1]), float(ask[-1]))
        open_bps = (entry / float(ask[-1]) - 1.0) * 10000.0

    total = cash + open_u - start_balance
    realised = cash - start_balance
    win_rate = wins / trades * 100.0 if trades else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
    return TradeResult(
        pair, "psar", params, timeframe, 0.0, 0.0, point_size,
        realised, open_u, total, trades, wins, losses, win_rate, pf,
        max_dd, long_trades, short_trades, 0, signal_exits, liquidations,
        account_dead, open_side_name, open_bps,
    )


def main() -> None:
    ap = build_parser("Parabolic SAR candle/tick stop-order backtest", "forex_parabolic_sar_tick_results.csv")
    ap.set_defaults(timeframes="1s,10s,15s,30s,1m")
    ap.add_argument(
        "--mode",
        default="stop_reverse,trend_follow,trend_contrarian,sar_slope",
        help=(
            "comma list. stop_reverse=TradingView stop-order sample; "
            "trend_follow=long on PSAR uptrend, short on downtrend; "
            "trend_contrarian=exact opposite of trend_follow timing; "
            "sar_slope=long when nextSAR>SAR, short when nextSAR<SAR"
        ),
    )
    ap.add_argument("--start", dest="psar_start", default="0.01,0.02,0.03,0.04")
    ap.add_argument("--increment", default="0.01,0.02,0.03,0.04")
    ap.add_argument("--maximum", default="0.1,0.2,0.3")
    ap.add_argument("--execution-mode", default="tick,candle",
                    help="comma list: tick=execute on ticks after candle signal; candle=execute only at closed candle tick")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    timeframes = parse_str_list(args.timeframes, ["1m"])
    starts = parse_num_list(args.psar_start, [0.02])
    increments = parse_num_list(args.increment, [0.02])
    maximums = parse_num_list(args.maximum, [0.2])
    modes = parse_str_list(args.mode, ["stop_reverse"])
    execution_modes = parse_str_list(args.execution_mode, ["tick"])
    bad_exec = [m for m in execution_modes if m not in ("tick", "candle")]
    if bad_exec:
        raise SystemExit(f"unsupported --execution-mode values: {bad_exec}")
    valid_modes = {"stop_reverse", "trend_follow", "trend_contrarian", "sar_slope"}
    bad_modes = [m for m in modes if m not in valid_modes]
    if bad_modes:
        raise SystemExit(f"unsupported --mode values: {bad_modes}")

    ticks, _ = load_market(args)
    results = []
    for pair, g in ticks.groupby("pair", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        bid = g["bid"].to_numpy(np.float64)
        ask = g["ask"].to_numpy(np.float64)
        # MT5 native XAUUSD candles match bid OHLC, not mid/ask.
        mid = bid
        ts_ns = g["timestamp"].astype("int64").to_numpy()
        day_id, max_days = day_ids_from_timestamps(ts_ns)
        point_size = args.point_size or default_point_size(pair)
        print(f"[psar] {pair} ticks={len(g):,}", flush=True)

        combos = list(product(modes, execution_modes, timeframes, starts, increments, maximums))

        def run_combo(combo):
            mode, execution_mode, tf, start, inc, maximum = combo
            timeframe_to_ns(tf)
            _, high, low, close, close_idx = closed_ohlc(mid, ts_ns, tf)
            if len(close) < 5:
                return None
            uptrend, sar, next_sar = parabolic_sar(high, low, close, start, inc, maximum)
            params = f"mode={mode};exec={execution_mode};start={start:g};inc={inc:g};max={maximum:g}"
            if mode in ("trend_follow", "trend_contrarian", "sar_slope"):
                if mode == "sar_slope":
                    state = np.where(next_sar > sar, 1.0, np.where(next_sar < sar, -1.0, 0.0))
                else:
                    state = np.where(uptrend, 1.0, -1.0)
                if mode == "trend_contrarian":
                    state = -state
                prev = np.roll(state, 1)
                prev[0] = 0.0
                candle_long = (state == 1.0) & (prev != 1.0)
                candle_short = (state == -1.0) & (prev != -1.0)
                candle_long_exit = state == -1.0
                candle_short_exit = state == 1.0
                if execution_mode == "candle":
                    long_trigger = candle_only_signals(len(bid), close_idx, candle_long)
                    short_trigger = candle_only_signals(len(bid), close_idx, candle_short)
                    long_exit = candle_only_signals(len(bid), close_idx, candle_long_exit)
                    short_exit = candle_only_signals(len(bid), close_idx, candle_short_exit)
                else:
                    long_trigger = candle_state_to_ticks(len(bid), close_idx, candle_long.astype(float)) == 1
                    short_trigger = candle_state_to_ticks(len(bid), close_idx, candle_short.astype(float)) == 1
                    long_exit = candle_state_to_ticks(len(bid), close_idx, candle_long_exit.astype(float)) == 1
                    short_exit = candle_state_to_ticks(len(bid), close_idx, candle_short_exit.astype(float)) == 1
                return simulate_triggers(
                    pair, "psar", params, tf, bid, ask, long_trigger, short_trigger,
                    long_exit, short_exit, 0.0, 0.0, point_size, args.amount,
                    args.compound, args.leverage, args.commission_per_million,
                    args.side, reverse_on_flip=True,
                    day_id=day_id, max_days=max_days,
                )
            if execution_mode == "candle":
                return None
            return simulate_sar_stops(
                pair, bid, ask, close_idx, uptrend, next_sar, params, tf,
                args.amount, args.compound, args.leverage,
                args.commission_per_million, args.side,
            )

        if args.workers > 1 and len(combos) > 1:
            print(f"[psar] workers={args.workers} combos={len(combos)}", flush=True)
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(run_combo, c) for c in combos]
                for i, fut in enumerate(as_completed(futs), 1):
                    res = fut.result()
                    if res is not None:
                        results.append(res)
                    if i % max(1, len(combos) // 10) == 0:
                        print(f"[psar] progress {i}/{len(combos)}", flush=True)
        else:
            for combo in combos:
                res = run_combo(combo)
                if res is not None:
                    results.append(res)

    write_results(args.out, [r for r in results if r.trades >= args.min_trades], args.top, args.sort_by)
    print(f"[psar] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
