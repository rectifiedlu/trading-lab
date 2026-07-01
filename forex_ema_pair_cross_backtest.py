"""Fast/slow EMA cross backtest with cumulative-PnL drawdown stats.

Rules:
    - Compute fast and slow EMA on closed candles.
    - Entry needs N consecutive candles in the same EMA regime.
    - Long regime: fast EMA > slow EMA.
    - Short regime: fast EMA < slow EMA.
    - If sl_points == 0, opposite EMA regime acts as the stop/exit.
    - If sl_points > 0, hard SL replaces the EMA regime exit.
    - TP is always tick-based.
"""

from __future__ import annotations

from itertools import product
import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from forex_strategy_common import (
    TradeResult,
    build_parser,
    candle_state_to_ticks,
    closed_candle_series,
    day_ids_from_timestamps,
    default_point_size,
    load_market,
    njit,
    parse_num_list,
    parse_str_list,
    simulate_triggers,
    write_results,
)
from forex_ema_cross_backtest import simulate_trailing_signals


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
    def _confirmed_regime_numba(
        fast: np.ndarray,
        slow: np.ndarray,
        confirm: int,
        deadband: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        regime = np.zeros(len(fast), dtype=np.float64)
        bullish = np.zeros(len(fast), dtype=np.bool_)
        bearish = np.zeros(len(fast), dtype=np.bool_)
        above_count = 0
        below_count = 0
        for i in range(len(fast)):
            f = fast[i]
            s = slow[i]
            if not np.isfinite(f) or not np.isfinite(s):
                above_count = 0
                below_count = 0
            elif f - s > deadband:
                bullish[i] = True
                above_count += 1
                below_count = 0
            elif s - f > deadband:
                bearish[i] = True
                below_count += 1
                above_count = 0
            else:
                above_count = 0
                below_count = 0
            if above_count >= confirm:
                regime[i] = 1.0
            elif below_count >= confirm:
                regime[i] = -1.0
        return regime, bullish, bearish


    @njit(cache=True)
    def _simulate_pair_switch_numba(
        bid: np.ndarray,
        ask: np.ndarray,
        long_trigger: np.ndarray,
        short_trigger: np.ndarray,
        long_exit: np.ndarray,
        short_exit: np.ndarray,
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
        reverse_on_flip: bool,
        reverse_on_tp: bool,
    ):
        tp_dist = tp_points * point_size if tp_points > 0.0 else 0.0
        sl_dist = sl_points * point_size if sl_points > 0.0 else 0.0
        allow_long = side_mode == 1 or side_mode == 3
        allow_short = side_mode == 2 or side_mode == 3
        cash = amount
        equity_peak = amount
        max_dd = 0.0
        gross_win = 0.0
        gross_loss = 0.0
        trades = wins = losses = long_trades = short_trades = 0
        stop_losses = signal_exits = liquidations = 0
        account_dead = False
        pos = 0
        entry = 0.0
        units = 0.0
        daily_pnl = np.zeros(max_days, dtype=np.float64)
        active_days = np.zeros(max_days, dtype=np.int64)
        cur_trade_drawdown = 0.0
        max_trade_drawdown = 0.0
        sum_trade_drawdown = 0.0
        worst_trade_pnl = 0.0
        loss_values = np.empty(100000, dtype=np.float64)
        loss_count = 0

        def add_daily(idx, pnl_value):
            d = day_id[idx]
            if d >= 0 and d < max_days:
                daily_pnl[d] += pnl_value
                active_days[d] = 1

        def open_side(new_pos, px, idx):
            nonlocal cash, pos, entry, units, cur_trade_drawdown
            margin = cash if compound else amount
            if margin <= 0.0:
                return False
            entry = px
            units = (margin * leverage) / entry
            fee = abs(entry * units) / 1_000_000.0 * commission_per_million
            cash -= fee
            add_daily(idx, -fee)
            pos = new_pos
            cur_trade_drawdown = 0.0
            return True

        def close_side(exit_px, idx, is_stop, is_signal):
            nonlocal cash, pos, entry, units, trades, wins, losses
            nonlocal long_trades, short_trades, stop_losses, signal_exits
            nonlocal gross_win, gross_loss, cur_trade_drawdown
            nonlocal max_trade_drawdown, sum_trade_drawdown
            nonlocal worst_trade_pnl, loss_values, loss_count
            if pos == 1:
                pnl = (exit_px - entry) * units
                long_trades += 1
            else:
                pnl = (entry - exit_px) * units
                short_trades += 1
            pnl -= abs(exit_px * units) / 1_000_000.0 * commission_per_million
            cash += pnl
            add_daily(idx, pnl)
            trades += 1
            if is_stop:
                stop_losses += 1
            if is_signal:
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
                if pnl < worst_trade_pnl:
                    worst_trade_pnl = pnl
                if loss_count < len(loss_values):
                    loss_values[loss_count] = pnl
                    loss_count += 1
            pos = 0
            entry = 0.0
            units = 0.0
            return pnl

        for i in range(len(bid) - 1):
            j = i + 1
            b = bid[j]
            a = ask[j]
            reverse_to = 0

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
                open_pnl = (b - entry) * units
                if sl_dist > 0.0 and b <= entry - sl_dist:
                    close_side(b, j, True, False)
                    continue
                if tp_dist > 0.0 and b >= entry + tp_dist:
                    close_side(b, j, False, False)
                    if reverse_on_tp and allow_short:
                        reverse_to = -1
                    else:
                        continue
                if long_exit[i]:
                    should_exit = (open_pnl >= 0.0 and tp_points <= 0.0) or (open_pnl < 0.0 and sl_points <= 0.0)
                    if should_exit:
                        close_side(b, j, False, True)
                        if reverse_on_flip and allow_short:
                            reverse_to = -1
                        else:
                            continue
            elif pos == -1:
                open_pnl = (entry - a) * units
                if sl_dist > 0.0 and a >= entry + sl_dist:
                    close_side(a, j, True, False)
                    continue
                if tp_dist > 0.0 and a <= entry - tp_dist:
                    close_side(a, j, False, False)
                    if reverse_on_tp and allow_long:
                        reverse_to = 1
                    else:
                        continue
                if short_exit[i]:
                    should_exit = (open_pnl >= 0.0 and tp_points <= 0.0) or (open_pnl < 0.0 and sl_points <= 0.0)
                    if should_exit:
                        close_side(a, j, False, True)
                        if reverse_on_flip and allow_long:
                            reverse_to = 1
                        else:
                            continue

            if pos == 0:
                if reverse_to == 1:
                    open_side(1, a, j)
                elif reverse_to == -1:
                    open_side(-1, b, j)
                elif allow_long and long_trigger[i]:
                    open_side(1, a, j)
                elif allow_short and short_trigger[i]:
                    open_side(-1, b, j)

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
        median_loss = 0.0
        if loss_count > 0:
            loss_sorted = loss_values[:loss_count].copy()
            loss_sorted.sort()
            if loss_count % 2 == 1:
                median_loss = loss_sorted[loss_count // 2]
            else:
                median_loss = 0.5 * (
                    loss_sorted[loss_count // 2 - 1] +
                    loss_sorted[loss_count // 2]
                )
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
            long_trades, short_trades, stop_losses, signal_exits,
            liquidations, account_dead, open_side_code, open_bps,
            max_trade_drawdown, trade_avg_drawdown, avg_day, median_day,
            active_day_count, worst_trade_pnl, median_loss,
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


def confirmed_regime(
    fast: np.ndarray,
    slow: np.ndarray,
    confirm: int,
    deadband: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if njit is not None:
        return _confirmed_regime_numba(
            fast.astype(np.float64),
            slow.astype(np.float64),
            int(confirm),
            float(deadband),
        )
    regime = np.zeros(len(fast), dtype=np.float64)
    above_count = 0
    below_count = 0
    for i, (f, s) in enumerate(zip(fast, slow)):
        if not np.isfinite(f) or not np.isfinite(s):
            above_count = below_count = 0
        elif f - s > deadband:
            above_count += 1
            below_count = 0
        elif s - f > deadband:
            below_count += 1
            above_count = 0
        else:
            above_count = below_count = 0
        if above_count >= confirm:
            regime[i] = 1.0
        elif below_count >= confirm:
            regime[i] = -1.0
    bearish = (slow - fast) > deadband
    bullish = (fast - slow) > deadband
    return regime, bullish, bearish


def add_drawdown_metrics(result: TradeResult, trades: list, start_balance: float) -> TradeResult:
    equity = [float(start_balance)]
    equity.extend(float(t.equity) for t in trades)
    final_equity = start_balance + result.total
    if abs(equity[-1] - final_equity) > 1e-9:
        equity.append(float(final_equity))
    y = np.array(equity, dtype=np.float64)
    peaks = np.maximum.accumulate(y)
    drawdowns = peaks - y
    result.cum_max_drawdown = float(np.max(drawdowns)) if len(drawdowns) else 0.0
    result.cum_avg_drawdown = float(np.mean(drawdowns)) if len(drawdowns) else 0.0
    return result


def simulate_pair_switch(
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
    reverse_on_flip: bool,
    reverse_on_tp: bool,
) -> TradeResult:
    if njit is None:
        # Fallback keeps old behavior if numba is unavailable.
        if reverse_on_tp:
            raise RuntimeError("--reverse-on-tp requires numba")
        return simulate_triggers(
            pair, strategy, params, timeframe, bid, ask, long_trigger,
            short_trigger, long_exit, short_exit, tp_points, sl_points,
            point_size, amount, compound, leverage, commission_per_million,
            side, reverse_on_flip=reverse_on_flip,
            day_id=day_id, max_days=max_days,
        )
    side_mode = 3
    if side == "long":
        side_mode = 1
    elif side == "short":
        side_mode = 2
    out = _simulate_pair_switch_numba(
        bid, ask, long_trigger, short_trigger, long_exit, short_exit,
        day_id, max_days, tp_points, sl_points, point_size, amount,
        compound, leverage, commission_per_million, side_mode,
        reverse_on_flip, reverse_on_tp,
    )
    (
        realised, open_u, total, trades, wins, losses, pf, max_dd,
        long_trades, short_trades, stop_losses, signal_exits,
        liquidations, account_dead, open_side_code, open_bps,
        trade_max_drawdown, trade_avg_drawdown, avg_day, median_day,
        active_days, worst_trade_pnl, median_loss,
    ) = out
    open_side = "long" if int(open_side_code) == 1 else ("short" if int(open_side_code) == -1 else "-")
    win_rate = wins / trades * 100.0 if trades else 0.0
    result = TradeResult(
        pair, strategy, params, timeframe, tp_points, sl_points, point_size,
        float(realised), float(open_u), float(total), int(trades), int(wins),
        int(losses), float(win_rate), float(pf), float(max_dd),
        int(long_trades), int(short_trades), int(stop_losses),
        int(signal_exits), int(liquidations), bool(account_dead),
        open_side, float(open_bps),
    )
    result.trade_max_drawdown = float(trade_max_drawdown)
    result.trade_avg_drawdown = float(trade_avg_drawdown)
    result.worst_trade_pnl = float(worst_trade_pnl)
    result.median_loss = float(median_loss)
    result.avg_day = float(avg_day)
    result.median_day = float(median_day)
    result.active_days = int(active_days)
    result.cum_max_drawdown = result.max_drawdown
    result.cum_avg_drawdown = 0.0
    return result


def main() -> None:
    ap = build_parser("EMA pair cross backtest", "forex_ema_pair_cross_results.csv")
    ap.set_defaults(
        timeframes="1s,10s,15s,30s,1m",
        tp_points="0,400,600,800,1000",
        sl_points="0,400,600,800,1000",
    )
    ap.add_argument("--fast-ema", default="4,6,9,12,21")
    ap.add_argument("--slow-ema", default="63,84,105,150,200")
    ap.add_argument("--confirm-candles", default="1")
    ap.add_argument("--signal-mode", default="candle,live",
                    help="comma list: candle=closed candle EMA cross; live=tick EMA cross")
    ap.add_argument("--deadband-points", default="0",
                    help="EMA gap neutral band in points; 0 keeps normal cross behavior")
    ap.add_argument("--reverse-on-flip", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--reverse-on-tp", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--trail-points", default="0",
                    help="0=off; only active when tp=0 and sl=0")
    ap.add_argument("--trades-out", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    timeframes = parse_str_list(args.timeframes, ["10s", "20s", "30s", "1m"])
    fasts = [int(x) for x in parse_num_list(args.fast_ema, [5, 8, 13, 21])]
    slows = [int(x) for x in parse_num_list(args.slow_ema, [21, 34, 42, 63, 84, 105])]
    confirms = [int(x) for x in parse_num_list(args.confirm_candles, [1, 2, 3])]
    modes = parse_str_list(args.signal_mode, ["candle"])
    bad_modes = [m for m in modes if m not in ("candle", "live")]
    if bad_modes:
        raise SystemExit(f"unsupported --signal-mode values: {bad_modes}")
    tps = parse_num_list(args.tp_points, [0])
    sls = parse_num_list(args.sl_points, [0])
    trails = parse_num_list(args.trail_points, [0])
    deadbands = parse_num_list(args.deadband_points, [0.0])

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
        print(f"[ema-pair] {pair} ticks={len(g):,}", flush=True)

        combos = [
            c for c in product(modes, timeframes, fasts, slows, confirms, deadbands, tps, sls, trails)
            if c[2] < c[3] and c[4] >= 1
        ]

        def run_combo(combo):
            mode, tf, fast_len, slow_len, confirm, deadband_points, tp, sl, trail = combo
            deadband = deadband_points * point_size
            if mode == "live":
                if len(mid) < slow_len + confirm + 2:
                    return None
                fast = ema(mid, fast_len)
                slow = ema(mid, slow_len)
                regime, bullish, bearish = confirmed_regime(fast, slow, confirm, deadband)
                prev_regime = np.roll(regime, 1)
                prev_regime[0] = 0.0
                long_trigger = (regime == 1) & (prev_regime != 1)
                short_trigger = (regime == -1) & (prev_regime != -1)
                long_exit = bearish
                short_exit = bullish
            else:
                close, close_tick_idx = closed_candle_series(mid, ts, tf)
                if len(close) < slow_len + confirm + 2:
                    return None

                fast = ema(close, fast_len)
                slow = ema(close, slow_len)
                regime, bullish, bearish = confirmed_regime(fast, slow, confirm, deadband)
                prev_regime = np.roll(regime, 1)
                prev_regime[0] = 0.0

                candle_long = (regime == 1) & (prev_regime != 1)
                candle_short = (regime == -1) & (prev_regime != -1)
                candle_long_exit = bearish
                candle_short_exit = bullish

                long_trigger = candle_state_to_ticks(
                    len(bid), close_tick_idx, candle_long.astype(float),
                ) == 1
                short_trigger = candle_state_to_ticks(
                    len(bid), close_tick_idx, candle_short.astype(float),
                ) == 1
                long_exit = candle_state_to_ticks(
                    len(bid), close_tick_idx, candle_long_exit.astype(float),
                ) == 1
                short_exit = candle_state_to_ticks(
                    len(bid), close_tick_idx, candle_short_exit.astype(float),
                ) == 1

            reverse_on_flip = bool(args.reverse_on_flip and confirm == 1 and sl <= 0)
            reverse_on_tp = bool(args.reverse_on_tp and tp > 0)
            params = (
                f"mode={mode};fast={fast_len};slow={slow_len};confirm={confirm};"
                f"deadband={deadband_points:g};"
                f"reverse={int(reverse_on_flip)};reverse_tp={int(reverse_on_tp)};"
                f"trail={trail:g}"
            )
            want_trades = bool(args.trades_out)
            if trail > 0 and tp == 0 and sl == 0:
                if want_trades:
                    return None
                res = simulate_trailing_signals(
                    pair, "emapair", params, tf, bid, ask, long_trigger,
                    short_trigger, long_exit, short_exit, trail, point_size,
                    args.amount, args.compound, args.leverage,
                    args.commission_per_million, args.side, reverse_on_flip,
                )
                res.cum_max_drawdown = res.max_drawdown
                res.cum_avg_drawdown = 0.0
                return res, []
            res = simulate_pair_switch(
                pair, "emapair", params, tf, bid, ask, long_trigger, short_trigger,
                long_exit, short_exit, day_id, max_days, tp, sl, point_size,
                args.amount, args.compound, args.leverage,
                args.commission_per_million, args.side, reverse_on_flip,
                reverse_on_tp,
            )
            return res, []

        if args.workers > 1 and len(combos) > 1 and not args.trades_out:
            print(f"[ema-pair] workers={args.workers} combos={len(combos)}", flush=True)
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(run_combo, c) for c in combos]
                for i, fut in enumerate(as_completed(futs), 1):
                    out = fut.result()
                    if out is not None:
                        res, _ = out
                        results.append(res)
                    if i % max(1, len(combos) // 10) == 0:
                        print(f"[ema-pair] progress {i}/{len(combos)}", flush=True)
        else:
            for combo in combos:
                out = run_combo(combo)
                if out is None:
                    continue
                res, trades = out
                results.append(res)
                if args.trades_out:
                    all_trades.extend(trades)

    write_results(
        args.out,
        [r for r in results if r.trades >= args.min_trades],
        args.top,
        args.sort_by,
    )
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
        print(f"[ema-pair] wrote trades {args.trades_out}", flush=True)
    print(f"[ema-pair] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
