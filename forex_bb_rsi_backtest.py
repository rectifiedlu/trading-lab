"""Bollinger + RSI mean-reversion backtest with dynamic exits.

Setup:
    Long arms when close is below lower BB and RSI <= oversold.
    Long enters when close reclaims above lower BB.
    Short arms when close is above upper BB and RSI >= overbought.
    Short enters when close rejects below upper BB.

Exits:
    tp_points > 0: fixed TP.
    tp_points == 0: dynamic TP by --tp-mode basis|opposite|signal.
    sl_points > 0: fixed SL.
    sl_points == 0: setup stretch low/high plus buffer.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

import numpy as np

from forex_signal_sweep_common import build_bid_ohlc, rma, side_mode_value
from forex_strategy_common import (
    TradeResult,
    active_session_allowed,
    build_parser,
    day_ids_from_timestamps,
    default_point_size,
    load_market,
    parse_num_list,
    parse_str_list,
    write_results,
    njit,
)


DEFAULT_TIMEFRAMES = ["30s", "1m", "3m","5m"]
DEFAULT_BB_LENGTHS = [13, 20, 34]
DEFAULT_BB_MULTS = [1.5, 2.0, 2.5]
DEFAULT_RSI_LENGTHS = [7, 10, 14]
DEFAULT_OVERSOLD = [25, 30, 35]
DEFAULT_OVERBOUGHT = [65, 70, 75]
GOLD_TP = [0, 200, 300, 400, 500]
GOLD_SL = [0, 200, 300, 400]
FX_TP = [0, 20, 30, 50, 80, 100]
FX_SL = [0, 20, 30, 50, 80, 100]
DEFAULT_TP_MODES = ["basis", "opposite", "signal"]
DEFAULT_MODES = ["normal", "invert"]
DEFAULT_SESSIONS = [-1, 0, 1]
DEFAULT_SL_BUFFER = [0, 50]


def rolling_sma(x: np.ndarray, length: int) -> np.ndarray:
    out = np.full(len(x), np.nan, dtype=np.float64)
    if len(x) < length:
        return out
    csum = np.cumsum(np.insert(x.astype(np.float64), 0, 0.0))
    out[length - 1:] = (csum[length:] - csum[:-length]) / float(length)
    return out


def bb_rsi_signals(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    bb_length: int,
    bb_mult: float,
    rsi_length: int,
    oversold: float,
    overbought: float,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    avg_gain = rma(gains, rsi_length)
    avg_loss = rma(losses, rsi_length)
    rs = avg_gain / np.maximum(avg_loss, 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    signal = np.zeros(len(closes), dtype=np.float64)
    stretch_low = np.full(len(closes), np.nan, dtype=np.float64)
    stretch_high = np.full(len(closes), np.nan, dtype=np.float64)
    long_armed = False
    short_armed = False
    low_ref = np.nan
    high_ref = np.nan

    for i in range(len(closes)):
        if not np.isfinite(lower[i]) or not np.isfinite(upper[i]) or not np.isfinite(rsi[i]):
            continue
        if closes[i] < lower[i] and rsi[i] <= oversold:
            long_armed = True
            low_ref = lows[i] if not np.isfinite(low_ref) else min(low_ref, lows[i])
        elif long_armed:
            low_ref = min(low_ref, lows[i])
            if closes[i] > lower[i]:
                signal[i] = 1.0
                stretch_low[i] = low_ref
                long_armed = False
                low_ref = np.nan

        if closes[i] > upper[i] and rsi[i] >= overbought:
            short_armed = True
            high_ref = highs[i] if not np.isfinite(high_ref) else max(high_ref, highs[i])
        elif short_armed:
            high_ref = max(high_ref, highs[i])
            if closes[i] < upper[i]:
                signal[i] = -1.0
                stretch_high[i] = high_ref
                short_armed = False
                high_ref = np.nan

    if mode == "invert":
        signal = -signal
        tmp = stretch_low.copy()
        stretch_low = stretch_high
        stretch_high = tmp
    return signal, basis, upper, lower, np.where(signal > 0, stretch_low, stretch_high)


if njit is not None:
    @njit(cache=True)
    def _simulate_bb_rsi_numba(
        bid, ask, close_idx, signal, basis, upper, lower, stretch_ref,
        entry_allowed_candle, day_id_candle, max_days, tp_points, sl_points,
        sl_buffer_points, tp_mode_code, point_size, amount, compound, leverage,
        commission_per_million, side_mode,
    ):
        cash = amount
        peak = amount
        max_dd = 0.0
        gross_win = 0.0
        gross_loss = 0.0
        trades = wins = losses = long_trades = short_trades = stops = sig_exits = liq = 0
        pos = 0
        entry = 0.0
        units = 0.0
        stop_px = 0.0
        take_px = 0.0
        cur_dd = 0.0
        max_trade_dd = 0.0
        loss_values = np.empty(len(close_idx), dtype=np.float64)
        loss_count = 0
        daily = np.zeros(max_days, dtype=np.float64)
        allow_long = side_mode == 1 or side_mode == 3
        allow_short = side_mode == 2 or side_mode == 3

        def add_daily(candle_i, pnl):
            d = day_id_candle[candle_i]
            if 0 <= d < max_days:
                daily[d] += pnl

        for ci in range(len(close_idx)):
            ti = int(close_idx[ci])
            b = bid[ti]
            a = ask[ti]
            if pos != 0:
                unreal = (b - entry) * units if pos == 1 else (entry - a) * units
                eq = cash + unreal
                if eq > peak:
                    peak = eq
                dd = peak - eq
                if dd > max_dd:
                    max_dd = dd
                if -unreal > cur_dd:
                    cur_dd = -unreal

            close = False
            exit_px = 0.0
            is_stop = False
            if pos == 1:
                if tp_points > 0 and b >= take_px:
                    close = True; exit_px = b
                elif b <= stop_px:
                    close = True; exit_px = b; is_stop = True
                elif tp_points <= 0:
                    if tp_mode_code == 1 and b >= basis[ci]:
                        close = True; exit_px = b
                    elif tp_mode_code == 2 and b >= upper[ci]:
                        close = True; exit_px = b
                    elif tp_mode_code == 3 and signal[ci] < 0:
                        close = True; exit_px = b
                if close:
                    pnl = (exit_px - entry) * units - abs(exit_px * units) / 1_000_000.0 * commission_per_million
                    cash += pnl; add_daily(ci, pnl); trades += 1; long_trades += 1
                    if is_stop: stops += 1
                    else: sig_exits += 1
                    if cur_dd > max_trade_dd: max_trade_dd = cur_dd
                    cur_dd = 0.0
                    if pnl >= 0:
                        wins += 1; gross_win += pnl
                    else:
                        losses += 1; gross_loss += -pnl
                        loss_values[loss_count] = pnl; loss_count += 1
                    pos = 0
            elif pos == -1:
                if tp_points > 0 and a <= take_px:
                    close = True; exit_px = a
                elif a >= stop_px:
                    close = True; exit_px = a; is_stop = True
                elif tp_points <= 0:
                    if tp_mode_code == 1 and a <= basis[ci]:
                        close = True; exit_px = a
                    elif tp_mode_code == 2 and a <= lower[ci]:
                        close = True; exit_px = a
                    elif tp_mode_code == 3 and signal[ci] > 0:
                        close = True; exit_px = a
                if close:
                    pnl = (entry - exit_px) * units - abs(exit_px * units) / 1_000_000.0 * commission_per_million
                    cash += pnl; add_daily(ci, pnl); trades += 1; short_trades += 1
                    if is_stop: stops += 1
                    else: sig_exits += 1
                    if cur_dd > max_trade_dd: max_trade_dd = cur_dd
                    cur_dd = 0.0
                    if pnl >= 0:
                        wins += 1; gross_win += pnl
                    else:
                        losses += 1; gross_loss += -pnl
                        loss_values[loss_count] = pnl; loss_count += 1
                    pos = 0

            if pos == 0 and entry_allowed_candle[ci]:
                margin = cash if compound else amount
                if margin > 0:
                    if signal[ci] > 0 and allow_long:
                        entry = a
                        units = margin * leverage / entry
                        fee = abs(entry * units) / 1_000_000.0 * commission_per_million
                        cash -= fee; add_daily(ci, -fee)
                        take_px = entry + tp_points * point_size
                        if sl_points > 0:
                            stop_px = entry - sl_points * point_size
                        else:
                            stop_px = stretch_ref[ci] - sl_buffer_points * point_size
                            if not np.isfinite(stop_px) or stop_px >= entry:
                                stop_px = entry - sl_buffer_points * point_size
                        pos = 1
                    elif signal[ci] < 0 and allow_short:
                        entry = b
                        units = margin * leverage / entry
                        fee = abs(entry * units) / 1_000_000.0 * commission_per_million
                        cash -= fee; add_daily(ci, -fee)
                        take_px = entry - tp_points * point_size
                        if sl_points > 0:
                            stop_px = entry + sl_points * point_size
                        else:
                            stop_px = stretch_ref[ci] + sl_buffer_points * point_size
                            if not np.isfinite(stop_px) or stop_px <= entry:
                                stop_px = entry + sl_buffer_points * point_size
                        pos = -1

        open_u = 0.0
        open_side = 0
        open_bps = 0.0
        if pos == 1:
            open_side = 1
            open_u = (bid[-1] - entry) * units
            open_bps = (bid[-1] / entry - 1.0) * 10000.0
        elif pos == -1:
            open_side = -1
            open_u = (entry - ask[-1]) * units
            open_bps = (entry / ask[-1] - 1.0) * 10000.0
        realised = cash - amount
        total = realised + open_u
        pf = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
        daily_sorted = daily.copy(); daily_sorted.sort()
        avg_day = np.mean(daily) if max_days > 0 else 0.0
        med_day = daily_sorted[max_days // 2] if max_days % 2 == 1 else 0.5 * (daily_sorted[max_days // 2 - 1] + daily_sorted[max_days // 2])
        med_loss = 0.0
        if loss_count > 0:
            loss_sorted = loss_values[:loss_count].copy(); loss_sorted.sort()
            med_loss = loss_sorted[loss_count // 2] if loss_count % 2 == 1 else 0.5 * (loss_sorted[loss_count // 2 - 1] + loss_sorted[loss_count // 2])
        return realised, open_u, total, trades, wins, losses, pf, max_dd, long_trades, short_trades, stops, sig_exits, liq, open_side, open_bps, max_trade_dd, avg_day, med_day, med_loss


def tp_mode_code(mode: str) -> int:
    return {"basis": 1, "opposite": 2, "signal": 3}[mode]


def default_tp_sl_for_pair(pair: str) -> tuple[list[float], list[float]]:
    if pair.upper() == "XAUUSD":
        return GOLD_TP, GOLD_SL
    return FX_TP, FX_SL


def pair_rsi_levels(oversolds: list[float], overboughts: list[float]) -> list[tuple[float, float]]:
    if len(oversolds) == len(overboughts):
        return list(zip(oversolds, overboughts))
    if len(oversolds) == 1:
        return [(oversolds[0], ob) for ob in overboughts]
    if len(overboughts) == 1:
        return [(os_, overboughts[0]) for os_ in oversolds]
    raise SystemExit("--oversold and --overbought must have same length, or one side must have one value")


def simulate_bb_rsi(pair, params, tf, bid, ask, close_idx, signal, basis, upper, lower, stretch_ref,
                    entry_allowed, candle_day_id, tp, sl, sl_buffer, tp_mode, point_size,
                    amount, compound, leverage, commission_per_million, side):
    if njit is None:
        raise RuntimeError("numba is required")
    max_days = int(np.max(candle_day_id)) + 1 if len(candle_day_id) else 1
    out = _simulate_bb_rsi_numba(
        bid.astype(np.float64), ask.astype(np.float64), close_idx.astype(np.int64),
        signal.astype(np.float64), basis.astype(np.float64), upper.astype(np.float64),
        lower.astype(np.float64), stretch_ref.astype(np.float64),
        entry_allowed.astype(np.bool_), candle_day_id.astype(np.int64), max_days,
        float(tp), float(sl), float(sl_buffer), tp_mode_code(tp_mode), point_size,
        amount, compound, leverage, commission_per_million, side_mode_value(side),
    )
    realised, open_u, total, trades, wins, losses, pf, max_dd, longs, shorts, stops, sig, liq, open_side_code, open_bps, tr_dd, avg_day, med_day, med_loss = out
    win_rate = wins / trades * 100.0 if trades else 0.0
    open_side = "long" if int(open_side_code) == 1 else ("short" if int(open_side_code) == -1 else "-")
    r = TradeResult(pair, "bb_rsi", params, tf, tp, sl, point_size, realised, open_u, total, int(trades), int(wins), int(losses), win_rate, pf, max_dd, int(longs), int(shorts), int(stops), int(sig), int(liq), bool(liq), open_side, open_bps)
    r.trade_max_drawdown = float(tr_dd)
    r.avg_day = float(avg_day)
    r.median_day = float(med_day)
    r.median_loss = float(med_loss)
    return r


def main() -> None:
    ap = build_parser("BB+RSI dynamic mean-reversion sweep", "forex_bb_rsi_results.csv")
    ap.add_argument("--bb-lengths", default=None)
    ap.add_argument("--bb-mults", default=None)
    ap.add_argument("--rsi-lengths", default=None)
    ap.add_argument("--oversold", default=None)
    ap.add_argument("--overbought", default=None)
    ap.add_argument("--tp-mode", default=None, help="basis,opposite,signal")
    ap.add_argument("--modes", default=None, help="normal,invert")
    ap.add_argument("--sessions", default=None, help="-1,0,1")
    ap.add_argument("--sl-buffer-points", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    timeframes = parse_str_list(args.timeframes, DEFAULT_TIMEFRAMES)
    bb_lengths = [int(x) for x in parse_num_list(args.bb_lengths, DEFAULT_BB_LENGTHS)]
    bb_mults = parse_num_list(args.bb_mults, DEFAULT_BB_MULTS)
    rsi_lengths = [int(x) for x in parse_num_list(args.rsi_lengths, DEFAULT_RSI_LENGTHS)]
    oversolds = parse_num_list(args.oversold, DEFAULT_OVERSOLD)
    overboughts = parse_num_list(args.overbought, DEFAULT_OVERBOUGHT)
    rsi_level_pairs = pair_rsi_levels(oversolds, overboughts)
    tp_modes = parse_str_list(args.tp_mode, DEFAULT_TP_MODES)
    modes = parse_str_list(args.modes, DEFAULT_MODES)
    sessions = [int(x) for x in parse_num_list(args.sessions, DEFAULT_SESSIONS)]
    buffers = parse_num_list(args.sl_buffer_points, DEFAULT_SL_BUFFER)

    ticks, t0 = load_market(args)
    results = []
    base_combo_count = len(timeframes) * len(bb_lengths) * len(bb_mults) * len(rsi_lengths) * len(rsi_level_pairs) * len(modes) * len(sessions)
    print(f"[bb-rsi] base_combos={base_combo_count:,} workers={args.workers}", flush=True)

    for pair, g in ticks.groupby("pair", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        bid = g["bid"].to_numpy(np.float64)
        ask = g["ask"].to_numpy(np.float64)
        ts_ns = g["timestamp"].astype("int64").to_numpy(np.int64)
        point_size = float(args.point_size or default_point_size(pair))
        default_tp, default_sl = default_tp_sl_for_pair(pair)
        tps = parse_num_list(args.tp_points, default_tp)
        sls = parse_num_list(args.sl_points, default_sl)
        dynamic_tp_count = sum(len(tp_modes) if tp <= 0 else 1 for tp in tps)
        dynamic_sl_count = sum(len(buffers) if sl <= 0 else 1 for sl in sls)
        combo_count = base_combo_count * dynamic_tp_count * dynamic_sl_count
        session_tick_cache = {s: active_session_allowed(ts_ns, s) for s in sessions}
        print(f"[bb-rsi] {pair} ticks={len(g):,} point={point_size:g} combos={combo_count:,} tp={tps} sl={sls}", flush=True)

        tasks = []
        for tf in timeframes:
            _, highs, lows, closes, close_idx = build_bid_ohlc(bid, ts_ns, tf)
            candle_days, _ = day_ids_from_timestamps(ts_ns[close_idx])
            session_candle_cache = {s: session_tick_cache[s][close_idx] for s in sessions}
            for bb_len, bb_mult, rsi_len, (os_, ob), mode in product(bb_lengths, bb_mults, rsi_lengths, rsi_level_pairs, modes):
                signal, basis, upper, lower, stretch_ref = bb_rsi_signals(highs, lows, closes, bb_len, bb_mult, rsi_len, os_, ob, mode)
                for tp, sl, sess in product(tps, sls, sessions):
                    tp_mode_values = tp_modes if tp <= 0 else [tp_modes[0]]
                    buffer_values = buffers if sl <= 0 else [buffers[0]]
                    for tp_mode, buf in product(tp_mode_values, buffer_values):
                        params = f"bb={bb_len};mult={bb_mult:g};rsi={rsi_len};os={os_:g};ob={ob:g};mode={mode};session={sess};tp_mode={tp_mode};sl_buf={buf:g}"
                        tasks.append((params, tf, close_idx, signal, basis, upper, lower, stretch_ref, session_candle_cache[sess], candle_days, tp, sl, buf, tp_mode))

        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [
                ex.submit(simulate_bb_rsi, pair, params, tf, bid, ask, close_idx, signal, basis, upper, lower, stretch_ref,
                          allowed, days, tp, sl, buf, tp_mode, point_size, args.amount, args.compound,
                          args.leverage, args.commission_per_million, args.side)
                for params, tf, close_idx, signal, basis, upper, lower, stretch_ref, allowed, days, tp, sl, buf, tp_mode in tasks
            ]
            for fut in as_completed(futs):
                results.append(fut.result())
                done += 1
                if done % max(1, len(futs) // 10) == 0:
                    print(f"[bb-rsi] {pair} progress {done}/{len(futs)}", flush=True)

    filtered = [r for r in results if r.trades >= args.min_trades]
    write_results(args.out, filtered, args.top, args.sort_by)
    print(f"[bb-rsi] wrote {args.out} elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
