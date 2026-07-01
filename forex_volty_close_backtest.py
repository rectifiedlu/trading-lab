"""Tick-backed Volty Expan Close Strategy backtest.

Pine source matched:
    length = input(5)
    numATRs = input(0.75)
    atrs = sma(tr, length) * numATRs
    strategy.entry(long, stop=close + atrs)
    strategy.entry(short, stop=close - atrs)

Signals are based on closed candles. Stop entries/reversals execute on ticks.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

import numpy as np
import pandas as pd

from forex_strategy_common import (
    TradeResult,
    build_parser,
    commission,
    closed_candle_series,
    default_point_size,
    load_market,
    open_unrealized,
    parse_num_list,
    parse_str_list,
    units_for_margin,
    write_results,
)

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


def closed_ohlc(mid: np.ndarray, ts_ns: np.ndarray, timeframe: str):
    close, close_idx = closed_candle_series(mid, ts_ns, timeframe)
    if len(close_idx) == 0:
        return close, close, close, close, close_idx
    opens = []
    highs = []
    lows = []
    closes = []
    prev = 0
    for idx in close_idx:
        chunk = mid[prev:idx + 1]
        if len(chunk):
            opens.append(float(chunk[0]))
            highs.append(float(np.max(chunk)))
            lows.append(float(np.min(chunk)))
            closes.append(float(chunk[-1]))
        prev = idx + 1
    return (
        np.array(opens, dtype=np.float64),
        np.array(highs, dtype=np.float64),
        np.array(lows, dtype=np.float64),
        np.array(closes, dtype=np.float64),
        close_idx,
    )


def sma(x: np.ndarray, length: int) -> np.ndarray:
    out = np.full(len(x), np.nan, dtype=np.float64)
    if length <= 1:
        return x.astype(np.float64)
    s = pd.Series(x)
    return s.rolling(length, min_periods=length).mean().to_numpy(np.float64)


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    return np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])


if njit is not None:
    @njit(cache=True)
    def _simulate_volty_numba(
        bid, ask, close_idx, upper, lower, amount, compound, leverage,
        commission_per_million, side_mode, invert_signals,
    ):
        cash = amount
        equity_peak = amount
        max_dd = 0.0
        gross_win = 0.0
        gross_loss = 0.0
        trades = wins = losses = long_trades = short_trades = 0
        pos = 0
        entry = 0.0
        units = 0.0
        j = 0
        active_upper = np.nan
        active_lower = np.nan
        allow_long = side_mode == 1 or side_mode == 3
        allow_short = side_mode == 2 or side_mode == 3
        for i in range(len(bid) - 1):
            while j < len(close_idx) and i > int(close_idx[j]):
                active_upper = upper[j]
                active_lower = lower[j]
                j += 1
            if not np.isfinite(active_upper) or not np.isfinite(active_lower):
                continue
            next_bid = bid[i + 1]
            next_ask = ask[i + 1]
            live_u = 0.0
            if pos == 1:
                live_u = (next_bid - entry) * units
            elif pos == -1:
                live_u = (entry - next_ask) * units
            live_eq = cash + live_u
            if live_eq > equity_peak:
                equity_peak = live_eq
            dd = equity_peak - live_eq
            if dd > max_dd:
                max_dd = dd
            hit_long = next_ask >= active_upper
            hit_short = next_bid <= active_lower
            if hit_long and hit_short:
                cur_mid = (bid[i] + ask[i]) / 2.0
                hit_long = abs(active_upper - cur_mid) <= abs(cur_mid - active_lower)
                hit_short = not hit_long
            target_pos = 0
            if invert_signals:
                if hit_long and allow_short:
                    target_pos = -1
                elif hit_short and allow_long:
                    target_pos = 1
            else:
                if hit_long and allow_long:
                    target_pos = 1
                elif hit_short and allow_short:
                    target_pos = -1
            if target_pos == 0 or target_pos == pos:
                continue
            if pos != 0:
                exit_px = next_bid if pos == 1 else next_ask
                pnl = ((exit_px - entry) if pos == 1 else (entry - exit_px)) * units
                pnl -= abs(exit_px * units) / 1_000_000.0 * commission_per_million
                cash += pnl
                trades += 1
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
                entry = 0.0
                units = 0.0
            margin = cash if compound else amount
            if margin <= 0:
                break
            entry = next_ask if target_pos == 1 else next_bid
            units = (margin * leverage) / entry
            cash -= abs(entry * units) / 1_000_000.0 * commission_per_million
            pos = target_pos
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
            long_trades, short_trades, open_side_code, open_bps,
        )


def simulate_volty(
    pair: str,
    tf: str,
    bid: np.ndarray,
    ask: np.ndarray,
    close_idx: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    params: str,
    amount: float,
    compound: bool,
    leverage: float,
    commission_per_million: float,
    side_mode: str,
    point_size: float,
    invert_signals: bool = False,
) -> TradeResult:
    if njit is not None:
        side_num = 3 if side_mode == "both" else (1 if side_mode == "long" else 2)
        out = _simulate_volty_numba(
            bid.astype(np.float64, copy=False),
            ask.astype(np.float64, copy=False),
            close_idx.astype(np.int64, copy=False),
            upper.astype(np.float64, copy=False),
            lower.astype(np.float64, copy=False),
            amount,
            bool(compound),
            leverage,
            commission_per_million,
            side_num,
            bool(invert_signals),
        )
        (
            realised, open_u, total, trades, wins, losses, pf, max_dd,
            long_trades, short_trades, open_side_code, open_bps,
        ) = out
        win_rate = wins / trades * 100.0 if trades else 0.0
        open_side = "long" if open_side_code == 1 else ("short" if open_side_code == -1 else "-")
        return TradeResult(
            pair, "volty", params, tf, 0.0, 0.0, point_size, realised, open_u,
            total, int(trades), int(wins), int(losses), win_rate, pf, max_dd,
            int(long_trades), int(short_trades), 0, int(trades), 0, False,
            open_side, open_bps,
        )

    cash = amount
    start_balance = amount
    equity_peak = amount
    max_dd = 0.0
    gross_win = gross_loss = 0.0
    trades = wins = losses = long_trades = short_trades = 0
    pos = 0
    entry = units = 0.0
    open_i = 0

    j = 0
    active_upper = np.nan
    active_lower = np.nan
    for i in range(len(bid) - 1):
        while j < len(close_idx) and i > int(close_idx[j]):
            active_upper = upper[j]
            active_lower = lower[j]
            j += 1

        if not np.isfinite(active_upper) or not np.isfinite(active_lower):
            continue

        next_bid = float(bid[i + 1])
        next_ask = float(ask[i + 1])
        live_eq = cash + open_unrealized(pos, entry, units, next_bid, next_ask)
        equity_peak = max(equity_peak, live_eq)
        max_dd = max(max_dd, equity_peak - live_eq)

        hit_long = next_ask >= active_upper
        hit_short = next_bid <= active_lower
        if hit_long and hit_short:
            # Ambiguous intratick ordering. Pick the closer stop from current mid.
            mid = (float(bid[i]) + float(ask[i])) / 2.0
            hit_long = abs(active_upper - mid) <= abs(mid - active_lower)
            hit_short = not hit_long

        target_pos = 0
        if invert_signals:
            if hit_long and side_mode in ("short", "both"):
                target_pos = -1
            elif hit_short and side_mode in ("long", "both"):
                target_pos = 1
        else:
            if hit_long and side_mode in ("long", "both"):
                target_pos = 1
            elif hit_short and side_mode in ("short", "both"):
                target_pos = -1
        if target_pos == 0 or target_pos == pos:
            continue

        if pos != 0:
            exit_px = next_bid if pos == 1 else next_ask
            pnl = ((exit_px - entry) if pos == 1 else (entry - exit_px)) * units
            pnl -= commission(exit_px, units, commission_per_million)
            cash += pnl
            trades += 1
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

        margin = cash if compound else amount
        if margin <= 0:
            break
        entry = next_ask if target_pos == 1 else next_bid
        units = units_for_margin(margin, leverage, entry)
        cash -= commission(entry, units, commission_per_million)
        pos = target_pos
        open_i = i + 1

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
        pair, "volty", params, tf, 0.0, 0.0, point_size, realised, open_u,
        total, trades, wins, losses, win_rate, pf, max_dd, long_trades,
        short_trades, 0, trades, 0, False, open_side, open_bps,
    )


def main() -> None:
    ap = build_parser("Volty expansion close tick backtest", "forex_volty_close_results.csv")
    ap.add_argument("--lengths", default="4,5,6,8,10,14,20")
    ap.add_argument("--atr-mults", default="0.5,0.6,0.7,0.75,0.85,0.95")
    ap.set_defaults(timeframes="1s,10s,15s,30s,1m", tp_points="0", sl_points="0")
    ap.add_argument("--mode", default="normal,invert",
                    help="comma list: normal=upper stop long/lower stop short; invert=fade those stops")
    ap.add_argument("--invert-signals", action="store_true",
                    help="deprecated shortcut for --mode invert")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    timeframes = parse_str_list(args.timeframes, ["1m"])
    modes = parse_str_list(args.mode, ["normal", "invert"])
    if args.invert_signals:
        modes = ["invert"]
    valid_modes = {"normal", "invert"}
    bad_modes = [m for m in modes if m not in valid_modes]
    if bad_modes:
        raise SystemExit(f"unsupported --mode values: {bad_modes}")
    lengths = [int(x) for x in parse_num_list(args.lengths, [5])]
    atr_mults = parse_num_list(args.atr_mults, [0.75])
    # Pine has no explicit TP/SL. 0 means disabled in our shared engine.
    tps = parse_num_list(args.tp_points, [0.0])
    sls = parse_num_list(args.sl_points, [0.0])

    ticks, _ = load_market(args)
    results = []
    for pair, g in ticks.groupby("pair", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        bid = g["bid"].to_numpy(np.float64)
        ask = g["ask"].to_numpy(np.float64)
        # MT5 native XAUUSD candles match bid OHLC, not mid/ask.
        mid = bid
        ts = g["timestamp"].astype("int64").to_numpy()
        point_size = args.point_size or default_point_size(pair)
        print(f"[volty] {pair} ticks={len(g):,}", flush=True)

        combos = list(product(modes, timeframes, lengths, atr_mults, tps, sls))

        def run_combo(combo):
            mode, tf, length, mult, tp, sl = combo
            _, high, low, close, close_idx = closed_ohlc(mid, ts, tf)
            if len(close) < length + 2:
                return None
            atrs = sma(true_range(high, low, close), length) * mult
            upper = close + atrs
            lower = close - atrs
            invert = mode == "invert"
            params = f"mode={mode};length={length};atr_mult={mult};invert={int(invert)}"
            return simulate_volty(
                pair, tf, bid, ask, close_idx, upper, lower, params, args.amount,
                args.compound, args.leverage, args.commission_per_million,
                args.side, point_size, invert,
            )

        if args.workers > 1 and len(combos) > 1:
            print(f"[volty] workers={args.workers} combos={len(combos)}", flush=True)
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(run_combo, c) for c in combos]
                for i, fut in enumerate(as_completed(futs), 1):
                    res = fut.result()
                    if res is not None:
                        results.append(res)
                    if i % max(1, len(combos) // 10) == 0:
                        print(f"[volty] progress {i}/{len(combos)}", flush=True)
        else:
            for combo in combos:
                res = run_combo(combo)
                if res is not None:
                    results.append(res)

    write_results(args.out, [r for r in results if r.trades >= args.min_trades], args.top, args.sort_by)
    print(f"[volty] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
