"""XAUUSD Asia-range liquidity sweep backtest.

Rough, testable version of:
    Asia range forms -> London window sweeps range high/low -> reclaim confirms
    -> optionally require a post-sweep FVG -> enter from confirmation/FVG
    -> exit by fixed points, Asia target, RR, FVG, or sweep stop.

Candles are built from bid prices because MT5 native XAUUSD candles match bid OHLC.
Execution uses ask for buys and bid for sells.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from forex_strategy_common import (
    TradeResult,
    build_parser,
    commission,
    default_point_size,
    load_market,
    parse_num_list,
    parse_str_list,
    timeframe_to_ns,
    units_for_margin,
    write_results,
)


@dataclass
class ComboStats:
    days: int = 0
    inside_open_days: int = 0
    above_open_days: int = 0
    below_open_days: int = 0
    sweep_days: int = 0
    confirmations: int = 0
    fvg_pass: int = 0
    entry_fills: int = 0
    side_filtered: int = 0
    no_fvg: int = 0
    no_entry_fill: int = 0


@dataclass
class SetupTrade:
    pair: str
    params: str
    timeframe: str
    date: str
    side: str
    sweep: str
    entry_i: int
    exit_i: int
    entry_px: float
    exit_px: float
    pnl: float
    reason: str
    equity: float
    asia_high: float
    asia_low: float
    confirm_close: float
    fvg_low: float
    fvg_high: float


def _parse_hhmm(s: str) -> int:
    h, m = s.strip().split(":", 1)
    return int(h) * 60 + int(m)


def _minute_of_day(ts: pd.Series) -> pd.Series:
    return ts.dt.hour * 60 + ts.dt.minute


def _time_mask(ts: pd.Series, start_hhmm: str, end_hhmm: str) -> pd.Series:
    start = _parse_hhmm(start_hhmm)
    end = _parse_hhmm(end_hhmm)
    minute = _minute_of_day(ts)
    if start <= end:
        return (minute >= start) & (minute < end)
    return (minute >= start) | (minute < end)


def build_bid_candles(g: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    ts_ns = g["timestamp"].astype("int64").to_numpy()
    bid = g["bid"].to_numpy(np.float64)
    tf_ns = timeframe_to_ns(timeframe)
    bucket = ts_ns // tf_ns
    df = pd.DataFrame({"bucket": bucket, "bid": bid, "tick_i": np.arange(len(g), dtype=np.int64)})
    agg = df.groupby("bucket", sort=True).agg(
        open=("bid", "first"),
        high=("bid", "max"),
        low=("bid", "min"),
        close=("bid", "last"),
        close_i=("tick_i", "last"),
    )
    agg["timestamp"] = pd.to_datetime(agg.index.to_numpy(np.int64) * tf_ns, utc=True)
    return agg.reset_index(drop=True)


def _enter_index_close(confirm_i: int, n_ticks: int) -> int | None:
    idx = confirm_i + 1
    return idx if idx < n_ticks else None


def _enter_index_retrace50(
    side: int,
    confirm_candle: pd.Series,
    start_i: int,
    end_i: int,
    bid: np.ndarray,
    ask: np.ndarray,
) -> int | None:
    body_mid = (float(confirm_candle["open"]) + float(confirm_candle["close"])) / 2.0
    if start_i >= end_i:
        return None
    for i in range(start_i, min(end_i, len(bid))):
        mid = (float(bid[i]) + float(ask[i])) / 2.0
        if side == 1 and mid <= body_mid:
            return i
        if side == -1 and mid >= body_mid:
            return i
    return None


def _fvg_for_candle(candles: pd.DataFrame, idx: int, side: int) -> tuple[float, float] | None:
    """Return FVG zone [low, high] for candle idx if it matches trade side.

    Bullish FVG: current low > high two candles back. Long zone is
    [high[i-2], low[i]]. Bearish FVG: current high < low two candles back.
    Short zone is [high[i], low[i-2]].
    """
    if idx < 2:
        return None
    cur = candles.iloc[idx]
    prev2 = candles.iloc[idx - 2]
    if side == 1:
        low = float(prev2["high"])
        high = float(cur["low"])
        if high > low:
            return low, high
    elif side == -1:
        low = float(cur["high"])
        high = float(prev2["low"])
        if high > low:
            return low, high
    return None


def _enter_index_fvg_mid(
    side: int,
    fvg_low: float,
    fvg_high: float,
    start_i: int,
    end_i: int,
    bid: np.ndarray,
    ask: np.ndarray,
) -> int | None:
    target = (fvg_low + fvg_high) / 2.0
    if start_i >= end_i:
        return None
    for i in range(start_i, min(end_i, len(bid))):
        mid = (float(bid[i]) + float(ask[i])) / 2.0
        if side == 1 and mid <= target:
            return i
        if side == -1 and mid >= target:
            return i
    return None


def _simulate_exit_levels(
    side: int,
    entry_i: int,
    exit_end_i: int,
    bid: np.ndarray,
    ask: np.ndarray,
    tp_px: float | None,
    sl_px: float | None,
) -> tuple[int, float, str]:
    last_i = min(max(entry_i + 1, exit_end_i), len(bid) - 1)
    for i in range(entry_i + 1, min(exit_end_i + 1, len(bid))):
        if side == 1:
            if tp_px is not None and bid[i] >= tp_px:
                return i, float(bid[i]), "tp"
            if sl_px is not None and bid[i] <= sl_px:
                return i, float(bid[i]), "sl"
        else:
            if tp_px is not None and ask[i] <= tp_px:
                return i, float(ask[i]), "tp"
            if sl_px is not None and ask[i] >= sl_px:
                return i, float(ask[i]), "sl"
    exit_px = float(bid[last_i] if side == 1 else ask[last_i])
    return last_i, exit_px, "session_close"


def _exit_levels(
    side: int,
    entry_px: float,
    tp_points: float,
    sl_points: float,
    point_size: float,
    target_mode: str,
    stop_mode: str,
    rr: float,
    asia_high: float,
    asia_low: float,
    sweep_high: float,
    sweep_low: float,
    fvg_low: float | None,
    fvg_high: float | None,
    stop_buffer_points: float,
) -> tuple[float | None, float | None]:
    buffer = stop_buffer_points * point_size
    if stop_mode == "fixed":
        sl_px = entry_px - sl_points * point_size if side == 1 and sl_points > 0 else None
        if side == -1 and sl_points > 0:
            sl_px = entry_px + sl_points * point_size
    elif stop_mode == "sweep":
        sl_px = sweep_low - buffer if side == 1 else sweep_high + buffer
    elif stop_mode == "fvg":
        if fvg_low is None or fvg_high is None:
            sl_px = None
        else:
            sl_px = fvg_low - buffer if side == 1 else fvg_high + buffer
    else:
        raise ValueError(f"unsupported stop mode: {stop_mode}")

    if target_mode == "fixed":
        tp_px = entry_px + tp_points * point_size if side == 1 and tp_points > 0 else None
        if side == -1 and tp_points > 0:
            tp_px = entry_px - tp_points * point_size
    elif target_mode == "asia_mid":
        tp_px = (asia_high + asia_low) / 2.0
    elif target_mode == "asia_opposite":
        tp_px = asia_high if side == 1 else asia_low
    elif target_mode == "rr":
        if sl_px is None:
            tp_px = None
        else:
            risk = abs(entry_px - sl_px)
            tp_px = entry_px + risk * rr if side == 1 else entry_px - risk * rr
    else:
        raise ValueError(f"unsupported target mode: {target_mode}")
    return tp_px, sl_px


def simulate_combo(
    pair: str,
    g: pd.DataFrame,
    candles: pd.DataFrame,
    timeframe: str,
    mode: str,
    open_regime: str,
    entry_mode: str,
    asia_start: str,
    asia_end: str,
    trade_start: str,
    trade_end: str,
    sweep_buffer_points: float,
    min_body_points: float,
    fvg_mode: str,
    target_mode: str,
    stop_mode: str,
    rr: float,
    stop_buffer_points: float,
    tp_points: float,
    sl_points: float,
    point_size: float,
    amount: float,
    compound: bool,
    leverage: float,
    commission_per_million: float,
    side_filter: str,
    max_trades_per_day: int,
    return_stats: bool = False,
) -> tuple[TradeResult, list[SetupTrade], ComboStats]:
    bid = g["bid"].to_numpy(np.float64)
    ask = g["ask"].to_numpy(np.float64)
    tick_ts = g["timestamp"].reset_index(drop=True)
    cash = amount
    equity_peak = amount
    max_dd = 0.0
    gross_win = 0.0
    gross_loss = 0.0
    trades = wins = losses = long_trades = short_trades = stop_losses = signal_exits = 0
    out_trades: list[SetupTrade] = []
    stats = ComboStats()
    daily_pnl: dict[str, float] = {}

    c = candles.copy()
    c["date"] = c["timestamp"].dt.strftime("%Y-%m-%d")
    c["asia"] = _time_mask(c["timestamp"], asia_start, asia_end)
    c["trade"] = _time_mask(c["timestamp"], trade_start, trade_end)
    buffer = sweep_buffer_points * point_size
    min_body = min_body_points * point_size

    for date, day in c.groupby("date", sort=True):
        if cash <= 0:
            break
        stats.days += 1
        asia = day[day["asia"]]
        scan = day[day["trade"]]
        if len(asia) < 2 or len(scan) < 1:
            continue
        asia_high = float(asia["high"].max())
        asia_low = float(asia["low"].min())
        first_scan = scan.iloc[0]
        first_open = float(first_scan["open"])
        if first_open > asia_high + buffer:
            day_regime = "above"
            stats.above_open_days += 1
        elif first_open < asia_low - buffer:
            day_regime = "below"
            stats.below_open_days += 1
        else:
            day_regime = "inside"
            stats.inside_open_days += 1
        if open_regime != "any" and day_regime != open_regime:
            continue
        day_trade_count = 0
        state = "pre_above" if day_regime == "above" else ("pre_below" if day_regime == "below" else "")
        sweep_candle_i = -1
        sweep_high = asia_high
        sweep_low = asia_low

        for candle_idx, row in scan.iterrows():
            if day_trade_count >= max_trades_per_day:
                break
            high = float(row["high"])
            low = float(row["low"])
            open_px = float(row["open"])
            close_px = float(row["close"])
            body = abs(close_px - open_px)

            if not state:
                if high >= asia_high + buffer:
                    state = "sweep_high"
                    stats.sweep_days += 1
                    sweep_candle_i = int(row["close_i"])
                    sweep_high = high
                    sweep_low = low
                elif low <= asia_low - buffer:
                    state = "sweep_low"
                    stats.sweep_days += 1
                    sweep_candle_i = int(row["close_i"])
                    sweep_high = high
                    sweep_low = low
                else:
                    continue
            else:
                sweep_high = max(sweep_high, high)
                sweep_low = min(sweep_low, low)

            side = 0
            if state in {"sweep_high", "pre_above"}:
                if mode == "fade":
                    confirmed = close_px < asia_high and close_px < open_px and body >= min_body
                    if confirmed:
                        side = -1
                else:
                    confirmed = close_px > asia_high and close_px > open_px and body >= min_body
                    if confirmed:
                        side = 1
            elif state in {"sweep_low", "pre_below"}:
                if mode == "fade":
                    confirmed = close_px > asia_low and close_px > open_px and body >= min_body
                    if confirmed:
                        side = 1
                else:
                    confirmed = close_px < asia_low and close_px < open_px and body >= min_body
                    if confirmed:
                        side = -1

            if side == 0:
                continue
            stats.confirmations += 1
            if side_filter == "long" and side != 1:
                stats.side_filtered += 1
                state = ""
                continue
            if side_filter == "short" and side != -1:
                stats.side_filtered += 1
                state = ""
                continue

            confirm_i = int(row["close_i"])
            fvg = _fvg_for_candle(c, int(candle_idx), side)
            if fvg_mode == "require" and fvg is None:
                stats.no_fvg += 1
                continue
            if fvg is not None:
                stats.fvg_pass += 1
            fvg_low = fvg[0] if fvg is not None else None
            fvg_high = fvg[1] if fvg is not None else None
            trade_window = scan[scan["close_i"] >= confirm_i]
            if len(trade_window):
                exit_end_i = int(trade_window["close_i"].iloc[-1])
            else:
                exit_end_i = min(len(bid) - 1, confirm_i)

            if entry_mode == "close":
                entry_i = _enter_index_close(confirm_i, len(bid))
            elif entry_mode == "retrace50":
                entry_i = _enter_index_retrace50(side, row, confirm_i + 1, exit_end_i, bid, ask)
            elif entry_mode == "fvg_mid":
                if fvg_low is None or fvg_high is None:
                    state = ""
                    continue
                entry_i = _enter_index_fvg_mid(
                    side, fvg_low, fvg_high, confirm_i + 1, exit_end_i, bid, ask,
                )
            else:
                raise ValueError(f"unsupported entry mode: {entry_mode}")
            if entry_i is None:
                stats.no_entry_fill += 1
                state = ""
                continue
            stats.entry_fills += 1

            entry_px = float(ask[entry_i] if side == 1 else bid[entry_i])
            margin = cash if compound else amount
            if margin <= 0:
                break
            units = units_for_margin(margin, leverage, entry_px)
            entry_fee = commission(entry_px, units, commission_per_million)
            cash -= entry_fee
            tp_px, sl_px = _exit_levels(
                side, entry_px, tp_points, sl_points, point_size,
                target_mode, stop_mode, rr, asia_high, asia_low,
                sweep_high, sweep_low, fvg_low, fvg_high, stop_buffer_points,
            )
            exit_i, exit_px, reason = _simulate_exit_levels(
                side, entry_i, exit_end_i, bid, ask, tp_px, sl_px,
            )
            pnl = ((exit_px - entry_px) if side == 1 else (entry_px - exit_px)) * units
            pnl -= commission(exit_px, units, commission_per_million)
            cash += pnl
            net_trade_pnl = pnl - entry_fee
            daily_pnl[date] = daily_pnl.get(date, 0.0) + net_trade_pnl

            trades += 1
            day_trade_count += 1
            if side == 1:
                long_trades += 1
            else:
                short_trades += 1
            if reason == "sl":
                stop_losses += 1
            else:
                signal_exits += int(reason == "session_close")
            if net_trade_pnl >= 0:
                wins += 1
                gross_win += net_trade_pnl
            else:
                losses += 1
                gross_loss += -net_trade_pnl

            equity = cash
            equity_peak = max(equity_peak, equity)
            max_dd = max(max_dd, equity_peak - equity)
            out_trades.append(SetupTrade(
                pair=pair,
                params="",
                timeframe=timeframe,
                date=date,
                side="long" if side == 1 else "short",
                sweep=state,
                entry_i=entry_i,
                exit_i=exit_i,
                entry_px=entry_px,
                exit_px=exit_px,
                pnl=net_trade_pnl,
                reason=reason,
                equity=equity,
                asia_high=asia_high,
                asia_low=asia_low,
                confirm_close=close_px,
                fvg_low=float(fvg_low) if fvg_low is not None else float("nan"),
                fvg_high=float(fvg_high) if fvg_high is not None else float("nan"),
            ))
            state = ""

    realised = cash - amount
    total = realised
    days = np.array(list(daily_pnl.values()), dtype=np.float64) if daily_pnl else np.array([0.0])
    pf = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
    wr = wins / trades * 100.0 if trades else 0.0
    params = (
        f"mode={mode};open={open_regime};entry={entry_mode};asia={asia_start}-{asia_end};"
        f"trade={trade_start}-{trade_end};sweep={sweep_buffer_points:g};"
        f"body={min_body_points:g};fvg={fvg_mode};target={target_mode};"
        f"stop={stop_mode};rr={rr:g};stop_buf={stop_buffer_points:g};"
        f"max_day={max_trades_per_day}"
    )
    for t in out_trades:
        t.params = params
    res = TradeResult(
        pair, "asiasweep", params, timeframe, tp_points, sl_points, point_size,
        realised, 0.0, total, trades, wins, losses, wr, pf, max_dd,
        long_trades, short_trades, stop_losses, signal_exits, 0, False, "-", 0.0,
    )
    res.avg_day = float(np.mean(days))
    res.median_day = float(np.median(days))
    res.active_days = int(len(daily_pnl))
    return res, out_trades, stats


def main() -> None:
    ap = build_parser("Asia range liquidity sweep backtest", "forex_asia_sweep_results.csv")
    ap.set_defaults(timeframes="1m,3m,5m", tp_points="300,500,800,1000", sl_points="300,500,800,1000")
    ap.add_argument("--mode", default="fade", help="comma list: fade,continue")
    ap.add_argument("--open-regime", default="any,inside,above,below",
                    help="comma list: any,inside,above,below; filters where price is at trade window open")
    ap.add_argument("--entry-mode", default="close,retrace50,fvg_mid",
                    help="comma list: close,retrace50,fvg_mid")
    ap.add_argument("--fvg-mode", default="off,require",
                    help="comma list: off,require")
    ap.add_argument("--target-mode", default="fixed,asia_mid,asia_opposite,rr",
                    help="comma list: fixed,asia_mid,asia_opposite,rr")
    ap.add_argument("--stop-mode", default="fixed,sweep,fvg",
                    help="comma list: fixed,sweep,fvg")
    ap.add_argument("--rr", default="1,1.5,2",
                    help="comma list of reward:risk targets when --target-mode includes rr")
    ap.add_argument("--stop-buffer-points", default="0,50",
                    help="extra points beyond sweep/FVG stop")
    ap.add_argument("--asia-start", default="00:00")
    ap.add_argument("--asia-end", default="06:00")
    ap.add_argument("--trade-start", default="07:00")
    ap.add_argument("--trade-end", default="11:00")
    ap.add_argument("--sweep-buffer-points", default="0,50,100")
    ap.add_argument("--min-body-points", default="0,50,100")
    ap.add_argument("--max-trades-per-day", type=int, default=1)
    ap.add_argument("--debug-stats", action="store_true",
                    help="print filter counts for the top result after the sweep")
    ap.add_argument("--trades-out", default=None)
    args = ap.parse_args()

    modes = parse_str_list(args.mode, ["fade"])
    open_regimes = parse_str_list(args.open_regime, ["any"])
    entry_modes = parse_str_list(args.entry_mode, ["close"])
    fvg_modes = parse_str_list(args.fvg_mode, ["off"])
    target_modes = parse_str_list(args.target_mode, ["fixed"])
    stop_modes = parse_str_list(args.stop_mode, ["fixed"])
    valid_modes = {"fade", "continue"}
    valid_open_regimes = {"any", "inside", "above", "below"}
    valid_entries = {"close", "retrace50", "fvg_mid"}
    valid_fvg = {"off", "require"}
    valid_targets = {"fixed", "asia_mid", "asia_opposite", "rr"}
    valid_stops = {"fixed", "sweep", "fvg"}
    bad_modes = [m for m in modes if m not in valid_modes]
    bad_open_regimes = [m for m in open_regimes if m not in valid_open_regimes]
    bad_entries = [m for m in entry_modes if m not in valid_entries]
    bad_fvg = [m for m in fvg_modes if m not in valid_fvg]
    bad_targets = [m for m in target_modes if m not in valid_targets]
    bad_stops = [m for m in stop_modes if m not in valid_stops]
    if bad_modes:
        raise SystemExit(f"unsupported --mode values: {bad_modes}")
    if bad_open_regimes:
        raise SystemExit(f"unsupported --open-regime values: {bad_open_regimes}")
    if bad_entries:
        raise SystemExit(f"unsupported --entry-mode values: {bad_entries}")
    if bad_fvg:
        raise SystemExit(f"unsupported --fvg-mode values: {bad_fvg}")
    if bad_targets:
        raise SystemExit(f"unsupported --target-mode values: {bad_targets}")
    if bad_stops:
        raise SystemExit(f"unsupported --stop-mode values: {bad_stops}")

    timeframes = parse_str_list(args.timeframes, ["1m"])
    tps = parse_num_list(args.tp_points, [500.0])
    sls = parse_num_list(args.sl_points, [500.0])
    sweep_buffers = parse_num_list(args.sweep_buffer_points, [0.0])
    min_bodies = parse_num_list(args.min_body_points, [0.0])
    rrs = parse_num_list(args.rr, [1.0])
    stop_buffers = parse_num_list(args.stop_buffer_points, [0.0])

    ticks, _ = load_market(args)
    results: list[TradeResult] = []
    all_trades: list[SetupTrade] = []
    for pair, g0 in ticks.groupby("pair", sort=False):
        g = g0.sort_values("timestamp").reset_index(drop=True)
        point_size = args.point_size or default_point_size(pair)
        print(f"[asiasweep] {pair} ticks={len(g):,}", flush=True)
        candle_cache = {tf: build_bid_candles(g, tf) for tf in timeframes}
        combos = [
            c for c in product(
                timeframes, modes, open_regimes, entry_modes, fvg_modes, target_modes, stop_modes,
                rrs, stop_buffers, sweep_buffers, min_bodies, tps, sls,
            )
            if not (c[3] == "fvg_mid" and c[4] == "off")
            if not (c[6] == "fvg" and c[4] == "off")
            if not (c[5] == "rr" and c[6] == "fixed" and c[12] <= 0)
        ]
        print(f"[asiasweep] combos={len(combos):,}", flush=True)
        for i, combo in enumerate(combos, 1):
            (
                tf, mode, open_regime, entry_mode, fvg_mode, target_mode, stop_mode,
                rr, stop_buffer, sweep_buffer, min_body, tp, sl,
            ) = combo
            res, trades, stats = simulate_combo(
                pair, g, candle_cache[tf], tf, mode, open_regime, entry_mode,
                args.asia_start, args.asia_end, args.trade_start, args.trade_end,
                sweep_buffer, min_body, fvg_mode, target_mode, stop_mode,
                rr, stop_buffer, tp, sl, point_size, args.amount,
                args.compound, args.leverage, args.commission_per_million,
                args.side, args.max_trades_per_day,
            )
            results.append(res)
            if args.debug_stats:
                setattr(res, "_debug_stats", stats)
            if args.trades_out:
                all_trades.extend(trades)
            if i % max(1, len(combos) // 10) == 0:
                print(f"[asiasweep] progress {i}/{len(combos)}", flush=True)

    write_results(args.out, [r for r in results if r.trades >= args.min_trades], args.top, args.sort_by)
    if args.debug_stats and results:
        best = max(results, key=lambda r: r.total)
        stats = getattr(best, "_debug_stats", None)
        if stats is not None:
            print(
                "[asiasweep] best filter stats "
                f"days={stats.days} sweep_days={stats.sweep_days} "
                f"confirm={stats.confirmations} fvg={stats.fvg_pass} "
                f"entry={stats.entry_fills} no_fvg={stats.no_fvg} "
                f"no_entry={stats.no_entry_fill} side_filtered={stats.side_filtered}",
                flush=True,
            )
    if args.trades_out:
        os.makedirs(os.path.dirname(args.trades_out) or ".", exist_ok=True)
        with open(args.trades_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "pair", "params", "timeframe", "date", "side", "sweep",
                "entry_i", "exit_i", "entry_px", "exit_px", "pnl", "reason",
                "equity", "asia_high", "asia_low", "confirm_close",
                "fvg_low", "fvg_high",
            ])
            for t in all_trades:
                w.writerow([
                    t.pair, t.params, t.timeframe, t.date, t.side, t.sweep,
                    t.entry_i, t.exit_i, round(t.entry_px, 5), round(t.exit_px, 5),
                    round(t.pnl, 6), t.reason, round(t.equity, 6),
                    round(t.asia_high, 5), round(t.asia_low, 5),
                    round(t.confirm_close, 5), round(t.fvg_low, 5),
                    round(t.fvg_high, 5),
                ])
        print(f"[asiasweep] wrote trades {args.trades_out}", flush=True)
    print(f"[asiasweep] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
