#!/usr/bin/env python3
"""Backtest quote-pressure sign trading on historical MT5 bid/ask ticks."""

from __future__ import annotations

import argparse
import csv
import math
import time
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
from numba import njit


FIELDS = [
    "rank_pnl",
    "rank_pnl_dd",
    "asset_class",
    "symbol",
    "days",
    "ticks",
    "window",
    "threshold",
    "deadband",
    "signal_mode",
    "tp_points",
    "sl_points",
    "max_spread_points",
    "retreat_weight",
    "lot",
    "trades",
    "wins",
    "losses",
    "tp_exits",
    "sl_exits",
    "signal_exits",
    "win_rate_pct",
    "net_points",
    "net_pnl",
    "max_dd_points",
    "max_dd",
    "pnl_dd",
    "avg_trade_points",
    "avg_hold_seconds",
    "start_utc",
    "end_utc",
]


def parse_number_list(value: str, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Historical MT5 quote-pressure backtest")
    symbols = parser.add_mutually_exclusive_group()
    symbols.add_argument("--symbol", help="single symbol (kept for compatibility)")
    symbols.add_argument(
        "--pairs",
        nargs="+",
        help="one or more symbols, separated by spaces or commas",
    )
    parser.add_argument("--days", type=float, default=7.0)
    parser.add_argument(
        "--windows",
        default="10,20,30,50,75,100,150,200,300,500,750,1000,1250,1500,1750,2000,2500,3000,4000,5000",
    )
    parser.add_argument(
        "--thresholds",
        default="0,0.025,0.05,0.075,0.1,0.15,0.2,0.3,0.4,0.5,0.65,0.8",
    )
    parser.add_argument("--deadbands", default="0,1")
    parser.add_argument(
        "--signal-modes",
        default="normal,invert",
        help="comma-separated signal directions: normal,invert",
    )
    parser.add_argument(
        "--tp-points",
        default="0",
        help="comma-separated take-profit distances in points; 0 uses signal exits",
    )
    parser.add_argument(
        "--sl-points",
        default="0",
        help="comma-separated stop-loss distances in points; 0 uses signal exits",
    )
    parser.add_argument(
        "--max-spread-points",
        default="5,10,20,30",
        help="comma-separated maximum entry spreads; entries require spread below the limit",
    )
    parser.add_argument("--retreat-weight", type=float, default=0.25)
    parser.add_argument("--lot", type=float, default=0.01)
    parser.add_argument("--commission-per-lot", type=float, default=0.0)
    parser.add_argument(
        "--out",
        default="data/forex/quote_pressure_backtest.csv",
    )
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="print progress every N completed configurations (0 disables it)",
    )
    return parser.parse_args()


def parse_symbols(args: argparse.Namespace) -> list[str]:
    raw = args.pairs if args.pairs else [args.symbol or "USDJPY"]
    symbols: list[str] = []
    for value in raw:
        for item in value.split(","):
            symbol = item.strip().upper()
            if symbol and symbol not in symbols:
                symbols.append(symbol)
    return symbols


def quote_components(
    bid: np.ndarray,
    ask: np.ndarray,
    retreat_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bid_change = np.diff(bid)
    ask_change = np.diff(ask)
    residual_bid = bid_change.copy()
    residual_ask = ask_change.copy()
    contribution = np.zeros_like(bid_change)

    shared = bid_change * ask_change > 0.0
    direction = np.where(bid_change > 0.0, 1.0, -1.0)
    common = np.where(
        shared,
        direction * np.minimum(np.abs(bid_change), np.abs(ask_change)),
        0.0,
    )
    contribution += 2.0 * common
    residual_bid -= common
    residual_ask -= common
    contribution += np.maximum(residual_bid, 0.0)
    contribution -= np.maximum(-residual_ask, 0.0)
    contribution += retreat_weight * np.maximum(residual_ask, 0.0)
    contribution -= retreat_weight * np.maximum(-residual_bid, 0.0)

    activity = np.abs(bid_change) + np.abs(ask_change)
    mid_movement = (bid_change + ask_change) / 2.0
    return contribution, activity, mid_movement


def rolling_ratio(numerator: np.ndarray, denominator: np.ndarray, window: int) -> np.ndarray:
    size = len(numerator) + 1
    result = np.zeros(size, dtype=np.float64)
    num_cumulative = np.empty(size, dtype=np.float64)
    den_cumulative = np.empty(size, dtype=np.float64)
    num_cumulative[0] = 0.0
    den_cumulative[0] = 0.0
    np.cumsum(numerator, out=num_cumulative[1:])
    np.cumsum(denominator, out=den_cumulative[1:])
    rolling_num = num_cumulative[window:] - num_cumulative[:-window]
    rolling_den = den_cumulative[window:] - den_cumulative[:-window]
    np.divide(
        rolling_num,
        rolling_den,
        out=result[window:],
        where=rolling_den > 0.0,
    )
    return result


@njit(cache=True)
def simulate(
    bid: np.ndarray,
    ask: np.ndarray,
    time_msc: np.ndarray,
    quote_pressure: np.ndarray,
    start_index: int,
    threshold: float,
    deadband: int,
    signal_multiplier: int,
    tp_points: float,
    sl_points: float,
    max_spread_points: float,
    point: float,
    money_per_price: float,
    commission: float,
):
    position = 0
    entry_price = 0.0
    entry_time = 0
    cash = 0.0
    net_price = 0.0
    peak_equity = 0.0
    max_drawdown = 0.0
    trades = 0
    wins = 0
    losses = 0
    total_hold_ms = 0
    tp_exits = 0
    sl_exits = 0
    signal_exits = 0
    long_blocked = False
    short_blocked = False

    for index in range(start_index, len(bid)):
        pressure = quote_pressure[index] * signal_multiplier
        desired = 0
        if pressure > 0.0 and pressure >= threshold:
            desired = 1
        elif pressure < 0.0 and pressure <= -threshold:
            desired = -1

        if desired == 0:
            long_blocked = False
            short_blocked = False

        close_position = False
        exit_reason = 0  # 1=TP, 2=SL, 3=signal
        position_points = 0.0
        if position == 1:
            position_points = (bid[index] - entry_price) / point
        elif position == -1:
            position_points = (entry_price - ask[index]) / point

        if position != 0 and tp_points > 0.0 and position_points >= tp_points:
            close_position = True
            exit_reason = 1
        elif position != 0 and sl_points > 0.0 and position_points <= -sl_points:
            close_position = True
            exit_reason = 2

        signal_close = False
        if position == 1:
            if deadband == 1:
                signal_close = pressure < threshold
            else:
                signal_close = desired == -1
        elif position == -1:
            if deadband == 1:
                signal_close = pressure > -threshold
            else:
                signal_close = desired == 1

        if not close_position and signal_close:
            tp_controls = position_points > 0.0 and tp_points > 0.0
            sl_controls = position_points < 0.0 and sl_points > 0.0
            if not tp_controls and not sl_controls:
                close_position = True
                exit_reason = 3

        if close_position:
            closed_side = position
            exit_price = bid[index] if position == 1 else ask[index]
            price_pnl = (exit_price - entry_price) * position
            trade_pnl = price_pnl * money_per_price - commission
            cash += trade_pnl
            net_price += price_pnl
            trades += 1
            total_hold_ms += time_msc[index] - entry_time
            if trade_pnl > 0.0:
                wins += 1
            else:
                losses += 1
            if exit_reason == 1:
                tp_exits += 1
            elif exit_reason == 2:
                sl_exits += 1
            else:
                signal_exits += 1
            if exit_reason in (1, 2):
                # A neutral score on the exit tick already satisfies the reset.
                if desired != 0:
                    if closed_side == 1:
                        long_blocked = True
                    else:
                        short_blocked = True
            position = 0

        direction_blocked = (desired == 1 and long_blocked) or (desired == -1 and short_blocked)
        spread_points = (ask[index] - bid[index]) / point
        spread_allowed = spread_points < max_spread_points
        if position == 0 and desired != 0 and not direction_blocked and spread_allowed:
            position = desired
            entry_price = ask[index] if desired == 1 else bid[index]
            entry_time = time_msc[index]

        equity = cash
        if position == 1:
            equity += (bid[index] - entry_price) * money_per_price - commission
        elif position == -1:
            equity += (entry_price - ask[index]) * money_per_price - commission
        if equity > peak_equity:
            peak_equity = equity
        drawdown = peak_equity - equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    if position != 0:
        index = len(bid) - 1
        exit_price = bid[index] if position == 1 else ask[index]
        price_pnl = (exit_price - entry_price) * position
        trade_pnl = price_pnl * money_per_price - commission
        cash += trade_pnl
        net_price += price_pnl
        trades += 1
        total_hold_ms += time_msc[index] - entry_time
        if trade_pnl > 0.0:
            wins += 1
        else:
            losses += 1
        if cash > peak_equity:
            peak_equity = cash
        drawdown = peak_equity - cash
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    max_dd_points = max_drawdown / money_per_price / point if money_per_price > 0.0 else 0.0
    net_points = net_price / point
    avg_hold_seconds = total_hold_ms / trades / 1000.0 if trades else 0.0
    return (
        cash,
        net_points,
        max_drawdown,
        max_dd_points,
        trades,
        wins,
        losses,
        avg_hold_seconds,
        tp_exits,
        sl_exits,
        signal_exits,
    )


def load_ticks(symbol: str, days: float):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    ticks = mt5.copy_ticks_range(symbol, start, end, mt5.COPY_TICKS_INFO)
    if ticks is None:
        raise RuntimeError(f"copy_ticks_range failed: {mt5.last_error()}")
    if not len(ticks):
        raise RuntimeError(f"No historical ticks returned for {symbol}")
    valid = (
        np.isfinite(ticks["bid"])
        & np.isfinite(ticks["ask"])
        & (ticks["bid"] > 0.0)
        & (ticks["ask"] >= ticks["bid"])
    )
    ticks = ticks[valid]
    if len(ticks) < 2:
        return ticks
    changed = np.ones(len(ticks), dtype=np.bool_)
    changed[1:] = (ticks["bid"][1:] != ticks["bid"][:-1]) | (
        ticks["ask"][1:] != ticks["ask"][:-1]
    )
    return ticks[changed]


def print_ranking_table(
    title: str,
    rows: list[dict[str, object]],
    key: str,
    top: int,
) -> None:
    candidates = rows
    if key == "pnl_dd":
        candidates = [
            row for row in rows
            if int(row["trades"]) > 0
            and float(row["net_pnl"]) > 0.0
            and float(row["max_dd"]) > 0.0
        ]
    ranked = sorted(candidates, key=lambda row: float(row[key]), reverse=True)[:top]
    print(f"\n  {title}", flush=True)
    if not ranked:
        print("  no qualifying results", flush=True)
        return

    headers = [
        "#", "pair", "mode", "w", "th", "db", "tp", "sl", "spr<",
        "pnl", "dd", "pnl/dd", "tr", "wr%", "avg_tr", "hold_s",
    ]
    table = []
    for rank, row in enumerate(ranked, 1):
        table.append([
            str(rank),
            str(row["symbol"]),
            str(row["signal_mode"]),
            str(row["window"]),
            f"{float(row['threshold']):g}",
            str(row["deadband"]),
            f"{float(row['tp_points']):g}",
            f"{float(row['sl_points']):g}",
            f"{float(row['max_spread_points']):g}",
            f"${float(row['net_pnl']):+.2f}",
            f"${float(row['max_dd']):.2f}",
            f"{float(row['pnl_dd']):.2f}",
            str(row["trades"]),
            f"{float(row['win_rate_pct']):.1f}",
            f"{float(row['avg_trade_points']):+.2f}",
            f"{float(row['avg_hold_seconds']):.1f}",
        ])
    widths = [len(header) for header in headers]
    for row in table:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    print("  " + " ".join(value.rjust(widths[i]) for i, value in enumerate(headers)), flush=True)
    print("  " + "-" * (sum(widths) + len(widths) - 1), flush=True)
    for row in table:
        print("  " + " ".join(value.rjust(widths[i]) for i, value in enumerate(row)), flush=True)


def print_grouped_rankings(rows: list[dict[str, object]], top: int) -> None:
    groups = [
        ("XAUUSD", [row for row in rows if row["asset_class"] == "XAUUSD"]),
        ("FX", [row for row in rows if row["asset_class"] == "FX"]),
    ]
    for group_name, group_rows in groups:
        if not group_rows:
            continue
        print_ranking_table(f"{group_name} top by total PnL", group_rows, "net_pnl", top)
        print_ranking_table(
            f"{group_name} top by total/account DD (PnL/DD)",
            group_rows,
            "pnl_dd",
            top,
        )


def print_pair_summary(rows: list[dict[str, object]]) -> None:
    print("\nBest configuration per pair:")
    print(
        "PAIR     RUNS | BEST PNL:      PNL       DD   RATIO   W    TH DB MODE   TP   SL  SPR "
        "| BEST PNL/DD:    PNL       DD   RATIO   W    TH DB MODE   TP   SL  SPR"
    )
    print("-" * 165)
    for symbol in sorted({str(row["symbol"]) for row in rows}):
        pair_rows = [row for row in rows if row["symbol"] == symbol]
        best_pnl = max(pair_rows, key=lambda row: float(row["net_pnl"]))
        best_ratio = max(pair_rows, key=lambda row: float(row["pnl_dd"]))
        print(
            f"{symbol:<8} {len(pair_rows):4d} | "
            f"{float(best_pnl['net_pnl']):+9.2f} {float(best_pnl['max_dd']):8.2f} "
            f"{float(best_pnl['pnl_dd']):+7.3f} {int(best_pnl['window']):4d} "
            f"{float(best_pnl['threshold']):5g} {int(best_pnl['deadband']):2d} "
            f"{str(best_pnl['signal_mode']):<6} "
            f"{float(best_pnl['tp_points']):4g} {float(best_pnl['sl_points']):4g} "
            f"{float(best_pnl['max_spread_points']):4g} | "
            f"{float(best_ratio['net_pnl']):+9.2f} {float(best_ratio['max_dd']):8.2f} "
            f"{float(best_ratio['pnl_dd']):+7.3f} {int(best_ratio['window']):4d} "
            f"{float(best_ratio['threshold']):5g} {int(best_ratio['deadband']):2d} "
            f"{str(best_ratio['signal_mode']):<6} "
            f"{float(best_ratio['tp_points']):4g} {float(best_ratio['sl_points']):4g} "
            f"{float(best_ratio['max_spread_points']):4g}",
            flush=True,
        )


def run_symbol(
    args: argparse.Namespace,
    symbol: str,
    windows: list[int],
    thresholds: list[float],
    deadbands: list[int],
    signal_modes: list[str],
    take_profits: list[float],
    stop_losses: list[float],
    max_spreads: list[float],
    pair_index: int,
    pair_count: int,
) -> list[dict[str, object]]:
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Could not select {symbol}: {mt5.last_error()}")
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"No symbol info for {symbol}")

    ticks = load_ticks(symbol, args.days)
    bid = np.ascontiguousarray(ticks["bid"], dtype=np.float64)
    ask = np.ascontiguousarray(ticks["ask"], dtype=np.float64)
    time_msc = np.ascontiguousarray(ticks["time_msc"], dtype=np.int64)
    point = float(info.point)
    tick_size = float(info.trade_tick_size or point)
    tick_value = float(info.trade_tick_value or info.trade_tick_value_profit)
    if point <= 0.0 or tick_size <= 0.0 or tick_value <= 0.0:
        raise RuntimeError(
            f"Invalid contract data for {symbol}: point={point}, "
            f"tick_size={tick_size}, tick_value={tick_value}"
        )
    money_per_price = args.lot * tick_value / tick_size
    commission = args.lot * args.commission_per_lot
    contribution, activity, _ = quote_components(bid, ask, args.retreat_weight)
    eligible_windows = [window for window in windows if window < len(ticks)]
    combinations = (
        len(eligible_windows)
        * len(thresholds)
        * len(deadbands)
        * len(signal_modes)
        * len(take_profits)
        * len(stop_losses)
        * len(max_spreads)
    )
    if not combinations:
        raise RuntimeError(f"Not enough ticks for any requested window on {symbol}")

    print(
        f"\n[quote-bt] pair {pair_index}/{pair_count} {symbol} ticks={len(ticks):,} "
        f"range={datetime.fromtimestamp(int(ticks[0]['time']), timezone.utc)} -> "
        f"{datetime.fromtimestamp(int(ticks[-1]['time']), timezone.utc)} "
        f"sweeps={combinations} lot={args.lot:g}",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    completed = 0
    pair_started = time.perf_counter()
    for window in eligible_windows:
        pressure = rolling_ratio(contribution, activity, window)
        configurations = product(
            thresholds,
            deadbands,
            signal_modes,
            take_profits,
            stop_losses,
            max_spreads,
        )
        for (
            threshold,
            deadband,
            signal_mode,
            tp_points,
            sl_points,
            max_spread_points,
        ) in configurations:
            signal_multiplier = 1 if signal_mode == "normal" else -1
            result = simulate(
                bid,
                ask,
                time_msc,
                pressure,
                window,
                threshold,
                deadband,
                signal_multiplier,
                tp_points,
                sl_points,
                max_spread_points,
                point,
                money_per_price,
                commission,
            )
            (
                net_pnl,
                net_points,
                max_dd,
                max_dd_points,
                trades,
                wins,
                losses,
                avg_hold,
                tp_exits,
                sl_exits,
                signal_exits,
            ) = result
            completed += 1
            win_rate = wins / trades * 100.0 if trades else 0.0
            pnl_dd = net_pnl / max_dd if max_dd > 1e-12 else 0.0
            rows.append({
                "rank_pnl": 0,
                "rank_pnl_dd": 0,
                "asset_class": "XAUUSD" if symbol.startswith("XAU") else "FX",
                "symbol": symbol,
                "days": args.days,
                "ticks": len(ticks),
                "window": window,
                "threshold": threshold,
                "deadband": deadband,
                "signal_mode": signal_mode,
                "tp_points": tp_points,
                "sl_points": sl_points,
                "max_spread_points": max_spread_points,
                "retreat_weight": args.retreat_weight,
                "lot": args.lot,
                "trades": trades,
                "wins": wins,
                "losses": losses,
                "tp_exits": tp_exits,
                "sl_exits": sl_exits,
                "signal_exits": signal_exits,
                "win_rate_pct": win_rate,
                "net_points": net_points,
                "net_pnl": net_pnl,
                "max_dd_points": max_dd_points,
                "max_dd": max_dd,
                "pnl_dd": pnl_dd,
                "avg_trade_points": net_points / trades if trades else 0.0,
                "avg_hold_seconds": avg_hold,
                "start_utc": datetime.fromtimestamp(
                    int(ticks[0]["time"]), timezone.utc
                ).isoformat(),
                "end_utc": datetime.fromtimestamp(
                    int(ticks[-1]["time"]), timezone.utc
                ).isoformat(),
            })
            should_report = (
                args.progress_every > 0
                and (
                    completed % args.progress_every == 0
                    or completed == combinations
                )
            )
            if should_report:
                elapsed = time.perf_counter() - pair_started
                eta = elapsed / completed * (combinations - completed)
                print(
                    f"[quote-bt] {symbol} {completed:4d}/{combinations} "
                    f"({completed / combinations:5.1%}) elapsed={elapsed:6.1f}s "
                    f"eta={eta:6.1f}s latest pnl={net_pnl:+.2f} "
                    f"mode={signal_mode} "
                    f"tp={tp_points:g} sl={sl_points:g} "
                    f"spread<{max_spread_points:g}",
                    flush=True,
                )
    return rows


def main() -> None:
    args = parse_args()
    symbols = parse_symbols(args)
    windows = sorted(set(parse_number_list(args.windows, int)))
    thresholds = sorted(set(parse_number_list(args.thresholds, float)))
    deadbands = sorted(set(parse_number_list(args.deadbands, int)))
    signal_modes = list(dict.fromkeys(
        item.strip().lower() for item in args.signal_modes.split(",") if item.strip()
    ))
    take_profits = sorted(set(parse_number_list(args.tp_points, float)))
    stop_losses = sorted(set(parse_number_list(args.sl_points, float)))
    max_spreads = sorted(set(parse_number_list(args.max_spread_points, float)))
    if args.days <= 0.0:
        raise SystemExit("--days must be positive")
    if not windows or min(windows) < 2:
        raise SystemExit("--windows values must be at least 2")
    if not thresholds or min(thresholds) < 0.0 or max(thresholds) > 1.0:
        raise SystemExit("--thresholds values must be between 0 and 1")
    if not deadbands or any(value not in (0, 1) for value in deadbands):
        raise SystemExit("--deadbands values must be 0 or 1")
    if not signal_modes or any(value not in ("normal", "invert") for value in signal_modes):
        raise SystemExit("--signal-modes values must be normal or invert")
    if not take_profits or min(take_profits) < 0.0:
        raise SystemExit("--tp-points values cannot be negative")
    if not stop_losses or min(stop_losses) < 0.0:
        raise SystemExit("--sl-points values cannot be negative")
    if not max_spreads or min(max_spreads) <= 0.0:
        raise SystemExit("--max-spread-points values must be positive")
    if not 0.0 <= args.retreat_weight <= 1.0:
        raise SystemExit("--retreat-weight must be between 0 and 1")
    if args.lot <= 0.0:
        raise SystemExit("--lot must be positive")
    if args.top < 1:
        raise SystemExit("--top must be at least 1")
    if args.progress_every < 0:
        raise SystemExit("--progress-every cannot be negative")

    if not mt5.initialize():
        raise SystemExit(f"MT5 initialize failed: {mt5.last_error()}")
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    try:
        print(
            f"[quote-bt] plan pairs={symbols} days={args.days:g} "
            f"windows={len(windows)} thresholds={len(thresholds)} "
            f"deadbands={deadbands} signal_modes={signal_modes} "
            f"tp_points={take_profits} sl_points={stop_losses} "
            f"max_spread_points={max_spreads}",
            flush=True,
        )
        for pair_index, symbol in enumerate(symbols, 1):
            try:
                rows.extend(
                    run_symbol(
                        args,
                        symbol,
                        windows,
                        thresholds,
                        deadbands,
                        signal_modes,
                        take_profits,
                        stop_losses,
                        max_spreads,
                        pair_index,
                        len(symbols),
                    )
                )
            except Exception as exc:
                message = f"{symbol}: {exc}"
                failures.append(message)
                print(f"[quote-bt] FAILED {message}", flush=True)

        if not rows:
            raise RuntimeError("No pair produced backtest results")

        for asset_class in ("XAUUSD", "FX"):
            indices = [
                index for index, row in enumerate(rows)
                if row["asset_class"] == asset_class
            ]
            pnl_order = sorted(
                indices,
                key=lambda index: float(rows[index]["net_pnl"]),
                reverse=True,
            )
            ratio_order = sorted(
                indices,
                key=lambda index: float(rows[index]["pnl_dd"]),
                reverse=True,
            )
            for rank, index in enumerate(pnl_order, 1):
                rows[index]["rank_pnl"] = rank
            for rank, index in enumerate(ratio_order, 1):
                rows[index]["rank_pnl_dd"] = rank
        class_order = {"XAUUSD": 0, "FX": 1}
        rows.sort(key=lambda row: (class_order[str(row["asset_class"])], int(row["rank_pnl"])))

        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print_pair_summary(rows)
        print_grouped_rankings(rows, args.top)
        print(
            f"\n[quote-bt] wrote {len(rows):,} rows to {output} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        if failures:
            print(f"[quote-bt] completed with {len(failures)} failure(s):", flush=True)
            for failure in failures:
                print(f"  - {failure}", flush=True)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
