#!/usr/bin/env python3
"""Backtest quote-pressure sign trading on historical MT5 bid/ask ticks."""

from __future__ import annotations

import argparse
import csv
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
from numba import njit


FIELDS = [
    "rank_pnl",
    "rank_pnl_dd",
    "symbol",
    "days",
    "ticks",
    "window",
    "threshold",
    "deadband",
    "retreat_weight",
    "lot",
    "trades",
    "wins",
    "losses",
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
    parser.add_argument("--symbol", default="USDJPY")
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
    parser.add_argument("--retreat-weight", type=float, default=0.25)
    parser.add_argument("--lot", type=float, default=0.01)
    parser.add_argument("--commission-per-lot", type=float, default=0.0)
    parser.add_argument(
        "--out",
        default="data/forex/quote_pressure_backtest.csv",
    )
    parser.add_argument("--top", type=int, default=10)
    return parser.parse_args()


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

    for index in range(start_index, len(bid)):
        pressure = quote_pressure[index]
        desired = 0
        if pressure > 0.0 and pressure >= threshold:
            desired = 1
        elif pressure < 0.0 and pressure <= -threshold:
            desired = -1

        close_position = False
        if position == 1:
            if deadband == 1:
                close_position = pressure < threshold
            else:
                close_position = desired == -1
        elif position == -1:
            if deadband == 1:
                close_position = pressure > -threshold
            else:
                close_position = desired == 1

        if close_position:
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
            position = 0

        if position == 0 and desired != 0:
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
    return cash, net_points, max_drawdown, max_dd_points, trades, wins, losses, avg_hold_seconds


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
    return ticks[valid]


def print_rankings(rows: list[dict[str, object]], key: str, top: int) -> None:
    print(f"\nTop {min(top, len(rows))} by {key}:")
    ordered = sorted(rows, key=lambda row: float(row[key]), reverse=True)
    for rank, row in enumerate(ordered[:top], 1):
        print(
            f"#{rank:02d} pnl={float(row['net_pnl']):+9.2f} "
            f"dd={float(row['max_dd']):8.2f} ratio={float(row['pnl_dd']):+7.3f} "
            f"trades={int(row['trades']):6d} win={float(row['win_rate_pct']):5.1f}% "
            f"w={row['window']} th={float(row['threshold']):g} db={row['deadband']}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    windows = sorted(set(parse_number_list(args.windows, int)))
    thresholds = sorted(set(parse_number_list(args.thresholds, float)))
    deadbands = sorted(set(parse_number_list(args.deadbands, int)))
    if args.days <= 0.0:
        raise SystemExit("--days must be positive")
    if not windows or min(windows) < 2:
        raise SystemExit("--windows values must be at least 2")
    if not thresholds or min(thresholds) < 0.0 or max(thresholds) > 1.0:
        raise SystemExit("--thresholds values must be between 0 and 1")
    if not deadbands or any(value not in (0, 1) for value in deadbands):
        raise SystemExit("--deadbands values must be 0 or 1")
    if not 0.0 <= args.retreat_weight <= 1.0:
        raise SystemExit("--retreat-weight must be between 0 and 1")
    if args.lot <= 0.0:
        raise SystemExit("--lot must be positive")

    if not mt5.initialize():
        raise SystemExit(f"MT5 initialize failed: {mt5.last_error()}")
    symbol = args.symbol.upper()
    try:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Could not select {symbol}: {mt5.last_error()}")
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"No symbol info for {symbol}")
        started = time.perf_counter()
        ticks = load_ticks(symbol, args.days)
        bid = np.ascontiguousarray(ticks["bid"], dtype=np.float64)
        ask = np.ascontiguousarray(ticks["ask"], dtype=np.float64)
        time_msc = np.ascontiguousarray(ticks["time_msc"], dtype=np.int64)
        point = float(info.point)
        tick_size = float(info.trade_tick_size or point)
        tick_value = float(info.trade_tick_value or info.trade_tick_value_profit)
        money_per_price = args.lot * tick_value / tick_size
        commission = args.lot * args.commission_per_lot
        contribution, activity, _ = quote_components(bid, ask, args.retreat_weight)
        combinations = len(windows) * len(thresholds) * len(deadbands)
        print(
            f"[quote-bt] {symbol} ticks={len(ticks):,} "
            f"range={datetime.fromtimestamp(int(ticks[0]['time']), timezone.utc)} -> "
            f"{datetime.fromtimestamp(int(ticks[-1]['time']), timezone.utc)} "
            f"sweeps={combinations} lot={args.lot:g}",
            flush=True,
        )

        rows: list[dict[str, object]] = []
        completed = 0
        for window in windows:
            if window >= len(ticks):
                continue
            pressure = rolling_ratio(contribution, activity, window)
            for threshold in thresholds:
                for deadband in deadbands:
                    result = simulate(
                        bid,
                        ask,
                        time_msc,
                        pressure,
                        window,
                        threshold,
                        deadband,
                        point,
                        money_per_price,
                        commission,
                    )
                    net_pnl, net_points, max_dd, max_dd_points, trades, wins, losses, avg_hold = result
                    completed += 1
                    win_rate = wins / trades * 100.0 if trades else 0.0
                    pnl_dd = net_pnl / max_dd if max_dd > 1e-12 else 0.0
                    rows.append({
                        "rank_pnl": 0,
                        "rank_pnl_dd": 0,
                        "symbol": symbol,
                        "days": args.days,
                        "ticks": len(ticks),
                        "window": window,
                        "threshold": threshold,
                        "deadband": deadband,
                        "retreat_weight": args.retreat_weight,
                        "lot": args.lot,
                        "trades": trades,
                        "wins": wins,
                        "losses": losses,
                        "win_rate_pct": win_rate,
                        "net_points": net_points,
                        "net_pnl": net_pnl,
                        "max_dd_points": max_dd_points,
                        "max_dd": max_dd,
                        "pnl_dd": pnl_dd,
                        "avg_trade_points": net_points / trades if trades else 0.0,
                        "avg_hold_seconds": avg_hold,
                        "start_utc": datetime.fromtimestamp(int(ticks[0]["time"]), timezone.utc).isoformat(),
                        "end_utc": datetime.fromtimestamp(int(ticks[-1]["time"]), timezone.utc).isoformat(),
                    })
                    print(
                        f"[quote-bt] {completed:3d}/{combinations} w={window} "
                        f"th={threshold:g} db={deadband} pnl={net_pnl:+.2f} "
                        f"dd={max_dd:.2f} trades={trades:,}",
                        flush=True,
                    )

        pnl_order = sorted(range(len(rows)), key=lambda index: float(rows[index]["net_pnl"]), reverse=True)
        ratio_order = sorted(range(len(rows)), key=lambda index: float(rows[index]["pnl_dd"]), reverse=True)
        for rank, index in enumerate(pnl_order, 1):
            rows[index]["rank_pnl"] = rank
        for rank, index in enumerate(ratio_order, 1):
            rows[index]["rank_pnl_dd"] = rank
        rows.sort(key=lambda row: int(row["rank_pnl"]))

        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print_rankings(rows, "net_pnl", args.top)
        print_rankings(rows, "pnl_dd", args.top)
        print(
            f"\n[quote-bt] wrote {output} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
