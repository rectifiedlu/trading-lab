from __future__ import annotations

import argparse
import itertools
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from mt5_ohlc_dump import load_mt5_rates_range


GOLD_TP = [200, 300, 400, 500]
GOLD_SL = [200, 300, 400, 500]
FX_TP = [20, 30, 50, 80, 100]
FX_SL = [20, 30, 50, 80, 100]
JPY_TP = [20, 30, 50, 80, 100]
JPY_SL = [20, 30, 50, 80, 100]


@dataclass
class ScanResult:
    pair: str
    tendency: str
    timeframe: str
    tp: float
    sl: float
    trades: int
    wins: int
    losses: int
    realised: float
    max_dd: float
    median_day: float
    avg_day: float
    win_rate: float
    profit_factor: float
    params: str


def point_size(pair: str) -> float:
    pair = pair.upper()
    if pair == "XAUUSD":
        return 0.01
    if pair.endswith("JPY"):
        return 0.001
    return 0.00001


def tp_sl_grid(pair: str) -> tuple[list[int], list[int]]:
    pair = pair.upper()
    if pair == "XAUUSD":
        return GOLD_TP, GOLD_SL
    if pair.endswith("JPY"):
        return JPY_TP, JPY_SL
    return FX_TP, FX_SL


def parse_dt(value: str | None, fallback: datetime) -> datetime:
    if not value:
        return fallback
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def atr(df: pd.DataFrame, length: int) -> np.ndarray:
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    prev = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    return pd.Series(tr).rolling(length, min_periods=length).mean().to_numpy(float)


def trade_pnl(side: int, entry: float, exit_px: float, amount: float, leverage: float, fee_per_million: float) -> float:
    units = amount * leverage / max(entry, 1e-12)
    gross = (exit_px - entry) * units if side == 1 else (entry - exit_px) * units
    fee = amount * leverage / 1_000_000.0 * fee_per_million * 2.0
    return gross - fee


def run_signals(
    df: pd.DataFrame,
    signals: np.ndarray,
    tp_points: float,
    sl_points: float,
    point: float,
    horizon: int,
    amount: float,
    leverage: float,
    fee_per_million: float,
) -> tuple[int, int, int, float, float, float, float, float, float]:
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    times = pd.to_datetime(df["time"], utc=True)
    realised = 0.0
    peak = 0.0
    max_dd = 0.0
    wins = losses = 0
    gross_win = gross_loss = 0.0
    daily: dict[object, float] = {}
    i = 1
    n = len(df)
    while i < n - 2:
        side = int(signals[i])
        if side == 0:
            i += 1
            continue
        entry_i = i + 1
        if entry_i >= n:
            break
        entry = float(o[entry_i])
        if side == 1:
            tp_px = entry + tp_points * point
            sl_px = entry - sl_points * point
        else:
            tp_px = entry - tp_points * point
            sl_px = entry + sl_points * point
        exit_px = float(c[min(entry_i + horizon, n - 1)])
        exit_i = min(entry_i + horizon, n - 1)
        for j in range(entry_i, min(entry_i + horizon + 1, n)):
            if side == 1:
                tp_hit = h[j] >= tp_px
                sl_hit = l[j] <= sl_px
                if sl_hit and tp_hit:
                    exit_px, exit_i = sl_px, j
                    break
                if sl_hit:
                    exit_px, exit_i = sl_px, j
                    break
                if tp_hit:
                    exit_px, exit_i = tp_px, j
                    break
            else:
                tp_hit = l[j] <= tp_px
                sl_hit = h[j] >= sl_px
                if sl_hit and tp_hit:
                    exit_px, exit_i = sl_px, j
                    break
                if sl_hit:
                    exit_px, exit_i = sl_px, j
                    break
                if tp_hit:
                    exit_px, exit_i = tp_px, j
                    break
        pnl = trade_pnl(side, entry, exit_px, amount, leverage, fee_per_million)
        realised += pnl
        peak = max(peak, realised)
        max_dd = max(max_dd, peak - realised)
        day = times.iloc[exit_i].date()
        daily[day] = daily.get(day, 0.0) + pnl
        if pnl >= 0:
            wins += 1
            gross_win += pnl
        else:
            losses += 1
            gross_loss += -pnl
        i = exit_i + 1
    trades = wins + losses
    wr = wins / trades * 100 if trades else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
    days = np.array(list(daily.values()), dtype=float) if daily else np.array([0.0])
    return trades, wins, losses, realised, max_dd, float(np.median(days)), float(np.mean(days)), wr, pf


def expansion_signals(df: pd.DataFrame, atr_len: int, mult: float, mode: str) -> np.ndarray:
    a = atr(df, atr_len)
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    rng = h - l
    body = c - o
    sig = np.zeros(len(df), dtype=np.int8)
    good = (rng >= a * mult) & np.isfinite(a)
    sig[good & (body > 0)] = 1
    sig[good & (body < 0)] = -1
    if mode == "fade":
        sig *= -1
    return sig


def compression_breakout_signals(df: pd.DataFrame, lookback: int, range_mult: float, mode: str) -> np.ndarray:
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    a = atr(df, 20)
    sig = np.zeros(len(df), dtype=np.int8)
    for i in range(lookback + 1, len(df)):
        hi = np.max(h[i - lookback:i])
        lo = np.min(l[i - lookback:i])
        if not np.isfinite(a[i]) or hi - lo > a[i] * range_mult:
            continue
        if c[i] > hi:
            sig[i] = 1
        elif c[i] < lo:
            sig[i] = -1
    if mode == "fade":
        sig *= -1
    return sig


def session_range_signals(df: pd.DataFrame, range_minutes: int, trade_minutes: int, mode: str) -> np.ndarray:
    # UTC sessions approximating Tokyo/London/NY starts.
    starts = [0, 7, 13]
    times = pd.to_datetime(df["time"], utc=True)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    sig = np.zeros(len(df), dtype=np.int8)
    frame_min = max(int((times.iloc[1] - times.iloc[0]).total_seconds() // 60), 1) if len(times) > 1 else 1
    range_bars = max(range_minutes // frame_min, 1)
    trade_bars = max(trade_minutes // frame_min, 1)
    by_day: dict[object, list[int]] = {}
    for i, t in enumerate(times):
        by_day.setdefault(t.date(), []).append(i)
    for idxs in by_day.values():
        for start_hour in starts:
            session = [i for i in idxs if times.iloc[i].hour == start_hour and times.iloc[i].minute == 0]
            if not session:
                continue
            s = session[0]
            if s + range_bars >= len(df):
                continue
            hi = np.max(h[s:s + range_bars])
            lo = np.min(l[s:s + range_bars])
            for i in range(s + range_bars, min(s + range_bars + trade_bars, len(df))):
                if c[i] > hi:
                    sig[i] = 1
                    break
                if c[i] < lo:
                    sig[i] = -1
                    break
    if mode == "fade":
        sig *= -1
    return sig


def prev_day_sweep_signals(df: pd.DataFrame, mode: str) -> np.ndarray:
    times = pd.to_datetime(df["time"], utc=True)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    sig = np.zeros(len(df), dtype=np.int8)
    daily = df.assign(day=times.dt.date).groupby("day").agg(high=("high", "max"), low=("low", "min"))
    prev = daily.shift(1)
    for i, t in enumerate(times):
        row = prev.loc[t.date()] if t.date() in prev.index else None
        if row is None or pd.isna(row["high"]):
            continue
        # Sweep then close back inside = fade. Close beyond = continue.
        if h[i] > row["high"]:
            sig[i] = -1 if c[i] < row["high"] else 1
        elif l[i] < row["low"]:
            sig[i] = 1 if c[i] > row["low"] else -1
    if mode == "continue":
        sig *= -1
    return sig


def scan_pair(args: argparse.Namespace, pair: str, timeframe: str) -> list[ScanResult]:
    end = parse_dt(args.to, datetime.now(timezone.utc))
    start = parse_dt(args.start, end - timedelta(days=args.days))
    df, _ = load_mt5_rates_range(pair, timeframe, start, end)
    if df.empty or len(df) < 500:
        print(f"[scan] skip {pair} {timeframe}: bars={len(df)}", flush=True)
        return []
    point = point_size(pair)
    tps, sls = tp_sl_grid(pair)
    results: list[ScanResult] = []
    signal_specs: list[tuple[str, str, np.ndarray]] = []
    for atr_len, mult, mode in itertools.product([7, 14, 21], [1.2, 1.6, 2.0, 2.5], ["follow", "fade"]):
        signal_specs.append(("expansion", f"atr={atr_len};mult={mult:g};mode={mode}", expansion_signals(df, atr_len, mult, mode)))
    for lookback, mult, mode in itertools.product([5, 10, 20, 34], [0.8, 1.0, 1.2, 1.5], ["breakout", "fade"]):
        signal_specs.append(("compression", f"lookback={lookback};range_mult={mult:g};mode={mode}", compression_breakout_signals(df, lookback, mult, mode)))
    for range_min, trade_min, mode in itertools.product([15, 30, 60], [120, 240], ["breakout", "fade"]):
        signal_specs.append(("session_range", f"range_min={range_min};trade_min={trade_min};mode={mode}", session_range_signals(df, range_min, trade_min, mode)))
    for mode in ["fade", "continue"]:
        signal_specs.append(("prev_day_sweep", f"mode={mode}", prev_day_sweep_signals(df, mode)))

    total = len(signal_specs) * len(tps) * len(sls)
    done = 0
    print(f"[scan] {pair} {timeframe} bars={len(df):,} signal_specs={len(signal_specs)} combos={total:,}", flush=True)
    for tendency, params, sig in signal_specs:
        if np.count_nonzero(sig) < args.min_signals:
            continue
        for tp, sl in itertools.product(tps, sls):
            done += 1
            trades, wins, losses, realised, max_dd, med_day, avg_day, wr, pf = run_signals(
                df, sig, tp, sl, point, args.horizon_bars, args.amount, args.leverage, args.commission_per_million
            )
            if trades < args.min_trades:
                continue
            results.append(ScanResult(
                pair=pair,
                tendency=tendency,
                timeframe=timeframe,
                tp=tp,
                sl=sl,
                trades=trades,
                wins=wins,
                losses=losses,
                realised=realised,
                max_dd=max_dd,
                median_day=med_day,
                avg_day=avg_day,
                win_rate=wr,
                profit_factor=pf,
                params=params,
            ))
    return results


def print_table(title: str, rows: list[ScanResult], top: int) -> None:
    print(f"\n  {title}")
    print("   # pair    tend            tf   tp  sl  trades   wr%    pf   realised     dd  med/day avg/day params")
    print("  ---------------------------------------------------------------------------------------------------------")
    for n, r in enumerate(rows[:top], 1):
        print(
            f"{n:4d} {r.pair:7} {r.tendency:15} {r.timeframe:>3} "
            f"{r.tp:4.0f} {r.sl:3.0f} {r.trades:7d} {r.win_rate:5.1f} {r.profit_factor:5.2f} "
            f"${r.realised:+9.2f} ${r.max_dd:6.2f} ${r.median_day:+7.2f} ${r.avg_day:+7.2f} {r.params}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Raw market tendency scanner over native MT5 OHLC.")
    ap.add_argument("--pairs", nargs="+", default=["XAUUSD", "EURUSD", "USDJPY", "GBPUSD"])
    ap.add_argument("--timeframes", default="1m,3m,5m,15m")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--to", default=None)
    ap.add_argument("--horizon-bars", type=int, default=100)
    ap.add_argument("--amount", type=float, default=50.0)
    ap.add_argument("--leverage", type=float, default=100.0)
    ap.add_argument("--commission-per-million", type=float, default=30.0)
    ap.add_argument("--min-trades", type=int, default=25)
    ap.add_argument("--min-signals", type=int, default=10)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out", default=os.path.join("data", "forex", "market_tendency_scan_results.csv"))
    args = ap.parse_args()

    all_results: list[ScanResult] = []
    for pair in args.pairs:
        for tf in [x.strip() for x in args.timeframes.split(",") if x.strip()]:
            try:
                all_results.extend(scan_pair(args, pair, tf))
            except SystemExit as exc:
                print(f"[scan] skip {pair} {tf}: {exc}", flush=True)

    if not all_results:
        raise SystemExit("[scan] no results")
    df = pd.DataFrame([r.__dict__ for r in all_results])
    df["real_dd"] = df["realised"] / df["max_dd"].replace(0, np.nan)
    df["med_dd"] = df["median_day"] / df["max_dd"].replace(0, np.nan)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.sort_values(["realised", "median_day"], ascending=[False, False]).to_csv(args.out, index=False)
    print(f"[scan] wrote {args.out} rows={len(df):,}", flush=True)
    positive = [r for r in all_results if r.realised > 0 and r.max_dd > 0]
    print_table("top by realised", sorted(positive, key=lambda r: (r.realised, r.median_day), reverse=True), args.top)
    print_table("top by median/day", sorted(positive, key=lambda r: (r.median_day, r.realised), reverse=True), args.top)
    print_table("top by realised/DD", sorted(positive, key=lambda r: (r.realised / max(r.max_dd, 1e-9), r.realised), reverse=True), args.top)


if __name__ == "__main__":
    main()
