"""Forex tick backtester with pluggable data sources.

Single-file workflow:
  - load local bid/ask ticks, or download/cache Dukascopy hourly tick files
  - sweep trailing-entry/trailing-exit params
  - execute on the next tick's bid/ask to avoid same-tick cheating

Examples:
  python forex_backtest.py --source dukascopy --pairs EURUSD GBPUSD --from 2026-04-20 --to 2026-04-21
  python forex_backtest.py --source local --csv data/forex/EURUSD_ticks.csv --pair EURUSD
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import lzma
import os
import struct
import sys
import time
import urllib.error
import urllib.request
import socket
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FOREX_DIR = os.path.join(BASE_DIR, "data", "forex")
DUKAS_CACHE_DIR = os.path.join(FOREX_DIR, "dukascopy")
DEFAULT_MT5_TICK_CSV = os.path.join(FOREX_DIR, "mt5_ticks_XAUUSD_20260416_20260516.csv")

DEFAULT_PAIRS = [
    # USD majors first: deepest books, usually smoothest movement/spreads.
    "XAUUSD"
]

# Basis points are not percentages:
#   1 bps = 0.01%, 100 bps = 1%, 10,000 bps = 100%.
# With $10 at 100x leverage, 1 bps is roughly $0.10 before spread/fees.
DEFAULT_ENTRY_BPS = []
for i in range(1,20):
    DEFAULT_ENTRY_BPS.append(i)
DEFAULT_BUY_BPS = list(DEFAULT_ENTRY_BPS)
DEFAULT_SELL_BPS = list(DEFAULT_ENTRY_BPS)
DEFAULT_LOSS_BPS = [10.0, 20.0, 30.0, 40.0]
DEFAULT_ARM_MULT = [2.0]
DEFAULT_COMMISSION_PER_MILLION = 30.0
DUKAS_DOWNLOAD_RETRIES = 3
DUKAS_DOWNLOAD_TIMEOUT_SEC = 15

def _simulate_pair_combos(args):
    pair, ticks, combos, amount, leverage, side, commission_per_million = args
    return [
        simulate_pair(pair, ticks, buy, sell, loss, arm, amount, leverage, side,
                      commission_per_million)
        for buy, sell, loss, arm in combos
    ]


def _best_with_min_trades(results: list[SimResult], min_trades: int) -> SimResult | None:
    eligible = [r for r in results if r.trades >= min_trades]
    if not eligible:
        return None
    return max(eligible, key=lambda r: (
        r.pnl + r.open_unrealized,
        r.pnl,
        r.profit_factor,
        r.trades,
    ))


@dataclass
class SimResult:
    pair: str
    buy_bps: float
    sell_bps: float
    loss_bps: float
    arm_mult: float
    pnl: float
    trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    trail_exits: int = 0
    loss_stop_exits: int = 0
    open_side: int = 0
    open_unrealized: float = 0.0
    open_pnl_bps: float = 0.0
    open_peak_bps: float = 0.0
    open_adverse_bps: float = 0.0
    open_ticks: int = 0


def _parse_datetime(s: str) -> dt.datetime:
    try:
        d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        d = dt.datetime.strptime(s, "%Y-%m-%d")
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d.astimezone(dt.timezone.utc)


def _default_date_window(days: int) -> tuple[str, str]:
    today = dt.datetime.now(dt.timezone.utc).date()
    end = today
    start = end - dt.timedelta(days=max(1, int(days)))
    return start.isoformat(), end.isoformat()


def _default_hour_window(hours: int) -> tuple[str, str]:
    end = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - dt.timedelta(hours=max(1, int(hours)))
    return start.isoformat(), end.isoformat()


def _hour_range(start: dt.datetime, end: dt.datetime) -> Iterable[dt.datetime]:
    cur = start.replace(minute=0, second=0, microsecond=0)
    while cur < end:
        yield cur
        cur += dt.timedelta(hours=1)


def _dukascopy_scale(pair: str) -> float:
    if pair.upper() == "XAUUSD":
        return 1000.0
    return 1000.0 if "JPY" in pair.upper() else 100000.0


def _dukascopy_url(pair: str, hour: dt.datetime) -> str:
    # Dukascopy months are zero-based in the datafeed path.
    return (
        f"https://datafeed.dukascopy.com/datafeed/{pair.upper()}/"
        f"{hour.year}/{hour.month - 1:02d}/{hour.day:02d}/{hour.hour:02d}h_ticks.bi5"
    )


def _dukascopy_cache_path(pair: str, hour: dt.datetime) -> str:
    return os.path.join(
        DUKAS_CACHE_DIR,
        pair.upper(),
        f"{hour.year}",
        f"{hour.month:02d}",
        f"{hour.day:02d}",
        f"{hour.hour:02d}h_ticks.bi5",
    )


def _download_dukascopy_hour(pair: str, hour: dt.datetime) -> bytes:
    path = _dukascopy_cache_path(pair, hour)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    req = urllib.request.Request(
        _dukascopy_url(pair, hour),
        headers={"User-Agent": "Mozilla/5.0"},
    )
    for attempt in range(1, DUKAS_DOWNLOAD_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=DUKAS_DOWNLOAD_TIMEOUT_SEC) as r:
                blob = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return b""
            if attempt >= DUKAS_DOWNLOAD_RETRIES:
                print(f"[load] {pair.upper()} {hour:%Y-%m-%d %H}: "
                      f"http {e.code}; skip", flush=True)
                return b""
        except (TimeoutError, socket.timeout, urllib.error.URLError, OSError) as e:
            if attempt >= DUKAS_DOWNLOAD_RETRIES:
                print(f"[load] {pair.upper()} {hour:%Y-%m-%d %H}: "
                      f"{type(e).__name__}; skip", flush=True)
                return b""
            time.sleep(0.5 * attempt)

    with open(path, "wb") as f:
        f.write(blob)
    return blob


def load_dukascopy_ticks(pair: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    t0 = time.time()
    rows = []
    scale = _dukascopy_scale(pair)
    hours = list(_hour_range(start, end))
    print(f"[load] {pair.upper()} Dukascopy {start.date()}->{end.date()} "
          f"hours={len(hours)}", flush=True)
    for idx, hour in enumerate(hours, 1):
        blob = _download_dukascopy_hour(pair, hour)
        if not blob:
            continue
        try:
            raw = lzma.decompress(blob)
        except lzma.LZMAError:
            continue
        # Each record: ms-from-hour, ask-int, bid-int, ask-vol-float, bid-vol-float.
        for ms, ask_i, bid_i, ask_vol, bid_vol in struct.iter_unpack(">IIIff", raw):
            ts = hour + dt.timedelta(milliseconds=int(ms))
            if start <= ts < end:
                rows.append((ts, pair.upper(), bid_i / scale, ask_i / scale, bid_vol, ask_vol))

        if idx % 12 == 0 or idx == len(hours):
            print(f"[load] {pair.upper()} {idx}/{len(hours)}h "
                  f"ticks={len(rows):,}", flush=True)

    df = pd.DataFrame(rows, columns=["timestamp", "pair", "bid", "ask", "bid_vol", "ask_vol"])
    print(f"[load] {pair.upper()} done ticks={len(df):,} "
          f"{time.time() - t0:.1f}s", flush=True)
    return df


def load_local_ticks(
    csv_path: str,
    pair: str | None = None,
    start: str | dt.datetime | pd.Timestamp | None = None,
    end: str | dt.datetime | pd.Timestamp | None = None,
    pairs: Iterable[str] | None = None,
) -> pd.DataFrame:
    print(f"[load] local CSV {csv_path}", flush=True)
    t0 = time.time()

    start_ts = pd.to_datetime(start, utc=True, format="mixed") if start is not None else None
    end_ts = pd.to_datetime(end, utc=True, format="mixed") if end is not None else None
    pair_set = {p.upper() for p in pairs} if pairs else set()
    if pair:
        pair_set.add(pair.upper())

    header = pd.read_csv(csv_path, nrows=0)
    cols = {c.lower(): c for c in header.columns}
    required = {"timestamp", "bid", "ask"}
    missing = required - set(cols)
    if missing:
        raise ValueError(f"local CSV missing columns: {sorted(missing)}")

    usecols = [cols["timestamp"], cols["bid"], cols["ask"]]
    has_pair_col = "pair" in cols
    if has_pair_col:
        usecols.append(cols["pair"])

    frames = []
    inferred_pair = (
        pair.upper() if pair
        else os.path.splitext(os.path.basename(csv_path))[0].split("_")[0].upper()
    )
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=1_000_000):
        ts = pd.to_datetime(chunk[cols["timestamp"]], utc=True, format="mixed", errors="coerce")
        mask = ts.notna()
        if start_ts is not None:
            mask &= ts >= start_ts
        if end_ts is not None:
            mask &= ts < end_ts
        if not mask.any():
            continue

        chunk = chunk.loc[mask].copy()
        out = pd.DataFrame({
            "timestamp": ts.loc[mask],
            "bid": pd.to_numeric(chunk[cols["bid"]], errors="coerce"),
            "ask": pd.to_numeric(chunk[cols["ask"]], errors="coerce"),
        })
        if has_pair_col:
            out["pair"] = chunk[cols["pair"]].astype(str).str.upper()
        else:
            out["pair"] = inferred_pair
        if pair_set:
            out = out[out["pair"].isin(pair_set)]
        out = out.dropna(subset=["timestamp", "bid", "ask"])
        out = out[(out["bid"] > 0) & (out["ask"] > 0) & (out["ask"] >= out["bid"])]
        if not out.empty:
            frames.append(out[["timestamp", "pair", "bid", "ask"]])

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["timestamp", "pair", "bid", "ask"]
    )
    if not out.empty:
        out = out.sort_values(["pair", "timestamp"]).reset_index(drop=True)
    print(f"[load] local done ticks={len(out):,} pairs={out['pair'].nunique()} "
          f"{time.time() - t0:.1f}s", flush=True)
    return out


def simulate_pair(
    pair: str,
    ticks: pd.DataFrame,
    buy_bps: float,
    sell_bps: float,
    loss_bps: float,
    arm_mult: float,
    margin_usd: float,
    leverage: float,
    side_mode: str,
    commission_per_million: float,
) -> SimResult:
    if len(ticks) < 3:
        return SimResult(pair, buy_bps, sell_bps, loss_bps, arm_mult, 0.0, 0, 0, 0, 0.0, 0.0, 0.0)

    bid = ticks["bid"].to_numpy(np.float64)
    ask = ticks["ask"].to_numpy(np.float64)
    # Use bid as the strategy/reference stream; MT5 native XAUUSD candles match bid OHLC.
    mid = bid

    cash = 0.0
    equity_peak = 0.0
    max_dd = 0.0
    trades = wins = losses = 0
    trail_exits = loss_stop_exits = 0
    gross_win = gross_loss = 0.0

    pos = 0  # 1 long, -1 short
    trail_armed = False
    entry = peak = trough = 0.0
    entry_i = -1
    units = 0.0
    ref_low = mid[0]
    ref_high = mid[0]

    def calc_units(px: float) -> float:
        if px <= 0:
            return 0.0
        return (margin_usd * leverage) / px

    def commission(px: float, qty: float) -> float:
        notional = abs(qty * px)
        return notional / 1_000_000.0 * commission_per_million

    for i in range(1, len(mid) - 1):
        px = mid[i]
        next_bid = bid[i + 1]
        next_ask = ask[i + 1]

        if pos == 0:
            if px < ref_low:
                ref_low = px
            if px > ref_high:
                ref_high = px

            long_rise_bps = (px / ref_low - 1.0) * 10000.0 if ref_low > 0 else 0.0
            short_drop_bps = (1.0 - px / ref_high) * 10000.0 if ref_high > 0 else 0.0

            if side_mode in ("long", "both") and long_rise_bps >= buy_bps:
                pos = 1
                entry = next_ask
                units = calc_units(entry)
                peak = entry
                trough = entry
                trail_armed = False
                entry_i = i + 1
                cash -= commission(entry, units)
                ref_low = px
                ref_high = px
                continue
            if side_mode in ("short", "both") and short_drop_bps >= buy_bps:
                pos = -1
                entry = next_bid
                units = calc_units(entry)
                peak = entry
                trough = entry
                trail_armed = False
                entry_i = i + 1
                cash -= commission(entry, units)
                ref_low = px
                ref_high = px
                continue

        elif pos == 1:
            if px > peak:
                peak = px
            pnl_bps = (px / entry - 1.0) * 10000.0 if entry > 0 else 0.0
            if pnl_bps >= sell_bps * arm_mult:
                trail_armed = True
            drawdown_bps = (1.0 - px / peak) * 10000.0 if peak > 0 else 0.0
            hit_trail = trail_armed and drawdown_bps >= sell_bps
            hit_loss = not trail_armed and loss_bps > 0 and drawdown_bps >= loss_bps
            if hit_trail or hit_loss:
                exit_px = next_bid
                pnl = (exit_px - entry) * units - commission(exit_px, units)
                cash += pnl
                trades += 1
                trail_exits += 1 if hit_trail else 0
                loss_stop_exits += 1 if hit_loss else 0
                wins += 1 if pnl > 0 else 0
                losses += 1 if pnl <= 0 else 0
                gross_win += max(0.0, pnl)
                gross_loss += max(0.0, -pnl)
                pos = 0
                trail_armed = False
                units = 0.0
                entry_i = -1
                ref_low = px
                ref_high = px

        else:
            if px < trough:
                trough = px
            pnl_bps = (entry / px - 1.0) * 10000.0 if px > 0 else 0.0
            if pnl_bps >= sell_bps * arm_mult:
                trail_armed = True
            drawdown_bps = (px / trough - 1.0) * 10000.0 if trough > 0 else 0.0
            hit_trail = trail_armed and drawdown_bps >= sell_bps
            hit_loss = not trail_armed and loss_bps > 0 and drawdown_bps >= loss_bps
            if hit_trail or hit_loss:
                exit_px = next_ask
                pnl = (entry - exit_px) * units - commission(exit_px, units)
                cash += pnl
                trades += 1
                trail_exits += 1 if hit_trail else 0
                loss_stop_exits += 1 if hit_loss else 0
                wins += 1 if pnl > 0 else 0
                losses += 1 if pnl <= 0 else 0
                gross_win += max(0.0, pnl)
                gross_loss += max(0.0, -pnl)
                pos = 0
                trail_armed = False
                units = 0.0
                entry_i = -1
                ref_low = px
                ref_high = px

        equity_peak = max(equity_peak, cash)
        max_dd = max(max_dd, equity_peak - cash)

    win_rate = wins / trades * 100.0 if trades else 0.0
    profit_factor = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
    open_unrealized = 0.0
    open_pnl_bps = 0.0
    open_peak_bps = 0.0
    open_adverse_bps = 0.0
    open_ticks = 0
    if pos == 1 and units > 0:
        last_bid = bid[-1]
        last_mid = mid[-1]
        open_unrealized = (last_bid - entry) * units
        open_pnl_bps = (last_mid / entry - 1.0) * 10000.0 if entry > 0 else 0.0
        open_peak_bps = (peak / entry - 1.0) * 10000.0 if entry > 0 else 0.0
        open_adverse_bps = max(0.0, -open_pnl_bps)
        open_ticks = max(0, len(mid) - 1 - entry_i)
    elif pos == -1 and units > 0:
        last_ask = ask[-1]
        last_mid = mid[-1]
        open_unrealized = (entry - last_ask) * units
        open_pnl_bps = (entry / last_mid - 1.0) * 10000.0 if last_mid > 0 else 0.0
        open_peak_bps = (entry / trough - 1.0) * 10000.0 if trough > 0 else 0.0
        open_adverse_bps = max(0.0, -open_pnl_bps)
        open_ticks = max(0, len(mid) - 1 - entry_i)

    return SimResult(
        pair, buy_bps, sell_bps, loss_bps, arm_mult, cash, trades, wins, losses,
        win_rate, profit_factor, max_dd, trail_exits, loss_stop_exits, pos,
        open_unrealized, open_pnl_bps, open_peak_bps, open_adverse_bps, open_ticks,
    )


def _parse_num_list(s: str | None, default: list[float]) -> list[float]:
    if not s:
        return [float(x) for x in default]
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _chunked(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def movement_stats(ticks: pd.DataFrame, atr_window_sec: int = 3600) -> pd.DataFrame:
    rows = []
    for pair, g in ticks.groupby("pair", sort=False):
        g = g.sort_values("timestamp")
        bid = g["bid"].to_numpy(np.float64)
        ask = g["ask"].to_numpy(np.float64)
        if len(g) < 3:
            continue
        # Use bid for movement stats so ranges line up with MT5 native candles.
        mid = bid
        spread_bps = (ask - bid) / mid * 10000.0
        tick_abs_bps = np.abs(np.diff(mid) / mid[:-1] * 10000.0)
        tick_abs_bps = tick_abs_bps[np.isfinite(tick_abs_bps)]
        tmp = pd.DataFrame({"timestamp": g["timestamp"].to_numpy(), "mid": mid})
        tmp = tmp.set_index("timestamp")
        ohlc = tmp["mid"].resample(f"{int(atr_window_sec)}s").ohlc().dropna()
        if ohlc.empty:
            continue
        range_bps = (ohlc["high"] - ohlc["low"]) / ohlc["close"] * 10000.0
        body_bps = (ohlc["close"] - ohlc["open"]).abs() / ohlc["open"] * 10000.0
        half_range_bps = range_bps / 2.0
        spread_p95 = float(np.nanpercentile(spread_bps, 95))
        tick_p95 = float(np.nanpercentile(tick_abs_bps, 95)) if len(tick_abs_bps) else 0.0
        # A practical lower bound for trigger grids: stay above bad-normal
        # spread and several ticks of micro jitter. Anything below this tends
        # to churn rather than capture a directional leg.
        noise_floor = max(spread_p95 * 1.5, tick_p95 * 3.0)
        half_atr_p50 = float(np.nanpercentile(half_range_bps, 50))
        half_atr_p90 = float(np.nanpercentile(half_range_bps, 90))
        rows.append({
            "pair": pair,
            "ticks": int(len(g)),
            "spread_p50": float(np.nanpercentile(spread_bps, 50)),
            "spread_p95": spread_p95,
            "tick_abs_p50": float(np.nanpercentile(tick_abs_bps, 50)) if len(tick_abs_bps) else 0.0,
            "tick_abs_p75": float(np.nanpercentile(tick_abs_bps, 75)) if len(tick_abs_bps) else 0.0,
            "tick_abs_p95": tick_p95,
            "noise_floor": noise_floor,
            "atr_p50": float(np.nanpercentile(range_bps, 50)),
            "atr_p90": float(np.nanpercentile(range_bps, 90)),
            "atr_p99": float(np.nanpercentile(range_bps, 99)),
            "half_atr_p50": half_atr_p50,
            "half_atr_p90": half_atr_p90,
            "body_p50": float(np.nanpercentile(body_bps, 50)),
            "body_p90": float(np.nanpercentile(body_bps, 90)),
            "grid_min": max(noise_floor, half_atr_p50 * 0.15),
            "grid_max": max(half_atr_p90, noise_floor * 3.0),
            "windows": int(len(ohlc)),
        })
    return pd.DataFrame(rows)


def load_ticks(args) -> pd.DataFrame:
    if args.source == "mt5":
        from mt5_tick_dump import load_mt5_ticks_range

        start = _parse_datetime(args.start)
        end = _parse_datetime(args.to)
        frames = [
            frame for pair in args.pairs
            for frame in [load_mt5_ticks_range(pair, start, end)]
            if not frame.empty
        ]
        ticks = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if ticks.empty:
            return ticks
        print(f"[load] MT5 direct done ticks={len(ticks):,} pairs={ticks['pair'].nunique()}", flush=True)
        return ticks.sort_values(["pair", "timestamp"]).reset_index(drop=True)
    if args.source == "local":
        return load_local_ticks(
            args.csv,
            args.pair,
            getattr(args, "start", None),
            getattr(args, "to", None),
            getattr(args, "pairs", None),
        )
    else:
        start = _parse_datetime(args.start)
        end = _parse_datetime(args.to)
        frames = [
            frame for pair in args.pairs
            for frame in [load_dukascopy_ticks(pair, start, end)]
            if not frame.empty
        ]
        ticks = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if ticks.empty:
            return ticks
        print(f"[load] sorting combined ticks={len(ticks):,}", flush=True)
        return ticks.sort_values(["pair", "timestamp"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["mt5", "local", "dukascopy"], default="mt5")
    ap.add_argument("--csv", default=DEFAULT_MT5_TICK_CSV,
                    help="local tick CSV with timestamp,bid,ask[,pair]")
    ap.add_argument("--pair", help="pair for local CSV if no pair column exists")
    ap.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    ap.add_argument("--hours", type=int, default=24,
                    help="default lookback in full UTC hours when --from/--to are omitted")
    ap.add_argument("--days", type=int, default=None,
                    help="optional full UTC day lookback when --from/--to are omitted")
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--to", default=None)
    ap.add_argument("--buy-bps", default=None,
                    help="comma list in basis points; 2 means 2 bps = 0.02%")
    ap.add_argument("--sell-bps", default=None,
                    help="trailing drawdown list in basis points")
    ap.add_argument("--loss-bps", default=None,
                    help="hard loss stop list in basis points; 0 disables")
    ap.add_argument("--arm-mult", default=None,
                    help="trail arms after sell_bps * arm_mult profit")
    ap.add_argument("--combo-mode", choices=["sell_lt_buy", "all"],
                    default="all",
                    help="parameter relationship filter")
    ap.add_argument("--amount", type=float, default=10.0,
                    help="margin/equity used per trade in USD")
    ap.add_argument("--leverage", type=float, default=100.0,
                    help="notional multiplier; notional = amount * leverage")
    ap.add_argument("--side", choices=["long", "short", "both"], default="long")
    ap.add_argument("--commission-per-million", type=float,
                    default=DEFAULT_COMMISSION_PER_MILLION)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count()) - 3))
    ap.add_argument("--chunk-size", type=int, default=250)
    ap.add_argument("--min-trades", type=int, default=0,
                    help="discard result rows below this trade count")
    ap.add_argument("--rank-min-trades", type=int, default=0,
                    help="rank results with at least this many trades before low-sample rows")
    ap.add_argument("--stats-only", action="store_true",
                    help="load ticks and print ATR/local movement stats without sweeping")
    ap.add_argument("--atr-window-sec", type=int, default=3600,
                    help="window size for --stats-only ATR stats")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(FOREX_DIR, "forex_backtest_results.csv"))
    args = ap.parse_args()

    if args.source == "local" and not args.csv:
        raise SystemExit("--csv is required for --source local")
    if args.start is None or args.to is None:
        if args.days is not None:
            default_start, default_end = _default_date_window(args.days)
        else:
            default_start, default_end = _default_hour_window(args.hours)
        args.start = args.start or default_start
        args.to = args.to or default_end

    os.makedirs(FOREX_DIR, exist_ok=True)
    print(f"[forex] source={args.source} from={args.start} to={args.to} "
          f"amount=${args.amount:g} lev={args.leverage:g}x "
          f"commission=${args.commission_per_million:g}/mm side={args.side} "
          f"rank_min_trades={args.rank_min_trades} combo_mode={args.combo_mode}",
          flush=True)
    run_t0 = time.time()
    ticks = load_ticks(args)
    if ticks.empty:
        raise SystemExit("no ticks loaded")

    if args.stats_only:
        stats = movement_stats(ticks, args.atr_window_sec)
        if not stats.empty:
            print(f"[stats] ATR/local movement in bps "
                  f"window={args.atr_window_sec}s:", flush=True)
            print(stats.sort_values("atr_p90", ascending=False).to_string(
                index=False,
                formatters={
                    "spread_p50": "{:.3f}".format,
                    "spread_p95": "{:.3f}".format,
                    "tick_abs_p50": "{:.3f}".format,
                    "tick_abs_p75": "{:.3f}".format,
                    "tick_abs_p95": "{:.3f}".format,
                    "noise_floor": "{:.2f}".format,
                    "atr_p50": "{:.2f}".format,
                    "atr_p90": "{:.2f}".format,
                    "atr_p99": "{:.2f}".format,
                    "half_atr_p50": "{:.2f}".format,
                    "half_atr_p90": "{:.2f}".format,
                    "body_p50": "{:.2f}".format,
                    "body_p90": "{:.2f}".format,
                    "grid_min": "{:.2f}".format,
                    "grid_max": "{:.2f}".format,
                },
            ), flush=True)
        return

    buy_values = _parse_num_list(args.buy_bps, DEFAULT_BUY_BPS)
    sell_values = _parse_num_list(args.sell_bps, DEFAULT_SELL_BPS)
    loss_values = _parse_num_list(args.loss_bps, DEFAULT_LOSS_BPS)
    arm_values = _parse_num_list(args.arm_mult, DEFAULT_ARM_MULT)

    results = []
    combos_per_pair = len(buy_values) * len(sell_values) * len(loss_values) * len(arm_values)
    print(f"[sweep] pairs={ticks['pair'].nunique()} combos_per_pair={combos_per_pair:,} "
          f"total={ticks['pair'].nunique() * combos_per_pair:,} "
          f"workers={args.workers} chunk={args.chunk_size}", flush=True)

    pair_ticks = {
        pair: g.sort_values("timestamp").reset_index(drop=True)
        for pair, g in ticks.groupby("pair", sort=False)
    }
    combos = [
        (buy, sell, loss, arm)
        for buy in buy_values
        for sell in sell_values
        for loss in loss_values
        for arm in arm_values
        if args.combo_mode == "all" or sell < buy
    ]
    if not combos:
        raise SystemExit("no param combos after combo-mode filter")
    futures = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for pair, g in pair_ticks.items():
            print(f"[sweep] {pair} ticks={len(g):,} combos={len(combos):,}",
                  flush=True)
            fut = pool.submit(
                _simulate_pair_combos,
                (pair, g, combos, args.amount, args.leverage, args.side,
                 args.commission_per_million),
            )
            futures[fut] = pair

        last_log = time.time()
        completed = 0
        total_pairs = len(futures)
        for fut in as_completed(futures):
            pair = futures[fut]
            pair_results = fut.result()
            results.extend(pair_results)
            completed += 1
            best = _best_with_min_trades(pair_results, args.min_trades)
            if best:
                print(f"[sweep] {pair} done best total={best.pnl + best.open_unrealized:.5f} "
                      f"realised={best.pnl:.5f} open={best.open_unrealized:+.5f} "
                      f"buy={best.buy_bps:g} sell={best.sell_bps:g} "
                      f"loss={best.loss_bps:g} arm={best.arm_mult:g} "
                      f"trades={best.trades} "
                      f"(min_trades={args.min_trades})",
                      flush=True)
            else:
                print(f"[sweep] {pair} done no result with "
                      f"trades>={args.min_trades}", flush=True)
            now = time.time()
            if now - last_log >= 5.0 or completed == total_pairs:
                last_log = now
                print(f"[sweep] progress pairs={completed:,}/{total_pairs:,} "
                      f"results={len(results):,}", flush=True)

    filtered_results = [r for r in results if r.trades >= args.min_trades]
    if not filtered_results:
        filtered_results = results
        print(f"[forex] WARN no rows had trades >= {args.min_trades}; "
              f"ranking all rows", flush=True)

    filtered_results.sort(
        key=lambda r: (
            r.trades >= args.rank_min_trades,
            r.pnl + r.open_unrealized,
            r.pnl,
            r.profit_factor,
            r.trades,
        ),
        reverse=True,
    )
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair", "buy_bps", "sell_bps", "loss_bps", "arm_mult", "realised_pnl",
                    "open_unrealized", "total_pnl", "trades",
                    "wins", "losses", "trail_exits", "loss_stop_exits",
                    "win_rate", "profit_factor", "max_drawdown",
                    "open_side", "open_pnl_bps", "open_peak_bps",
                    "open_adverse_bps", "open_ticks"])
        for r in filtered_results:
            w.writerow([r.pair, r.buy_bps, r.sell_bps, r.loss_bps, r.arm_mult, round(r.pnl, 6),
                        round(r.open_unrealized, 6), round(r.pnl + r.open_unrealized, 6),
                        r.trades, r.wins, r.losses, r.trail_exits, r.loss_stop_exits,
                        round(r.win_rate, 2),
                        round(r.profit_factor, 4), round(r.max_drawdown, 6),
                        r.open_side, round(r.open_pnl_bps, 4),
                        round(r.open_peak_bps, 4), round(r.open_adverse_bps, 4),
                        r.open_ticks])

    print(f"[forex] ticks={len(ticks):,} pairs={ticks['pair'].nunique()} "
          f"results={len(results):,} filtered={len(filtered_results):,} "
          f"min_trades={args.min_trades} elapsed={time.time() - run_t0:.1f}s")
    print(f"[forex] wrote {args.out}")
    for r in filtered_results[:args.top]:
        print(
            f"{r.pair:<7} buy={r.buy_bps:>5g}bps sell={r.sell_bps:>5g}bps "
            f"loss={r.loss_bps:>5g}bps arm={r.arm_mult:>4g} "
            f"total={r.pnl + r.open_unrealized:>10.5f} "
            f"realised={r.pnl:>10.5f} "
            f"trades={r.trades:>4} trail={r.trail_exits:>4} stop={r.loss_stop_exits:>4} "
            f"wr={r.win_rate:>5.1f}% pf={r.profit_factor:>6.2f} dd={r.max_drawdown:.5f} "
            f"open={r.open_side} u={r.open_unrealized:+.4f} obps={r.open_pnl_bps:+.2f}"
        )


if __name__ == "__main__":
    main()
