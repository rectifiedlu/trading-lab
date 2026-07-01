"""Compare MT5 native candles against candles rebuilt from MT5 ticks.

Purpose:
    Determine whether a broker's MT5 candles are closest to bid, ask, mid, or
    last-price OHLC. This matters for synthetic seconds candles and SAR/ATR
    strategies where high/low differences change signals.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


def timeframe_value(mt5, tf: str):
    table = {
        "1m": mt5.TIMEFRAME_M1,
        "2m": mt5.TIMEFRAME_M2,
        "3m": mt5.TIMEFRAME_M3,
        "5m": mt5.TIMEFRAME_M5,
        "10m": mt5.TIMEFRAME_M10,
        "15m": mt5.TIMEFRAME_M15,
        "30m": mt5.TIMEFRAME_M30,
        "1h": mt5.TIMEFRAME_H1,
        "4h": mt5.TIMEFRAME_H4,
    }
    key = tf.lower().strip()
    if key not in table:
        raise SystemExit(f"unsupported native timeframe: {tf}")
    return table[key]


def timeframe_seconds(tf: str) -> int:
    key = tf.lower().strip()
    if key.endswith("s"):
        return int(float(key[:-1]))
    if key.endswith("m"):
        return int(float(key[:-1]) * 60)
    if key.endswith("h"):
        return int(float(key[:-1]) * 3600)
    raise SystemExit(f"unsupported timeframe: {tf}")


def parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    return pd.to_datetime(s, utc=True, format="mixed").to_pydatetime()


def rebuild_ohlc(ticks: pd.DataFrame, source: str, tf_sec: int) -> pd.DataFrame:
    if source == "bid":
        px = ticks["bid"]
    elif source == "ask":
        px = ticks["ask"]
    elif source == "mid":
        px = (ticks["bid"] + ticks["ask"]) / 2.0
    elif source == "last":
        if "last" not in ticks.columns:
            return pd.DataFrame()
        px = ticks["last"].replace(0, np.nan)
    else:
        raise ValueError(source)

    df = pd.DataFrame({
        "timestamp": ticks["timestamp"],
        "price": px,
    }).dropna()
    if df.empty:
        return df
    epoch = df["timestamp"].astype("int64") // 1_000_000_000
    bucket = (epoch // tf_sec) * tf_sec
    df["time"] = pd.to_datetime(bucket, unit="s", utc=True)
    out = df.groupby("time")["price"].ohlc()
    return out.reset_index()


def compare(native: pd.DataFrame, rebuilt: pd.DataFrame, source: str) -> dict:
    if rebuilt.empty:
        return {
            "source": source,
            "bars": 0,
            "mae_total": np.inf,
            "max_total": np.inf,
            "open_mae": np.inf,
            "high_mae": np.inf,
            "low_mae": np.inf,
            "close_mae": np.inf,
        }
    merged = native.merge(rebuilt, on="time", suffixes=("_mt5", "_rebuilt"))
    if merged.empty:
        return {
            "source": source,
            "bars": 0,
            "mae_total": np.inf,
            "max_total": np.inf,
            "open_mae": np.inf,
            "high_mae": np.inf,
            "low_mae": np.inf,
            "close_mae": np.inf,
        }
    diffs = {}
    total = np.zeros(len(merged), dtype=np.float64)
    max_total = 0.0
    for col in ("open", "high", "low", "close"):
        d = (merged[f"{col}_mt5"] - merged[f"{col}_rebuilt"]).abs().to_numpy(np.float64)
        diffs[f"{col}_mae"] = float(np.mean(d))
        total += d
        max_total = max(max_total, float(np.max(d)))
    return {
        "source": source,
        "bars": int(len(merged)),
        "mae_total": float(np.mean(total)),
        "max_total": max_total,
        **diffs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--timeframe", default="1m")
    ap.add_argument("--hours", type=int, default=6)
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--to", default=None)
    args = ap.parse_args()

    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise SystemExit(f"mt5.initialize failed: {mt5.last_error()}")
    if not mt5.symbol_select(args.symbol, True):
        raise SystemExit(f"symbol_select failed: {args.symbol}")

    end = parse_dt(args.to) or datetime.now(timezone.utc)
    start = parse_dt(args.start) or (end - timedelta(hours=args.hours))

    rates = mt5.copy_rates_range(args.symbol, timeframe_value(mt5, args.timeframe), start, end)
    ticks = mt5.copy_ticks_range(args.symbol, start, end, mt5.COPY_TICKS_ALL)
    info = mt5.symbol_info(args.symbol)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise SystemExit("no native candles returned")
    if ticks is None or len(ticks) == 0:
        raise SystemExit("no ticks returned")

    native = pd.DataFrame(rates)
    native["time"] = pd.to_datetime(native["time"], unit="s", utc=True)
    native = native[["time", "open", "high", "low", "close"]]

    tick_df = pd.DataFrame(ticks)
    tick_df["timestamp"] = pd.to_datetime(tick_df["time_msc"], unit="ms", utc=True)

    tf_sec = timeframe_seconds(args.timeframe)
    point = float(getattr(info, "point", 0.01) or 0.01) if info else 0.01

    print(
        f"[mt5-candle-compare] symbol={args.symbol} tf={args.timeframe} "
        f"from={start.isoformat()} to={end.isoformat()} "
        f"native_bars={len(native):,} ticks={len(tick_df):,} point={point:g}",
        flush=True,
    )

    rows = []
    for source in ("bid", "ask", "mid", "last"):
        rebuilt = rebuild_ohlc(tick_df, source, tf_sec)
        rows.append(compare(native, rebuilt, source))

    rows.sort(key=lambda r: r["mae_total"])
    print()
    print(" source bars  total_mae  max_err  open_mae high_mae low_mae close_mae   total_pts")
    print(" --------------------------------------------------------------------------------------")
    for r in rows:
        total_pts = r["mae_total"] / point if np.isfinite(r["mae_total"]) else np.inf
        print(
            f" {r['source']:<6} {r['bars']:>4} "
            f"{r['mae_total']:>10.5f} {r['max_total']:>8.5f} "
            f"{r['open_mae']:>9.5f} {r['high_mae']:>8.5f} "
            f"{r['low_mae']:>7.5f} {r['close_mae']:>9.5f} "
            f"{total_pts:>10.2f}",
            flush=True,
        )

    best = rows[0]
    print()
    print(
        f"[mt5-candle-compare] best={best['source']} "
        f"avg_total_error={best['mae_total']:.5f} "
        f"({best['mae_total'] / point:.2f} points)",
        flush=True,
    )


if __name__ == "__main__":
    main()
