from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError as exc:
    raise SystemExit("MetaTrader5 package missing. Run: pip install MetaTrader5") from exc


TIMEFRAMES = {
    "1m": mt5.TIMEFRAME_M1,
    "2m": mt5.TIMEFRAME_M2,
    "3m": mt5.TIMEFRAME_M3,
    "4m": mt5.TIMEFRAME_M4,
    "5m": mt5.TIMEFRAME_M5,
    "6m": mt5.TIMEFRAME_M6,
    "10m": mt5.TIMEFRAME_M10,
    "12m": mt5.TIMEFRAME_M12,
    "15m": mt5.TIMEFRAME_M15,
    "20m": mt5.TIMEFRAME_M20,
    "30m": mt5.TIMEFRAME_M30,
    "1h": mt5.TIMEFRAME_H1,
    "2h": mt5.TIMEFRAME_H2,
    "3h": mt5.TIMEFRAME_H3,
    "4h": mt5.TIMEFRAME_H4,
    "1d": mt5.TIMEFRAME_D1,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Dump broker OHLC bars from the open MT5 terminal.")
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--timeframe", default="1m", choices=sorted(TIMEFRAMES))
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--from", dest="start", default=None, help="UTC start, e.g. 2025-06-01")
    ap.add_argument("--to", default=None, help="UTC end, e.g. 2026-06-01")
    ap.add_argument("--out", default=None)
    ap.add_argument("--probe-only", action="store_true", help="print available range without writing CSV")
    return ap.parse_args()


def parse_dt(value: str | None, fallback: datetime) -> datetime:
    if not value:
        return fallback
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_mt5_rates_range(symbol: str, timeframe: str, start: datetime, end: datetime) -> tuple[pd.DataFrame, object]:
    if not mt5.initialize():
        raise SystemExit(f"mt5.initialize failed: {mt5.last_error()}")

    try:
        info = mt5.symbol_info(symbol)
        if info is None:
            raise SystemExit(f"symbol not found in MT5: {symbol}")
        if not info.visible and not mt5.symbol_select(symbol, True):
            raise SystemExit(f"could not select symbol: {symbol}")

        tf = TIMEFRAMES[timeframe]
        print(
            f"[mt5-ohlc] symbol={symbol} tf={timeframe} from={start.isoformat()} to={end.isoformat()} "
            f"digits={info.digits} point={info.point} spread={info.spread}",
            flush=True,
        )
        rates = mt5.copy_rates_range(symbol, tf, start, end)
        if rates is None:
            raise SystemExit(f"copy_rates_range failed: {mt5.last_error()}")
    finally:
        mt5.shutdown()

    df = pd.DataFrame(rates)
    if df.empty:
        return pd.DataFrame(columns=["time", "pair", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]), info

    out = pd.DataFrame({
        "time": pd.to_datetime(df["time"], unit="s", utc=True),
        "pair": symbol.upper(),
        "open": df["open"].astype(float),
        "high": df["high"].astype(float),
        "low": df["low"].astype(float),
        "close": df["close"].astype(float),
        "tick_volume": df["tick_volume"].astype(float),
        "spread": df["spread"].astype(float),
        "real_volume": df["real_volume"].astype(float),
    })
    return out.sort_values("time").reset_index(drop=True), info


def main() -> None:
    args = parse_args()
    end = parse_dt(args.to, datetime.now(timezone.utc))
    start = parse_dt(args.start, end - timedelta(days=args.days))
    df, _info = load_mt5_rates_range(args.symbol, args.timeframe, start, end)
    if df.empty:
        raise SystemExit("[mt5-ohlc] no bars returned")

    print(
        f"[mt5-ohlc] bars={len(df):,} range={df['time'].iloc[0]} -> {df['time'].iloc[-1]}",
        flush=True,
    )
    print(
        f"[mt5-ohlc] spread p50={df['spread'].quantile(0.5):.2f} "
        f"p90={df['spread'].quantile(0.9):.2f} p99={df['spread'].quantile(0.99):.2f}",
        flush=True,
    )
    if args.probe_only:
        return

    out = args.out
    if not out:
        safe_start = start.strftime("%Y%m%d")
        safe_end = end.strftime("%Y%m%d")
        out = os.path.join("data", "forex", f"mt5_ohlc_{args.symbol}_{args.timeframe}_{safe_start}_{safe_end}.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[mt5-ohlc] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
