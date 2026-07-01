from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError as exc:
    raise SystemExit("MetaTrader5 package missing. Run: pip install MetaTrader5") from exc


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Dump broker ticks from the open MT5 terminal.")
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--from", dest="start", default=None, help="UTC start, e.g. 2026-05-01")
    ap.add_argument("--to", default=None, help="UTC end, e.g. 2026-05-16")
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def parse_dt(value: str | None, fallback: datetime) -> datetime:
    if not value:
        return fallback
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_mt5_ticks_range(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Load bid/ask ticks directly from the currently configured MT5 terminal."""
    if not mt5.initialize():
        raise SystemExit(f"mt5.initialize failed: {mt5.last_error()}")

    try:
        info = mt5.symbol_info(symbol)
        if info is None:
            raise SystemExit(f"symbol not found in MT5: {symbol}")
        if not info.visible and not mt5.symbol_select(symbol, True):
            raise SystemExit(f"could not select symbol: {symbol}")

        print(
            f"[mt5-dump] symbol={symbol} from={start.isoformat()} to={end.isoformat()} "
            f"digits={info.digits} point={info.point} spread={info.spread}",
            flush=True,
        )
        ticks = mt5.copy_ticks_range(symbol, start, end, mt5.COPY_TICKS_ALL)
        if ticks is None:
            raise SystemExit(f"copy_ticks_range failed: {mt5.last_error()}")
    finally:
        mt5.shutdown()

    df = pd.DataFrame(ticks)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "pair", "bid", "ask"])
    if "time_msc" not in df.columns:
        raise SystemExit(f"[mt5-dump] unexpected columns: {list(df.columns)}")

    df = df[(df["bid"] > 0) & (df["ask"] > 0) & (df["ask"] >= df["bid"])].copy()
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "pair", "bid", "ask"])

    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df["time_msc"], unit="ms", utc=True),
        "bid": df["bid"].astype(float),
        "ask": df["ask"].astype(float),
        "pair": symbol.upper(),
    })
    return out[["timestamp", "pair", "bid", "ask"]].sort_values("timestamp").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    end = parse_dt(args.to, datetime.now(timezone.utc))
    start = parse_dt(args.start, end - timedelta(days=args.days))

    out_df = load_mt5_ticks_range(args.symbol, start, end)
    if out_df.empty:
        raise SystemExit("[mt5-dump] no bid/ask ticks returned")

    out = args.out
    if not out:
        safe_start = start.strftime("%Y%m%d")
        safe_end = end.strftime("%Y%m%d")
        out = os.path.join("data", "forex", f"mt5_ticks_{args.symbol}_{safe_start}_{safe_end}.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    out_df.to_csv(out, index=False)

    spread = (out_df["ask"] - out_df["bid"]).describe(percentiles=[0.5, 0.9, 0.95, 0.99])
    print(f"[mt5-dump] wrote {out} ticks={len(out_df):,}", flush=True)
    print(
        "[mt5-dump] spread "
        f"p50={spread['50%']:.5f} p90={spread['90%']:.5f} "
        f"p95={spread['95%']:.5f} p99={spread['99%']:.5f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
