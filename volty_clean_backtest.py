from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

from forex_backtest import DEFAULT_MT5_TICK_CSV
from mt5_tick_dump import load_mt5_ticks_range


@dataclass
class Candle:
    bucket: int
    open: float
    high: float
    low: float
    close: float


@dataclass
class Trade:
    side: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry: float
    exit: float
    pnl: float
    reason: str
    equity: float


def parse_dt(value: str | None, fallback: pd.Timestamp) -> pd.Timestamp:
    if not value:
        return fallback
    ts = pd.Timestamp(datetime.fromisoformat(value.replace("Z", "+00:00")))
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def timeframe_seconds(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("s"):
        return int(float(tf[:-1]))
    if tf.endswith("m"):
        return int(float(tf[:-1]) * 60)
    if tf.endswith("h"):
        return int(float(tf[:-1]) * 3600)
    raise ValueError(f"unsupported timeframe: {tf}")


def iter_ticks(path: str, start: pd.Timestamp, end: pd.Timestamp, chunksize: int):
    usecols = ["timestamp", "bid", "ask"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        ts = pd.to_datetime(chunk["timestamp"], utc=True, format="mixed")
        mask = (ts >= start) & (ts < end)
        if not mask.any():
            continue
        part = chunk.loc[mask].copy()
        part["timestamp"] = ts.loc[mask]
        part["bid"] = pd.to_numeric(part["bid"], errors="coerce")
        part["ask"] = pd.to_numeric(part["ask"], errors="coerce")
        part = part.dropna(subset=["timestamp", "bid", "ask"])
        part = part[(part["bid"] > 0) & (part["ask"] >= part["bid"])]
        for row in part.itertuples(index=False):
            yield row.timestamp, float(row.bid), float(row.ask)


def iter_mt5_ticks(symbol: str, start: pd.Timestamp, end: pd.Timestamp):
    df = load_mt5_ticks_range(symbol, start.to_pydatetime(), end.to_pydatetime())
    for row in df.itertuples(index=False):
        yield row.timestamp, float(row.bid), float(row.ask)


def true_range(c: Candle, prev_close: float) -> float:
    return max(c.high - c.low, abs(c.high - prev_close), abs(c.low - prev_close))


def units_for_margin(margin: float, leverage: float, px: float) -> float:
    return margin * leverage / px


def commission(px: float, units: float, commission_per_million: float) -> float:
    return abs(px * units) / 1_000_000.0 * commission_per_million


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Clean Volty Expan Close backtest on MT5 tick CSV.")
    ap.add_argument("--source", choices=["mt5", "local"], default="mt5")
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--csv", default=DEFAULT_MT5_TICK_CSV)
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--to", default=None)
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--timeframe", default="1m")
    ap.add_argument("--length", type=int, default=5)
    ap.add_argument("--atr-mult", type=float, default=0.75)
    ap.add_argument("--hold-close-mult", type=float, default=None,
                    help="ATR multiplier for close-only exits while holding. Default = atr-mult.")
    ap.add_argument("--min-atr-points", default="0",
                    help="comma list; 0 disables. Uses point size.")
    ap.add_argument("--point-size", type=float, default=0.01)
    ap.add_argument("--candle-price", default="bid",
                    help="comma list: bid,mid,ask. Default bid because MT5 XAUUSD candles match bid OHLC.")
    ap.add_argument("--execution", choices=["tick", "candle"], default="tick")
    ap.add_argument("--amount", type=float, default=50.0)
    ap.add_argument("--leverage", type=float, default=100.0)
    ap.add_argument("--commission-per-million", type=float, default=30.0)
    ap.add_argument("--compound", action="store_true")
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument("--trades-out", default=None)
    return ap.parse_args()


def tick_price(bid: float, ask: float, mode: str) -> float:
    if mode == "bid":
        return bid
    if mode == "ask":
        return ask
    return (bid + ask) / 2.0


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def run_one(
    args: argparse.Namespace,
    start: pd.Timestamp,
    end: pd.Timestamp,
    mode: str,
    min_atr_points: float,
) -> None:
    tf_sec = timeframe_seconds(args.timeframe)
    candles: list[Candle] = []
    tr_values: list[float] = []
    trades: list[Trade] = []
    cur: Candle | None = None
    prev_close: float | None = None
    upper = lower = None
    hold_upper = hold_lower = None
    level_atr_pass = False

    pos = 0
    entry = units = 0.0
    cash = args.amount
    equity_peak = args.amount
    max_dd = 0.0
    ticks = 0

    def close_trade(ts, px: float, reason: str) -> None:
        nonlocal pos, entry, units, cash, equity_peak, max_dd
        side = "long" if pos == 1 else "short"
        if pos == 1:
            pnl = (px - entry) * units - commission(px, units, args.commission_per_million)
        else:
            pnl = (entry - px) * units - commission(px, units, args.commission_per_million)
        cash += pnl
        trades.append(Trade(side, entry_ts, ts, entry, px, pnl, reason, cash))
        equity_peak = max(equity_peak, cash)
        max_dd = max(max_dd, equity_peak - cash)
        pos = 0
        entry = units = 0.0

    def open_long(ts, px: float) -> None:
        nonlocal pos, entry, units, cash, entry_ts
        margin = cash if args.compound else args.amount
        units = units_for_margin(margin, args.leverage, px)
        cash -= commission(px, units, args.commission_per_million)
        pos = 1
        entry = px
        entry_ts = ts

    def open_short(ts, px: float) -> None:
        nonlocal pos, entry, units, cash, entry_ts
        margin = cash if args.compound else args.amount
        units = units_for_margin(margin, args.leverage, px)
        cash -= commission(px, units, args.commission_per_million)
        pos = -1
        entry = px
        entry_ts = ts

    entry_ts = start

    def process_breakout(
        ts,
        bid_px: float,
        ask_px: float,
        high_px: float,
        low_px: float,
        open_px: float | None = None,
        close_px: float | None = None,
    ) -> None:
        nonlocal upper, lower, hold_upper, hold_lower
        if upper is None or lower is None:
            return

        # Close-only levels are checked first. If they fire, do not also open
        # the opposite side on this same tick/candle.
        if pos == 1 and hold_lower is not None and low_px <= hold_lower:
            close_trade(ts, bid_px, "hold_close")
            return
        if pos == -1 and hold_upper is not None and high_px >= hold_upper:
            close_trade(ts, ask_px, "hold_close")
            return

        hit_upper = high_px >= upper
        hit_lower = low_px <= lower
        if hit_upper and hit_lower:
            # Without tick path, same-candle double hits are ambiguous. Use the
            # candle direction as a deterministic approximation.
            hit_upper = (close_px or ask_px) >= (open_px or bid_px)
            hit_lower = not hit_upper
        if pos <= 0 and hit_upper:
            if pos == -1:
                close_trade(ts, ask_px, "reverse")
                if level_atr_pass:
                    open_long(ts, ask_px)
            elif level_atr_pass:
                open_long(ts, ask_px)
        elif pos >= 0 and hit_lower:
            if pos == 1:
                close_trade(ts, bid_px, "reverse")
                if level_atr_pass:
                    open_short(ts, bid_px)
            elif level_atr_pass:
                open_short(ts, bid_px)

    tick_iter = (
        iter_mt5_ticks(args.symbol, start, end)
        if args.source == "mt5"
        else iter_ticks(args.csv, start, end, args.chunksize)
    )
    for ts, bid, ask in tick_iter:
        ticks += 1
        mid = tick_price(bid, ask, mode)
        bucket = int(ts.timestamp()) // tf_sec

        if cur is None:
            cur = Candle(bucket, mid, mid, mid, mid)
        elif bucket != cur.bucket:
            if args.execution == "candle" and upper is not None and lower is not None:
                process_breakout(
                    ts,
                    cur.close,
                    cur.close,
                    cur.high,
                    cur.low,
                    cur.open,
                    cur.close,
                )
            candles.append(cur)
            if prev_close is not None:
                tr_values.append(true_range(cur, prev_close))
            prev_close = cur.close
            if len(tr_values) >= args.length:
                raw_atr = sum(tr_values[-args.length:]) / args.length
                min_atr = min_atr_points * args.point_size
                level_atr_pass = min_atr_points <= 0 or raw_atr >= min_atr
                hold_mult = args.hold_close_mult
                if hold_mult is None:
                    hold_mult = args.atr_mult
                hold_atrs = raw_atr * hold_mult
                hold_upper = cur.close + hold_atrs
                hold_lower = cur.close - hold_atrs
                if level_atr_pass or pos != 0:
                    atrs = raw_atr * args.atr_mult
                    upper = cur.close + atrs
                    lower = cur.close - atrs
                else:
                    upper = lower = None
            cur = Candle(bucket, mid, mid, mid, mid)
        else:
            cur.high = max(cur.high, mid)
            cur.low = min(cur.low, mid)
            cur.close = mid

        if upper is None or lower is None:
            continue
        if args.execution == "candle":
            continue

        process_breakout(ts, bid, ask, ask, bid)

        if pos != 0:
            live_u = (bid - entry) * units if pos == 1 else (entry - ask) * units
            live_eq = cash + live_u
            equity_peak = max(equity_peak, live_eq)
            max_dd = max(max_dd, equity_peak - live_eq)

    open_u = 0.0
    if pos == 1:
        open_u = (bid - entry) * units
    elif pos == -1:
        open_u = (entry - ask) * units

    realised = cash - args.amount
    total = realised + open_u
    wins = sum(1 for t in trades if t.pnl >= 0)
    losses = len(trades) - wins
    gross_win = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = -sum(t.pnl for t in trades if t.pnl < 0)
    pf = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
    wr = wins / len(trades) * 100.0 if trades else 0.0

    print(
        f"[volty-clean] price={mode:<3} ticks={ticks:,} candles={len(candles):,} trades={len(trades)} "
        f"exec={args.execution} min_atr={min_atr_points:g}pt hold_mult="
        f"{args.hold_close_mult if args.hold_close_mult is not None else args.atr_mult:g} "
        f"wins={wins} losses={losses} wr={wr:.1f}% pf={pf:.2f} "
        f"realised=${realised:+.4f} open=${open_u:+.4f} total=${total:+.4f} "
        f"dd=${max_dd:.4f}",
        flush=True,
    )

    if args.trades_out:
        out_path = args.trades_out
        if "," in args.candle_price:
            dot = out_path.rfind(".")
            out_path = f"{out_path[:dot]}_{mode}{out_path[dot:]}" if dot >= 0 else f"{out_path}_{mode}"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["side", "entry_ts", "exit_ts", "entry", "exit", "pnl", "reason", "equity"])
            for t in trades:
                w.writerow([t.side, t.entry_ts, t.exit_ts, t.entry, t.exit, t.pnl, t.reason, t.equity])
        print(f"[volty-clean] wrote {out_path}", flush=True)


def main() -> None:
    args = parse_args()
    now = pd.Timestamp.now(tz="UTC")
    end = parse_dt(args.to, now)
    start = parse_dt(args.start, end - pd.Timedelta(days=args.days))
    modes = [x.strip().lower() for x in args.candle_price.split(",") if x.strip()]
    min_atrs = parse_float_list(args.min_atr_points)
    bad = sorted(set(modes) - {"mid", "bid", "ask"})
    if bad:
        raise SystemExit(f"unsupported --candle-price values: {bad}")

    print(
        f"[volty-clean] source={args.source} symbol={args.symbol} csv={args.csv} "
        f"from={start.isoformat()} to={end.isoformat()} "
        f"tf={args.timeframe} length={args.length} mult={args.atr_mult} "
        f"candle_price={','.join(modes)} execution={args.execution} "
        f"min_atr_points={','.join(str(x) for x in min_atrs)}",
        flush=True,
    )
    for mode in modes:
        for min_atr in min_atrs:
            run_one(args, start, end, mode, min_atr)


if __name__ == "__main__":
    main()
