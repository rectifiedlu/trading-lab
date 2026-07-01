from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import product

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError as exc:
    raise SystemExit("MetaTrader5 package missing. Run: pip install MetaTrader5") from exc


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


def timeframe_value(tf: str):
    table = {
        "1m": mt5.TIMEFRAME_M1,
        "2m": mt5.TIMEFRAME_M2,
        "3m": mt5.TIMEFRAME_M3,
        "5m": mt5.TIMEFRAME_M5,
        "10m": mt5.TIMEFRAME_M10,
        "15m": mt5.TIMEFRAME_M15,
        "30m": mt5.TIMEFRAME_M30,
        "1h": mt5.TIMEFRAME_H1,
    }
    key = tf.lower()
    if key not in table:
        raise SystemExit(f"unsupported MT5 timeframe: {tf}")
    return table[key]


def parse_dt(value: str | None, fallback: datetime) -> datetime:
    if not value:
        return fallback
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def commission(px: float, units: float, commission_per_million: float) -> float:
    return abs(px * units) / 1_000_000.0 * commission_per_million


def units_for_margin(margin: float, leverage: float, px: float) -> float:
    return margin * leverage / px


def true_range(high: float, low: float, prev_close: float) -> float:
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="MT5 native-candle + MT5 tick Volty hybrid backtest.")
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--timeframe", default="1m")
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--to", default=None)
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--length", default="4,5,6,8,10,12,14,16,20")
    ap.add_argument("--atr-mult", default="0.5,0.6,0.7,0.75,0.85,0.95")
    ap.add_argument("--hold-close-mult", default="0")
    ap.add_argument("--min-atr-points", type=float, default=0.0)
    ap.add_argument("--amount", type=float, default=50.0)
    ap.add_argument("--leverage", type=float, default=100.0)
    ap.add_argument("--commission-per-million", type=float, default=30.0)
    ap.add_argument("--compound", action="store_true")
    ap.add_argument("--entry-filter", choices=["none", "prev_color"], default="none")
    ap.add_argument("--trades-out", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    return ap.parse_args()


def parse_float_list(value: str | None, default: list[float]) -> list[float]:
    if value is None or str(value).strip() == "":
        return default
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


def parse_int_list(value: str | None, default: list[int]) -> list[int]:
    return [int(x) for x in parse_float_list(value, [float(x) for x in default])]


def simulate(
    candles: pd.DataFrame,
    tick_df: pd.DataFrame,
    args: argparse.Namespace,
    point: float,
    length: int,
    atr_mult: float,
    hold_mult: float,
) -> tuple[dict, list[Trade]]:
    trades: list[Trade] = []
    pos = 0
    entry = units = 0.0
    entry_ts = pd.Timestamp(candles.iloc[0].time)
    cash = args.amount
    equity_peak = args.amount
    max_dd = 0.0

    def close_trade(ts, px: float, reason: str) -> None:
        nonlocal pos, entry, units, cash, equity_peak, max_dd
        side = "long" if pos == 1 else "short"
        pnl = (
            (px - entry) * units if pos == 1 else (entry - px) * units
        ) - commission(px, units, args.commission_per_million)
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

    tick_i = 0
    tick_times = tick_df["time"].to_numpy()
    bids = tick_df["bid"].to_numpy(np.float64)
    asks = tick_df["ask"].to_numpy(np.float64)

    tr_values: list[float] = []
    for i in range(1, len(candles) - 1):
        row = candles.iloc[i]
        prev = candles.iloc[i - 1]
        tr_values.append(true_range(float(row.high), float(row.low), float(prev.close)))
        if len(tr_values) < length:
            continue

        raw_atr = float(sum(tr_values[-length:]) / length)
        atr_pass = args.min_atr_points <= 0 or raw_atr >= args.min_atr_points * point
        c = float(row.close)
        prev_green = float(row.close) > float(row.open)
        prev_red = float(row.close) < float(row.open)
        upper = c + raw_atr * atr_mult
        lower = c - raw_atr * atr_mult
        hold_upper = c + raw_atr * hold_mult
        hold_lower = c - raw_atr * hold_mult

        win_start = candles.iloc[i + 1].time
        win_end = candles.iloc[i + 2].time if i + 2 < len(candles) else candles.iloc[-1].time

        while tick_i < len(tick_times) and pd.Timestamp(tick_times[tick_i]) < win_start:
            tick_i += 1
        j = tick_i
        while j < len(tick_times) and pd.Timestamp(tick_times[j]) < win_end:
            ts = pd.Timestamp(tick_times[j])
            bid = float(bids[j])
            ask = float(asks[j])

            hold_active = hold_mult > 0
            if hold_active and pos == 1 and bid <= hold_lower:
                close_trade(ts, bid, "hold_close")
                j += 1
                continue
            if hold_active and pos == -1 and ask >= hold_upper:
                close_trade(ts, ask, "hold_close")
                j += 1
                continue

            if pos <= 0 and ask >= upper:
                color_allows = args.entry_filter != "prev_color" or prev_green
                if pos == -1 and not color_allows:
                    close_trade(ts, ask, "color_close")
                    j += 1
                    continue
                if not color_allows:
                    j += 1
                    continue
                if pos == -1:
                    close_trade(ts, ask, "reverse")
                    if atr_pass:
                        open_long(ts, ask)
                elif atr_pass:
                    open_long(ts, ask)
            elif pos >= 0 and bid <= lower:
                color_allows = args.entry_filter != "prev_color" or prev_red
                if pos == 1 and not color_allows:
                    close_trade(ts, bid, "color_close")
                    j += 1
                    continue
                if not color_allows:
                    j += 1
                    continue
                if pos == 1:
                    close_trade(ts, bid, "reverse")
                    if atr_pass:
                        open_short(ts, bid)
                elif atr_pass:
                    open_short(ts, bid)

            if pos != 0:
                live_u = (bid - entry) * units if pos == 1 else (entry - ask) * units
                live_eq = cash + live_u
                equity_peak = max(equity_peak, live_eq)
                max_dd = max(max_dd, equity_peak - live_eq)
            j += 1

    open_u = 0.0
    if pos == 1:
        open_u = (float(bids[-1]) - entry) * units
    elif pos == -1:
        open_u = (entry - float(asks[-1])) * units

    realised = cash - args.amount
    total = realised + open_u
    wins = sum(1 for t in trades if t.pnl >= 0)
    losses = len(trades) - wins
    gross_win = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = -sum(t.pnl for t in trades if t.pnl < 0)
    pf = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
    wr = wins / len(trades) * 100.0 if trades else 0.0
    return {
        "length": length,
        "atr_mult": atr_mult,
        "hold_mult": hold_mult,
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "wr": wr,
        "pf": pf,
        "realised": realised,
        "open": open_u,
        "total": total,
        "dd": max_dd,
    }, trades


def main() -> None:
    args = parse_args()
    end = parse_dt(args.to, datetime.now(timezone.utc))
    start = parse_dt(args.start, end - timedelta(days=args.days))
    lengths = parse_int_list(args.length, [4])
    atr_mults = parse_float_list(args.atr_mult, [0.73])
    hold_mults_arg = parse_float_list(args.hold_close_mult, []) if args.hold_close_mult is not None else []

    if not mt5.initialize():
        raise SystemExit(f"mt5.initialize failed: {mt5.last_error()}")
    try:
        if not mt5.symbol_select(args.symbol, True):
            raise SystemExit(f"symbol_select failed: {args.symbol}")
        info = mt5.symbol_info(args.symbol)
        if info is None:
            raise SystemExit(f"symbol_info failed: {args.symbol}")
        point = float(info.point or 0.01)

        tf = timeframe_value(args.timeframe)
        rates = mt5.copy_rates_range(args.symbol, tf, start, end)
        ticks = mt5.copy_ticks_range(args.symbol, start, end, mt5.COPY_TICKS_ALL)
    finally:
        mt5.shutdown()

    max_len = max(lengths)
    if rates is None or len(rates) < max_len + 3:
        raise SystemExit(f"not enough MT5 candles: {0 if rates is None else len(rates)}")
    if ticks is None or len(ticks) == 0:
        raise SystemExit("no MT5 ticks returned")

    candles = pd.DataFrame(rates)
    candles["time"] = pd.to_datetime(candles["time"], unit="s", utc=True)
    tick_df = pd.DataFrame(ticks)
    tick_df["time"] = pd.to_datetime(tick_df["time_msc"], unit="ms", utc=True)
    tick_df = tick_df[(tick_df["bid"] > 0) & (tick_df["ask"] > 0)].copy()

    print(
        f"[mt5-volty] symbol={args.symbol} tf={args.timeframe} from={start.isoformat()} "
        f"to={end.isoformat()} candles={len(candles):,} ticks={len(tick_df):,} "
        f"length={args.length} mult={args.atr_mult} hold_mult={args.hold_close_mult or 'same'} "
        f"min_atr={args.min_atr_points:g}pt filter={args.entry_filter}",
        flush=True,
    )

    results = []
    best_trades: list[Trade] = []
    combos = []
    for length, atr_mult in product(lengths, atr_mults):
        hold_mults = hold_mults_arg or [atr_mult]
        for hold_mult in hold_mults:
            effective_hold_mult = hold_mult
            if effective_hold_mult > atr_mult:
                continue
            combos.append((length, atr_mult, hold_mult, effective_hold_mult))

    def run_combo(combo):
        length, atr_mult, hold_mult, effective_hold_mult = combo
        res, trades = simulate(candles, tick_df, args, point, length, atr_mult, effective_hold_mult)
        res["hold_arg"] = hold_mult
        return res, trades

    if args.workers > 1 and len(combos) > 1 and not args.trades_out:
        print(f"[mt5-volty] workers={args.workers} combos={len(combos)}", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_combo, c) for c in combos]
            for i, fut in enumerate(as_completed(futs), 1):
                res, trades = fut.result()
                results.append(res)
                if not best_trades or res["total"] > max(r["total"] for r in results[:-1]):
                    best_trades = trades
                if i % max(1, len(combos) // 10) == 0:
                    print(f"[mt5-volty] progress {i}/{len(combos)}", flush=True)
    else:
        for combo in combos:
            res, trades = run_combo(combo)
            results.append(res)
            if not best_trades or res["total"] > max(r["total"] for r in results[:-1]):
                best_trades = trades
            print(
                f"[mt5-volty] len={res['length']:<3} mult={res['atr_mult']:<5g} hold={res['hold_arg']:<5g} "
                f"total=${res['total']:+.4f} real=${res['realised']:+.4f} "
                f"tr={res['trades']} wr={res['wr']:.1f}% pf={res['pf']:.2f} dd=${res['dd']:.2f}",
                flush=True,
            )

    results.sort(key=lambda r: (r["total"], r["pf"]), reverse=True)
    print("\n[mt5-volty] top by total", flush=True)
    for r in results[:20]:
        print(
            f"len={r['length']:<3} mult={r['atr_mult']:<5g} hold={r.get('hold_arg', r['hold_mult']):<5g} "
            f"total=${r['total']:+.4f} real=${r['realised']:+.4f} "
            f"tr={r['trades']} wr={r['wr']:.1f}% pf={r['pf']:.2f} dd=${r['dd']:.2f}",
            flush=True,
        )

    if args.trades_out:
        with open(args.trades_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["side", "entry_ts", "exit_ts", "entry", "exit", "pnl", "reason", "equity"])
            for t in best_trades:
                w.writerow([t.side, t.entry_ts, t.exit_ts, t.entry, t.exit, t.pnl, t.reason, t.equity])
        print(f"[mt5-volty] wrote {args.trades_out}", flush=True)


if __name__ == "__main__":
    main()
