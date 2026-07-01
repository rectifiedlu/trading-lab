"""Tick-level RSI bounce backtester for forex.

Strategy:
  - Build RSI from timeframe candles, but update the current candle on every tick.
  - Long arms when RSI <= long_level, then buys after RSI bounces up by bounce.
  - Short arms when RSI >= short_level, then shorts after RSI falls by bounce.
  - Exits use a fixed TP in symbol points. One position per pair.

Examples:
  python forex_rsi_backtest.py --pairs EURUSD --days 7
  python forex_rsi_backtest.py --pairs EURUSD XAUUSD --timeframes 1m,5m --rsi-periods 10,12,14
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd

from forex_backtest import (
    DEFAULT_COMMISSION_PER_MILLION,
    DEFAULT_MT5_TICK_CSV,
    FOREX_DIR,
    _default_date_window,
    _default_hour_window,
    load_ticks,
)
from forex_strategy_common import print_ranked_sections

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


DEFAULT_PAIRS = ["XAUUSD"]
DEFAULT_TIMEFRAMES = ["5m"]
DEFAULT_RSI_PERIODS = [10.0, 12.0, 14.0, 16.0, 18.0]
DEFAULT_LONG_LEVELS = [28.0, 32.0]
DEFAULT_SHORT_LEVELS = [68.0, 72.0]
DEFAULT_BOUNCES = [1.0, 2.0, 3.0, 4.0]
DEFAULT_TP_POINTS = [150.0, 400.0, 750.0, 950.0]
DEFAULT_SL_POINTS = [0, 400.0, 750.0, 1000.0, 1500.0, 1750.0]
DEFAULT_AMOUNT = 50.0


@dataclass(frozen=True)
class Combo:
    timeframe: str
    rsi_period: int
    long_level: float
    short_level: float
    bounce: float
    tp_points: float
    sl_points: float


@dataclass
class Result:
    pair: str
    side: str
    timeframe: str
    rsi_period: int
    long_level: float
    short_level: float
    bounce: float
    tp_points: float
    sl_points: float
    point_size: float
    realised: float
    open_unrealized: float
    total: float
    trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    long_trades: int
    short_trades: int
    stop_losses: int
    liquidations: int
    account_dead: bool
    open_side: str
    open_bps: float


def _parse_num_list(s: str | None, default: Iterable[float]) -> list[float]:
    if not s:
        return [float(x) for x in default]
    return [float(p.strip()) for p in s.split(",") if p.strip()]


def _parse_str_list(s: str | None, default: Iterable[str]) -> list[str]:
    if not s:
        return [str(x) for x in default]
    return [p.strip() for p in s.split(",") if p.strip()]


def _chunked(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _timeframe_to_ns(timeframe: str) -> int:
    tf = timeframe.strip().lower()
    if tf.endswith("m"):
        return int(float(tf[:-1]) * 60 * 1_000_000_000)
    if tf.endswith("h"):
        return int(float(tf[:-1]) * 3600 * 1_000_000_000)
    if tf.endswith("s"):
        return int(float(tf[:-1]) * 1_000_000_000)
    raise ValueError(f"unsupported timeframe: {timeframe}")


def _live_simple_rsi(mid: np.ndarray, ts_ns: np.ndarray, timeframe: str, period: int) -> np.ndarray:
    """Compute RSI using closed candle closes plus the current tick as forming close.

    This is intentionally tick-reactive: inside a 1m/5m candle, the current tick
    changes the latest RSI value instead of waiting for candle close.
    """
    if len(mid) == 0:
        return np.array([], dtype=np.float64)

    tf_ns = _timeframe_to_ns(timeframe)
    bucket = ts_ns // tf_ns
    closed_closes: list[float] = []
    cur_bucket = int(bucket[0])
    last_mid = float(mid[0])
    rsi = np.full(len(mid), np.nan, dtype=np.float64)

    for i, px in enumerate(mid):
        b = int(bucket[i])
        if b != cur_bucket:
            closed_closes.append(last_mid)
            cur_bucket = b

        last_mid = float(px)
        if len(closed_closes) < period:
            continue

        window = np.array(closed_closes[-period:] + [float(px)], dtype=np.float64)
        delta = np.diff(window)
        gains = np.clip(delta, 0.0, None)
        losses = np.clip(-delta, 0.0, None)
        avg_gain = float(gains.mean())
        avg_loss = float(losses.mean())
        if avg_loss <= 0.0:
            rsi[i] = 100.0
        elif avg_gain <= 0.0:
            rsi[i] = 0.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def _commission(px: float, units: float, commission_per_million: float) -> float:
    notional = abs(px * units)
    return notional / 1_000_000.0 * commission_per_million


def _units(amount: float, leverage: float, px: float) -> float:
    return (amount * leverage) / px


def _default_point_size(pair: str) -> float:
    p = pair.upper()
    if p == "XAUUSD":
        return 0.01
    if "JPY" in p:
        return 0.001
    return 0.00001


def _open_unrealized(pos: int, entry: float, units: float, bid: float, ask: float) -> float:
    if pos == 1:
        return (bid - entry) * units
    if pos == -1:
        return (entry - ask) * units
    return 0.0


def simulate_combo(
    pair: str,
    bid: np.ndarray,
    ask: np.ndarray,
    rsi: np.ndarray,
    combo: Combo,
    side_mode: str,
    amount: float,
    compound: bool,
    leverage: float,
    commission_per_million: float,
    point_size: float,
) -> Result:
    tp_distance = combo.tp_points * point_size
    sl_distance = combo.sl_points * point_size if combo.sl_points > 0 else 0.0

    start_balance = amount
    cash = start_balance
    equity_peak = start_balance
    max_dd = 0.0
    gross_win = 0.0
    gross_loss = 0.0
    trades = wins = losses = long_trades = short_trades = 0
    stop_losses = 0
    liquidations = 0
    account_dead = False
    liquidation_bps = 10000.0 / max(leverage, 1e-9)

    pos = 0
    entry = 0.0
    units = 0.0

    long_armed = False
    long_low = np.inf
    short_armed = False
    short_high = -np.inf

    for i in range(len(bid) - 1):
        val = rsi[i]
        if not np.isfinite(val):
            continue

        next_bid = float(bid[i + 1])
        next_ask = float(ask[i + 1])

        if pos == 1:
            live_equity = cash + _open_unrealized(pos, entry, units, next_bid, next_ask)
            equity_peak = max(equity_peak, live_equity)
            max_dd = max(max_dd, equity_peak - live_equity)

            adverse_bps = (entry / next_bid - 1.0) * 10000.0 if next_bid > 0 else np.inf
            if adverse_bps >= liquidation_bps:
                loss = max(0.0, cash)
                pnl = -loss
                cash = 0.0
                trades += 1
                losses += 1
                long_trades += 1
                liquidations += 1
                account_dead = True
                gross_loss += -pnl
                pos = 0
                entry = units = 0.0
                equity_peak = max(equity_peak, cash)
                max_dd = max(max_dd, equity_peak - cash)
                break

            if sl_distance > 0 and next_bid <= entry - sl_distance:
                exit_px = next_bid
                pnl = (exit_px - entry) * units - _commission(exit_px, units, commission_per_million)
                cash += pnl
                trades += 1
                losses += 1
                long_trades += 1
                stop_losses += 1
                gross_loss += -pnl if pnl < 0 else 0.0
                gross_win += pnl if pnl >= 0 else 0.0
                pos = 0
                entry = units = 0.0
                equity_peak = max(equity_peak, cash)
                max_dd = max(max_dd, equity_peak - cash)
                continue

            if next_bid >= entry + tp_distance:
                exit_px = next_bid
                pnl = (exit_px - entry) * units - _commission(exit_px, units, commission_per_million)
                cash += pnl
                trades += 1
                long_trades += 1
                if pnl >= 0:
                    wins += 1
                    gross_win += pnl
                else:
                    losses += 1
                    gross_loss += -pnl
                pos = 0
                entry = units = 0.0
                equity_peak = max(equity_peak, cash)
                max_dd = max(max_dd, equity_peak - cash)
            continue

        if pos == -1:
            live_equity = cash + _open_unrealized(pos, entry, units, next_bid, next_ask)
            equity_peak = max(equity_peak, live_equity)
            max_dd = max(max_dd, equity_peak - live_equity)

            adverse_bps = (next_ask / entry - 1.0) * 10000.0 if entry > 0 else np.inf
            if adverse_bps >= liquidation_bps:
                loss = max(0.0, cash)
                pnl = -loss
                cash = 0.0
                trades += 1
                losses += 1
                short_trades += 1
                liquidations += 1
                account_dead = True
                gross_loss += -pnl
                pos = 0
                entry = units = 0.0
                equity_peak = max(equity_peak, cash)
                max_dd = max(max_dd, equity_peak - cash)
                break

            if sl_distance > 0 and next_ask >= entry + sl_distance:
                exit_px = next_ask
                pnl = (entry - exit_px) * units - _commission(exit_px, units, commission_per_million)
                cash += pnl
                trades += 1
                losses += 1
                short_trades += 1
                stop_losses += 1
                gross_loss += -pnl if pnl < 0 else 0.0
                gross_win += pnl if pnl >= 0 else 0.0
                pos = 0
                entry = units = 0.0
                equity_peak = max(equity_peak, cash)
                max_dd = max(max_dd, equity_peak - cash)
                continue

            if next_ask <= entry - tp_distance:
                exit_px = next_ask
                pnl = (entry - exit_px) * units - _commission(exit_px, units, commission_per_million)
                cash += pnl
                trades += 1
                short_trades += 1
                if pnl >= 0:
                    wins += 1
                    gross_win += pnl
                else:
                    losses += 1
                    gross_loss += -pnl
                pos = 0
                entry = units = 0.0
                equity_peak = max(equity_peak, cash)
                max_dd = max(max_dd, equity_peak - cash)
            continue

        allow_long = side_mode in ("long", "both")
        allow_short = side_mode in ("short", "both")

        if allow_long and val <= combo.long_level:
            long_armed = True
            long_low = min(long_low, val)
        if allow_short and val >= combo.short_level:
            short_armed = True
            short_high = max(short_high, val)

        long_ready = long_armed and val >= long_low + combo.bounce
        short_ready = short_armed and val <= short_high - combo.bounce

        if long_ready and short_ready:
            long_score = val - (long_low + combo.bounce)
            short_score = (short_high - combo.bounce) - val
            short_ready = short_score > long_score
            long_ready = not short_ready

        if long_ready:
            entry = next_ask
            margin = cash if compound else amount
            if margin <= 0:
                break
            units = _units(margin, leverage, entry)
            cash -= _commission(entry, units, commission_per_million)
            pos = 1
            long_armed = short_armed = False
            long_low = np.inf
            short_high = -np.inf
        elif short_ready:
            entry = next_bid
            margin = cash if compound else amount
            if margin <= 0:
                break
            units = _units(margin, leverage, entry)
            cash -= _commission(entry, units, commission_per_million)
            pos = -1
            long_armed = short_armed = False
            long_low = np.inf
            short_high = -np.inf

        equity_peak = max(equity_peak, cash)
        max_dd = max(max_dd, equity_peak - cash)

    open_unrealized = 0.0
    open_side = "-"
    open_bps = 0.0
    if pos == 1:
        open_side = "long"
        open_unrealized = (float(bid[-1]) - entry) * units
        open_bps = (float(bid[-1]) / entry - 1.0) * 10000.0
    elif pos == -1:
        open_side = "short"
        open_unrealized = (entry - float(ask[-1])) * units
        open_bps = (entry / float(ask[-1]) - 1.0) * 10000.0

    equity = cash + open_unrealized
    total = equity - start_balance
    realised = cash - start_balance
    win_rate = (wins / trades * 100.0) if trades else 0.0
    profit_factor = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
    return Result(
        pair=pair,
        side=side_mode,
        timeframe=combo.timeframe,
        rsi_period=combo.rsi_period,
        long_level=combo.long_level,
        short_level=combo.short_level,
        bounce=combo.bounce,
        tp_points=combo.tp_points,
        sl_points=combo.sl_points,
        point_size=point_size,
        realised=realised,
        open_unrealized=open_unrealized,
        total=total,
        trades=trades,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_drawdown=max_dd,
        long_trades=long_trades,
        short_trades=short_trades,
        stop_losses=stop_losses,
        liquidations=liquidations,
        account_dead=account_dead,
        open_side=open_side,
        open_bps=open_bps,
    )


def _build_combos(args: argparse.Namespace) -> list[Combo]:
    timeframes = _parse_str_list(args.timeframes, DEFAULT_TIMEFRAMES)
    periods = [int(x) for x in _parse_num_list(args.rsi_periods, DEFAULT_RSI_PERIODS)]
    long_levels = _parse_num_list(args.long_levels, DEFAULT_LONG_LEVELS)
    short_levels = _parse_num_list(args.short_levels, DEFAULT_SHORT_LEVELS)
    bounces = _parse_num_list(args.bounces, DEFAULT_BOUNCES)
    tp_points = _parse_num_list(args.tp_points, DEFAULT_TP_POINTS)
    sl_points = _parse_num_list(args.sl_points, DEFAULT_SL_POINTS)

    return [
        Combo(tf, period, ll, sl, bounce, tp, stop)
        for tf, period, ll, sl, bounce, tp, stop in product(
            timeframes, periods, long_levels, short_levels, bounces, tp_points, sl_points,
        )
    ]


def _simulate_chunk(payload) -> list[Result]:
    pair, bid, ask, rsi_cache, combos, side, amount, compound, leverage, commission, point_size = payload
    out = []
    for combo in combos:
        rsi = rsi_cache[(combo.timeframe, combo.rsi_period)]
        out.append(simulate_combo(
            pair, bid, ask, rsi, combo, side, amount, compound, leverage, commission, point_size,
        ))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["mt5", "local", "dukascopy"], default="mt5")
    ap.add_argument("--csv", default=DEFAULT_MT5_TICK_CSV,
                    help="local tick CSV with timestamp,bid,ask[,pair]")
    ap.add_argument("--pair", help="pair for local CSV if no pair column exists")
    ap.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--to", default=None)
    ap.add_argument("--side", choices=["long", "short", "both"], default="both")
    ap.add_argument("--timeframes", default=None, help="comma list, e.g. 1m,5m")
    ap.add_argument("--rsi-periods", default=None, help="comma list, e.g. 10,12,14")
    ap.add_argument("--long-levels", default=None, help="oversold arm levels")
    ap.add_argument("--short-levels", default=None, help="overbought arm levels")
    ap.add_argument("--bounces", default=None,
                    help="shared RSI bounce points after long/short arm")
    ap.add_argument("--tp-points", default=None,
                    help="comma list of take-profit distances in symbol points")
    ap.add_argument("--sl-points", default=None,
                    help="comma list of stop-loss distances in symbol points; 0 disables")
    ap.add_argument("--point-size", type=float, default=None,
                    help="override point size; XAUUSD default=0.01, JPY=0.001, others=0.00001")
    ap.add_argument("--amount", type=float, default=DEFAULT_AMOUNT,
                    help="fixed margin per trade by default; starting balance with --compound")
    ap.add_argument("--compound", action="store_true",
                    help="use current equity as trade margin")
    ap.add_argument("--leverage", type=float, default=100.0)
    ap.add_argument("--commission-per-million", type=float, default=DEFAULT_COMMISSION_PER_MILLION)
    ap.add_argument("--min-trades", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--chunk-size", type=int, default=5000)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(FOREX_DIR, "forex_rsi_results.csv"))
    args = ap.parse_args()

    if args.source == "local" and not args.csv:
        raise SystemExit("--csv is required for --source local")
    if args.start is None or args.to is None:
        if args.days is not None:
            args.start, args.to = _default_date_window(args.days)
        else:
            args.start, args.to = _default_hour_window(args.hours)

    os.makedirs(FOREX_DIR, exist_ok=True)
    combos = _build_combos(args)
    print(
        f"[rsi] source={args.source} from={args.start} to={args.to} "
        f"pairs={len(args.pairs)} side={args.side} combos_per_pair={len(combos):,} "
        f"amount=${args.amount:g} lev={args.leverage:g}x "
        f"compound={int(args.compound)} "
        f"workers={args.workers} chunk={args.chunk_size}",
        flush=True,
    )
    t0 = time.time()
    ticks = load_ticks(args)
    if ticks.empty:
        raise SystemExit("no ticks loaded")

    results: list[Result] = []
    for pair, g in ticks.groupby("pair", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        print(f"[rsi] {pair} ticks={len(g):,} combos={len(combos):,}", flush=True)

        bid = g["bid"].to_numpy(np.float64)
        ask = g["ask"].to_numpy(np.float64)
        # MT5 native XAUUSD candles match bid OHLC, not mid/ask.
        mid = bid
        ts = pd.to_datetime(g["timestamp"], utc=True).astype("int64").to_numpy()
        point_size = args.point_size or _default_point_size(pair)

        rsi_cache = {}
        for tf, period in sorted({(c.timeframe, c.rsi_period) for c in combos}):
            print(f"[rsi] {pair} precompute RSI {tf}/{period}", flush=True)
            rsi_cache[(tf, period)] = _live_simple_rsi(mid, ts, tf, period)
        print(
            f"[rsi] {pair} RSI ready sets={len(rsi_cache)} "
            f"point_size={point_size:g}",
            flush=True,
        )

        chunks = list(_chunked(combos, max(1, args.chunk_size)))
        print(
            f"[rsi] {pair} chunks ready chunks={len(chunks):,} "
            f"chunk_size={max(1, args.chunk_size):,} combos={len(combos):,}",
            flush=True,
        )
        pair_results: list[Result] = []
        if args.workers <= 1 or len(chunks) <= 1:
            print(f"[rsi] {pair} running inline", flush=True)
            for n, chunk in enumerate(chunks, 1):
                pair_results.extend(_simulate_chunk(
                    (pair, bid, ask, rsi_cache, chunk, args.side, args.amount,
                     args.compound, args.leverage, args.commission_per_million, point_size)
                ))
                print(
                    f"[rsi] {pair} progress chunks={n:,}/{len(chunks):,} "
                    f"combos={min(n * args.chunk_size, len(combos)):,}/{len(combos):,} "
                    f"results={len(pair_results):,}",
                    flush=True,
                )
        else:
            print(
                f"[rsi] {pair} launching workers={args.workers} "
                f"tasks={len(chunks):,}",
                flush=True,
            )
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [
                    pool.submit(
                        _simulate_chunk,
                        (pair, bid, ask, rsi_cache, chunk, args.side, args.amount,
                         args.compound, args.leverage, args.commission_per_million, point_size),
                    )
                    for chunk in chunks
                ]
                done = 0
                for fut in as_completed(futures):
                    pair_results.extend(fut.result())
                    done += 1
                    print(
                        f"[rsi] {pair} progress chunks={done:,}/{len(chunks):,} "
                        f"combos={min(done * args.chunk_size, len(combos)):,}/{len(combos):,} "
                        f"results={len(pair_results):,}",
                        flush=True,
                    )
        print(
            f"[rsi] {pair} simulation complete raw_results={len(pair_results):,}",
            flush=True,
        )

        eligible = [r for r in pair_results if r.trades >= args.min_trades]
        results.extend(eligible)
        best = max(
            eligible,
            key=lambda r: (r.total, r.realised, r.profit_factor),
            default=None,
        )
        if best:
            print(
                f"[rsi] {pair} best total=${best.total:+.4f} realised=${best.realised:+.4f} "
                f"{best.timeframe}/{best.rsi_period} L={best.long_level:g} "
                f"S={best.short_level:g} bounce={best.bounce:g} "
                f"tp={best.tp_points:g}pt sl={best.sl_points:g}pt "
                f"trades={best.trades} wr={best.win_rate:.1f}% "
                f"dd=${best.max_drawdown:.4f} stops={best.stop_losses} "
                f"liq={best.liquidations}",
                flush=True,
            )

    results.sort(key=lambda r: (r.total, r.realised, r.profit_factor), reverse=True)
    top_rows = results[: max(1, args.top)]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "pair", "side", "timeframe", "rsi_period",
            "long_level", "short_level", "bounce",
            "tp_points", "sl_points", "point_size",
            "realised", "open_unrealized", "total", "trades", "wins", "losses",
            "win_rate", "profit_factor", "max_drawdown", "long_trades",
            "short_trades", "stop_losses", "liquidations", "account_dead",
            "open_side", "open_bps",
        ])
        for r in results:
            w.writerow([
                r.pair, r.side, r.timeframe, r.rsi_period,
                r.long_level, r.short_level, r.bounce,
                r.tp_points, r.sl_points, r.point_size,
                round(r.realised, 6), round(r.open_unrealized, 6), round(r.total, 6),
                r.trades, r.wins, r.losses, round(r.win_rate, 2),
                round(r.profit_factor, 4), round(r.max_drawdown, 6),
                r.long_trades, r.short_trades, r.stop_losses, r.liquidations,
                int(r.account_dead), r.open_side, round(r.open_bps, 4),
            ])

    print(f"[rsi] wrote {args.out} elapsed={time.time() - t0:.1f}s", flush=True)
    for r in results:
        r.strategy = "rsi"
        r.params = (
            f"period={r.rsi_period};L={r.long_level:g};S={r.short_level:g};"
            f"bounce={r.bounce:g}"
        )
        r.signal_exits = 0
    print_ranked_sections(results, args.top)


if __name__ == "__main__":
    main()
