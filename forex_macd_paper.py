"""MT5 MACD paper/demo trader for XAUUSD.

Default behavior sends orders to the currently connected MT5 account. Use
--dry-run to log decisions without sending orders.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
from forex_synthetic_candles import Candle, SyntheticBidOHLC

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "forex")
TRADES_CSV = os.path.join(DATA_DIR, "forex_macd_paper_trades.csv")
DEFAULT_MAGIC = 260513


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_trade_log(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "ts", "symbol", "side", "ticket", "entry", "exit", "volume",
            "profit", "reason", "macd", "fast", "slow", "macd_sma",
            "timeframe", "tp_points", "sl_points",
        ])


def timeframe_value(mt5, tf: str):
    table = {
        "1m": mt5.TIMEFRAME_M1,
        "2m": mt5.TIMEFRAME_M2,
        "3m": mt5.TIMEFRAME_M3,
        "4m": mt5.TIMEFRAME_M4,
        "5m": mt5.TIMEFRAME_M5,
        "10m": mt5.TIMEFRAME_M10,
        "15m": mt5.TIMEFRAME_M15,
        "30m": mt5.TIMEFRAME_M30,
        "1h": mt5.TIMEFRAME_H1,
    }
    key = tf.lower()
    if key not in table:
        raise SystemExit(f"unsupported timeframe: {tf}")
    return table[key]


def ema(values: np.ndarray, length: int) -> np.ndarray:
    if length <= 1:
        return values.astype(np.float64)
    out = np.empty(len(values), dtype=np.float64)
    alpha = 2.0 / (length + 1.0)
    val = float(values[0])
    for i, x in enumerate(values):
        val = alpha * float(x) + (1.0 - alpha) * val
        out[i] = val
    return out


def sma(values: np.ndarray, length: int) -> np.ndarray:
    if length <= 1:
        return values.astype(np.float64)
    out = np.full(len(values), np.nan, dtype=np.float64)
    csum = np.cumsum(np.insert(values.astype(np.float64), 0, 0.0))
    out[length - 1:] = (csum[length:] - csum[:-length]) / float(length)
    first = np.where(np.isfinite(out))[0]
    if len(first):
        out[:first[0]] = out[first[0]]
    return out


def calc_closed_macd(mt5, symbol: str, timeframe: str, fast: int, slow: int,
                     macd_sma: int, bars: int) -> tuple[float, int] | None:
    # Start from position 1 to exclude the currently forming candle.
    rates = mt5.copy_rates_from_pos(symbol, timeframe_value(mt5, timeframe), 1, bars)
    if rates is None or len(rates) < max(fast, slow) + macd_sma + 2:
        return None
    close = np.array(rates["close"], dtype=np.float64)
    raw = ema(close, fast) - ema(close, slow)
    line = sma(raw, macd_sma)
    return float(line[-1]), int(rates["time"][-1])


def calc_synthetic_macd(candles: SyntheticBidOHLC, fast: int, slow: int,
                        macd_sma: int, bars: int) -> tuple[float, int] | None:
    need = max(fast, slow) + macd_sma + 2
    if len(candles.closed) < need:
        return None
    rows = list(candles.closed)[-max(bars, need):]
    close = np.array([c.close for c in rows], dtype=np.float64)
    raw = ema(close, fast) - ema(close, slow)
    line = sma(raw, macd_sma)
    return float(line[-1]), int(candles.last_closed_bucket * candles.tf_sec)


def seed_synthetic_candles_from_mt5(mt5, symbol: str, candles: SyntheticBidOHLC,
                                    timeframe: str, bars: int) -> int:
    """Seed synthetic bid candles with MT5's already closed native candles.

    MT5 native XAUUSD OHLC matches bid candles closely. This avoids waiting for
    a full warmup after process start while still using our synthetic candles
    for everything that closes after startup.
    """
    try:
        tf_value = timeframe_value(mt5, timeframe)
    except SystemExit:
        return 0
    rates = mt5.copy_rates_from_pos(symbol, tf_value, 1, bars)
    if rates is None or len(rates) == 0:
        return 0
    candles.closed.clear()
    for r in rates:
        bucket = int(r["time"]) // candles.tf_sec
        candles.closed.append(Candle(
            bucket=bucket,
            open=float(r["open"]),
            high=float(r["high"]),
            low=float(r["low"]),
            close=float(r["close"]),
        ))
    candles.last_closed_bucket = int(candles.closed[-1].bucket)
    return len(candles.closed)


def macd_region(macd: float, deadband: float, direction: str) -> str:
    if abs(macd) <= deadband:
        return "neutral"
    if direction == "contrarian":
        return "long" if macd < 0 else "short"
    return "long" if macd > 0 else "short"


def round_volume(info, volume: float) -> float:
    step = float(info.volume_step or 0.01)
    vmin = float(info.volume_min or step)
    vmax = float(info.volume_max or volume)
    volume = max(vmin, min(vmax, volume))
    steps = math.floor((volume - vmin) / step + 1e-9)
    return round(vmin + steps * step, 8)


def max_volume_for_margin(mt5, symbol: str, order_type: int, price: float) -> float:
    info = mt5.symbol_info(symbol)
    acc = mt5.account_info()
    if info is None or acc is None:
        return 0.0
    free_margin = float(acc.margin_free)
    vmin = float(info.volume_min or 0.01)
    vmax = float(info.volume_max or 100.0)
    margin_min = mt5.order_calc_margin(order_type, symbol, vmin, price)
    if margin_min is None or margin_min <= 0 or margin_min > free_margin:
        return 0.0
    lo, hi = vmin, vmax
    for _ in range(36):
        mid = (lo + hi) / 2.0
        margin = mt5.order_calc_margin(order_type, symbol, mid, price)
        if margin is not None and margin <= free_margin:
            lo = mid
        else:
            hi = mid
    return round_volume(info, lo)


def filling_candidates(mt5, symbol: str):
    info = mt5.symbol_info(symbol)
    first = getattr(info, "filling_mode", None) if info else None
    modes = [first, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN]
    out = []
    for mode in modes:
        if mode is not None and mode not in out:
            out.append(mode)
    return out


def find_position(mt5, symbol: str, magic: int):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return None
    for p in positions:
        if int(getattr(p, "magic", 0) or 0) == int(magic):
            return p
    return None


def send_order(mt5, args, side: str, volume: float, reason: str, close_ticket: int | None = None):
    tick = mt5.symbol_info_tick(args.symbol)
    if tick is None:
        print(f"[macd-paper] ORDER SKIP no tick", flush=True)
        return None
    is_buy = side == "buy"
    order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
    price = float(tick.ask if is_buy else tick.bid)
    if close_ticket is None and args.compound:
        volume = max_volume_for_margin(mt5, args.symbol, order_type, price)
        if volume <= 0:
            print(f"[macd-paper] ORDER SKIP compound no free margin side={side}", flush=True)
            return None
    req_base = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": args.symbol,
        "volume": float(volume),
        "type": order_type,
        "price": price,
        "deviation": args.deviation,
        "magic": args.magic,
        "comment": f"macd_{reason}",
        "type_time": mt5.ORDER_TIME_GTC,
    }
    if close_ticket is not None:
        req_base["position"] = int(close_ticket)

    if args.dry_run:
        print(f"[macd-paper] DRY ORDER {side.upper()} vol={volume:g} "
              f"px={price:.3f} reason={reason}", flush=True)
        return {"dry_run": True, "price": price, "volume": volume}

    last = None
    for filling in filling_candidates(mt5, args.symbol):
        req = dict(req_base)
        req["type_filling"] = filling
        check = mt5.order_check(req)
        if check is not None and getattr(check, "retcode", 0) == 10030:
            continue
        res = mt5.order_send(req)
        last = res
        if res is not None and res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[macd-paper] ORDER OK {side.upper()} vol={volume:g} "
                  f"px={price:.3f} reason={reason} order={getattr(res, 'order', 0)}",
                  flush=True)
            return res
        print(f"[macd-paper] ORDER FAIL fill={filling} "
              f"ret={getattr(res, 'retcode', None)}", flush=True)
    print(f"[macd-paper] ORDER FAIL all fillings last={getattr(last, 'retcode', None)}", flush=True)
    return None


def log_trade(args, pos, exit_px: float, reason: str, macd: float):
    with open(args.trades_csv, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            now_iso(), args.symbol, "long" if pos.type == 0 else "short",
            int(pos.ticket), float(pos.price_open), exit_px, float(pos.volume),
            float(getattr(pos, "profit", 0.0)), reason, macd, args.fast,
            args.slow, args.macd_sma, args.timeframe, args.tp_points,
            args.sl_points,
        ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--timeframe", default="1m")
    ap.add_argument("--fast", type=int, default=12)
    ap.add_argument("--slow", type=int, default=13)
    ap.add_argument("--macd-sma", type=int, default=1)
    ap.add_argument("--deadband", type=float, default=0.1,
                    help="neutral MACD band around 0; default disabled")
    ap.add_argument("--direction", choices=["trend", "contrarian"], default="contrarian",
                    help="trend: positive MACD -> long, negative -> short")
    ap.add_argument("--tp-points", type=float, default=0.0,
                    help="0=close winning trades on MACD side switch")
    ap.add_argument("--sl-points", type=float, default=800.0,
                    help="0=close losing trades on MACD side switch; >0 uses hard SL")
    ap.add_argument("--point-size", type=float, default=0.01)
    ap.add_argument("--lot", type=float, default=0.01)
    ap.add_argument("--compound", action="store_true",
                    help="open new trades with max volume allowed by MT5 free margin")
    ap.add_argument("--poll-sec", type=float, default=0.5)
    ap.add_argument("--bars", type=int, default=300)
    ap.add_argument("--magic", type=int, default=DEFAULT_MAGIC)
    ap.add_argument("--deviation", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reenter-after-tp", action="store_true",
                    help="allow immediate same-region re-entry after TP")
    ap.add_argument("--reverse-on-flip", action="store_true",
                    help="close and immediately open opposite side on MACD flip")
    ap.add_argument("--trades-csv", default=TRADES_CSV)
    args = ap.parse_args()

    ensure_trade_log(args.trades_csv)
    try:
        import MetaTrader5 as mt5
    except ImportError as e:
        raise SystemExit("pip install MetaTrader5 and run this on the MT5 machine") from e

    if not mt5.initialize():
        raise SystemExit(f"mt5.initialize failed: {mt5.last_error()}")
    if not mt5.symbol_select(args.symbol, True):
        raise SystemExit(f"symbol_select failed for {args.symbol}: {mt5.last_error()}")
    info = mt5.symbol_info(args.symbol)
    if info is None:
        raise SystemExit(f"symbol_info failed for {args.symbol}")
    volume = round_volume(info, args.lot)
    blocked_side: str | None = None
    last_region: str | None = None
    last_bar_time: int | None = None
    seen_region = False
    last_ticket: int | None = None
    last_status = 0.0
    print(f"[macd-paper] start symbol={args.symbol} tf={args.timeframe} "
          f"fast={args.fast} slow={args.slow} macd_sma={args.macd_sma} "
          f"direction={args.direction} deadband={args.deadband} "
          f"tp={args.tp_points:g}pt sl={args.sl_points:g}pt lot={volume:g} "
          f"compound={int(args.compound)} dry={int(args.dry_run)} "
          f"candles=synthetic_bid", flush=True)

    candle_builder = SyntheticBidOHLC(args.timeframe, maxlen=max(args.bars + 10, 1000))
    seeded = seed_synthetic_candles_from_mt5(
        mt5, args.symbol, candle_builder, args.timeframe, max(args.bars, 100)
    )
    if seeded:
        print(f"[macd-paper] seeded {seeded} closed MT5 bid candles; "
              f"new candles will be synthetic from startup", flush=True)
    else:
        print(f"[macd-paper] no MT5 candle seed for tf={args.timeframe}; "
              f"waiting for synthetic warmup", flush=True)

    try:
        while True:
            tick = mt5.symbol_info_tick(args.symbol)
            synthetic_new_bar = False
            if tick is not None:
                synthetic_new_bar = candle_builder.update(tick)
            macd_row = calc_synthetic_macd(candle_builder, args.fast, args.slow,
                                           args.macd_sma, args.bars)
            pos = find_position(mt5, args.symbol, args.magic)
            if tick is None or macd_row is None:
                time.sleep(args.poll_sec)
                continue
            macd, bar_time = macd_row
            bid = float(tick.bid)
            ask = float(tick.ask)
            region = macd_region(macd, args.deadband, args.direction)
            new_closed_bar = synthetic_new_bar and (last_bar_time is None or bar_time != last_bar_time)
            region_changed = new_closed_bar and seen_region and region != last_region
            fresh_long = region_changed and region == "long"
            fresh_short = region_changed and region == "short"

            current_ticket = int(pos.ticket) if pos is not None else None
            if last_ticket is not None and current_ticket is None:
                # Position disappeared without this loop closing it: manual close,
                # broker-side close, or restart sync. Do not re-enter same region.
                if last_region in ("long", "short"):
                    blocked_side = last_region
                print(f"[macd-paper] SYNC flat after external close; "
                      f"blocking {blocked_side or '-'} until region changes", flush=True)
            last_ticket = current_ticket

            if blocked_side and region != blocked_side:
                blocked_side = None

            if pos is not None:
                side = "long" if int(pos.type) == mt5.ORDER_TYPE_BUY else "short"
                entry = float(pos.price_open)
                pnl_points = ((bid - entry) if side == "long" else (entry - ask)) / args.point_size
                tp_hit = args.tp_points > 0 and pnl_points >= args.tp_points
                sl_hit = args.sl_points > 0 and pnl_points <= -args.sl_points
                flip = new_closed_bar and (
                    (side == "long" and region == "short") or
                    (side == "short" and region == "long")
                )
                if tp_hit:
                    close_side = "sell" if side == "long" else "buy"
                    res = send_order(mt5, args, close_side, float(pos.volume), "tp", int(pos.ticket))
                    if res is not None:
                        log_trade(args, pos, bid if side == "long" else ask, "tp", macd)
                        if not args.reenter_after_tp:
                            blocked_side = side
                        last_ticket = None
                elif sl_hit:
                    close_side = "sell" if side == "long" else "buy"
                    res = send_order(mt5, args, close_side, float(pos.volume), "sl", int(pos.ticket))
                    if res is not None:
                        log_trade(args, pos, bid if side == "long" else ask, "sl", macd)
                        blocked_side = side
                        last_ticket = None
                elif flip:
                    profitable = pnl_points >= 0
                    should_close = (profitable and args.tp_points <= 0) or ((not profitable) and args.sl_points <= 0)
                    if should_close:
                        close_side = "sell" if side == "long" else "buy"
                        res = send_order(mt5, args, close_side, float(pos.volume), "flip_close", int(pos.ticket))
                        if res is not None:
                            log_trade(args, pos, bid if side == "long" else ask, "flip", macd)
                            last_ticket = None
                            if args.reverse_on_flip:
                                open_side = "sell" if side == "long" else "buy"
                                send_order(mt5, args, open_side, volume, "flip_open")
                            else:
                                blocked_side = region
            else:
                if fresh_long and blocked_side != "long":
                    send_order(mt5, args, "buy", volume, "entry_long")
                elif fresh_short and blocked_side != "short":
                    send_order(mt5, args, "sell", volume, "entry_short")

            now = time.time()
            if now - last_status >= 5:
                acc = mt5.account_info()
                pos_txt = "-"
                if pos is not None:
                    side = "L" if int(pos.type) == mt5.ORDER_TYPE_BUY else "S"
                    pos_txt = f"{side} entry={float(pos.price_open):.3f} p=${float(pos.profit):+.2f}"
                print(f"[macd-paper] bal=${float(acc.balance):.2f} eq=${float(acc.equity):.2f} "
                      f"{args.symbol} bid={bid:.3f} ask={ask:.3f} "
                      f"closed_macd={macd:+.4f} region={region} "
                      f"bar={bar_time} block={blocked_side or '-'} pos={pos_txt}",
                      flush=True)
                last_status = now
            if new_closed_bar:
                last_region = region
                last_bar_time = bar_time
                seen_region = True
            time.sleep(args.poll_sec)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
