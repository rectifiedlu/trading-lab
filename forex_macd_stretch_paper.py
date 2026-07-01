"""MT5 MACD stretch-gate paper/demo trader for XAUUSD.

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

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "forex")
TRADES_CSV = os.path.join(DATA_DIR, "forex_macd_stretch_paper_trades.csv")
DEFAULT_MAGIC = 260514


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
            "timeframe", "tp_points", "stretch_ema", "stretch_atr",
            "stretch_mult",
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


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    return np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])


def calc_closed_signal(mt5, args) -> dict[str, float | int | bool] | None:
    # Start from position 1 to exclude the currently forming candle.
    rates = mt5.copy_rates_from_pos(
        args.symbol, timeframe_value(mt5, args.timeframe), 1, args.bars,
    )
    needed = max(args.fast, args.slow, args.macd_sma,
                 args.stretch_ema, args.stretch_atr) + 2
    if rates is None or len(rates) < needed:
        return None
    close = np.array(rates["close"], dtype=np.float64)
    high = np.array(rates["high"], dtype=np.float64)
    low = np.array(rates["low"], dtype=np.float64)
    raw = ema(close, args.fast) - ema(close, args.slow)
    line = sma(raw, args.macd_sma)
    basis = ema(close, args.stretch_ema)
    atr = sma(true_range(high, low, close), args.stretch_atr)
    upper = basis + atr * args.stretch_mult
    lower = basis - atr * args.stretch_mult
    last_close = float(close[-1])
    return {
        "macd": float(line[-1]),
        "bar_time": int(rates["time"][-1]),
        "close": last_close,
        "basis": float(basis[-1]),
        "atr": float(atr[-1]),
        "upper": float(upper[-1]),
        "lower": float(lower[-1]),
        "long_allowed": bool(last_close <= float(upper[-1])),
        "short_allowed": bool(last_close >= float(lower[-1])),
    }


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
        print(f"[macd-stretch-paper] ORDER SKIP no tick", flush=True)
        return None
    is_buy = side == "buy"
    order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
    price = float(tick.ask if is_buy else tick.bid)
    if close_ticket is None and args.compound:
        volume = max_volume_for_margin(mt5, args.symbol, order_type, price)
        if volume <= 0:
            print(f"[macd-stretch-paper] ORDER SKIP compound no free margin side={side}", flush=True)
            return None
    req_base = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": args.symbol,
        "volume": float(volume),
        "type": order_type,
        "price": price,
        "deviation": args.deviation,
        "magic": args.magic,
        "comment": f"macdstr_{reason}",
        "type_time": mt5.ORDER_TIME_GTC,
    }
    if close_ticket is not None:
        req_base["position"] = int(close_ticket)

    if args.dry_run:
        print(f"[macd-stretch-paper] DRY ORDER {side.upper()} vol={volume:g} "
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
            print(f"[macd-stretch-paper] ORDER OK {side.upper()} vol={volume:g} "
                  f"px={price:.3f} reason={reason} order={getattr(res, 'order', 0)}",
                  flush=True)
            return res
        print(f"[macd-stretch-paper] ORDER FAIL fill={filling} "
              f"ret={getattr(res, 'retcode', None)}", flush=True)
    print(f"[macd-stretch-paper] ORDER FAIL all fillings last={getattr(last, 'retcode', None)}", flush=True)
    return None


def log_trade(args, pos, exit_px: float, reason: str, macd: float):
    with open(args.trades_csv, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            now_iso(), args.symbol, "long" if pos.type == 0 else "short",
            int(pos.ticket), float(pos.price_open), exit_px, float(pos.volume),
            float(getattr(pos, "profit", 0.0)), reason, macd, args.fast,
            args.slow, args.macd_sma, args.timeframe, args.tp_points,
            args.stretch_ema, args.stretch_atr, args.stretch_mult,
        ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--timeframe", default="5m")
    ap.add_argument("--fast", type=int, default=8)
    ap.add_argument("--slow", type=int, default=13)
    ap.add_argument("--macd-sma", type=int, default=1)
    ap.add_argument("--deadband", type=float, default=0.1,
                    help="neutral MACD band around 0")
    ap.add_argument("--direction", choices=["trend", "contrarian"], default="trend",
                    help="trend: positive MACD -> long, negative -> short")
    ap.add_argument("--tp-points", type=float, default=600.0)
    ap.add_argument("--point-size", type=float, default=0.01)
    ap.add_argument("--stretch-ema", type=int, default=20)
    ap.add_argument("--stretch-atr", type=int, default=7)
    ap.add_argument("--stretch-mult", type=float, default=0.7)
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
    tp_dist = args.tp_points * args.point_size
    blocked_side: str | None = None
    last_region: str | None = None
    last_bar_time: int | None = None
    seen_region = False
    last_ticket: int | None = None
    last_status = 0.0
    print(f"[macd-stretch-paper] start symbol={args.symbol} tf={args.timeframe} "
          f"fast={args.fast} slow={args.slow} macd_sma={args.macd_sma} "
          f"direction={args.direction} deadband={args.deadband} "
          f"stretch={args.stretch_ema}/{args.stretch_atr}x{args.stretch_mult:g} "
          f"tp={args.tp_points:g}pt lot={volume:g} "
          f"compound={int(args.compound)} dry={int(args.dry_run)}", flush=True)

    try:
        while True:
            tick = mt5.symbol_info_tick(args.symbol)
            sig = calc_closed_signal(mt5, args)
            pos = find_position(mt5, args.symbol, args.magic)
            if tick is None or sig is None:
                time.sleep(args.poll_sec)
                continue
            macd = float(sig["macd"])
            bar_time = int(sig["bar_time"])
            bid = float(tick.bid)
            ask = float(tick.ask)
            region = macd_region(macd, args.deadband, args.direction)
            new_closed_bar = last_bar_time is None or bar_time != last_bar_time
            region_changed = new_closed_bar and seen_region and region != last_region
            fresh_long = region_changed and region == "long" and bool(sig["long_allowed"])
            fresh_short = region_changed and region == "short" and bool(sig["short_allowed"])

            current_ticket = int(pos.ticket) if pos is not None else None
            if last_ticket is not None and current_ticket is None:
                # Position disappeared without this loop closing it: manual close,
                # broker-side close, or restart sync. Do not re-enter same region.
                if last_region in ("long", "short"):
                    blocked_side = last_region
                print(f"[macd-stretch-paper] SYNC flat after external close; "
                      f"blocking {blocked_side or '-'} until region changes", flush=True)
            last_ticket = current_ticket

            if blocked_side and region != blocked_side:
                blocked_side = None

            if pos is not None:
                side = "long" if int(pos.type) == mt5.ORDER_TYPE_BUY else "short"
                entry = float(pos.price_open)
                pnl_points = ((bid - entry) if side == "long" else (entry - ask)) / args.point_size
                tp_hit = pnl_points >= args.tp_points
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
                elif flip:
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
                elif region_changed and region == "long":
                    print(f"[macd-stretch-paper] ENTRY SKIP long stretch "
                          f"close={float(sig['close']):.3f} upper={float(sig['upper']):.3f}",
                          flush=True)
                elif fresh_short and blocked_side != "short":
                    send_order(mt5, args, "sell", volume, "entry_short")
                elif region_changed and region == "short":
                    print(f"[macd-stretch-paper] ENTRY SKIP short stretch "
                          f"close={float(sig['close']):.3f} lower={float(sig['lower']):.3f}",
                          flush=True)

            now = time.time()
            if now - last_status >= 5:
                acc = mt5.account_info()
                pos_txt = "-"
                if pos is not None:
                    side = "L" if int(pos.type) == mt5.ORDER_TYPE_BUY else "S"
                    pos_txt = f"{side} entry={float(pos.price_open):.3f} p=${float(pos.profit):+.2f}"
                print(f"[macd-stretch-paper] bal=${float(acc.balance):.2f} eq=${float(acc.equity):.2f} "
                      f"{args.symbol} bid={bid:.3f} ask={ask:.3f} "
                      f"closed_macd={macd:+.4f} region={region} "
                      f"close={float(sig['close']):.3f} ema={float(sig['basis']):.3f} "
                      f"band=[{float(sig['lower']):.3f},{float(sig['upper']):.3f}] "
                      f"allow=L{int(bool(sig['long_allowed']))}/S{int(bool(sig['short_allowed']))} "
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
