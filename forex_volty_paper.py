"""MT5 paper/demo trader for Volty Expan Close.

Logic:
    - On each new closed candle, calculate SMA(TrueRange, length) * atr_mult.
    - upper = close + atrs, lower = close - atrs.
    - Tick execution: ask >= upper opens/reverses long, bid <= lower opens/reverses short.
    - If ATR is below min while flat, no new entries.
    - If ATR is below min while holding, opposite breakout closes only.
    - No TP/SL. Exit is opposite breakout close/reversal.

Default sends orders to the currently connected MT5 account. Use --dry-run to log only.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime, timezone

import numpy as np
from forex_synthetic_candles import SyntheticBidOHLC, timeframe_seconds

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


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
    }
    key = tf.lower()
    if key not in table:
        raise SystemExit(f"unsupported timeframe: {tf}")
    return table[key]


def is_synthetic_timeframe(tf: str) -> bool:
    return tf.lower().strip().endswith("s")


def round_volume(info, volume: float) -> float:
    step = float(info.volume_step or 0.01)
    vmin = float(info.volume_min or step)
    vmax = float(info.volume_max or volume)
    volume = max(vmin, min(vmax, volume))
    steps = math.floor((volume - vmin) / step + 1e-9)
    return round(vmin + steps * step, 8)


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


def send_order(mt5, args, side: str, reason: str, close_ticket: int | None = None):
    tick = mt5.symbol_info_tick(args.symbol)
    info = mt5.symbol_info(args.symbol)
    if tick is None or info is None:
        print("[volty-paper] ORDER SKIP no tick/info", flush=True)
        return None
    volume = round_volume(info, args.lot)
    is_buy = side == "buy"
    order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
    price = float(tick.ask if is_buy else tick.bid)
    req_base = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": args.symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "deviation": args.deviation,
        "magic": args.magic,
        "comment": f"volty_{reason}",
        "type_time": mt5.ORDER_TIME_GTC,
    }
    if close_ticket is not None:
        req_base["position"] = int(close_ticket)
    if args.dry_run:
        print(f"[volty-paper] DRY {side.upper()} vol={volume:g} px={price:.3f} reason={reason}", flush=True)
        return {"dry": True}
    last = None
    for filling in filling_candidates(mt5, args.symbol):
        req = dict(req_base)
        req["type_filling"] = filling
        res = mt5.order_send(req)
        last = res
        if res is not None and res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[volty-paper] ORDER OK {side.upper()} vol={volume:g} px={price:.3f} reason={reason}", flush=True)
            return res
    print(f"[volty-paper] ORDER FAIL side={side} last={getattr(last, 'retcode', None)}", flush=True)
    return None


def calc_levels_from_candles(mt5, args, candles: SyntheticBidOHLC):
    if len(candles.closed) < args.length + 1:
        return None
    rows = list(candles.closed)[-(args.length + 1):]
    high = np.array([c.high for c in rows], dtype=np.float64)
    low = np.array([c.low for c in rows], dtype=np.float64)
    close = np.array([c.close for c in rows], dtype=np.float64)
    open_ = np.array([c.open for c in rows], dtype=np.float64)
    info = mt5.symbol_info(args.symbol)
    point = float(getattr(info, "point", 0.01) or 0.01) if info else 0.01
    trs = []
    for i in range(len(rows) - args.length, len(rows)):
        prev_close = close[i - 1]
        tr = max(high[i] - low[i], abs(high[i] - prev_close), abs(low[i] - prev_close))
        trs.append(tr)
    raw_atr = float(np.mean(trs))
    hold_mult = args.hold_close_mult
    if hold_mult is None:
        hold_mult = args.atr_mult
    atrs = raw_atr * args.atr_mult
    hold_atrs = raw_atr * hold_mult
    c = float(close[-1])
    return {
        "bar_time": int(candles.last_closed_bucket * candles.tf_sec),
        "prev_green": bool(close[-1] > open_[-1]),
        "prev_red": bool(close[-1] < open_[-1]),
        "close": c,
        "raw_atr": raw_atr,
        "atrs": atrs,
        "hold_atrs": hold_atrs,
        "atr_pass": args.min_atr_points <= 0 or raw_atr >= args.min_atr_points * point,
        "upper": c + atrs,
        "lower": c - atrs,
        "hold_upper": c + hold_atrs,
        "hold_lower": c - hold_atrs,
    }


def calc_live_sampled_atr_levels(mt5, args):
    rates = mt5.copy_rates_from_pos(args.symbol, timeframe_value(mt5, args.atr_timeframe), 0, args.length + 2)
    tick = mt5.symbol_info_tick(args.symbol)
    if rates is None or tick is None or len(rates) < args.length + 1:
        return None
    high = np.array(rates["high"], dtype=np.float64)
    low = np.array(rates["low"], dtype=np.float64)
    close = np.array(rates["close"], dtype=np.float64)
    info = mt5.symbol_info(args.symbol)
    point = float(getattr(info, "point", 0.01) or 0.01) if info else 0.01
    trs = []
    # Includes the currently forming MT5 candle at index -1, matching the live
    # ATR-style behavior the user sees moving on MT5.
    for i in range(len(rates) - args.length, len(rates)):
        prev_close = close[i - 1]
        trs.append(max(high[i] - low[i], abs(high[i] - prev_close), abs(low[i] - prev_close)))
    raw_atr = float(np.mean(trs))
    hold_mult = args.hold_close_mult
    if hold_mult is None:
        hold_mult = args.atr_mult
    base = float(tick.bid)
    atrs = raw_atr * args.atr_mult
    hold_atrs = raw_atr * hold_mult
    prev_open = float(rates["open"][-2])
    prev_close = float(rates["close"][-2])
    return {
        "bar_time": int(time.time() // timeframe_seconds(args.timeframe)),
        "prev_green": bool(prev_close > prev_open),
        "prev_red": bool(prev_close < prev_open),
        "close": base,
        "raw_atr": raw_atr,
        "atrs": atrs,
        "hold_atrs": hold_atrs,
        "atr_pass": args.min_atr_points <= 0 or raw_atr >= args.min_atr_points * point,
        "upper": base + atrs,
        "lower": base - atrs,
        "hold_upper": base + hold_atrs,
        "hold_lower": base - hold_atrs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--timeframe", default="1m")
    ap.add_argument("--length", type=int, default=5)
    ap.add_argument("--atr-mult", type=float, default=0.75)
    ap.add_argument("--hold-close-mult", type=float, default=None,
                    help="ATR multiplier for close-only exits while holding. Default = atr-mult.")
    ap.add_argument("--min-atr-points", type=float, default=0.0)
    ap.add_argument("--entry-filter", choices=["none", "prev_color"], default="none")
    ap.add_argument("--lot", type=float, default=0.01)
    ap.add_argument("--magic", type=int, default=26051701)
    ap.add_argument("--deviation", type=int, default=50)
    ap.add_argument("--poll", type=float, default=0.25)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise SystemExit(f"mt5.initialize failed: {mt5.last_error()}")
    if not mt5.symbol_select(args.symbol, True):
        raise SystemExit(f"symbol_select failed: {args.symbol}")

    print(
        f"[volty-paper] start {args.symbol} tf={args.timeframe} length={args.length} "
        f"mult={args.atr_mult} hold_mult="
        f"{args.hold_close_mult if args.hold_close_mult is not None else args.atr_mult:g} "
        f"min_atr={args.min_atr_points:g}pt filter={args.entry_filter} "
        f"lot={args.lot:g} candles=synthetic_bid",
        flush=True,
    )
    last_bar = 0
    levels = None
    candles = SyntheticBidOHLC(args.timeframe, maxlen=max(args.length + 10, 1000))
    try:
        while True:
            tick = mt5.symbol_info_tick(args.symbol)
            if tick is not None and candles.update(tick):
                new_levels = calc_levels_from_candles(mt5, args, candles)
                if new_levels and int(new_levels["bar_time"]) != last_bar:
                    levels = new_levels
                    last_bar = int(levels["bar_time"])
                    print(
                        f"[volty-paper] levels close={levels['close']:.3f} "
                        f"raw_atr={levels['raw_atr']:.3f} atrs={levels['atrs']:.3f} "
                        f"hold_atrs={levels['hold_atrs']:.3f} "
                        f"pass={int(levels['atr_pass'])} upper={levels['upper']:.3f} "
                        f"lower={levels['lower']:.3f} hold_up={levels['hold_upper']:.3f} "
                        f"hold_dn={levels['hold_lower']:.3f} green={int(levels['prev_green'])} "
                        f"red={int(levels['prev_red'])}",
                        flush=True,
                    )
            pos = find_position(mt5, args.symbol, args.magic)
            side = 0 if pos is None else (1 if int(pos.type) == mt5.POSITION_TYPE_BUY else -1)
            if tick and levels:
                # Close-only hold levels fire first and never open the opposite
                # side on the same tick.
                if side == 1 and float(tick.bid) <= float(levels["hold_lower"]):
                    send_order(mt5, args, "sell", "hold_close_long", int(pos.ticket))
                elif side == -1 and float(tick.ask) >= float(levels["hold_upper"]):
                    send_order(mt5, args, "buy", "hold_close_short", int(pos.ticket))
                elif side <= 0 and float(tick.ask) >= float(levels["upper"]):
                    color_allows = args.entry_filter != "prev_color" or levels["prev_green"]
                    if side == -1 and not color_allows:
                        send_order(mt5, args, "buy", "color_close_short", int(pos.ticket))
                    elif not color_allows:
                        time.sleep(args.poll)
                        continue
                    elif pos is not None:
                        send_order(mt5, args, "buy", "reverse_close_short", int(pos.ticket))
                        if levels["atr_pass"]:
                            send_order(mt5, args, "buy", "break_upper")
                    elif levels["atr_pass"]:
                        send_order(mt5, args, "buy", "break_upper")
                elif side >= 0 and float(tick.bid) <= float(levels["lower"]):
                    color_allows = args.entry_filter != "prev_color" or levels["prev_red"]
                    if side == 1 and not color_allows:
                        send_order(mt5, args, "sell", "color_close_long", int(pos.ticket))
                    elif not color_allows:
                        time.sleep(args.poll)
                        continue
                    elif pos is not None:
                        send_order(mt5, args, "sell", "reverse_close_long", int(pos.ticket))
                        if levels["atr_pass"]:
                            send_order(mt5, args, "sell", "break_lower")
                    elif levels["atr_pass"]:
                        send_order(mt5, args, "sell", "break_lower")

                acc = mt5.account_info()
                ptxt = "-" if pos is None else ("L" if side == 1 else "S")
                pnl = 0.0 if pos is None else float(pos.profit)
                print(
                    f"[volty-paper] eq=${float(acc.equity):.2f} {args.symbol} "
                    f"bid={float(tick.bid):.3f} ask={float(tick.ask):.3f} pos={ptxt} p=${pnl:+.2f}",
                    flush=True,
                )
            time.sleep(args.poll)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
