"""MT5 paper/demo trader for Parabolic SAR stop-reversal.

Default behavior mirrors the TradingView sample:
    - Calculate SAR on closed candles.
    - If SAR state is uptrend, watch nextBarSAR as a short stop.
    - If SAR state is downtrend, watch nextBarSAR as a long stop.
    - When stop is hit, close current position and open/reverse.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

from forex_parabolic_sar_tick_backtest import parabolic_sar
from forex_synthetic_candles import SyntheticBidOHLC

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
        "4h": mt5.TIMEFRAME_H4,
    }
    key = tf.lower().strip()
    if key not in table:
        raise SystemExit(f"unsupported timeframe: {tf}")
    return table[key]


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


def pos_side(mt5, pos) -> int:
    if pos is None:
        return 0
    typ = int(getattr(pos, "type", -1))
    if typ == mt5.POSITION_TYPE_BUY:
        return 1
    if typ == mt5.POSITION_TYPE_SELL:
        return -1
    return 0


def send_order(mt5, args, side: str, reason: str, close_ticket: int | None = None):
    tick = mt5.symbol_info_tick(args.symbol)
    info = mt5.symbol_info(args.symbol)
    if tick is None or info is None:
        print("[psar-paper] ORDER SKIP no tick/info", flush=True)
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
        "comment": f"psar_{reason}",
        "type_time": mt5.ORDER_TIME_GTC,
    }
    if close_ticket is not None:
        req_base["position"] = int(close_ticket)
    if args.dry_run:
        print(f"[psar-paper] DRY {side.upper()} vol={volume:g} px={price:.3f} reason={reason}", flush=True)
        return {"dry": True}
    last = None
    for filling in filling_candidates(mt5, args.symbol):
        req = dict(req_base)
        req["type_filling"] = filling
        res = mt5.order_send(req)
        last = res
        if res is not None and res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[psar-paper] ORDER OK {side.upper()} vol={volume:g} px={price:.3f} reason={reason}", flush=True)
            return res
    print(f"[psar-paper] ORDER FAIL side={side} last={getattr(last, 'retcode', None)}", flush=True)
    return None


def calc_psar_state(candles: SyntheticBidOHLC, args):
    need = max(5, args.bars)
    if len(candles.closed) < need:
        return None
    rows = list(candles.closed)[-need:]
    high = np.array([c.high for c in rows], dtype=np.float64)
    low = np.array([c.low for c in rows], dtype=np.float64)
    close = np.array([c.close for c in rows], dtype=np.float64)
    uptrend, sar, next_sar = parabolic_sar(high, low, close, args.start, args.increment, args.maximum)
    idx = len(close) - 1
    level = float(next_sar[idx])
    if not np.isfinite(level):
        return None
    return {
        "bar_time": int(candles.last_closed_bucket * candles.tf_sec),
        "uptrend": bool(uptrend[idx]),
        "sar": float(sar[idx]) if np.isfinite(sar[idx]) else float("nan"),
        "next_sar": level,
        "high": float(high[-1]),
        "low": float(low[-1]),
        "close": float(close[-1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--timeframe", default="1m")
    ap.add_argument("--start", type=float, default=0.03)
    ap.add_argument("--increment", type=float, default=0.03)
    ap.add_argument("--maximum", type=float, default=0.2)
    ap.add_argument("--bars", type=int, default=300)
    ap.add_argument("--lot", type=float, default=0.01)
    ap.add_argument("--magic", type=int, default=26051903)
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
        f"[psar-paper] start {args.symbol} tf={args.timeframe} "
        f"start={args.start:g} inc={args.increment:g} max={args.maximum:g} "
        f"lot={args.lot:g} dry={int(args.dry_run)} candles=synthetic_bid",
        flush=True,
    )

    state = None
    last_bar = 0
    consumed_bar = 0
    last_log = 0.0
    candles = SyntheticBidOHLC(args.timeframe, maxlen=max(args.bars + 10, 1000))

    try:
        while True:
            tick = mt5.symbol_info_tick(args.symbol)
            if tick is not None and candles.update(tick):
                new_state = calc_psar_state(candles, args)
                if new_state and int(new_state["bar_time"]) != last_bar:
                    state = new_state
                    last_bar = int(state["bar_time"])
                    consumed_bar = 0
                    direction = "uptrend -> short stop" if state["uptrend"] else "downtrend -> long stop"
                    print(
                        f"[psar-paper] bar close={state['close']:.3f} sar={state['sar']:.3f} "
                        f"next={state['next_sar']:.3f} {direction}",
                        flush=True,
                    )
            pos = find_position(mt5, args.symbol, args.magic)
            side = pos_side(mt5, pos)
            if tick is None or state is None:
                time.sleep(args.poll)
                continue

            bid = float(tick.bid)
            ask = float(tick.ask)
            level = float(state["next_sar"])
            wants_short = bool(state["uptrend"])
            wants_long = not wants_short
            hit_short = wants_short and bid <= level
            hit_long = wants_long and ask >= level

            if consumed_bar != last_bar and (hit_short or hit_long):
                consumed_bar = last_bar
                if hit_short:
                    if side == 1:
                        send_order(mt5, args, "sell", "flip_close", int(pos.ticket))
                        time.sleep(0.1)
                    pos = find_position(mt5, args.symbol, args.magic)
                    if pos_side(mt5, pos) == 0:
                        send_order(mt5, args, "sell", "flip_open")
                elif hit_long:
                    if side == -1:
                        send_order(mt5, args, "buy", "flip_close", int(pos.ticket))
                        time.sleep(0.1)
                    pos = find_position(mt5, args.symbol, args.magic)
                    if pos_side(mt5, pos) == 0:
                        send_order(mt5, args, "buy", "flip_open")

            now = time.time()
            if now - last_log >= 2.0:
                acc = mt5.account_info()
                pos = find_position(mt5, args.symbol, args.magic)
                side = pos_side(mt5, pos)
                pos_txt = "-" if side == 0 else ("L" if side == 1 else "S")
                p = float(getattr(pos, "profit", 0.0) or 0.0) if pos else 0.0
                trigger = "short" if wants_short else "long"
                dist = (bid - level) if wants_short else (level - ask)
                print(
                    f"[psar-paper] eq=${float(acc.equity if acc else 0):.2f} "
                    f"bid={bid:.3f} ask={ask:.3f} pos={pos_txt} p=${p:+.2f} "
                    f"next={level:.3f} wait={trigger} dist={dist:+.3f}",
                    flush=True,
                )
                last_log = now

            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("[psar-paper] stopped", flush=True)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
