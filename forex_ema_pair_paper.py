"""MT5 paper/demo trader for fast/slow EMA pair cross confirmation."""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from collections import deque

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def timeframe_value(mt5, tf: str):
    table = {
        "1m": mt5.TIMEFRAME_M1, "2m": mt5.TIMEFRAME_M2, "3m": mt5.TIMEFRAME_M3,
        "5m": mt5.TIMEFRAME_M5, "10m": mt5.TIMEFRAME_M10, "15m": mt5.TIMEFRAME_M15,
        "30m": mt5.TIMEFRAME_M30, "1h": mt5.TIMEFRAME_H1,
    }
    key = tf.lower()
    if key not in table:
        raise SystemExit(f"unsupported timeframe: {tf}")
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


def is_synthetic_timeframe(tf: str) -> bool:
    return tf.lower().strip().endswith("s")


def ema(values: np.ndarray, length: int) -> np.ndarray:
    out = np.empty(len(values), dtype=np.float64)
    alpha = 2.0 / (length + 1.0)
    val = float(values[0])
    for i, x in enumerate(values):
        val = alpha * float(x) + (1.0 - alpha) * val
        out[i] = val
    return out


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
        print("[ema-pair-paper] ORDER SKIP no tick/info", flush=True)
        return None
    volume = round_volume(info, args.lot)
    is_buy = side == "buy"
    order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
    price = float(tick.ask if is_buy else tick.bid)
    tp = sl = 0.0
    if close_ticket is None:
        if args.tp_points > 0:
            tp = price + args.tp_points * info.point if is_buy else price - args.tp_points * info.point
        if args.sl_points > 0:
            sl = price - args.sl_points * info.point if is_buy else price + args.sl_points * info.point
        tp = round(tp, info.digits) if tp else 0.0
        sl = round(sl, info.digits) if sl else 0.0
    req_base = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": args.symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": args.deviation,
        "magic": args.magic,
        "comment": f"emapair_{reason}",
        "type_time": mt5.ORDER_TIME_GTC,
    }
    if close_ticket is not None:
        req_base["position"] = int(close_ticket)
    if args.dry_run:
        print(f"[ema-pair-paper] DRY {side.upper()} vol={volume:g} px={price:.3f} sl={sl:.3f} tp={tp:.3f} reason={reason}", flush=True)
        return {"dry": True}
    last = None
    for filling in filling_candidates(mt5, args.symbol):
        req = dict(req_base)
        req["type_filling"] = filling
        res = mt5.order_send(req)
        last = res
        if res is not None and res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[ema-pair-paper] ORDER OK {side.upper()} vol={volume:g} px={price:.3f} reason={reason}", flush=True)
            return res
    print(f"[ema-pair-paper] ORDER FAIL side={side} last={getattr(last, 'retcode', None)}", flush=True)
    return None


def calc_state(mt5, args):
    need = max(args.fast_ema, args.slow_ema) + args.confirm_candles + 5
    rates = mt5.copy_rates_from_pos(args.symbol, timeframe_value(mt5, args.timeframe), 1, need)
    if rates is None or len(rates) < need - 2:
        return None
    close = np.array(rates["close"], dtype=np.float64)
    fast = ema(close, args.fast_ema)
    slow = ema(close, args.slow_ema)
    above = fast > slow
    below = fast < slow
    above_count = below_count = 0
    for a, b in zip(above, below):
        if a:
            above_count += 1; below_count = 0
        elif b:
            below_count += 1; above_count = 0
        else:
            above_count = below_count = 0
    regime = 1 if above_count >= args.confirm_candles else (-1 if below_count >= args.confirm_candles else 0)
    return {
        "bar_time": int(rates["time"][-1]),
        "fast": float(fast[-1]),
        "slow": float(slow[-1]),
        "regime": regime,
        "above_count": above_count,
        "below_count": below_count,
    }


class SyntheticBidCandles:
    def __init__(self, timeframe: str, maxlen: int):
        self.tf_sec = timeframe_seconds(timeframe)
        self.maxlen = maxlen
        self.bucket = None
        self.close = None
        self.closed = deque(maxlen=maxlen)
        self.last_closed_bucket = 0

    def update(self, tick) -> bool:
        bid = float(tick.bid)
        ts = int(getattr(tick, "time", 0) or time.time())
        bucket = ts // self.tf_sec
        if self.bucket is None:
            self.bucket = bucket
            self.close = bid
            return False
        if bucket != self.bucket:
            self.closed.append(float(self.close))
            self.last_closed_bucket = int(self.bucket)
            self.bucket = bucket
            self.close = bid
            return True
        self.close = bid
        return False

    def state(self, args):
        need = max(args.fast_ema, args.slow_ema) + args.confirm_candles + 5
        if len(self.closed) < need - 2:
            return None
        close = np.array(self.closed, dtype=np.float64)
        fast = ema(close, args.fast_ema)
        slow = ema(close, args.slow_ema)
        above = fast > slow
        below = fast < slow
        above_count = below_count = 0
        for a, b in zip(above, below):
            if a:
                above_count += 1; below_count = 0
            elif b:
                below_count += 1; above_count = 0
            else:
                above_count = below_count = 0
        regime = 1 if above_count >= args.confirm_candles else (-1 if below_count >= args.confirm_candles else 0)
        return {
            "bar_time": int(self.last_closed_bucket * self.tf_sec),
            "fast": float(fast[-1]),
            "slow": float(slow[-1]),
            "regime": regime,
            "above_count": above_count,
            "below_count": below_count,
        }


def latest_close_deal(mt5, args, since: datetime):
    deals = mt5.history_deals_get(since, datetime.now(timezone.utc))
    if not deals:
        return None
    ours = [
        d for d in deals
        if getattr(d, "symbol", "") == args.symbol
        and int(getattr(d, "magic", 0) or 0) == int(args.magic)
        and int(getattr(d, "entry", -1)) in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT)
    ]
    if not ours:
        return None
    return sorted(ours, key=lambda d: int(getattr(d, "time_msc", 0) or 0))[-1]


def classify_close_reason(deal, side: int, last_tp: float, last_sl: float, point: float) -> str:
    if deal is None:
        return "unknown"
    px = float(getattr(deal, "price", 0.0) or 0.0)
    tol = max(point * 10.0, 1e-9)
    if side == 1:
        if last_tp > 0 and px >= last_tp - tol:
            return "tp"
        if last_sl > 0 and px <= last_sl + tol:
            return "sl"
    elif side == -1:
        if last_tp > 0 and px <= last_tp + tol:
            return "tp"
        if last_sl > 0 and px >= last_sl - tol:
            return "sl"
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--timeframe", default="1m")
    ap.add_argument("--fast-ema", type=int, default=9)
    ap.add_argument("--slow-ema", type=int, default=150)
    ap.add_argument("--confirm-candles", type=int, default=1)
    ap.add_argument("--reverse-on-flip", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--reverse-on-tp", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--reverse-on-sl", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--tp-points", type=float, default=950)
    ap.add_argument("--sl-points", type=float, default=450)
    ap.add_argument("--lot", type=float, default=0.01)
    ap.add_argument("--magic", type=int, default=26051702)
    ap.add_argument("--deviation", type=int, default=50)
    ap.add_argument("--poll", type=float, default=0.5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.fast_ema >= args.slow_ema:
        raise SystemExit("--fast-ema must be < --slow-ema")

    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise SystemExit(f"mt5.initialize failed: {mt5.last_error()}")
    if not mt5.symbol_select(args.symbol, True):
        raise SystemExit(f"symbol_select failed: {args.symbol}")

    print(
        f"[ema-pair-paper] start {args.symbol} tf={args.timeframe} fast={args.fast_ema} "
        f"slow={args.slow_ema} confirm={args.confirm_candles} tp={args.tp_points:g} "
        f"sl={args.sl_points:g} flip={int(args.reverse_on_flip)} "
        f"rtp={int(args.reverse_on_tp)} rsl={int(args.reverse_on_sl)} "
        f"lot={args.lot:g} candles=synthetic_bid",
        flush=True,
    )
    last_bar = 0
    last_regime = None
    had_side = 0
    had_tp = 0.0
    had_sl = 0.0
    last_deal_ticket = 0
    hist_since = datetime.now(timezone.utc) - timedelta(days=2)
    need = max(args.fast_ema, args.slow_ema) + args.confirm_candles + 10
    candle_builder = SyntheticBidCandles(args.timeframe, need * 2)
    print("[ema-pair-paper] synthetic candles from BID ticks", flush=True)
    try:
        while True:
            pos = find_position(mt5, args.symbol, args.magic)
            side = 0 if pos is None else (1 if int(pos.type) == mt5.POSITION_TYPE_BUY else -1)

            if had_side != 0 and side == 0:
                deal = latest_close_deal(mt5, args, hist_since)
                ticket = int(getattr(deal, "ticket", 0) or 0) if deal else 0
                if deal and ticket != last_deal_ticket:
                    last_deal_ticket = ticket
                    info = mt5.symbol_info(args.symbol)
                    profit = float(getattr(deal, "profit", 0.0) or 0.0)
                    close_reason = classify_close_reason(
                        deal, had_side, had_tp, had_sl,
                        float(getattr(info, "point", 0.01) or 0.01) if info else 0.01,
                    )
                    print(
                        f"[ema-pair-paper] CLOSED reason={close_reason} "
                        f"profit=${profit:+.2f} px={float(getattr(deal, 'price', 0.0) or 0.0):.3f} "
                        f"tp={had_tp:.3f} sl={had_sl:.3f}",
                        flush=True,
                    )
                    if (close_reason == "tp" and args.reverse_on_tp) or (
                        close_reason == "sl" and args.reverse_on_sl
                    ):
                        send_order(mt5, args, "sell" if had_side == 1 else "buy", f"reverse_after_{close_reason}")
                had_side = 0
                had_tp = 0.0
                had_sl = 0.0

            tick = mt5.symbol_info_tick(args.symbol)
            synthetic_new_bar = False
            if tick is not None:
                synthetic_new_bar = candle_builder.update(tick)
            state = candle_builder.state(args)
            if state and int(state["bar_time"]) != last_bar:
                if not synthetic_new_bar:
                    time.sleep(args.poll)
                    continue
                last_bar = int(state["bar_time"])
                regime = int(state["regime"])
                pos = find_position(mt5, args.symbol, args.magic)
                side = 0 if pos is None else (1 if int(pos.type) == mt5.POSITION_TYPE_BUY else -1)

                if last_regime is None:
                    last_regime = regime
                    print(
                        f"[ema-pair-paper] baseline reg={regime} "
                        f"fast-slow={state['fast'] - state['slow']:+.3f} "
                        f"fast={state['fast']:.3f} slow={state['slow']:.3f}",
                        flush=True,
                    )
                    continue

                prev_regime = last_regime
                if side == 1 and args.sl_points <= 0 and regime == -1:
                    send_order(mt5, args, "sell", "flip_close_long", int(pos.ticket))
                    if args.reverse_on_flip and args.confirm_candles == 1:
                        send_order(mt5, args, "sell", "flip_open_short")
                elif side == -1 and args.sl_points <= 0 and regime == 1:
                    send_order(mt5, args, "buy", "flip_close_short", int(pos.ticket))
                    if args.reverse_on_flip and args.confirm_candles == 1:
                        send_order(mt5, args, "buy", "flip_open_long")
                elif side == 0 and regime != 0 and regime != last_regime:
                    send_order(mt5, args, "buy" if regime == 1 else "sell", "signal")

                last_regime = regime
                print(
                    f"[ema-pair-paper] bar reg={regime} prev={prev_regime} "
                    f"fast-slow={state['fast'] - state['slow']:+.3f} "
                    f"fast={state['fast']:.3f} slow={state['slow']:.3f} "
                    f"ac={state['above_count']} bc={state['below_count']} pos={side}",
                    flush=True,
                )

            pos = find_position(mt5, args.symbol, args.magic)
            if pos is not None:
                had_side = 1 if int(pos.type) == mt5.POSITION_TYPE_BUY else -1
                had_tp = float(getattr(pos, "tp", 0.0) or 0.0)
                had_sl = float(getattr(pos, "sl", 0.0) or 0.0)
            acc = mt5.account_info()
            tick = mt5.symbol_info_tick(args.symbol)
            ptxt = "-" if pos is None else ("L" if had_side == 1 else "S")
            pnl = 0.0 if pos is None else float(pos.profit)
            if acc and tick:
                print(
                    f"[ema-pair-paper] eq=${float(acc.equity):.2f} bid={float(tick.bid):.3f} "
                    f"ask={float(tick.ask):.3f} pos={ptxt} p=${pnl:+.2f}",
                    flush=True,
                )
            time.sleep(args.poll)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
