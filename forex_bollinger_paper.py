"""MT5 paper/demo trader for Bollinger band tick execution.

Matches `forex_bollinger_tick_backtest.py` mechanics:
    - Build closed bid candles from live MT5 ticks.
    - Calculate SMA/stdev Bollinger levels from closed candle closes.
    - Normal entry: cross back above lower band -> long, cross back below upper -> short.
    - `opposite` exit reverses/closes on the opposite signal.
    - `basis` exits at the middle band.
    - `basis_trail` arms after basis touch and trails by band width * multiplier.

Default params use the current stable candidate:
    1m length=55 mult=2 exit=basis_trail trail_mult=0.15
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

from forex_synthetic_candles import Candle
from forex_synthetic_candles import SyntheticBidOHLC

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


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
        print("[bb-paper] ORDER SKIP no tick/info", flush=True)
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
        "comment": f"bb_{reason}",
        "type_time": mt5.ORDER_TIME_GTC,
    }
    if close_ticket is not None:
        req_base["position"] = int(close_ticket)
    if args.dry_run:
        print(f"[bb-paper] DRY {side.upper()} vol={volume:g} px={price:.3f} reason={reason}", flush=True)
        return {"dry": True}

    last = None
    for filling in filling_candidates(mt5, args.symbol):
        req = dict(req_base)
        req["type_filling"] = filling
        res = mt5.order_send(req)
        last = res
        if res is not None and res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[bb-paper] ORDER OK {side.upper()} vol={volume:g} px={price:.3f} reason={reason}", flush=True)
            return res
    print(f"[bb-paper] ORDER FAIL side={side} last={getattr(last, 'retcode', None)} reason={reason}", flush=True)
    return None


def calc_bands(args, candles: SyntheticBidOHLC):
    if len(candles.closed) < args.length:
        return None
    rows = list(candles.closed)[-args.length:]
    closes = np.array([c.close for c in rows], dtype=np.float64)
    basis = float(np.mean(closes))
    std = float(np.std(closes))
    upper = basis + args.mult * std
    lower = basis - args.mult * std
    half_width = max(abs(basis - lower), abs(upper - basis))
    return {
        "bar_time": int(candles.last_closed_bucket * candles.tf_sec),
        "basis": basis,
        "upper": upper,
        "lower": lower,
        "half_width": half_width,
        "close": float(closes[-1]),
    }


def preload_candles(mt5, args, candles: SyntheticBidOHLC) -> bool:
    """Seed closed bid candles from MT5 history so live mode starts immediately."""
    tf_map = {
        "1m": mt5.TIMEFRAME_M1,
        "2m": mt5.TIMEFRAME_M2,
        "3m": mt5.TIMEFRAME_M3,
        "5m": mt5.TIMEFRAME_M5,
        "10m": mt5.TIMEFRAME_M10,
        "15m": mt5.TIMEFRAME_M15,
        "30m": mt5.TIMEFRAME_M30,
        "1h": mt5.TIMEFRAME_H1,
    }
    tf = args.timeframe.lower().strip()
    if tf not in tf_map:
        return False
    need = args.length + 5
    rates = mt5.copy_rates_from_pos(args.symbol, tf_map[tf], 1, need)
    if rates is None or len(rates) < args.length:
        return False
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
    return True


def region_text(px: float, levels: dict) -> str:
    if px >= levels["upper"]:
        return "above_upper"
    if px <= levels["lower"]:
        return "below_lower"
    if px >= levels["basis"]:
        return "upper_half"
    return "lower_half"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--timeframe", default="1m")
    ap.add_argument("--length", type=int, default=55)
    ap.add_argument("--mult", type=float, default=2.0)
    ap.add_argument("--exit-mode", choices=["opposite", "basis", "basis_trail"], default="basis_trail")
    ap.add_argument("--trail-band-mult", type=float, default=0.15)
    ap.add_argument("--trail-points", type=float, default=0.0)
    ap.add_argument("--invert-signals", action="store_true")
    ap.add_argument("--reverse-on-flip", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--lot", type=float, default=0.01)
    ap.add_argument("--magic", type=int, default=26052055)
    ap.add_argument("--deviation", type=int, default=50)
    ap.add_argument("--poll", type=float, default=0.25)
    ap.add_argument("--log-every", type=float, default=3.0,
                    help="seconds between reference logs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise SystemExit(f"mt5.initialize failed: {mt5.last_error()}")
    if not mt5.symbol_select(args.symbol, True):
        raise SystemExit(f"symbol_select failed: {args.symbol}")

    print(
        f"[bb-paper] start {args.symbol} tf={args.timeframe} length={args.length} "
        f"mult={args.mult:g} exit={args.exit_mode} trail_mult={args.trail_band_mult:g} "
        f"reverse={int(args.reverse_on_flip)} invert={int(args.invert_signals)} "
        f"lot={args.lot:g} candles=synthetic_bid",
        flush=True,
    )

    candles = SyntheticBidOHLC(args.timeframe, maxlen=max(args.length + 20, 1000))
    levels = None
    last_bar = 0
    prev_bid = prev_ask = None
    armed = False
    best_px = None
    last_log = 0.0
    if preload_candles(mt5, args, candles):
        levels = calc_bands(args, candles)
        if levels:
            last_bar = int(levels["bar_time"])
            print(
                f"[bb-paper] preload candles={len(candles.closed)} close={levels['close']:.3f} "
                f"basis={levels['basis']:.3f} upper={levels['upper']:.3f} "
                f"lower={levels['lower']:.3f} half={levels['half_width']:.3f}",
                flush=True,
            )
    else:
        print(
            f"[bb-paper] preload unavailable for tf={args.timeframe}; "
            f"waiting for {args.length} closed synthetic candles",
            flush=True,
        )

    try:
        while True:
            tick = mt5.symbol_info_tick(args.symbol)
            info = mt5.symbol_info(args.symbol)
            if tick is None or info is None:
                time.sleep(args.poll)
                continue

            if candles.update(tick):
                new_levels = calc_bands(args, candles)
                if new_levels and int(new_levels["bar_time"]) != last_bar:
                    levels = new_levels
                    last_bar = int(levels["bar_time"])
                    print(
                        f"[bb-paper] levels close={levels['close']:.3f} "
                        f"basis={levels['basis']:.3f} upper={levels['upper']:.3f} "
                        f"lower={levels['lower']:.3f} half={levels['half_width']:.3f}",
                        flush=True,
                    )

            bid = float(tick.bid)
            ask = float(tick.ask)
            mid = bid
            if prev_bid is None:
                prev_bid = bid
                prev_ask = ask
                time.sleep(args.poll)
                continue

            pos = find_position(mt5, args.symbol, args.magic)
            side = 0 if pos is None else (1 if int(pos.type) == mt5.POSITION_TYPE_BUY else -1)

            if levels:
                long_sig = prev_bid <= levels["lower"] and bid > levels["lower"]
                short_sig = prev_bid >= levels["upper"] and bid < levels["upper"]
                if args.invert_signals:
                    long_sig, short_sig = short_sig, long_sig

                if side == 1:
                    if args.exit_mode == "basis_trail":
                        if bid >= levels["basis"]:
                            armed = True
                        if armed:
                            best_px = bid if best_px is None else max(best_px, bid)
                            dist = args.trail_points * info.point if args.trail_points > 0 else levels["half_width"] * args.trail_band_mult
                            if dist > 0 and bid <= best_px - dist:
                                send_order(mt5, args, "sell", "basis_trail_long", int(pos.ticket))
                                armed = False
                                best_px = None
                        # No same-tick reverse on basis_trail exits.
                    elif args.exit_mode == "basis" and bid >= levels["basis"]:
                        send_order(mt5, args, "sell", "basis_long", int(pos.ticket))
                    elif args.exit_mode == "opposite" and short_sig:
                        send_order(mt5, args, "sell", "flip_close_long", int(pos.ticket))
                        if args.reverse_on_flip:
                            send_order(mt5, args, "sell", "flip_open_short")
                elif side == -1:
                    if args.exit_mode == "basis_trail":
                        if ask <= levels["basis"]:
                            armed = True
                        if armed:
                            best_px = ask if best_px is None else min(best_px, ask)
                            dist = args.trail_points * info.point if args.trail_points > 0 else levels["half_width"] * args.trail_band_mult
                            if dist > 0 and ask >= best_px + dist:
                                send_order(mt5, args, "buy", "basis_trail_short", int(pos.ticket))
                                armed = False
                                best_px = None
                    elif args.exit_mode == "basis" and ask <= levels["basis"]:
                        send_order(mt5, args, "buy", "basis_short", int(pos.ticket))
                    elif args.exit_mode == "opposite" and long_sig:
                        send_order(mt5, args, "buy", "flip_close_short", int(pos.ticket))
                        if args.reverse_on_flip:
                            send_order(mt5, args, "buy", "flip_open_long")
                else:
                    armed = False
                    best_px = None
                    if long_sig:
                        send_order(mt5, args, "buy", "entry_long")
                    elif short_sig:
                        send_order(mt5, args, "sell", "entry_short")

                now = time.time()
                if now - last_log >= args.log_every:
                    last_log = now
                    point = float(getattr(info, "point", 0.01) or 0.01)
                    d_basis = (bid - levels["basis"]) / point
                    d_lower = (bid - levels["lower"]) / point
                    d_upper = (bid - levels["upper"]) / point
                    width_pts = (levels["upper"] - levels["lower"]) / point
                    trail_dist = args.trail_points if args.trail_points > 0 else (levels["half_width"] * args.trail_band_mult / point)
                    trail_ref = "-"
                    trail_stop = "-"
                    if armed and best_px is not None:
                        if side == 1:
                            stop_px = best_px - trail_dist * point
                            trail_ref = f"best={best_px:.3f}"
                            trail_stop = f"stop={stop_px:.3f} d_stop={(bid - stop_px) / point:+.0f}pt"
                        elif side == -1:
                            stop_px = best_px + trail_dist * point
                            trail_ref = f"best={best_px:.3f}"
                            trail_stop = f"stop={stop_px:.3f} d_stop={(stop_px - ask) / point:+.0f}pt"
                    sig_txt = []
                    raw_long = prev_bid <= levels["lower"] and bid > levels["lower"]
                    raw_short = prev_bid >= levels["upper"] and bid < levels["upper"]
                    if raw_long:
                        sig_txt.append("lower_reclaim")
                    if raw_short:
                        sig_txt.append("upper_reject")
                    sig = ",".join(sig_txt) if sig_txt else "-"
                    ptxt = "-"
                    if pos is not None:
                        pnl = float(getattr(pos, "profit", 0.0) or 0.0)
                        entry = float(getattr(pos, "price_open", 0.0) or 0.0)
                        side_name = "L" if side == 1 else "S"
                        ptxt = f"{side_name} entry={entry:.3f} p=${pnl:+.2f}"
                    print(
                        f"[bb-paper] px={bid:.3f}/{ask:.3f} reg={region_text(bid, levels)} "
                        f"dB={d_basis:+.0f}pt dL={d_lower:+.0f}pt dU={d_upper:+.0f}pt "
                        f"width={width_pts:.0f}pt basis={levels['basis']:.3f} "
                        f"L={levels['lower']:.3f} U={levels['upper']:.3f} "
                        f"sig={sig} pos={ptxt} armed={int(armed)} "
                        f"trail={trail_dist:.0f}pt {trail_ref} {trail_stop}",
                        flush=True,
                    )

            prev_bid = bid
            prev_ask = ask
            time.sleep(args.poll)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
