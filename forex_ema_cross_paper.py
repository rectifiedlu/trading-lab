"""MT5 EMA candle-confirmation cross paper/demo trader for XAUUSD.

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
TRADES_CSV = os.path.join(DATA_DIR, "forex_ema_cross_paper_trades.csv")
DEFAULT_MAGIC = 260515


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_trade_log(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "ts", "symbol", "side", "ticket", "entry", "exit", "volume",
            "profit", "reason", "close", "ema", "side_state",
            "above_count", "below_count", "ema_length", "confirm_candles",
            "timeframe", "tp_points",
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


def calc_closed_ema_state(mt5, args) -> dict[str, float | int | str] | None:
    rates = mt5.copy_rates_from_pos(
        args.symbol, timeframe_value(mt5, args.timeframe), 1, args.bars,
    )
    if rates is None or len(rates) < args.ema_length + args.confirm_candles + 2:
        return None
    close = np.array(rates["close"], dtype=np.float64)
    basis = ema(close, args.ema_length)
    above_count = 0
    below_count = 0
    for px, ma in zip(close, basis):
        if not np.isfinite(ma):
            above_count = 0
            below_count = 0
        elif px > ma:
            above_count += 1
            below_count = 0
        elif px < ma:
            below_count += 1
            above_count = 0
        else:
            above_count = 0
            below_count = 0
    side_state = "neutral"
    if above_count >= args.confirm_candles:
        side_state = "long"
    elif below_count >= args.confirm_candles:
        side_state = "short"
    return {
        "close": float(close[-1]),
        "ema": float(basis[-1]),
        "bar_time": int(rates["time"][-1]),
        "above_count": int(above_count),
        "below_count": int(below_count),
        "side_state": side_state,
        "above": bool(close[-1] > basis[-1]),
        "below": bool(close[-1] < basis[-1]),
    }


def calc_live_ema_state(mt5, args) -> dict[str, float | int | str] | None:
    # EMA/bands are from fully closed candles only. Live ticks only decide
    # whether price crosses those fixed bands.
    rates = mt5.copy_rates_from_pos(
        args.symbol, timeframe_value(mt5, args.timeframe), 1, args.bars,
    )
    tick = mt5.symbol_info_tick(args.symbol)
    if rates is None or tick is None or len(rates) < args.ema_length + 2:
        return None
    close = np.array(rates["close"], dtype=np.float64)
    basis = ema(close, args.ema_length)
    entry_deadband = args.entry_deadband_points * args.point_size
    exit_deadband = args.exit_deadband_points * args.point_size
    entry_upper = float(basis[-1] + entry_deadband)
    entry_lower = float(basis[-1] - entry_deadband)
    exit_upper = float(basis[-1] + exit_deadband)
    exit_lower = float(basis[-1] - exit_deadband)
    price = (float(tick.bid) + float(tick.ask)) / 2.0
    dist_points = (price - float(basis[-1])) / args.point_size
    side_state = "above_band" if price >= entry_upper else ("below_band" if price <= entry_lower else "inside_band")
    return {
        "close": price,
        "ema": float(basis[-1]),
        "upper": entry_upper,
        "lower": entry_lower,
        "entry_upper": entry_upper,
        "entry_lower": entry_lower,
        "exit_upper": exit_upper,
        "exit_lower": exit_lower,
        "bar_time": int(rates["time"][-1]),
        "above_count": int(price >= entry_upper),
        "below_count": int(price <= entry_lower),
        "side_state": side_state,
        "above": bool(price >= entry_upper),
        "below": bool(price <= entry_lower),
        "dist_points": dist_points,
    }


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
        print(f"[ema-cross-paper] ORDER SKIP no tick", flush=True)
        return None
    is_buy = side == "buy"
    order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
    price = float(tick.ask if is_buy else tick.bid)
    if close_ticket is None and args.compound:
        volume = max_volume_for_margin(mt5, args.symbol, order_type, price)
        if volume <= 0:
            print(f"[ema-cross-paper] ORDER SKIP compound no free margin side={side}", flush=True)
            return None
    req_base = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": args.symbol,
        "volume": float(volume),
        "type": order_type,
        "price": price,
        "deviation": args.deviation,
        "magic": args.magic,
        "comment": f"emacross_{reason}",
        "type_time": mt5.ORDER_TIME_GTC,
    }
    if close_ticket is not None:
        req_base["position"] = int(close_ticket)

    if args.dry_run:
        print(f"[ema-cross-paper] DRY ORDER {side.upper()} vol={volume:g} "
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
            print(f"[ema-cross-paper] ORDER OK {side.upper()} vol={volume:g} "
                  f"px={price:.3f} reason={reason} order={getattr(res, 'order', 0)}",
                  flush=True)
            return res
        print(f"[ema-cross-paper] ORDER FAIL fill={filling} "
              f"ret={getattr(res, 'retcode', None)}", flush=True)
    print(f"[ema-cross-paper] ORDER FAIL all fillings last={getattr(last, 'retcode', None)}", flush=True)
    return None


def log_trade(args, pos, exit_px: float, reason: str, state: dict[str, float | int | str]):
    with open(args.trades_csv, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            now_iso(), args.symbol, "long" if pos.type == 0 else "short",
            int(pos.ticket), float(pos.price_open), exit_px, float(pos.volume),
            float(getattr(pos, "profit", 0.0)), reason,
            float(state["close"]), float(state["ema"]), str(state["side_state"]),
            int(state["above_count"]), int(state["below_count"]),
            args.ema_length, args.confirm_candles, args.timeframe, args.tp_points,
        ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--timeframe", default="3m")
    ap.add_argument("--ema-length", type=int, default=21)
    ap.add_argument("--confirm-candles", type=int, default=2)
    ap.add_argument("--signal-mode", choices=["candle", "tick", "sample"], default="sample",
                    help="sample=check live price vs closed-candle EMA every N seconds")
    ap.add_argument("--sample-sec", type=float, default=10.0)
    ap.add_argument("--deadband-points", type=float, default=0.0,
                    help="shortcut used for entry/exit deadband if either is omitted")
    ap.add_argument("--entry-deadband-points", type=float, default=None)
    ap.add_argument("--exit-deadband-points", type=float, default=None)
    ap.add_argument("--tp-points", type=float, default=200.0)
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
    ap.add_argument("--reverse-on-flip", action=argparse.BooleanOptionalAction, default=True,
                    help="close and immediately open opposite side on EMA failure")
    ap.add_argument("--trades-csv", default=TRADES_CSV)
    args = ap.parse_args()
    if args.entry_deadband_points is None:
        args.entry_deadband_points = args.deadband_points
    if args.exit_deadband_points is None:
        args.exit_deadband_points = args.deadband_points

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
    prev_live_price: float | None = None
    prev_upper: float | None = None
    prev_lower: float | None = None
    last_sample_at = 0.0
    last_sample_region = "unknown"
    print(f"[ema-cross-paper] start symbol={args.symbol} tf={args.timeframe} "
          f"mode={args.signal_mode} ema={args.ema_length} "
          f"confirm={args.confirm_candles} "
          f"entry_db={args.entry_deadband_points:g}pt "
          f"exit_db={args.exit_deadband_points:g}pt "
          f"sample={args.sample_sec:g}s reverse={int(args.reverse_on_flip)} "
          f"tp={args.tp_points:g}pt lot={volume:g} "
          f"compound={int(args.compound)} dry={int(args.dry_run)}", flush=True)

    try:
        while True:
            tick = mt5.symbol_info_tick(args.symbol)
            state = (
                calc_live_ema_state(mt5, args)
                if args.signal_mode in ("tick", "sample")
                else calc_closed_ema_state(mt5, args)
            )
            pos = find_position(mt5, args.symbol, args.magic)
            if tick is None or state is None:
                time.sleep(args.poll_sec)
                continue
            bar_time = int(state["bar_time"])
            bid = float(tick.bid)
            ask = float(tick.ask)
            region = str(state["side_state"])
            new_closed_bar = last_bar_time is None or bar_time != last_bar_time
            should_sample = (
                args.signal_mode != "sample"
                or time.time() - last_sample_at >= args.sample_sec
            )
            if args.signal_mode == "tick":
                price = float(state["close"])
                upper = float(state["upper"])
                lower = float(state["lower"])
                fresh_long = (
                    prev_live_price is not None
                    and prev_upper is not None
                    and prev_live_price < prev_upper
                    and price >= upper
                )
                fresh_short = (
                    prev_live_price is not None
                    and prev_lower is not None
                    and prev_live_price > prev_lower
                    and price <= lower
                )
                region_changed = fresh_long or fresh_short
                region = "long" if fresh_long else ("short" if fresh_short else region)
            elif args.signal_mode == "sample":
                fresh_long = fresh_short = False
                if should_sample:
                    price = float(state["close"])
                    entry_upper = float(state["entry_upper"])
                    entry_lower = float(state["entry_lower"])
                    exit_upper = float(state["exit_upper"])
                    exit_lower = float(state["exit_lower"])
                    if last_sample_region == "long":
                        sample_region = "short" if price <= exit_lower else "long"
                    elif last_sample_region == "short":
                        sample_region = "long" if price >= exit_upper else "short"
                    else:
                        sample_region = (
                            "long" if price >= entry_upper
                            else ("short" if price <= entry_lower else "inside")
                        )
                    startup_sample = last_sample_region == "unknown"
                    fresh_long = (
                        not startup_sample
                        and sample_region == "long"
                        and last_sample_region != "long"
                    )
                    fresh_short = (
                        not startup_sample
                        and sample_region == "short"
                        and last_sample_region != "short"
                    )
                    region_changed = fresh_long or fresh_short
                    region = sample_region
                    print(
                        f"[ema-cross-paper] SAMPLE region={sample_region} "
                        f"price={price:.3f} ema={float(state['ema']):.3f} "
                        f"entry=[{entry_lower:.3f},{entry_upper:.3f}] "
                        f"exit=[{exit_lower:.3f},{exit_upper:.3f}] "
                        f"dist={float(state['dist_points']):+.1f}pt "
                        f"fresh=L{int(fresh_long)}/S{int(fresh_short)} "
                        f"startup={int(startup_sample)}",
                        flush=True,
                    )
                    last_sample_region = sample_region
                    last_sample_at = time.time()
                else:
                    region_changed = False
            else:
                signal_changed = seen_region and region != last_region
                region_changed = new_closed_bar and signal_changed
                fresh_long = region_changed and region == "long"
                fresh_short = region_changed and region == "short"

            current_ticket = int(pos.ticket) if pos is not None else None
            if last_ticket is not None and current_ticket is None:
                # Position disappeared without this loop closing it: manual close,
                # broker-side close, or restart sync. Do not re-enter same region.
                if last_region in ("long", "short"):
                    blocked_side = last_region
                print(f"[ema-cross-paper] SYNC flat after external close; "
                      f"blocking {blocked_side or '-'} until region changes", flush=True)
            last_ticket = current_ticket

            if blocked_side and region != blocked_side:
                blocked_side = None

            if pos is not None:
                side = "long" if int(pos.type) == mt5.ORDER_TYPE_BUY else "short"
                entry = float(pos.price_open)
                pnl_points = ((bid - entry) if side == "long" else (entry - ask)) / args.point_size
                tp_hit = pnl_points >= args.tp_points
                if args.signal_mode in ("tick", "sample"):
                    flip = (
                        (side == "long" and fresh_short) or
                        (side == "short" and fresh_long)
                    )
                else:
                    failed = (
                        (side == "long" and bool(state["below"])) or
                        (side == "short" and bool(state["above"]))
                    )
                    flip = new_closed_bar and failed
                if tp_hit:
                    close_side = "sell" if side == "long" else "buy"
                    res = send_order(mt5, args, close_side, float(pos.volume), "tp", int(pos.ticket))
                    if res is not None:
                        log_trade(args, pos, bid if side == "long" else ask, "tp", state)
                        if not args.reenter_after_tp:
                            blocked_side = side
                        last_ticket = None
                elif flip:
                    close_side = "sell" if side == "long" else "buy"
                    res = send_order(mt5, args, close_side, float(pos.volume), "flip_close", int(pos.ticket))
                    if res is not None:
                        log_trade(args, pos, bid if side == "long" else ask, "flip", state)
                        last_ticket = None
                        print(
                            f"[ema-cross-paper] FLIP close {side} "
                            f"price={float(state['close']):.3f} "
                            f"ema={float(state['ema']):.3f} "
                            f"band=[{float(state.get('lower', 0.0)):.3f},"
                            f"{float(state.get('upper', 0.0)):.3f}]",
                            flush=True,
                        )
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
                print(f"[ema-cross-paper] bal=${float(acc.balance):.2f} eq=${float(acc.equity):.2f} "
                      f"{args.symbol} bid={bid:.3f} ask={ask:.3f} "
                      f"close={float(state['close']):.3f} ema={float(state['ema']):.3f} "
                      f"dist={float(state.get('dist_points', (float(state['close']) - float(state['ema'])) / args.point_size)):+.1f}pt "
                      f"entry=[{float(state.get('entry_lower', state.get('lower', state['ema']))):.3f},"
                      f"{float(state.get('entry_upper', state.get('upper', state['ema']))):.3f}] "
                      f"exit=[{float(state.get('exit_lower', state.get('lower', state['ema']))):.3f},"
                      f"{float(state.get('exit_upper', state.get('upper', state['ema']))):.3f}] "
                      f"state={region} above={int(state['above_count'])} "
                      f"below={int(state['below_count'])} "
                      f"bar={bar_time} block={blocked_side or '-'} pos={pos_txt}",
                      flush=True)
                last_status = now
            if args.signal_mode in ("tick", "sample") or new_closed_bar:
                last_region = region
                last_bar_time = bar_time
                seen_region = True
            if args.signal_mode == "tick":
                prev_live_price = float(state["close"])
                prev_upper = float(state["upper"])
                prev_lower = float(state["lower"])
            time.sleep(args.poll_sec)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
