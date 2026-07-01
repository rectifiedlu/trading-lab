"""MT5 paper trader for Keltner signal strategy."""

from __future__ import annotations

import argparse
import time

import numpy as np

from forex_signal_paper_common import (
    SignalInfo,
    StateMachinePaper,
    add_common_args,
    ema,
    find_position,
    point_size,
    preload_bid_candles,
    rma,
)
from forex_synthetic_candles import SyntheticBidOHLC


def calc_signal(args, candles: SyntheticBidOHLC) -> SignalInfo | None:
    if len(candles.closed) < args.length + 2:
        return None
    rows = list(candles.closed)
    highs = np.array([c.high for c in rows], dtype=np.float64)
    lows = np.array([c.low for c in rows], dtype=np.float64)
    closes = np.array([c.close for c in rows], dtype=np.float64)
    prev_close = np.empty(len(closes), dtype=np.float64)
    prev_close[0] = closes[0]
    prev_close[1:] = closes[:-1]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    center = ema(closes, args.length)
    atr = rma(tr, args.length)
    if not np.isfinite(atr[-1]):
        return None
    upper = float(center[-1] + args.mult * atr[-1])
    lower = float(center[-1] - args.mult * atr[-1])
    exit_upper = float(center[-1] + args.mult * args.exit_mult * atr[-1])
    exit_lower = float(center[-1] - args.mult * args.exit_mult * atr[-1])
    close = float(closes[-1])
    state = 1 if close > upper else (-1 if close < lower else 0)
    exit_state = 1 if close > exit_upper else (-1 if close < exit_lower else 0)
    if args.mode == "invert":
        state = -state
        exit_state = -exit_state
    return SignalInfo(
        state,
        f"close={close:.3f} mid={center[-1]:.3f} upper={upper:.3f} lower={lower:.3f} "
        f"exit_upper={exit_upper:.3f} exit_lower={exit_lower:.3f} atr={atr[-1]:.3f}",
        exit_state,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap, default_magic=26053102, default_tf="30s", default_tp=800, default_cut=1200, default_hold=0)
    ap.add_argument("--length", type=int, default=34)
    ap.add_argument("--mult", type=float, default=1.5)
    ap.add_argument("--exit-mult", type=float, default=0.75)
    ap.add_argument("--mode", choices=["normal", "invert"], default="invert")
    ap.set_defaults(trail_points=50.0, trail_source="candle_open", block_entry_hours="12,22")
    args = ap.parse_args()

    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise SystemExit(f"mt5.initialize failed: {mt5.last_error()}")
    if not mt5.symbol_select(args.symbol, True):
        raise SystemExit(f"symbol_select failed: {args.symbol}")
    print(
        f"[keltner-paper] start {args.symbol} tf={args.timeframe} length={args.length} "
        f"mult={args.mult:g} mode={args.mode} tp={args.tp_points:g} trail={args.trail_points:g} "
        f"trail_source={args.trail_source} "
        f"cut={args.loss_cut_points:g} hold={args.max_hold_minutes:g} exit_mult={args.exit_mult:g} "
        f"block_hours={args.block_entry_hours or '-'} lot={args.lot:g}",
        flush=True,
    )

    candles = SyntheticBidOHLC(args.timeframe, maxlen=max(args.length + 50, 1000))
    preload_bid_candles(mt5, args, candles, args.length + 10, "keltner")
    sm = StateMachinePaper(args, "keltner")
    last_bar = 0
    last_log = 0.0
    try:
        sig = calc_signal(args, candles)
        if sig:
            sm.on_state(mt5, sig.state, sig.text, sig.exit_state)
        while True:
            tick = mt5.symbol_info_tick(args.symbol)
            if tick is None:
                time.sleep(args.poll)
                continue
            closed_new_bar = candles.update(tick)
            candle_open = float(candles.current.open) if candles.current else None
            sm.on_tick_exits(mt5, candle_open=candle_open)
            if closed_new_bar:
                sig = calc_signal(args, candles)
                if sig and candles.last_closed_bucket != last_bar:
                    last_bar = candles.last_closed_bucket
                    print(f"[keltner-paper] bar state={sig.state} exit_state={sig.exit_state} {sig.text}", flush=True)
                    sm.on_state(mt5, sig.state, sig.text, sig.exit_state)
            now = time.time()
            if now - last_log >= args.log_every:
                last_log = now
                pos = find_position(mt5, args.symbol, args.magic)
                ptxt = "-"
                if pos:
                    side = "L" if int(pos.type) == mt5.POSITION_TYPE_BUY else "S"
                    ptxt = f"{side} entry={float(pos.price_open):.3f} p=${float(pos.profit):+.2f}"
                print(
                    f"[keltner-paper] px={float(tick.bid):.3f}/{float(tick.ask):.3f} "
                    f"state={sm.prev_state} pos={ptxt} best={sm.best_px:.3f} "
                    f"copen={candle_open if candle_open is not None else 0.0:.3f} "
                    f"trail_src={args.trail_source} point={point_size(mt5, args.symbol):g}",
                    flush=True,
                )
            time.sleep(args.poll)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
