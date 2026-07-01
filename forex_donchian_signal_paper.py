"""MT5 paper trader for Donchian signal strategy."""

from __future__ import annotations

import argparse
import time

import numpy as np

from forex_signal_paper_common import (
    StateMachinePaper,
    add_common_args,
    find_position,
    point_size,
    preload_bid_candles,
    SignalInfo,
)
from forex_synthetic_candles import SyntheticBidOHLC


def calc_signal(args, candles: SyntheticBidOHLC) -> SignalInfo | None:
    if len(candles.closed) < args.length + 1:
        return None
    rows = list(candles.closed)
    hist = rows[-args.length - 1:-1]
    close = float(rows[-1].close)
    upper = max(c.high for c in hist)
    lower = min(c.low for c in hist)
    state = 1 if close > upper else (-1 if close < lower else 0)
    if args.mode == "invert":
        state = -state
    return SignalInfo(state, f"close={close:.3f} upper={upper:.3f} lower={lower:.3f} raw={'break' if state else 'flat'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap, default_magic=26053101, default_tf="30s", default_tp=800, default_cut=1200, default_hold=0)
    ap.add_argument("--length", type=int, default=12)
    ap.add_argument("--mode", choices=["normal", "invert"], default="invert")
    args = ap.parse_args()

    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise SystemExit(f"mt5.initialize failed: {mt5.last_error()}")
    if not mt5.symbol_select(args.symbol, True):
        raise SystemExit(f"symbol_select failed: {args.symbol}")
    print(
        f"[donchian-paper] start {args.symbol} tf={args.timeframe} length={args.length} "
        f"mode={args.mode} tp={args.tp_points:g} cut={args.loss_cut_points:g} "
        f"hold={args.max_hold_minutes:g} lot={args.lot:g}",
        flush=True,
    )

    candles = SyntheticBidOHLC(args.timeframe, maxlen=max(args.length + 20, 1000))
    preload_bid_candles(mt5, args, candles, args.length + 5, "donchian")
    sm = StateMachinePaper(args, "donchian")
    last_bar = 0
    last_log = 0.0
    try:
        sig = calc_signal(args, candles)
        if sig:
            sm.on_state(mt5, sig.state, sig.text)
        while True:
            tick = mt5.symbol_info_tick(args.symbol)
            if tick is None:
                time.sleep(args.poll)
                continue
            sm.on_tick_exits(mt5)
            if candles.update(tick):
                sig = calc_signal(args, candles)
                if sig and candles.last_closed_bucket != last_bar:
                    last_bar = candles.last_closed_bucket
                    print(f"[donchian-paper] bar state={sig.state} {sig.text}", flush=True)
                    sm.on_state(mt5, sig.state, sig.text)
            now = time.time()
            if now - last_log >= args.log_every:
                last_log = now
                pos = find_position(mt5, args.symbol, args.magic)
                ptxt = "-"
                if pos:
                    side = "L" if int(pos.type) == mt5.POSITION_TYPE_BUY else "S"
                    ptxt = f"{side} entry={float(pos.price_open):.3f} p=${float(pos.profit):+.2f}"
                ps = point_size(mt5, args.symbol)
                print(
                    f"[donchian-paper] px={float(tick.bid):.3f}/{float(tick.ask):.3f} "
                    f"state={sm.prev_state} pos={ptxt} point={ps:g}",
                    flush=True,
                )
            time.sleep(args.poll)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
