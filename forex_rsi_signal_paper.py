"""MT5 paper trader for RSI extreme signal strategy."""

from __future__ import annotations

import argparse
import time

import numpy as np

from forex_signal_paper_common import (
    SignalInfo,
    StateMachinePaper,
    add_common_args,
    find_position,
    point_size,
    preload_bid_candles,
    rma,
)
from forex_synthetic_candles import SyntheticBidOHLC


def calc_rsi(closes: np.ndarray, period: int) -> float:
    delta = np.empty(len(closes), dtype=np.float64)
    delta[0] = 0.0
    delta[1:] = closes[1:] - closes[:-1]
    gains = np.maximum(delta, 0.0)
    losses = np.maximum(-delta, 0.0)
    avg_gain = rma(gains, period)
    avg_loss = rma(losses, period)
    rs = avg_gain[-1] / max(avg_loss[-1], 1e-12)
    return float(100.0 - (100.0 / (1.0 + rs)))


def calc_signal(args, candles: SyntheticBidOHLC) -> SignalInfo | None:
    if len(candles.closed) < args.period + 2:
        return None
    closes = np.array([c.close for c in candles.closed], dtype=np.float64)
    rsi = calc_rsi(closes, args.period)
    if args.kind == "rsix":
        state = 1 if rsi < args.low else (-1 if rsi > args.high else 0)
    else:
        state = 1 if rsi > 50.0 else (-1 if rsi < 50.0 else 0)
    if args.mode == "invert":
        state = -state
    return SignalInfo(state, f"close={closes[-1]:.3f} rsi={rsi:.2f} low={args.low:g} high={args.high:g}")


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap, default_magic=26053103, default_tf="30s", default_tp=600, default_cut=600, default_hold=0)
    ap.add_argument("--period", type=int, default=7)
    ap.add_argument("--kind", choices=["rsix", "rsi50"], default="rsix")
    ap.add_argument("--mode", choices=["normal", "invert"], default="normal")
    ap.add_argument("--low", type=float, default=30.0)
    ap.add_argument("--high", type=float, default=70.0)
    args = ap.parse_args()

    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise SystemExit(f"mt5.initialize failed: {mt5.last_error()}")
    if not mt5.symbol_select(args.symbol, True):
        raise SystemExit(f"symbol_select failed: {args.symbol}")
    print(
        f"[rsi-paper] start {args.symbol} tf={args.timeframe} period={args.period} "
        f"kind={args.kind} mode={args.mode} tp={args.tp_points:g} "
        f"cut={args.loss_cut_points:g} hold={args.max_hold_minutes:g} lot={args.lot:g}",
        flush=True,
    )

    candles = SyntheticBidOHLC(args.timeframe, maxlen=max(args.period + 50, 1000))
    preload_bid_candles(mt5, args, candles, args.period + 10, "rsi")
    sm = StateMachinePaper(args, "rsi")
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
                    print(f"[rsi-paper] bar state={sig.state} {sig.text}", flush=True)
                    sm.on_state(mt5, sig.state, sig.text)
            now = time.time()
            if now - last_log >= args.log_every:
                last_log = now
                pos = find_position(mt5, args.symbol, args.magic)
                ptxt = "-"
                if pos:
                    side = "L" if int(pos.type) == mt5.POSITION_TYPE_BUY else "S"
                    ptxt = f"{side} entry={float(pos.price_open):.3f} p=${float(pos.profit):+.2f}"
                print(
                    f"[rsi-paper] px={float(tick.bid):.3f}/{float(tick.ask):.3f} "
                    f"state={sm.prev_state} pos={ptxt} point={point_size(mt5, args.symbol):g}",
                    flush=True,
                )
            time.sleep(args.poll)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
