"""MT5 paper trader for XAUUSD Keltner-regime + RSI/Stoch re-entry.

Defaults mirror the current XAU regime shortlist:
    k_tf=10m;k_len=55;k_mult=1.5;mean=rsi;mtf=15m;rsi=14;os=30;ob=70;session=0;tp=200;sl=0
"""

from __future__ import annotations

import argparse
import time
from types import SimpleNamespace

import numpy as np

from forex_signal_paper_common import (
    SignalInfo,
    StateMachinePaper,
    add_common_args,
    find_position,
    point_size,
    preload_bid_candles,
)
from forex_synthetic_candles import SyntheticBidOHLC
from forex_keltner_regime_backtest import keltner_regime, rsi_reentry_state, stoch_reentry_state


def rows_to_arrays(candles: SyntheticBidOHLC) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = list(candles.closed)
    highs = np.array([c.high for c in rows], dtype=np.float64)
    lows = np.array([c.low for c in rows], dtype=np.float64)
    closes = np.array([c.close for c in rows], dtype=np.float64)
    return highs, lows, closes


def preload_for_timeframe(mt5, args, candles: SyntheticBidOHLC, timeframe: str, need: int, tag: str) -> int:
    load_args = SimpleNamespace(**vars(args))
    load_args.timeframe = timeframe
    return preload_bid_candles(mt5, load_args, candles, need, tag)


def min_history(args) -> tuple[int, int]:
    k_need = int(args.keltner_length) + 10
    if args.mean == "rsi":
        m_need = int(args.rsi_period) + 10
    else:
        m_need = int(args.stoch_length) + 10
    return k_need, m_need


def rsi_debug(closes: np.ndarray, period: int) -> float:
    if len(closes) < period + 2:
        return float("nan")
    from forex_keltner_regime_backtest import rma
    delta = np.empty(len(closes), dtype=np.float64)
    delta[0] = 0.0
    delta[1:] = closes[1:] - closes[:-1]
    gains = np.maximum(delta, 0.0)
    losses = np.maximum(-delta, 0.0)
    avg_gain = rma(gains, period)
    avg_loss = rma(losses, period)
    rs = avg_gain[-1] / max(float(avg_loss[-1]), 1e-12)
    return float(100.0 - (100.0 / (1.0 + rs)))


def stoch_debug(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, length: int) -> float:
    if len(closes) < length:
        return float("nan")
    lo = float(np.min(lows[-length:]))
    hi = float(np.max(highs[-length:]))
    return float(100.0 * (closes[-1] - lo) / max(hi - lo, 1e-12))


def calc_signal(args, k_candles: SyntheticBidOHLC, m_candles: SyntheticBidOHLC) -> SignalInfo | None:
    k_need, m_need = min_history(args)
    if len(k_candles.closed) < k_need or len(m_candles.closed) < m_need:
        return None

    k_highs, k_lows, k_closes = rows_to_arrays(k_candles)
    m_highs, m_lows, m_closes = rows_to_arrays(m_candles)

    trend, inside = keltner_regime(k_highs, k_lows, k_closes, args.keltner_length, args.keltner_mult)
    trend_state = int(trend[-1])
    is_inside = bool(inside[-1] >= 0.5)

    k_center = float("nan")
    k_upper = float("nan")
    k_lower = float("nan")
    if len(k_closes) >= args.keltner_length:
        from forex_keltner_regime_backtest import ema, rma, true_range
        tr = true_range(k_highs, k_lows, k_closes)
        center = ema(k_closes, args.keltner_length)
        atr = rma(tr, args.keltner_length)
        k_center = float(center[-1])
        k_upper = float(center[-1] + args.keltner_mult * atr[-1])
        k_lower = float(center[-1] - args.keltner_mult * atr[-1])

    if args.mean == "rsi":
        mean_state_arr = rsi_reentry_state(m_closes, args.rsi_period, args.oversold, args.overbought)
        mean_state = int(mean_state_arr[-1])
        osc_text = f"rsi={rsi_debug(m_closes, args.rsi_period):.2f} os={args.oversold:g} ob={args.overbought:g}"
    else:
        mean_state_arr = stoch_reentry_state(m_highs, m_lows, m_closes, args.stoch_length, args.low, args.high)
        mean_state = int(mean_state_arr[-1])
        osc_text = f"stoch={stoch_debug(m_highs, m_lows, m_closes, args.stoch_length):.2f} low={args.low:g} high={args.high:g}"

    state = mean_state if is_inside else trend_state
    regime = "inside" if is_inside else ("above" if trend_state > 0 else "below" if trend_state < 0 else "outside_flat")
    text = (
        f"k={args.keltner_timeframe}/{args.keltner_length}/{args.keltner_mult:g} "
        f"m={args.mean}:{args.meanrev_timeframe} close={m_closes[-1]:.2f} "
        f"state={state} regime={regime} trend={trend_state} mean_sig={mean_state} "
        f"k_mid={k_center:.2f} k_lo={k_lower:.2f} k_hi={k_upper:.2f} {osc_text};session={args.session}"
    )
    return SignalInfo(state, text, state)


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap, default_magic=26060701, default_tf="15m", default_tp=200, default_cut=0, default_hold=0)
    ap.set_defaults(symbol="XAUUSD", lot=0.01, session=0, filling_mode="ioc", log_every=1.0)
    ap.add_argument("--keltner-timeframe", default="10m")
    ap.add_argument("--meanrev-timeframe", default="15m")
    ap.add_argument("--keltner-length", type=int, default=55)
    ap.add_argument("--keltner-mult", type=float, default=1.5)
    ap.add_argument("--mean", choices=["rsi", "stoch"], default="rsi")
    ap.add_argument("--rsi-period", type=int, default=14)
    ap.add_argument("--oversold", type=float, default=30.0)
    ap.add_argument("--overbought", type=float, default=70.0)
    ap.add_argument("--stoch-length", type=int, default=14)
    ap.add_argument("--low", type=float, default=20.0)
    ap.add_argument("--high", type=float, default=80.0)
    ap.add_argument("--sl-points", type=float, default=None, help="alias for --loss-cut-points")
    args = ap.parse_args()

    if args.sl_points is not None:
        args.loss_cut_points = args.sl_points
    args.ignore_signal_exit_when_bracket = args.tp_points > 0 and args.loss_cut_points > 0

    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise SystemExit(f"mt5.initialize failed: {mt5.last_error()}")
    if not mt5.symbol_select(args.symbol, True):
        raise SystemExit(f"symbol_select failed: {args.symbol}")

    tag = "xau-regime"
    k_need, m_need = min_history(args)
    print(
        f"[{tag}-paper] start {args.symbol} k={args.keltner_timeframe}/{args.keltner_length}/{args.keltner_mult:g} "
        f"mean={args.mean}/{args.meanrev_timeframe} rsi={args.rsi_period}/{args.oversold:g}/{args.overbought:g} "
        f"tp={args.tp_points:g} sl={args.loss_cut_points:g} session={args.session} "
        f"lot={args.lot:g} fill={args.filling_mode} dry={int(args.dry_run)}",
        flush=True,
    )

    k_candles = SyntheticBidOHLC(args.keltner_timeframe, maxlen=max(k_need + 100, 2000))
    m_candles = SyntheticBidOHLC(args.meanrev_timeframe, maxlen=max(m_need + 100, 2000))
    preload_for_timeframe(mt5, args, k_candles, args.keltner_timeframe, k_need + 5, tag)
    preload_for_timeframe(mt5, args, m_candles, args.meanrev_timeframe, m_need + 5, tag)

    sm = StateMachinePaper(args, tag)
    last_k_bar = 0
    last_m_bar = 0
    last_log = 0.0
    try:
        sig = calc_signal(args, k_candles, m_candles)
        if sig:
            sm.on_state(mt5, sig.state, sig.text, sig.exit_state)
        while True:
            tick = mt5.symbol_info_tick(args.symbol)
            if tick is None:
                time.sleep(args.poll)
                continue
            closed_k = k_candles.update(tick)
            closed_m = m_candles.update(tick)
            sm.on_tick_exits(mt5, candle_open=float(m_candles.current.open) if m_candles.current else None)

            if closed_k or closed_m:
                sig = calc_signal(args, k_candles, m_candles)
                if sig and (k_candles.last_closed_bucket != last_k_bar or m_candles.last_closed_bucket != last_m_bar):
                    last_k_bar = k_candles.last_closed_bucket
                    last_m_bar = m_candles.last_closed_bucket
                    print(f"[{tag}-paper] bar {sig.text}", flush=True)
                    sm.on_state(mt5, sig.state, sig.text, sig.exit_state)

            now = time.time()
            if now - last_log >= args.log_every:
                last_log = now
                pos = find_position(mt5, args.symbol, args.magic)
                ptxt = "-"
                if pos:
                    side = "L" if int(pos.type) == mt5.POSITION_TYPE_BUY else "S"
                    ptxt = f"{side} entry={float(pos.price_open):.2f} p=${float(pos.profit):+.2f}"
                ps = point_size(mt5, args.symbol)
                bid = float(tick.bid)
                ask = float(tick.ask)
                print(
                    f"[{tag}-paper] px={bid:.2f}/{ask:.2f} state={sm.prev_state} pos={ptxt} "
                    f"point={ps:g} k_closed={len(k_candles.closed)} m_closed={len(m_candles.closed)}",
                    flush=True,
                )
            time.sleep(args.poll)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
