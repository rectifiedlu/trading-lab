"""Unified MT5 paper trader for base-signal strategies.

This mirrors forex_unified_signal_backtest.py at paper-trading scale:
    - build bid candles from MT5 ticks, preloaded from MT5 bars/ticks
    - compute one strategy state on closed candles
    - mode=normal uses the state, mode=invert flips it
    - TP/loss-cut are monitored on live ticks by StateMachinePaper
    - if TP and loss-cut are both set, strategy signal exits are ignored,
      matching the unified backtest bracket behavior
"""

from __future__ import annotations

import argparse
import re
import time

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
from forex_unified_signal_backtest import (
    bb_rsi_legacy_state,
    bb_rsi_level_exit_state,
    bb_rsi_state,
    bollinger_level_exit_state,
    bollinger_state,
    bollinger_target_exit_state,
    cci_level_exit_state,
    cci_state,
    dmi_state,
    donchian_state,
    ema_pair_state,
    ema_price_state,
    keltner_inside_exit_state,
    keltner_neutral_exit_state,
    keltner_state,
    macd_state,
    psar_state,
    rma,
    rsi_level_exit_state,
    rsi_state,
    stochastic_level_exit_state,
    stochastic_state,
    supertrend_state,
)


def rows_to_arrays(candles: SyntheticBidOHLC) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = list(candles.closed)
    highs = np.array([c.high for c in rows], dtype=np.float64)
    lows = np.array([c.low for c in rows], dtype=np.float64)
    closes = np.array([c.close for c in rows], dtype=np.float64)
    return highs, lows, closes


def min_history(args) -> int:
    strategy = args.strategy
    if strategy == "ema_pair":
        return max(args.fast, args.slow) + 5
    if strategy in {"ema", "keltner", "donchian", "bollinger", "cci", "stoch", "rsi"}:
        return int(args.length if strategy != "rsi" else args.period) + 10
    if strategy in {"bb_rsi", "bb_rsi_legacy"}:
        return max(args.bb_length, args.rsi_period) + 10
    if strategy == "macd":
        return max(args.fast, args.slow, args.signal) + 200
    if strategy == "dmi":
        return max(args.di_length, args.adx_length) + 10
    if strategy == "supertrend":
        return args.length + 10
    if strategy == "psar":
        return 20
    return 100


def is_fx_symbol(symbol: str) -> bool:
    clean = re.sub(r"[^A-Za-z]", "", symbol).upper()
    if len(clean) < 6:
        return False
    base, quote = clean[:3], clean[3:6]
    currencies = {
        "USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF",
        "SEK", "NOK", "DKK", "MXN", "ZAR", "TRY", "CNH",
    }
    metals = {"XAU", "XAG", "XPT", "XPD"}
    return base in currencies and quote in currencies and base not in metals


def resolve_auto_lot(mt5, args, tag: str) -> None:
    if args.lot is not None:
        args.auto_lot_used = False
        return

    info = mt5.symbol_info(args.symbol)
    tick = mt5.symbol_info_tick(args.symbol)
    if info is None or tick is None:
        args.lot = 0.01
        args.auto_lot_used = True
        print(f"[{tag}-paper] auto lot fallback=0.01 no symbol info/tick", flush=True)
        return

    if not is_fx_symbol(args.symbol):
        args.lot = 0.01
        args.auto_lot_used = True
        print(f"[{tag}-paper] auto lot non-fx default=0.01", flush=True)
        return

    target_margin = float(args.auto_fx_margin)
    price = float(tick.ask or tick.bid)
    margin_one_lot = None
    if price > 0:
        margin_one_lot = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, args.symbol, 1.0, price)

    if margin_one_lot is not None and float(margin_one_lot) > 0:
        raw_lot = target_margin / float(margin_one_lot)
        source = f"margin_1lot=${float(margin_one_lot):.2f}"
    else:
        contract = float(getattr(info, "trade_contract_size", 100000.0) or 100000.0)
        raw_lot = (target_margin * float(args.auto_fx_leverage)) / contract
        source = f"fallback_notional=${target_margin * float(args.auto_fx_leverage):.2f}"

    from forex_signal_paper_common import round_volume
    args.lot = round_volume(info, raw_lot)
    args.auto_lot_used = True
    print(
        f"[{tag}-paper] auto FX lot target_margin=${target_margin:g} "
        f"raw={raw_lot:.6f} lot={args.lot:g} {source}",
        flush=True,
    )


def latest_bb_rsi_debug(closes: np.ndarray, bb_length: int, bb_mult: float, rsi_period: int) -> str:
    if len(closes) < max(bb_length, rsi_period) + 2:
        return ""
    window = closes[-bb_length:]
    basis = float(np.mean(window))
    dev = float(np.std(window))
    upper = basis + bb_mult * dev
    lower = basis - bb_mult * dev

    delta = np.empty(len(closes), dtype=np.float64)
    delta[0] = 0.0
    delta[1:] = closes[1:] - closes[:-1]
    gains = np.maximum(delta, 0.0)
    losses = np.maximum(-delta, 0.0)
    avg_gain = rma(gains, rsi_period)
    avg_loss = rma(losses, rsi_period)
    rs = avg_gain[-1] / max(float(avg_loss[-1]), 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    close = float(closes[-1])
    pos = "inside"
    if close < lower:
        pos = "below_lower"
    elif close > upper:
        pos = "above_upper"
    return (
        f" bb_mid={basis:.5f} bb_lo={lower:.5f} bb_hi={upper:.5f} "
        f"rsi={rsi:.2f} band_pos={pos}"
    )


def latest_macd_debug(closes: np.ndarray, fast: int, slow: int, signal: int) -> str:
    if len(closes) < max(fast, slow, signal) + 2:
        return ""
    from forex_unified_signal_backtest import ema, rolling_sma

    line = ema(closes, fast) - ema(closes, slow)
    if signal <= 1:
        return f" macd={line[-1]:+.8f} zero=0.00000000"
    sig = rolling_sma(line, signal)
    hist = line - sig
    return (
        f" macd={line[-1]:+.8f} sigline={sig[-1]:+.8f} hist={hist[-1]:+.8f} "
        f"prev_hist={hist[-2]:+.8f}"
    )


def calc_signal(args, candles: SyntheticBidOHLC) -> SignalInfo | None:
    need = min_history(args)
    if len(candles.closed) < need:
        return None
    highs, lows, closes = rows_to_arrays(candles)
    strategy = args.strategy

    if strategy == "keltner":
        state, params = keltner_state(highs, lows, closes, args.length, args.mult, args.mode)
        if args.sl0_exit_mode in {"inside", "neutral"}:
            if args.sl0_exit_mode == "inside":
                exit_state = keltner_inside_exit_state(highs, lows, closes, args.length, args.mult, args.mode)
            else:
                exit_state = keltner_neutral_exit_state(highs, lows, closes, args.length, args.mult, args.mode)
            st = int(state[-1]) if len(state) else 0
            ex = int(exit_state[-1]) if len(exit_state) else 0
            text = f"close={closes[-1]:.5f} state={st} exit={ex} {params};sl0_exit={args.sl0_exit_mode};session={args.session}"
            return SignalInfo(st, text, ex)
    elif strategy == "donchian":
        state, params = donchian_state(highs, lows, closes, args.length, args.mode)
    elif strategy == "bollinger":
        state, params = bollinger_state(highs, lows, closes, args.length, args.mult, args.mode)
        if args.sl0_exit_mode in {"level", "neutral", "opposite"}:
            if args.sl0_exit_mode == "level":
                exit_state = bollinger_level_exit_state(highs, lows, closes, args.length, args.mult, args.mode)
            else:
                exit_state = bollinger_target_exit_state(
                    highs, lows, closes, args.length, args.mult, args.mode, args.sl0_exit_mode
                )
            st = int(state[-1]) if len(state) else 0
            ex = int(exit_state[-1]) if len(exit_state) else 0
            text = f"close={closes[-1]:.5f} state={st} exit={ex} {params};sl0_exit={args.sl0_exit_mode};session={args.session}"
            return SignalInfo(st, text, ex)
    elif strategy == "bb_rsi":
        state, params = bb_rsi_state(
            highs, lows, closes, args.bb_length, args.mult, args.rsi_period,
            args.oversold, args.overbought, args.mode,
        )
        if args.sl0_exit_mode == "level":
            exit_state = bb_rsi_level_exit_state(
                highs, lows, closes, args.bb_length, args.mult, args.rsi_period,
                args.oversold, args.overbought, args.mode,
            )
            st = int(state[-1]) if len(state) else 0
            ex = int(exit_state[-1]) if len(exit_state) else 0
            params += latest_bb_rsi_debug(closes, args.bb_length, args.mult, args.rsi_period)
            text = f"close={closes[-1]:.5f} state={st} exit={ex} {params};sl0_exit=level;session={args.session}"
            return SignalInfo(st, text, ex)
        params += latest_bb_rsi_debug(closes, args.bb_length, args.mult, args.rsi_period)
    elif strategy == "bb_rsi_legacy":
        state, params = bb_rsi_legacy_state(
            highs, lows, closes, args.bb_length, args.mult, args.rsi_period,
            args.oversold, args.overbought, args.mode,
        )
        params += latest_bb_rsi_debug(closes, args.bb_length, args.mult, args.rsi_period)
    elif strategy == "rsi":
        state, params = rsi_state(closes, args.period, args.kind, args.mode)
        if args.sl0_exit_mode == "level":
            exit_state = rsi_level_exit_state(closes, args.period, args.kind, args.mode)
            state_text = int(state[-1]) if len(state) else 0
            exit_text = int(exit_state[-1]) if len(exit_state) else 0
            text = f"close={closes[-1]:.5f} state={state_text} exit={exit_text} {params};sl0_exit=level;session={args.session}"
            return SignalInfo(state_text, text, exit_text)
    elif strategy == "stoch":
        state, params = stochastic_state(highs, lows, closes, args.length, args.low, args.high, args.mode)
        if args.sl0_exit_mode == "level":
            exit_state = stochastic_level_exit_state(highs, lows, closes, args.length, args.low, args.high, args.mode)
            st = int(state[-1]) if len(state) else 0
            ex = int(exit_state[-1]) if len(exit_state) else 0
            text = f"close={closes[-1]:.5f} state={st} exit={ex} {params};sl0_exit=level;session={args.session}"
            return SignalInfo(st, text, ex)
    elif strategy == "macd":
        state, params = macd_state(closes, args.fast, args.slow, args.signal, args.deadband, args.mode)
        params += latest_macd_debug(closes, args.fast, args.slow, args.signal)
    elif strategy == "ema":
        state, params = ema_price_state(closes, args.length, args.mode)
    elif strategy == "ema_pair":
        state, params = ema_pair_state(closes, args.fast, args.slow, args.mode)
    elif strategy == "cci":
        state, params = cci_state(highs, lows, closes, args.length, args.threshold, args.mode)
        if args.sl0_exit_mode == "level":
            exit_state = cci_level_exit_state(highs, lows, closes, args.length, args.threshold, args.mode)
            st = int(state[-1]) if len(state) else 0
            ex = int(exit_state[-1]) if len(exit_state) else 0
            text = f"close={closes[-1]:.5f} state={st} exit={ex} {params};sl0_exit=level;session={args.session}"
            return SignalInfo(st, text, ex)
    elif strategy == "dmi":
        state, params = dmi_state(highs, lows, closes, args.di_length, args.adx_length, args.adx_min, args.mode)
    elif strategy == "supertrend":
        state, _, _ = supertrend_state(highs, lows, closes, args.length, args.mult, args.mode)
        params = f"length={args.length};mult={args.mult:g};mode={args.mode}"
    elif strategy == "psar":
        state, params = psar_state(highs, lows, closes, args.start, args.increment, args.maximum, args.mode)
    else:
        raise SystemExit(f"unsupported strategy for unified paper: {strategy}")

    st = int(state[-1]) if len(state) else 0
    text = f"close={closes[-1]:.5f} state={st} {params};session={args.session}"
    return SignalInfo(st, text, st)


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap, default_magic=26060401, default_tf="1m", default_tp=0, default_cut=0, default_hold=0)
    ap.set_defaults(lot=None)
    ap.add_argument(
        "--strategy",
        choices=[
            "keltner", "donchian", "bollinger", "bb_rsi", "bb_rsi_legacy", "rsi", "stoch",
            "macd", "ema", "ema_pair", "cci", "dmi", "supertrend", "psar",
        ],
        required=True,
    )
    ap.add_argument("--mode", choices=["normal", "invert"], default="normal")
    ap.add_argument("--sl0-exit-mode", choices=["signal", "level", "inside", "neutral", "opposite"], default="signal")
    ap.add_argument("--sl-points", type=float, default=None, help="alias for --loss-cut-points")

    ap.add_argument("--length", type=int, default=20)
    ap.add_argument("--mult", type=float, default=2.0)
    ap.add_argument("--bb-length", type=int, default=20)
    ap.add_argument("--rsi-period", type=int, default=14)
    ap.add_argument("--oversold", type=float, default=30.0)
    ap.add_argument("--overbought", type=float, default=70.0)
    ap.add_argument("--period", type=int, default=14)
    ap.add_argument("--kind", choices=["rsi50", "rsix", "rsix_reentry"], default="rsix")
    ap.add_argument("--low", type=float, default=20.0)
    ap.add_argument("--high", type=float, default=80.0)
    ap.add_argument("--fast", type=int, default=9)
    ap.add_argument("--slow", type=int, default=377)
    ap.add_argument("--signal", type=int, default=1)
    ap.add_argument("--deadband", type=float, default=0.0)
    ap.add_argument("--threshold", type=float, default=200.0)
    ap.add_argument("--di-length", type=int, default=14)
    ap.add_argument("--adx-length", type=int, default=14)
    ap.add_argument("--adx-min", type=float, default=20.0)
    ap.add_argument("--start", type=float, default=0.02)
    ap.add_argument("--increment", type=float, default=0.02)
    ap.add_argument("--maximum", type=float, default=0.2)
    ap.add_argument(
        "--auto-fx-margin",
        type=float,
        default=50.0,
        help="When --lot is omitted for FX symbols, size to roughly this account-currency margin.",
    )
    ap.add_argument(
        "--auto-fx-leverage",
        type=float,
        default=100.0,
        help="Fallback leverage only used if MT5 margin calculation is unavailable.",
    )
    args = ap.parse_args()

    if args.sl_points is not None:
        args.loss_cut_points = args.sl_points
    args.ignore_signal_exit_when_bracket = args.tp_points > 0 and args.loss_cut_points > 0

    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise SystemExit(f"mt5.initialize failed: {mt5.last_error()}")
    if not mt5.symbol_select(args.symbol, True):
        raise SystemExit(f"symbol_select failed: {args.symbol}")
    resolve_auto_lot(mt5, args, tag="unified")

    need = min_history(args)
    tag = "unified"
    print(
        f"[{tag}-paper] start {args.symbol} strat={args.strategy} tf={args.timeframe} "
        f"mode={args.mode} session={args.session} tp={args.tp_points:g} "
        f"sl={args.loss_cut_points:g} bracket_ignore_signal={int(args.ignore_signal_exit_when_bracket)} "
        f"lot={args.lot:g} auto_lot={int(getattr(args, 'auto_lot_used', False))} dry={int(args.dry_run)}",
        flush=True,
    )

    candles = SyntheticBidOHLC(args.timeframe, maxlen=max(need + 100, 2000))
    preload_bid_candles(mt5, args, candles, need + 5, tag)
    sm = StateMachinePaper(args, tag)
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
            sm.on_tick_exits(mt5, candle_open=float(candles.current.open) if candles.current else None)
            if closed_new_bar:
                sig = calc_signal(args, candles)
                if sig and candles.last_closed_bucket != last_bar:
                    last_bar = candles.last_closed_bucket
                    print(f"[{tag}-paper] bar {sig.text}", flush=True)
                    sm.on_state(mt5, sig.state, sig.text, sig.exit_state)
            now = time.time()
            if now - last_log >= args.log_every:
                last_log = now
                pos = find_position(mt5, args.symbol, args.magic)
                ptxt = "-"
                if pos:
                    side = "L" if int(pos.type) == mt5.POSITION_TYPE_BUY else "S"
                    ptxt = f"{side} entry={float(pos.price_open):.5f} p=${float(pos.profit):+.2f}"
                print(
                    f"[{tag}-paper] px={float(tick.bid):.5f}/{float(tick.ask):.5f} "
                    f"state={sm.prev_state} pos={ptxt} point={point_size(mt5, args.symbol):g}",
                    flush=True,
                )
            time.sleep(args.poll)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
