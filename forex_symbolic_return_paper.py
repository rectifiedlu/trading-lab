"""MT5 paper trader for symbolic future-return model artifacts.

Loads one artifact produced by ``forex_symbolic_return_train.py`` and applies
the same raw-OHLC lag order, threshold signal, session gate, normal/invert
mode, and TP/SL exit-mode semantics used by the symbolic backtest.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from forex_signal_paper_common import (
    find_position,
    point_size,
    preload_bid_candles,
    send_order,
)
from forex_strategy_common import active_session_allowed, default_point_size
from forex_synthetic_candles import SyntheticBidOHLC


EXIT_MODES = {"opposite", "neutral", "fixed", "fixed_signal"}
TAG = "symbolic"


def load_artifact(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    model_path = Path(meta["model_path"])
    if not model_path.is_absolute():
        model_path = path.parent / model_path.name
    with open(model_path, "rb") as handle:
        return meta, pickle.load(handle), model_path


def score_closed_candles(model, candles: SyntheticBidOHLC, window: int) -> float | None:
    rows = list(candles.closed)
    if len(rows) < window:
        return None
    values: list[float] = []
    for candle in reversed(rows[-window:]):
        values.extend((candle.open, candle.high, candle.low, candle.close))
    score = float(np.asarray(model.predict(np.asarray([values], dtype=np.float64))).reshape(-1)[0])
    return score if np.isfinite(score) else None


def signal_from_score(score: float | None, threshold: float, mode: str) -> int:
    if score is None:
        return 0
    side = 1 if score >= threshold else (-1 if score <= -threshold else 0)
    return -side if mode == "invert" else side


def side_allowed(args, side: int) -> bool:
    return args.side == "both" or (args.side == "long" and side == 1) or (args.side == "short" and side == -1)


def entry_allowed(meta: dict, candle_bucket: int, tf_sec: int) -> bool:
    ts_ns = np.array([candle_bucket * tf_sec * 1_000_000_000], dtype=np.int64)
    return bool(active_session_allowed(ts_ns, int(meta["session"]))[0])


def should_signal_exit(mode: str, side: int, signal: int) -> bool:
    if mode == "fixed":
        return False
    if mode in {"opposite", "fixed_signal"}:
        return signal == -side
    if mode == "neutral":
        return signal != side
    raise ValueError(f"unknown exit mode: {mode}")


def current_move_points(mt5, args, pos) -> float | None:
    tick = mt5.symbol_info_tick(args.symbol)
    if tick is None:
        return None
    ps = point_size(mt5, args.symbol)
    entry = float(pos.price_open)
    if int(pos.type) == mt5.POSITION_TYPE_BUY:
        return (float(tick.bid) - entry) / ps
    return (entry - float(tick.ask)) / ps


def close_fixed_if_needed(mt5, args) -> bool:
    pos = find_position(mt5, args.symbol, args.magic)
    if pos is None:
        return False
    move = current_move_points(mt5, args, pos)
    if move is None:
        return False
    side = 1 if int(pos.type) == mt5.POSITION_TYPE_BUY else -1
    close_side = "sell" if side == 1 else "buy"
    if args.tp_mode in {"fixed", "fixed_signal"} and args.tp_points > 0 and move >= args.tp_points:
        send_order(mt5, args, close_side, "tp_fixed", int(pos.ticket), tag=TAG)
        return True
    if args.sl_mode in {"fixed", "fixed_signal"} and args.sl_points > 0 and move <= -args.sl_points:
        send_order(mt5, args, close_side, "sl_fixed", int(pos.ticket), tag=TAG)
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="MT5 paper trader for symbolic future-return models")
    ap.add_argument("--model", required=True, help="symbolic artifact JSON produced by forex_symbolic_return_train.py")
    ap.add_argument("--symbol", default=None, help="override artifact pair")
    ap.add_argument("--timeframe", default=None, help="override artifact timeframe")
    ap.add_argument("--threshold", type=float, default=100.0)
    ap.add_argument("--mode", choices=["normal", "invert"], default="normal")
    ap.add_argument("--tp-mode", choices=sorted(EXIT_MODES), default="fixed_signal")
    ap.add_argument("--sl-mode", choices=sorted(EXIT_MODES), default="fixed_signal")
    ap.add_argument("--tp-points", type=float, default=0.0)
    ap.add_argument("--sl-points", type=float, default=0.0)
    ap.add_argument("--side", choices=["long", "short", "both"], default="both")
    ap.add_argument("--lot", type=float, default=0.01)
    ap.add_argument("--magic", type=int, default=941201)
    ap.add_argument("--deviation", type=int, default=50)
    ap.add_argument("--filling-mode", choices=["auto", "broker", "ioc", "fok", "return"], default="auto")
    ap.add_argument("--poll", type=float, default=0.25)
    ap.add_argument("--log-every", type=float, default=3.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    meta_path = Path(args.model)
    meta, model, model_path = load_artifact(meta_path)
    args.symbol = (args.symbol or str(meta["pair"])).upper()
    args.timeframe = args.timeframe or str(meta["timeframe"])
    window = int(meta["window"])

    import MetaTrader5 as mt5

    if not mt5.initialize():
        raise SystemExit(f"mt5.initialize failed: {mt5.last_error()}")
    if not mt5.symbol_select(args.symbol, True):
        raise SystemExit(f"symbol_select failed: {args.symbol}")

    candles = SyntheticBidOHLC(args.timeframe, maxlen=max(window + 500, 2000))
    if preload_bid_candles(mt5, args, candles, window + 5, TAG) < window:
        raise SystemExit(f"not enough closed candles for window={window}")

    last_log = 0.0
    last_bar = 0
    print(
        f"[{TAG}-paper] start symbol={args.symbol} tf={args.timeframe} window={window} "
        f"horizon={meta['horizon']} session={meta['session']} threshold={args.threshold:g} mode={args.mode} "
        f"tp={args.tp_mode}:{args.tp_points:g} sl={args.sl_mode}:{args.sl_points:g} "
        f"model={model_path.name} dry={int(args.dry_run)}",
        flush=True,
    )

    try:
        while True:
            tick = mt5.symbol_info_tick(args.symbol)
            if tick is None:
                time.sleep(args.poll)
                continue
            closed_new_bar = candles.update(tick)
            close_fixed_if_needed(mt5, args)

            if closed_new_bar and candles.last_closed_bucket != last_bar:
                last_bar = candles.last_closed_bucket
                score = score_closed_candles(model, candles, window)
                signal = signal_from_score(score, args.threshold, args.mode)
                session_ok = entry_allowed(meta, last_bar, candles.tf_sec)
                pos = find_position(mt5, args.symbol, args.magic)
                signal_text = "LONG" if signal == 1 else ("SHORT" if signal == -1 else "NEUTRAL")
                print(
                    f"[{TAG}-paper] bar={last_bar} score={score if score is not None else float('nan'):+.3f} "
                    f"signal={signal_text} session_ok={int(session_ok)}",
                    flush=True,
                )

                if pos is not None:
                    side = 1 if int(pos.type) == mt5.POSITION_TYPE_BUY else -1
                    move = current_move_points(mt5, args, pos)
                    if move is not None:
                        exit_mode = args.tp_mode if move >= 0 else args.sl_mode
                        if should_signal_exit(exit_mode, side, signal):
                            close_side = "sell" if side == 1 else "buy"
                            print(f"[{TAG}-paper] signal exit mode={exit_mode} move={move:+.1f}pt", flush=True)
                            send_order(mt5, args, close_side, f"exit_{exit_mode}", int(pos.ticket), tag=TAG)
                elif signal != 0:
                    if not session_ok:
                        print(f"[{TAG}-paper] entry blocked by session", flush=True)
                    elif not side_allowed(args, signal):
                        print(f"[{TAG}-paper] entry blocked by side filter", flush=True)
                    else:
                        send_order(mt5, args, "buy" if signal == 1 else "sell", f"entry_{signal_text.lower()}", tag=TAG)

            now = time.time()
            if now - last_log >= args.log_every:
                last_log = now
                pos = find_position(mt5, args.symbol, args.magic)
                pos_text = "-"
                if pos is not None:
                    move = current_move_points(mt5, args, pos)
                    side = "L" if int(pos.type) == mt5.POSITION_TYPE_BUY else "S"
                    pos_text = f"{side} entry={float(pos.price_open):.5f} move={move if move is not None else float('nan'):+.1f}pt"
                print(
                    f"[{TAG}-paper] tick={float(tick.bid):.5f}/{float(tick.ask):.5f} pos={pos_text}",
                    flush=True,
                )
            time.sleep(args.poll)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
