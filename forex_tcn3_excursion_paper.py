"""MT5 paper trader for TCN3 net-excursion regression models.

The model emits one score in points. This trader mirrors the TCN3 backtest:
thresholded normal/invert signals, independent TP/SL modes, tick-based fixed
exits, candle-close signal exits, session-gated entries, and mandatory
same-side reset after a fixed exit.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from forex_ml_barrier_cnn import BarrierData, make_time_features, make_window_features
from forex_ml_tick_simulator import CandleData, load_torch_model, parse_model_name
from forex_score_reset import DirectionalScoreReset
from forex_signal_paper_common import find_position, point_size, send_order
from forex_strategy_common import active_session_allowed, timeframe_to_ns


EXIT_MODES = {"opposite", "neutral", "fixed", "fixed_signal"}
EXPECTED_LABEL = "signed_net_future_excursion_points_v1"
TAG = "tcn3"


def signal_name(side: int) -> str:
    return "LONG" if side == 1 else "SHORT" if side == -1 else "NEUTRAL"


def signal_from_score(score: float | None, threshold: float, mode: str) -> int:
    if score is None or not np.isfinite(score):
        return 0
    side = 1 if score >= threshold else (-1 if score <= -threshold else 0)
    return -side if mode == "invert" else side


def side_allowed(args, side: int) -> bool:
    return args.side == "both" or (args.side == "long" and side == 1) or (args.side == "short" and side == -1)


def session_allowed(bucket_ns: int, session: int) -> bool:
    return bool(active_session_allowed(np.asarray([bucket_ns], dtype=np.int64), session)[0])


def ticks_to_arrays(ticks) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bid = np.asarray([float(t["bid"]) for t in ticks], dtype=np.float64)
    ask = np.asarray([float(t["ask"]) for t in ticks], dtype=np.float64)
    ts = np.asarray([
        int(t["time_msc"]) * 1_000_000 if int(t["time_msc"]) > 0 else int(t["time"]) * 1_000_000_000
        for t in ticks
    ], dtype=np.int64)
    return bid, ask, ts


def candles_from_ticks(bid: np.ndarray, ask: np.ndarray, ts: np.ndarray, tf_ns: int) -> CandleData:
    buckets = (ts // tf_ns) * tf_ns
    unique, starts = np.unique(buckets, return_index=True)
    ends = np.r_[starts[1:], len(bid)]
    ohlc = np.empty((len(unique), 4), dtype=np.float32)
    spread = np.empty(len(unique), dtype=np.float32)
    for i, (start, end) in enumerate(zip(starts, ends)):
        values = bid[start:end]
        ohlc[i] = (values[0], np.max(values), np.min(values), values[-1])
        spread[i] = np.mean(ask[start:end] - values)
    idx = np.arange(len(unique), dtype=np.int64)
    return CandleData(unique.astype("datetime64[ns]"), ohlc, spread, idx, idx)


def preload_candles(mt5, args, need: int, tf_ns: int) -> CandleData:
    tick = mt5.symbol_info_tick(args.symbol)
    now_sec = int(getattr(tick, "time", int(time.time()))) if tick is not None else int(time.time())
    current_bucket = (now_sec * 1_000_000_000 // tf_ns) * tf_ns
    end = datetime.fromtimestamp(current_bucket / 1e9, tz=timezone.utc)
    seconds = max(int((need + 20) * tf_ns / 1e9), 900)
    candles = None
    for _ in range(7):
        ticks = mt5.copy_ticks_range(args.symbol, end - timedelta(seconds=seconds), end, mt5.COPY_TICKS_INFO)
        if ticks is not None and len(ticks):
            bid, ask, ts = ticks_to_arrays(ticks)
            mask = ts < current_bucket
            if np.any(mask):
                candles = candles_from_ticks(bid[mask], ask[mask], ts[mask], tf_ns)
                if len(candles.ohlc) >= need:
                    break
        seconds *= 2
    if candles is None or len(candles.ohlc) < need:
        got = 0 if candles is None else len(candles.ohlc)
        raise SystemExit(f"could not preload tick-built candles: need={need} got={got}")
    if len(candles.ohlc) > need:
        sl = slice(len(candles.ohlc) - need, None)
        idx = np.arange(need, dtype=np.int64)
        candles = CandleData(candles.times[sl], candles.ohlc[sl], candles.spread[sl], idx, idx)
    print(f"[{TAG}-paper] preload tick-built candles={len(candles.ohlc)}", flush=True)
    return candles


def fetch_closed_bucket(mt5, args, bucket_ns: int, tf_ns: int, fallback):
    start = datetime.fromtimestamp(bucket_ns / 1e9, tz=timezone.utc)
    end = datetime.fromtimestamp((bucket_ns + tf_ns) / 1e9, tz=timezone.utc)
    ticks = mt5.copy_ticks_range(args.symbol, start, end, mt5.COPY_TICKS_INFO)
    if ticks is None or not len(ticks):
        return fallback
    bid, ask, ts = ticks_to_arrays(ticks)
    mask = ((ts // tf_ns) * tf_ns) == bucket_ns
    if not np.any(mask):
        return fallback
    bid, ask = bid[mask], ask[mask]
    return float(bid[0]), float(np.max(bid)), float(np.min(bid)), float(bid[-1]), float(np.mean(ask - bid))


def append_candle(candles: CandleData, bucket_ns: int, row, maxlen: int) -> CandleData:
    o, h, low, close, spread = row
    times = np.concatenate([candles.times, np.asarray([bucket_ns], dtype=np.int64).astype("datetime64[ns]")])[-maxlen:]
    ohlc = np.concatenate([candles.ohlc, np.asarray([[o, h, low, close]], dtype=np.float32)])[-maxlen:]
    spreads = np.concatenate([candles.spread, np.asarray([spread], dtype=np.float32)])[-maxlen:]
    idx = np.arange(len(ohlc), dtype=np.int64)
    return CandleData(times, ohlc, spreads, idx, idx)


def predict_score(model, ns, candles: CandleData, point: float, device: torch.device) -> float | None:
    window = int(ns.window)
    if len(candles.ohlc) < window:
        return None
    scale_points = float(getattr(ns, "barrier_points", getattr(ns, "move_scale_points", 100.0)))
    session = active_session_allowed(candles.times.astype("int64"), 1).astype(np.float32)
    data = BarrierData(
        times=candles.times,
        ohlc=candles.ohlc,
        spread=candles.spread,
        labels=np.zeros(len(candles.ohlc), dtype=np.float32),
        valid=np.ones(len(candles.ohlc), dtype=np.bool_),
        session=session,
        point_size=point,
    )
    i = len(candles.ohlc) - 1
    features = make_window_features(data, i, window, scale_points, ns.feature_set)
    extras = [data.spread[i] / max(point * scale_points, 1e-12), *make_time_features(data.times)[i]]
    if getattr(ns, "session_feature", False):
        extras.insert(1, data.session[i])
    with torch.inference_mode():
        raw = model(
            torch.from_numpy(features[None]).to(device),
            torch.from_numpy(np.asarray(extras, dtype=np.float32)[None]).to(device),
        )
    score = float(raw.reshape(-1)[0].cpu()) * float(getattr(ns, "move_scale_points", 100.0))
    return score if np.isfinite(score) else None


def current_move_points(mt5, args, pos) -> float | None:
    tick = mt5.symbol_info_tick(args.symbol)
    if tick is None:
        return None
    point = point_size(mt5, args.symbol)
    if int(pos.type) == mt5.POSITION_TYPE_BUY:
        return (float(tick.bid) - float(pos.price_open)) / point
    return (float(pos.price_open) - float(tick.ask)) / point


def fixed_exit_if_needed(mt5, args) -> int:
    pos = find_position(mt5, args.symbol, args.magic)
    if pos is None:
        return 0
    move = current_move_points(mt5, args, pos)
    if move is None:
        return 0
    side = 1 if int(pos.type) == mt5.POSITION_TYPE_BUY else -1
    if args.tp_mode in {"fixed", "fixed_signal"} and move >= args.tp_points:
        result = send_order(mt5, args, "sell" if side == 1 else "buy", "tp_fixed", int(pos.ticket), tag=TAG)
        return side if result is not None else 0
    if args.sl_mode in {"fixed", "fixed_signal"} and move <= -args.sl_points:
        result = send_order(mt5, args, "sell" if side == 1 else "buy", "sl_fixed", int(pos.ticket), tag=TAG)
        return side if result is not None else 0
    return 0


def signal_exit(mode: str, side: int, signal: int) -> bool:
    if mode == "fixed":
        return False
    if mode in {"opposite", "fixed_signal"}:
        return signal == -side
    return signal != side


def validate_exit(name: str, mode: str, points: float) -> None:
    valid = {"fixed", "fixed_signal"} if points > 0 else {"opposite", "neutral"}
    if mode not in valid:
        raise SystemExit(f"{name}: mode={mode} points={points:g} is not a backtest-valid combination; choose {sorted(valid)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="MT5 paper trader for TCN3 net-excursion models")
    ap.add_argument("--model", required=True)
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--mode", choices=["normal", "invert"], default="normal")
    ap.add_argument("--tp-mode", choices=sorted(EXIT_MODES), default="opposite")
    ap.add_argument("--sl-mode", choices=sorted(EXIT_MODES), default="opposite")
    ap.add_argument("--tp-points", type=float, default=0.0)
    ap.add_argument("--sl-points", type=float, default=0.0)
    ap.add_argument("--session", default="label", help="entry session; default uses the model label session")
    ap.add_argument("--side", choices=["long", "short", "both"], default="both")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--lot", type=float, default=0.01)
    ap.add_argument("--magic", type=int, default=943301)
    ap.add_argument("--deviation", type=int, default=50)
    ap.add_argument("--filling-mode", choices=["auto", "broker", "ioc", "fok", "return"], default="auto")
    ap.add_argument("--poll", type=float, default=0.25)
    ap.add_argument("--log-every", type=float, default=3.0)
    ap.add_argument("--bar-settle", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    validate_exit("TP", args.tp_mode, args.tp_points)
    validate_exit("SL", args.sl_mode, args.sl_points)

    model_path = Path(args.model)
    meta = parse_model_name(model_path)
    args.symbol = (args.symbol or str(meta.get("pair", ""))).upper()
    args.timeframe = str(meta.get("tf", "1m"))
    label_session = int(meta.get("label_session", 0))
    args.session = label_session if str(args.session).lower() == "label" else int(args.session)
    if str(meta.get("target", "")) != "excursion" or str(meta.get("model", "")) != "tcn3":
        raise SystemExit(f"not a TCN3 excursion model: {model_path.name}")

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    model, ns, model_point = load_torch_model(model_path)
    if str(getattr(ns, "excursion_label", "")) != EXPECTED_LABEL:
        raise SystemExit(f"incompatible excursion label; expected {EXPECTED_LABEL}, retrain this model")
    model = model.to(device).eval()

    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise SystemExit(f"mt5.initialize failed: {mt5.last_error()}")
    if not mt5.symbol_select(args.symbol, True):
        raise SystemExit(f"symbol_select failed: {args.symbol}")

    tf_ns = timeframe_to_ns(args.timeframe)
    maxlen = max(int(ns.window) + 500, 2000)
    candles = preload_candles(mt5, args, int(ns.window) + 20, tf_ns)
    reset = DirectionalScoreReset()
    current_bucket = None
    cur_o = cur_h = cur_l = cur_c = spread_sum = 0.0
    spread_n = 0
    last_score = None
    last_signal = 0
    last_session_ok = None
    last_log = 0.0
    print(
        f"[{TAG}-paper] START symbol={args.symbol} tf={args.timeframe} window={ns.window} "
        f"horizon={meta.get('horizon', getattr(ns, 'horizon', '?'))} label_session={label_session} "
        f"entry_session={args.session} threshold={args.threshold:g} mode={args.mode} "
        f"tp={args.tp_mode}:{args.tp_points:g} sl={args.sl_mode}:{args.sl_points:g} "
        f"device={device} reset=mandatory dry={int(args.dry_run)} model={model_path.name}", flush=True,
    )

    try:
        while True:
            tick = mt5.symbol_info_tick(args.symbol)
            if tick is None:
                time.sleep(args.poll)
                continue
            bid, ask = float(tick.bid), float(tick.ask)
            bucket = (int(getattr(tick, "time", int(time.time()))) * 1_000_000_000 // tf_ns) * tf_ns
            exited_side = fixed_exit_if_needed(mt5, args)
            if exited_side:
                reset.mark_fixed_exit(exited_side)
                print(f"[{TAG}-paper] fixed exit side={signal_name(exited_side)} {reset.text()}", flush=True)

            if current_bucket is None:
                current_bucket = bucket
                cur_o = cur_h = cur_l = cur_c = bid
                spread_sum, spread_n = ask - bid, 1
            elif bucket == current_bucket:
                cur_h, cur_l, cur_c = max(cur_h, bid), min(cur_l, bid), bid
                spread_sum += ask - bid
                spread_n += 1
            else:
                if args.bar_settle > 0:
                    time.sleep(args.bar_settle)
                fallback = (cur_o, cur_h, cur_l, cur_c, spread_sum / max(spread_n, 1))
                row = fetch_closed_bucket(mt5, args, current_bucket, tf_ns, fallback)
                closed_bucket = current_bucket
                candles = append_candle(candles, closed_bucket, row, maxlen)
                score = predict_score(model, ns, candles, model_point, device)
                signal = signal_from_score(score, args.threshold, args.mode)
                reset.observe(signal)
                # The order is sent on this new bucket's first tick, so apply
                # the entry session to that bucket, matching the backtest.
                allowed = session_allowed(bucket, args.session)
                last_score, last_signal, last_session_ok = score, signal, allowed
                print(
                    f"[{TAG}-paper] BAR {datetime.fromtimestamp(closed_bucket / 1e9, tz=timezone.utc):%m-%d %H:%M} "
                    f"close={row[3]:.5f} score={score if score is not None else float('nan'):+.3f} "
                    f"signal={signal_name(signal)} entry_ok={int(allowed)} {reset.text()}", flush=True,
                )

                pos = find_position(mt5, args.symbol, args.magic)
                if pos is not None:
                    side = 1 if int(pos.type) == mt5.POSITION_TYPE_BUY else -1
                    move = current_move_points(mt5, args, pos)
                    exit_mode = args.tp_mode if move is not None and move >= 0 else args.sl_mode
                    if signal_exit(exit_mode, side, signal):
                        result = send_order(mt5, args, "sell" if side == 1 else "buy", f"exit_{exit_mode}", int(pos.ticket), tag=TAG)
                        if result is not None and signal == -side:
                            if not allowed:
                                print(f"[{TAG}-paper] reverse blocked by session", flush=True)
                            elif reset.blocks(signal):
                                print(f"[{TAG}-paper] reverse blocked by {reset.text()}", flush=True)
                            elif not side_allowed(args, signal):
                                print(f"[{TAG}-paper] reverse blocked by side filter", flush=True)
                            else:
                                send_order(mt5, args, "buy" if signal == 1 else "sell", f"reverse_{exit_mode}", tag=TAG)
                elif signal != 0:
                    if not allowed:
                        print(f"[{TAG}-paper] entry blocked by session", flush=True)
                    elif reset.blocks(signal):
                        print(f"[{TAG}-paper] entry blocked by {reset.text()}", flush=True)
                    elif not side_allowed(args, signal):
                        print(f"[{TAG}-paper] entry blocked by side filter", flush=True)
                    else:
                        send_order(mt5, args, "buy" if signal == 1 else "sell", f"entry_{signal_name(signal).lower()}", tag=TAG)

                current_bucket = bucket
                cur_o = cur_h = cur_l = cur_c = bid
                spread_sum, spread_n = ask - bid, 1

            now = time.time()
            if now - last_log >= args.log_every:
                last_log = now
                pos = find_position(mt5, args.symbol, args.magic)
                pos_text = "flat" if pos is None else f"{signal_name(1 if int(pos.type) == mt5.POSITION_TYPE_BUY else -1)} move={current_move_points(mt5, args, pos):+.1f}pt"
                score_text = "waiting" if last_score is None else f"{last_score:+.3f}"
                print(
                    f"[{TAG}-paper] tick={bid:.5f}/{ask:.5f} pos={pos_text} score={score_text} "
                    f"signal={signal_name(last_signal)} entry_ok={'?' if last_session_ok is None else int(last_session_ok)} {reset.text()}", flush=True,
                )
            time.sleep(args.poll)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
