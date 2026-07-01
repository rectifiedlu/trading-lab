"""MT5 paper trader for clean move4 ML tick strategy.

Uses the same model inference and entry/exit semantics as
forex_ml_tick_simulator.py:
- entry: probability threshold + risk filter
- TP and SL modes are independent
- fixed exits are tick-based
- prob/signal exits are candle-close based
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import deque

import numpy as np
import torch

from forex_ml_tick_simulator import (
    CandleData,
    MODES,
    RISK_MODES,
    build_bid_candles,
    default_sl_grid,
    default_tp_grid,
    load_torch_model,
    parse_model_name,
)
from forex_signal_paper_common import find_position, mt5_timeframe, point_size, send_order
from forex_strategy_common import active_session_allowed, default_point_size, timeframe_to_ns
from forex_ml_barrier_cnn import BarrierData, make_time_features, make_window_features


@dataclass
class ActivePlan:
    ticket: int
    side: int
    entry: float
    tp_points: float
    sl_points: float
    long_prob: float
    short_prob: float
    exp_up: float
    exp_down: float


def signal_name(side: int) -> str:
    return "LONG" if side == 1 else "SHORT" if side == -1 else "WAIT"


def risk_ok(args, side: int, pred: np.ndarray) -> bool:
    p_long, p_short, exp_up, exp_down = map(float, pred)
    if side == 1:
        if p_long < args.threshold or exp_up < args.min_tp:
            return False
        if args.risk_mode == "none":
            return True
        if args.risk_mode == "rr":
            return exp_up / max(exp_down, 1e-9) >= args.risk_value
        if args.risk_mode == "edge":
            return exp_up - exp_down >= args.risk_value
        return p_long - p_short >= args.risk_value
    if p_short < args.threshold or exp_down < args.min_tp:
        return False
    if args.risk_mode == "none":
        return True
    if args.risk_mode == "rr":
        return exp_down / max(exp_up, 1e-9) >= args.risk_value
    if args.risk_mode == "edge":
        return exp_down - exp_up >= args.risk_value
    return p_short - p_long >= args.risk_value


def choose_signal(args, pred: np.ndarray) -> int:
    if args.side in {"both", "long"} and risk_ok(args, 1, pred):
        return 1
    if args.side in {"both", "short"} and risk_ok(args, -1, pred):
        return -1
    return 0


def mode_exit(args, mode: str, side: int, pred: np.ndarray) -> bool:
    p_long, p_short, exp_up, exp_down = map(float, pred)
    if mode == "fixed":
        return False
    if mode == "prob_flip":
        return p_short > p_long if side == 1 else p_long > p_short
    if mode == "signal_raw":
        return p_short >= args.threshold if side == 1 else p_long >= args.threshold
    if mode == "signal_exit":
        return risk_ok(args, -side, pred)
    if mode == "rr_floor":
        return exp_up / max(exp_down, 1e-9) <= args.rr_floor if side == 1 else exp_down / max(exp_up, 1e-9) <= args.rr_floor
    raise ValueError(f"bad mode: {mode}")


def entry_allowed_now(args) -> bool:
    if int(args.session) == -99:
        return True
    now_ns = int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)
    return bool(active_session_allowed(np.array([now_ns], dtype=np.int64), int(args.session))[0])


def build_candles_from_rates(rates) -> CandleData:
    times = np.array([int(r["time"]) * 1_000_000_000 for r in rates], dtype=np.int64).astype("datetime64[ns]")
    ohlc = np.column_stack([
        np.asarray([float(r["open"]) for r in rates], dtype=np.float32),
        np.asarray([float(r["high"]) for r in rates], dtype=np.float32),
        np.asarray([float(r["low"]) for r in rates], dtype=np.float32),
        np.asarray([float(r["close"]) for r in rates], dtype=np.float32),
    ])
    spread = np.asarray([float(r["spread"]) for r in rates], dtype=np.float32)
    idx = np.arange(len(rates), dtype=np.int64)
    return CandleData(times=times, ohlc=ohlc, spread=spread, close_tick_idx=idx, tick_to_candle=idx)


def preload_candles(mt5, args, need: int) -> CandleData:
    native = mt5_timeframe(mt5, args.timeframe)
    if native is not None:
        rates = mt5.copy_rates_from_pos(args.symbol, native, 1, need)
        if rates is not None and len(rates) >= max(need // 2, 10):
            print(f"[ml-paper] preload native candles={len(rates)}", flush=True)
            return build_candles_from_rates(rates)
    tf_ns = timeframe_to_ns(args.timeframe)
    seconds = max(int((need + 10) * tf_ns / 1_000_000_000), 300)
    start = datetime.now(timezone.utc) - timedelta(seconds=seconds)
    ticks = mt5.copy_ticks_from(args.symbol, start, 500000, mt5.COPY_TICKS_INFO)
    if ticks is None or len(ticks) == 0:
        raise SystemExit("could not preload candles")
    bid = np.asarray([float(t["bid"]) for t in ticks], dtype=np.float64)
    ask = np.asarray([float(t["ask"]) for t in ticks], dtype=np.float64)
    ts = np.asarray([int(t["time_msc"]) * 1_000_000 if int(t["time_msc"]) > 0 else int(t["time"]) * 1_000_000_000 for t in ticks], dtype=np.int64)
    candles = build_bid_candles(bid, ask, ts, args.timeframe)
    print(f"[ml-paper] preload tick candles={len(candles.ohlc)}", flush=True)
    return candles


def append_closed_candle(candles: CandleData, bucket_ns: int, o: float, h: float, l: float, c: float, spread: float, maxlen: int) -> CandleData:
    t = np.array([bucket_ns], dtype=np.int64).astype("datetime64[ns]")
    row = np.array([[o, h, l, c]], dtype=np.float32)
    sp = np.array([spread], dtype=np.float32)
    times = np.concatenate([candles.times, t])[-maxlen:]
    ohlc = np.concatenate([candles.ohlc, row])[-maxlen:]
    spreads = np.concatenate([candles.spread, sp])[-maxlen:]
    idx = np.arange(len(ohlc), dtype=np.int64)
    return CandleData(times=times, ohlc=ohlc, spread=spreads, close_tick_idx=idx, tick_to_candle=idx)


def predict_last(model, ns, candles: CandleData, ps: float, device: torch.device) -> np.ndarray | None:
    window = int(ns.window)
    if len(candles.ohlc) < window:
        return None
    barrier = float(getattr(ns, "barrier_points", getattr(ns, "move_scale_points", 100.0)))
    session = active_session_allowed(candles.times.astype("int64"), 1).astype(np.float32)
    data = BarrierData(
        times=candles.times,
        ohlc=candles.ohlc,
        spread=candles.spread,
        labels=np.zeros((len(candles.ohlc), 4), dtype=np.float32),
        valid=np.ones(len(candles.ohlc), dtype=np.bool_),
        session=session,
        point_size=ps,
    )
    i = len(candles.ohlc) - 1
    x = make_window_features(data, i, window, barrier, ns.feature_set)
    time_feat = make_time_features(data.times)
    extra_values = [data.spread[i] / (data.point_size * barrier), *time_feat[i]]
    if getattr(ns, "session_feature", False):
        extra_values.insert(1, data.session[i])
    with torch.no_grad():
        raw = model(
            torch.from_numpy(x[None, :, :]).to(device),
            torch.from_numpy(np.asarray(extra_values, dtype=np.float32)[None, :]).to(device),
        )
        if getattr(ns, "target", "") == "move4":
            p = torch.sigmoid(raw[:, :2])
            moves = torch.relu(raw[:, 2:]) * float(getattr(ns, "move_scale_points", 100.0))
            out = torch.cat([p, moves], dim=1).cpu().numpy()[0]
        else:
            p_up = torch.sigmoid(raw).reshape(-1, 1)
            huge = torch.full_like(p_up, 1.0e9)
            out = torch.cat([p_up, 1.0 - p_up, huge, huge], dim=1).cpu().numpy()[0]
    return out.astype(np.float32)


def plan_from_position(mt5, args, pos, pred: np.ndarray, tag: str) -> ActivePlan:
    side = 1 if int(pos.type) == mt5.POSITION_TYPE_BUY else -1
    plan = ActivePlan(
        int(pos.ticket),
        side,
        float(pos.price_open),
        float(args.tp_points if args.tp_mode == "fixed" else 0.0),
        float(args.sl_points if args.sl_mode == "fixed" else 0.0),
        float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3]),
    )
    print(f"[ml-paper] ADOPT {tag} {signal_name(side)} ticket={plan.ticket} entry={plan.entry:.5f}", flush=True)
    return plan


def live_text(mt5, args, plan: ActivePlan | None, bid: float, ask: float) -> str:
    pos = find_position(mt5, args.symbol, args.magic)
    if pos is None:
        return "-"
    side = 1 if int(pos.type) == mt5.POSITION_TYPE_BUY else -1
    entry = float(pos.price_open)
    ps = point_size(mt5, args.symbol)
    move = (bid - entry) / ps if side == 1 else (entry - ask) / ps
    pnl = float(getattr(pos, "profit", 0.0) or 0.0)
    pred = ""
    if plan and int(pos.ticket) == plan.ticket:
        pred = f" p=({plan.long_prob:.2f}/{plan.short_prob:.2f}) exp=({plan.exp_up:.0f}/{plan.exp_down:.0f})"
    return f"{signal_name(side)} entry={entry:.5f} move={move:+.1f}pt pnl=${pnl:+.2f}{pred}"


def close_fixed_if_needed(mt5, args, plan: ActivePlan | None) -> ActivePlan | None:
    pos = find_position(mt5, args.symbol, args.magic)
    if pos is None:
        return None
    tick = mt5.symbol_info_tick(args.symbol)
    if tick is None:
        return plan
    ps = point_size(mt5, args.symbol)
    side = 1 if int(pos.type) == mt5.POSITION_TYPE_BUY else -1
    entry = float(pos.price_open)
    move = (float(tick.bid) - entry) / ps if side == 1 else (entry - float(tick.ask)) / ps
    if args.tp_mode == "fixed" and args.tp_points > 0 and move >= args.tp_points:
        send_order(mt5, args, "sell" if side == 1 else "buy", "tp_fixed", int(pos.ticket), tag="ml")
        return None
    if args.sl_mode == "fixed" and args.sl_points > 0 and move <= -args.sl_points:
        send_order(mt5, args, "sell" if side == 1 else "buy", "sl_fixed", int(pos.ticket), tag="ml")
        return None
    return plan


def decision_text(args, pred: np.ndarray, state: int) -> str:
    p_long, p_short, exp_up, exp_down = map(float, pred)
    rr_long = exp_up / max(exp_down, 1e-9)
    rr_short = exp_down / max(exp_up, 1e-9)
    return (
        f"sig={signal_name(state):<5} pL={p_long:.3f} pS={p_short:.3f} "
        f"expUp={exp_up:6.1f} expDn={exp_down:6.1f} "
        f"rrL={rr_long:4.2f} rrS={rr_short:4.2f}"
    )


def smooth_live_prediction(history: deque[np.ndarray], pred: np.ndarray, ma: int) -> np.ndarray:
    history.append(pred.astype(np.float32))
    ma = max(1, int(ma))
    if ma <= 1:
        return pred
    vals = list(history)[-ma:]
    return np.nanmean(np.stack(vals), axis=0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean MT5 paper trader for move4 ML models")
    ap.add_argument("--model", required=True)
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--threshold", type=float, default=0.65)
    ap.add_argument("--risk-mode", choices=sorted(RISK_MODES), default="rr")
    ap.add_argument("--risk-value", type=float, default=2.0)
    ap.add_argument("--tp-mode", choices=sorted(MODES), default="prob_flip")
    ap.add_argument("--sl-mode", choices=sorted(MODES), default="prob_flip")
    ap.add_argument("--rr-floor", type=float, default=1.2)
    ap.add_argument("--prob-ma", type=int, default=1)
    ap.add_argument("--tp-points", type=float, default=0.0)
    ap.add_argument("--sl-points", type=float, default=0.0)
    ap.add_argument("--min-tp", type=float, default=None)
    ap.add_argument("--session", default="label")
    ap.add_argument("--side", choices=["long", "short", "both"], default="both")
    ap.add_argument("--lot", type=float, default=0.01)
    ap.add_argument("--magic", type=int, default=935501)
    ap.add_argument("--deviation", type=int, default=50)
    ap.add_argument("--filling-mode", choices=["auto", "broker", "ioc", "fok", "return"], default="ioc")
    ap.add_argument("--poll", type=float, default=0.25)
    ap.add_argument("--log-every", type=float, default=3.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    meta = parse_model_name(Path(args.model))
    args.symbol = (args.symbol or str(meta.get("pair", "AUDUSD"))).upper()
    args.timeframe = str(meta.get("tf", "1m"))
    args.session = int(meta.get("label_session", 0)) if str(args.session).lower() == "label" else int(args.session)
    if args.min_tp is None:
        args.min_tp = 25.0 if args.symbol.startswith("XAU") else 5.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ns, _ = load_torch_model(Path(args.model))
    model = model.to(device).eval()

    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise SystemExit(f"mt5.initialize failed: {mt5.last_error()}")
    if not mt5.symbol_select(args.symbol, True):
        raise SystemExit(f"symbol_select failed: {args.symbol}")

    ps = default_point_size(args.symbol)
    candles = preload_candles(mt5, args, int(ns.window) + 20)
    maxlen = max(int(ns.window) + 500, 2000)
    tf_ns = timeframe_to_ns(args.timeframe)
    current_bucket = None
    cur_o = cur_h = cur_l = cur_c = 0.0
    spread_sum = 0.0
    spread_n = 0
    last_log = 0.0
    plan: ActivePlan | None = None
    pred_history: deque[np.ndarray] = deque(maxlen=max(1, int(args.prob_ma)))
    print(
        f"[ml-paper] START {args.symbol} {args.timeframe} threshold={args.threshold:g} "
        f"risk={args.risk_mode}:{args.risk_value:g} tp={args.tp_mode}:{args.tp_points:g} "
        f"sl={args.sl_mode}:{args.sl_points:g} rr_floor={args.rr_floor:g} prob_ma={args.prob_ma} session={args.session} lot={args.lot:g} device={device}",
        flush=True,
    )
    try:
        while True:
            tick = mt5.symbol_info_tick(args.symbol)
            if tick is None:
                time.sleep(args.poll)
                continue
            bid = float(tick.bid)
            ask = float(tick.ask)
            now_sec = int(getattr(tick, "time", int(time.time())))
            bucket = ((now_sec * 1_000_000_000) // tf_ns) * tf_ns
            if current_bucket is None:
                current_bucket = bucket
                cur_o = cur_h = cur_l = cur_c = bid
                spread_sum = ask - bid
                spread_n = 1
            elif bucket != current_bucket:
                avg_spread = spread_sum / max(spread_n, 1)
                candles = append_closed_candle(candles, current_bucket, cur_o, cur_h, cur_l, cur_c, avg_spread, maxlen)
                pred = predict_last(model, ns, candles, ps, device)
                if pred is not None:
                    raw_pred = pred
                    pred = smooth_live_prediction(pred_history, raw_pred, args.prob_ma)
                    state = choose_signal(args, pred)
                    ts = datetime.fromtimestamp(current_bucket / 1_000_000_000, tz=timezone.utc)
                    print(f"[ml-paper] BAR {ts:%m-%d %H:%M} close={cur_c:.5f} | {decision_text(args, pred, state)} raw=({raw_pred[0]:.3f}/{raw_pred[1]:.3f}) session_ok={int(entry_allowed_now(args))}", flush=True)
                    pos = find_position(mt5, args.symbol, args.magic)
                    if pos is not None and (plan is None or int(pos.ticket) != plan.ticket):
                        plan = plan_from_position(mt5, args, pos, pred, "existing_position")
                    if pos is not None and plan is not None and int(pos.ticket) == plan.ticket:
                        move = ((cur_c - plan.entry) / ps) if plan.side == 1 else ((plan.entry - cur_c) / ps)
                        exit_mode = args.tp_mode if move >= 0 else args.sl_mode
                        if mode_exit(args, exit_mode, plan.side, pred):
                            old_side = plan.side
                            print(f"[ml-paper] SIGNAL EXIT mode={exit_mode} move={move:+.1f}pt", flush=True)
                            send_order(mt5, args, "sell" if old_side == 1 else "buy", f"exit_{exit_mode}", int(pos.ticket), tag="ml")
                            plan = None
                            new_side = -old_side
                            if risk_ok(args, new_side, pred):
                                print(f"[ml-paper] REVERSE {signal_name(new_side)}", flush=True)
                                res = send_order(mt5, args, "buy" if new_side == 1 else "sell", f"reverse_{exit_mode}", tag="ml")
                                if res is not None:
                                    pos2 = find_position(mt5, args.symbol, args.magic)
                                    if pos2 is not None:
                                        plan = plan_from_position(mt5, args, pos2, pred, "reverse")
                    elif pos is None and state != 0:
                        if not entry_allowed_now(args):
                            print("[ml-paper] ENTRY BLOCK session", flush=True)
                        else:
                            res = send_order(mt5, args, "buy" if state == 1 else "sell", f"entry_{signal_name(state).lower()}", tag="ml")
                            if res is not None:
                                pos2 = find_position(mt5, args.symbol, args.magic)
                                if pos2 is not None:
                                    plan = plan_from_position(mt5, args, pos2, pred, "entry")
                current_bucket = bucket
                cur_o = cur_h = cur_l = cur_c = bid
                spread_sum = ask - bid
                spread_n = 1
            else:
                cur_h = max(cur_h, bid)
                cur_l = min(cur_l, bid)
                cur_c = bid
                spread_sum += ask - bid
                spread_n += 1
            plan = close_fixed_if_needed(mt5, args, plan)
            now = time.time()
            if now - last_log >= args.log_every:
                last_log = now
                print(f"[ml-paper] TICK bid={bid:.5f} ask={ask:.5f} spread={(ask-bid)/point_size(mt5,args.symbol):.1f}pt | pos={live_text(mt5,args,plan,bid,ask)}", flush=True)
            time.sleep(args.poll)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
