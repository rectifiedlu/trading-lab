"""MT5 paper trader for clean TCN2 nextbar models.

Matches forex_tcn2_nextbar_simulator.py semantics:
- model outputs one probability: p_up
- p_up moving average maps to LONG/SHORT/WAIT via upper/lower thresholds
- TP and SL modes are independent: fixed, fixed_signal, neutral, opposite
- fixed exits are tick-based; signal exits are candle-close based
- ma_reset blocks same-side re-entry after a fixed TP/SL until p_ma leaves that side zone
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from forex_ml_barrier_cnn import BarrierData, make_time_features, make_window_features
from forex_ml_tick_simulator import CandleData, build_bid_candles, load_torch_model, parse_model_name
from forex_signal_paper_common import find_position, point_size, send_order
from forex_strategy_common import active_session_allowed, default_point_size, timeframe_to_ns
from forex_tcn2_nextbar_simulator import EXIT_MODES, REENTRY_MODES, parse_reentry_mode


def signal_name(side: int) -> str:
    return "LONG" if side == 1 else "SHORT" if side == -1 else "WAIT"


def signal_from_prob(p_up: float, upper: float, lower: float) -> int:
    if not np.isfinite(p_up):
        return 0
    if p_up >= upper:
        return 1
    if p_up <= lower:
        return -1
    return 0


def apply_signal_mode(p_up: float, signal_mode: str) -> float:
    if not np.isfinite(p_up):
        return p_up
    if signal_mode == "invert":
        return 1.0 - p_up
    return p_up


def side_allowed(args, side: int) -> bool:
    return args.side == "both" or (args.side == "long" and side == 1) or (args.side == "short" and side == -1)


def entry_session_allowed(args) -> bool:
    if int(args.session) == -99:
        return True
    import MetaTrader5 as mt5
    tick = mt5.symbol_info_tick(args.symbol)
    now_sec = int(getattr(tick, "time", int(time.time()))) if tick is not None else int(time.time())
    now_ns = now_sec * 1_000_000_000
    return bool(active_session_allowed(np.array([now_ns], dtype=np.int64), int(args.session))[0])


def ticks_to_arrays(ticks):
    bid = np.asarray([float(t["bid"]) for t in ticks], dtype=np.float64)
    ask = np.asarray([float(t["ask"]) for t in ticks], dtype=np.float64)
    ts = np.asarray(
        [int(t["time_msc"]) * 1_000_000 if int(t["time_msc"]) > 0 else int(t["time"]) * 1_000_000_000 for t in ticks],
        dtype=np.int64,
    )
    return bid, ask, ts


def preload_candles(mt5, args, need: int) -> CandleData:
    tf_ns = timeframe_to_ns(args.timeframe)
    tick = mt5.symbol_info_tick(args.symbol)
    now_sec = int(getattr(tick, "time", int(time.time()))) if tick is not None else int(time.time())
    now_ns = now_sec * 1_000_000_000
    current_bucket = (now_ns // tf_ns) * tf_ns
    end = datetime.fromtimestamp(current_bucket / 1_000_000_000, tz=timezone.utc)
    seconds = max(int((need + 20) * tf_ns / 1_000_000_000), 900)
    candles = None
    for _ in range(6):
        start = end - timedelta(seconds=seconds)
        ticks = mt5.copy_ticks_range(args.symbol, start, end, mt5.COPY_TICKS_INFO)
        if ticks is not None and len(ticks) > 0:
            bid, ask, ts = ticks_to_arrays(ticks)
            closed = ts < current_bucket
            bid = bid[closed]
            ask = ask[closed]
            ts = ts[closed]
            if len(ts) > 0:
                candles = build_bid_candles(bid, ask, ts, args.timeframe)
                if len(candles.ohlc) >= need:
                    break
        seconds *= 2
    if candles is None or len(candles.ohlc) < need:
        got = 0 if candles is None else len(candles.ohlc)
        raise SystemExit(f"could not preload required candles: need={need} got={got}")
    if len(candles.ohlc) > need:
        sl = slice(len(candles.ohlc) - need, len(candles.ohlc))
        idx = np.arange(need, dtype=np.int64)
        candles = CandleData(
            times=candles.times[sl],
            ohlc=candles.ohlc[sl],
            spread=candles.spread[sl],
            close_tick_idx=idx,
            tick_to_candle=idx,
        )
    print(f"[tcn2-paper] preload tick-built candles={len(candles.ohlc)}", flush=True)
    return candles


def fetch_closed_bucket(mt5, args, bucket_ns: int, tf_ns: int, fallback):
    start = datetime.fromtimestamp(bucket_ns / 1_000_000_000, tz=timezone.utc)
    end = datetime.fromtimestamp((bucket_ns + tf_ns) / 1_000_000_000, tz=timezone.utc)
    ticks = mt5.copy_ticks_range(args.symbol, start, end, mt5.COPY_TICKS_INFO)
    if ticks is None or len(ticks) == 0:
        return fallback
    bid, ask, ts = ticks_to_arrays(ticks)
    buckets = (ts // tf_ns) * tf_ns
    mask = buckets == bucket_ns
    if not np.any(mask):
        return fallback
    b = bid[mask]
    a = ask[mask]
    return (
        float(b[0]),
        float(np.max(b)),
        float(np.min(b)),
        float(b[-1]),
        float(np.mean(a - b)),
    )


def append_closed_candle(candles: CandleData, bucket_ns: int, o: float, h: float, l: float, c: float, spread: float, maxlen: int) -> CandleData:
    t = np.array([bucket_ns], dtype=np.int64).astype("datetime64[ns]")
    row = np.array([[o, h, l, c]], dtype=np.float32)
    sp = np.array([spread], dtype=np.float32)
    times = np.concatenate([candles.times, t])[-maxlen:]
    ohlc = np.concatenate([candles.ohlc, row])[-maxlen:]
    spreads = np.concatenate([candles.spread, sp])[-maxlen:]
    idx = np.arange(len(ohlc), dtype=np.int64)
    return CandleData(times=times, ohlc=ohlc, spread=spreads, close_tick_idx=idx, tick_to_candle=idx)


def predict_p_up(model, ns, candles: CandleData, point: float, device: torch.device) -> float | None:
    window = int(ns.window)
    if len(candles.ohlc) < window:
        return None
    scale = float(getattr(ns, "barrier_points", getattr(ns, "move_scale_points", 100.0)))
    session = active_session_allowed(candles.times.astype("int64"), 1).astype(np.float32)
    data = BarrierData(
        times=candles.times,
        ohlc=candles.ohlc,
        spread=candles.spread,
        labels=np.zeros((len(candles.ohlc), 4), dtype=np.float32),
        valid=np.ones(len(candles.ohlc), dtype=np.bool_),
        session=session,
        point_size=point,
    )
    i = len(candles.ohlc) - 1
    x = make_window_features(data, i, window, scale, ns.feature_set)
    time_feat = make_time_features(data.times)
    extra_values = [data.spread[i] / (data.point_size * scale), *time_feat[i]]
    if getattr(ns, "session_feature", False):
        extra_values.insert(1, data.session[i])
    with torch.no_grad():
        raw = model(
            torch.from_numpy(x[None, :, :]).to(device),
            torch.from_numpy(np.asarray(extra_values, dtype=np.float32)[None, :]).to(device),
        )
        return float(torch.sigmoid(raw.reshape(-1)[0]).detach().cpu().item())


def smooth_prob(history: deque[float], raw: float, ma: int) -> float:
    history.append(float(raw))
    vals = list(history)[-max(1, int(ma)):]
    return float(np.mean(vals))


def seed_probability_history(model, ns, candles: CandleData, point: float, device: torch.device, ma: int, signal_mode: str) -> deque[float]:
    history: deque[float] = deque(maxlen=max(1, int(ma)))
    needed = max(0, int(ma) - 1)
    if needed == 0:
        return history
    start = max(int(ns.window) - 1, len(candles.ohlc) - needed)
    for end_i in range(start, len(candles.ohlc)):
        idx = np.arange(end_i + 1, dtype=np.int64)
        prefix = CandleData(
            times=candles.times[:end_i + 1],
            ohlc=candles.ohlc[:end_i + 1],
            spread=candles.spread[:end_i + 1],
            close_tick_idx=idx,
            tick_to_candle=idx,
        )
        raw = predict_p_up(model, ns, prefix, point, device)
        if raw is not None:
            history.append(apply_signal_mode(raw, signal_mode))
    return history


@dataclass
class ActivePlan:
    ticket: int
    side: int
    entry: float
    p_entry: float
    p_raw_entry: float


def plan_from_position(mt5, args, pos, p_ma: float, p_raw: float, tag: str) -> ActivePlan:
    side = 1 if int(pos.type) == mt5.POSITION_TYPE_BUY else -1
    plan = ActivePlan(int(pos.ticket), side, float(pos.price_open), float(p_ma), float(p_raw))
    print(
        f"[tcn2-paper] ADOPT {tag} {signal_name(side)} ticket={plan.ticket} "
        f"entry={plan.entry:.5f} p={p_ma:.3f} raw={p_raw:.3f}",
        flush=True,
    )
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
        pred = f" entry_p={plan.p_entry:.3f}"
    return f"{signal_name(side)} entry={entry:.5f} move={move:+.1f}pt pnl=${pnl:+.2f}{pred}"


def fixed_exit_if_needed(mt5, args, plan: ActivePlan | None):
    pos = find_position(mt5, args.symbol, args.magic)
    if pos is None:
        return None, 0, ""
    tick = mt5.symbol_info_tick(args.symbol)
    if tick is None:
        return plan, 0, ""
    ps = point_size(mt5, args.symbol)
    side = 1 if int(pos.type) == mt5.POSITION_TYPE_BUY else -1
    entry = float(pos.price_open)
    move = (float(tick.bid) - entry) / ps if side == 1 else (entry - float(tick.ask)) / ps
    if args.tp_mode in ("fixed", "fixed_signal") and args.tp_points > 0 and move >= args.tp_points:
        send_order(mt5, args, "sell" if side == 1 else "buy", "tp_fixed", int(pos.ticket), tag="tcn2")
        return None, side, "TP"
    if args.sl_mode in ("fixed", "fixed_signal") and args.sl_points > 0 and move <= -args.sl_points:
        send_order(mt5, args, "sell" if side == 1 else "buy", "sl_fixed", int(pos.ticket), tag="tcn2")
        return None, side, "SL"
    return plan, 0, ""


def mode_exit(mode: str, side: int, sig: int) -> bool:
    if mode == "fixed":
        return False
    if mode == "opposite":
        return sig == -side
    if mode == "neutral":
        return sig != side
    if mode == "fixed_signal":
        return sig == -side
    raise ValueError(f"bad mode: {mode}")


def main() -> None:
    ap = argparse.ArgumentParser(description="MT5 paper trader for TCN2 nextbar models")
    ap.add_argument("--model", required=True)
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--upper", type=float, default=0.6)
    ap.add_argument("--lower", type=float, default=0.4)
    ap.add_argument("--prob-ma", type=int, default=1)
    ap.add_argument("--signal-mode", choices=["normal", "invert"], default="normal")
    ap.add_argument("--tp-mode", choices=sorted(EXIT_MODES), default="neutral")
    ap.add_argument("--sl-mode", choices=sorted(EXIT_MODES), default="fixed")
    ap.add_argument("--tp-points", type=float, default=0.0)
    ap.add_argument("--sl-points", type=float, default=0.0)
    ap.add_argument("--reentry-mode", choices=sorted(REENTRY_MODES), default="immediate")
    ap.add_argument("--session", default="label")
    ap.add_argument("--side", choices=["long", "short", "both"], default="both")
    ap.add_argument("--lot", type=float, default=0.01)
    ap.add_argument("--magic", type=int, default=936601)
    ap.add_argument("--deviation", type=int, default=50)
    ap.add_argument("--filling-mode", choices=["auto", "broker", "ioc", "fok", "return"], default="ioc")
    ap.add_argument("--poll", type=float, default=0.25)
    ap.add_argument("--log-every", type=float, default=3.0)
    ap.add_argument(
        "--bar-settle",
        type=float,
        default=1.0,
        help="seconds to wait after a candle boundary before fetching its complete MT5 tick history",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.reentry_mode = parse_reentry_mode(args.reentry_mode)

    meta = parse_model_name(Path(args.model))
    args.symbol = (args.symbol or str(meta.get("pair", "AUDUSD"))).upper()
    args.timeframe = str(meta.get("tf", "1m"))
    args.session = int(meta.get("label_session", 0)) if str(args.session).lower() == "label" else int(args.session)

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
    p_hist = seed_probability_history(model, ns, candles, ps, device, args.prob_ma, args.signal_mode)
    block_long = False
    block_short = False

    print(
        f"[tcn2-paper] START {args.symbol} {args.timeframe} upper={args.upper:g} lower={args.lower:g} "
        f"ma={args.prob_ma} signal_mode={args.signal_mode} tp={args.tp_mode}:{args.tp_points:g} sl={args.sl_mode}:{args.sl_points:g} "
        f"reentry={args.reentry_mode} session={args.session} lot={args.lot:g} fill={args.filling_mode} device={device}",
        flush=True,
    )
    print(
        f"[tcn2-paper] MODEL {Path(args.model).name} window={ns.window} "
        f"horizon={meta.get('horizon', getattr(ns, 'horizon', '?'))} features={ns.feature_set}",
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
            # Use the broker tick clock, matching MT5 history and backtest data.
            now_sec = int(getattr(tick, "time", int(time.time())))
            bucket = ((now_sec * 1_000_000_000) // tf_ns) * tf_ns
            if current_bucket is None:
                current_bucket = bucket
                cur_o = cur_h = cur_l = cur_c = bid
                spread_sum = ask - bid
                spread_n = 1
            elif bucket != current_bucket:
                if args.bar_settle > 0:
                    time.sleep(args.bar_settle)
                fallback = (cur_o, cur_h, cur_l, cur_c, spread_sum / max(spread_n, 1))
                exact_o, exact_h, exact_l, exact_c, avg_spread = fetch_closed_bucket(mt5, args, current_bucket, tf_ns, fallback)
                candles = append_closed_candle(candles, current_bucket, exact_o, exact_h, exact_l, exact_c, avg_spread, maxlen)
                p_raw = predict_p_up(model, ns, candles, ps, device)
                if p_raw is not None:
                    p_trade_raw = apply_signal_mode(p_raw, args.signal_mode)
                    p_ma = smooth_prob(p_hist, p_trade_raw, args.prob_ma)
                    if args.reentry_mode == "ma_reset":
                        if block_long and p_ma < args.upper:
                            block_long = False
                            print(f"[tcn2-paper] REENTRY RESET long p={p_ma:.3f} < upper={args.upper:g}", flush=True)
                        if block_short and p_ma > args.lower:
                            block_short = False
                            print(f"[tcn2-paper] REENTRY RESET short p={p_ma:.3f} > lower={args.lower:g}", flush=True)
                    sig = signal_from_prob(p_ma, args.upper, args.lower)
                    ts = datetime.fromtimestamp(current_bucket / 1_000_000_000, tz=timezone.utc)
                    print(
                        f"[tcn2-paper] BAR {ts:%m-%d %H:%M} close={exact_c:.5f} "
                        f"sig={signal_name(sig):<5} p={p_ma:.3f} raw={p_raw:.3f} trade_raw={p_trade_raw:.3f} "
                        f"zone={'LONG' if p_ma >= args.upper else 'SHORT' if p_ma <= args.lower else 'NEUTRAL'} "
                        f"blocks=L{int(block_long)}/S{int(block_short)} "
                        f"session_ok={int(entry_session_allowed(args))}",
                        flush=True,
                    )

                    pos = find_position(mt5, args.symbol, args.magic)
                    if pos is not None and (plan is None or int(pos.ticket) != plan.ticket):
                        plan = plan_from_position(mt5, args, pos, p_ma, p_raw, "existing_position")

                    if pos is not None and plan is not None and int(pos.ticket) == plan.ticket:
                        move = ((exact_c - plan.entry) / ps) if plan.side == 1 else ((plan.entry - exact_c) / ps)
                        exit_mode = args.tp_mode if move >= 0 else args.sl_mode
                        if mode_exit(exit_mode, plan.side, sig):
                            old_side = plan.side
                            print(f"[tcn2-paper] SIGNAL EXIT mode={exit_mode} move={move:+.1f}pt sig={signal_name(sig)}", flush=True)
                            send_order(mt5, args, "sell" if old_side == 1 else "buy", f"exit_{exit_mode}", int(pos.ticket), tag="tcn2")
                            plan = None
                            new_side = -old_side
                            if sig == new_side and side_allowed(args, new_side):
                                if not entry_session_allowed(args):
                                    print(f"[tcn2-paper] REVERSE BLOCK session {signal_name(new_side)}", flush=True)
                                elif args.reentry_mode == "ma_reset" and ((new_side == 1 and block_long) or (new_side == -1 and block_short)):
                                    print(f"[tcn2-paper] REVERSE BLOCK {signal_name(new_side)} ma_reset", flush=True)
                                else:
                                    print(f"[tcn2-paper] REVERSE {signal_name(new_side)}", flush=True)
                                    res = send_order(mt5, args, "buy" if new_side == 1 else "sell", f"reverse_{exit_mode}", tag="tcn2")
                                    if res is not None:
                                        pos2 = find_position(mt5, args.symbol, args.magic)
                                        if pos2 is not None:
                                            plan = plan_from_position(mt5, args, pos2, p_ma, p_raw, "reverse")
                    elif pos is None and sig != 0:
                        blocked = args.reentry_mode == "ma_reset" and ((sig == 1 and block_long) or (sig == -1 and block_short))
                        if not side_allowed(args, sig):
                            print(f"[tcn2-paper] ENTRY BLOCK side {signal_name(sig)}", flush=True)
                        elif blocked:
                            print(f"[tcn2-paper] ENTRY BLOCK ma_reset {signal_name(sig)}", flush=True)
                        elif not entry_session_allowed(args):
                            print("[tcn2-paper] ENTRY BLOCK session", flush=True)
                        else:
                            res = send_order(mt5, args, "buy" if sig == 1 else "sell", f"entry_{signal_name(sig).lower()}", tag="tcn2")
                            if res is not None:
                                pos2 = find_position(mt5, args.symbol, args.magic)
                                if pos2 is not None:
                                    plan = plan_from_position(mt5, args, pos2, p_ma, p_raw, "entry")

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

            plan, fixed_exit_side, fixed_exit_kind = fixed_exit_if_needed(mt5, args, plan)
            if fixed_exit_side and args.reentry_mode == "ma_reset":
                if fixed_exit_side == 1:
                    block_long = True
                else:
                    block_short = True
                print(
                    f"[tcn2-paper] REENTRY BLOCK {signal_name(fixed_exit_side)} "
                    f"after fixed {fixed_exit_kind}",
                    flush=True,
                )
            if fixed_exit_side:
                print(
                    f"[tcn2-paper] FIXED {fixed_exit_kind} closed intrabar; "
                    "next entry waits for candle-close signal",
                    flush=True,
                )

            now = time.time()
            if now - last_log >= args.log_every:
                last_log = now
                print(
                    f"[tcn2-paper] TICK bid={bid:.5f} ask={ask:.5f} "
                    f"spread={(ask-bid)/point_size(mt5,args.symbol):.1f}pt "
                    f"blocks=L{int(block_long)}/S{int(block_short)} | pos={live_text(mt5,args,plan,bid,ask)}",
                    flush=True,
                )
            time.sleep(args.poll)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
