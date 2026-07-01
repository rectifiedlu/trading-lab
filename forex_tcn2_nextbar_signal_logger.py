"""Live signal logger for TCN2 nextbar paper/backtest parity checks.

Logs the exact candle stream and model signals seen by the paper process.
Run it beside/without the paper bot for a few minutes, then compare against
tick-built backtest signals over the same UTC window.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import deque

import numpy as np
import torch

from forex_ml_tick_simulator import build_bid_candles, load_torch_model, parse_model_name
from forex_signal_paper_common import find_position, point_size
from forex_strategy_common import active_session_allowed, timeframe_to_ns
from forex_tcn2_nextbar_paper import (
    append_closed_candle,
    apply_signal_mode,
    predict_p_up,
    seed_probability_history,
    signal_from_prob,
    signal_name,
)


def session_allowed(mt5, symbol: str, session: int) -> bool:
    if session == -99:
        return True
    tick = mt5.symbol_info_tick(symbol)
    now_sec = int(getattr(tick, "time", int(time.time()))) if tick is not None else int(time.time())
    now_ns = now_sec * 1_000_000_000
    return bool(active_session_allowed(np.array([now_ns], dtype=np.int64), int(session))[0])


def ticks_to_arrays(ticks):
    bid = np.asarray([float(t["bid"]) for t in ticks], dtype=np.float64)
    ask = np.asarray([float(t["ask"]) for t in ticks], dtype=np.float64)
    ts = np.asarray(
        [int(t["time_msc"]) * 1_000_000 if int(t["time_msc"]) > 0 else int(t["time"]) * 1_000_000_000 for t in ticks],
        dtype=np.int64,
    )
    return bid, ask, ts


def preload_tick_candles(mt5, args, need: int):
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
        candles = type(candles)(
            times=candles.times[sl],
            ohlc=candles.ohlc[sl],
            spread=candles.spread[sl],
            close_tick_idx=idx,
            tick_to_candle=idx,
        )
    print(f"[tcn2-log] preload tick-built candles={len(candles.ohlc)}", flush=True)
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


def window_hash(candles, window: int) -> str:
    if len(candles.ohlc) < window:
        return ""
    payload = np.column_stack([
        candles.ohlc[-window:].astype(np.float64),
        candles.spread[-window:].astype(np.float64),
    ])
    payload = np.round(payload, 10)
    return hashlib.sha256(np.ascontiguousarray(payload).tobytes()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description="Log live TCN2 candle/model signals for parity checks")
    ap.add_argument("--model", required=True)
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--upper", type=float, default=0.5)
    ap.add_argument("--lower", type=float, default=0.5)
    ap.add_argument("--prob-ma", type=int, default=1)
    ap.add_argument("--signal-mode", choices=["normal", "invert"], default="normal")
    ap.add_argument("--session", default="label")
    ap.add_argument("--magic", type=int, default=936601)
    ap.add_argument("--poll", type=float, default=0.1)
    ap.add_argument(
        "--bar-settle",
        type=float,
        default=1.0,
        help="seconds to wait after a candle boundary before fetching its complete MT5 tick history",
    )
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--out", default="data/forex/analysis/tcn2_live_signal_log.csv")
    args = ap.parse_args()

    meta = parse_model_name(Path(args.model))
    args.symbol = (args.symbol or str(meta.get("pair", "AUDUSD"))).upper()
    args.timeframe = str(meta.get("tf", "1m"))
    args.session = int(meta.get("label_session", 0)) if str(args.session).lower() == "label" else int(args.session)

    import MetaTrader5 as mt5

    if not mt5.initialize():
        raise SystemExit(f"mt5.initialize failed: {mt5.last_error()}")
    if not mt5.symbol_select(args.symbol, True):
        raise SystemExit(f"symbol_select failed: {args.symbol}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ns, _ = load_torch_model(Path(args.model))
    model = model.to(device).eval()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    window_path = out_path.with_name(out_path.stem + "_windows.csv")

    candles = preload_tick_candles(mt5, args, int(ns.window) + 20)
    maxlen = max(int(ns.window) + 500, 2000)
    tf_ns = timeframe_to_ns(args.timeframe)
    current_bucket = None
    cur_o = cur_h = cur_l = cur_c = 0.0
    spread_sum = 0.0
    spread_n = 0
    p_hist = seed_probability_history(model, ns, candles, ps, device, args.prob_ma, args.signal_mode)
    end_at = time.time() + float(args.minutes) * 60.0
    ps = point_size(mt5, args.symbol)

    fields = [
        "wall_utc", "bucket_utc", "symbol", "timeframe",
        "open", "high", "low", "close", "avg_spread_points",
        "raw_p", "trade_p", "p_ma", "signal", "session_ok",
        "window_start_utc", "window_end_utc", "window_hash",
        "position_side", "position_entry", "position_profit",
        "upper", "lower", "prob_ma", "signal_mode", "model_file",
    ]
    window_fields = ["bucket_utc", "offset", "time_utc", "open", "high", "low", "close", "spread_points"]
    with out_path.open("w", newline="", encoding="utf-8") as f, window_path.open("w", newline="", encoding="utf-8") as fw:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        ww = csv.DictWriter(fw, fieldnames=window_fields)
        ww.writeheader()
        print(
            f"[tcn2-log] START {args.symbol} {args.timeframe} minutes={args.minutes:g} "
            f"upper={args.upper:g} lower={args.lower:g} ma={args.prob_ma} mode={args.signal_mode} "
            f"session={args.session} out={out_path}",
            flush=True,
        )
        try:
            while time.time() < end_at:
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
                    raw_p = predict_p_up(model, ns, candles, ps, device)
                    if raw_p is not None:
                        trade_p = apply_signal_mode(float(raw_p), args.signal_mode)
                        p_hist.append(float(trade_p))
                        vals = list(p_hist)[-max(1, int(args.prob_ma)):]
                        p_ma = float(sum(vals) / len(vals))
                        sig = signal_from_prob(p_ma, args.upper, args.lower)
                        pos = find_position(mt5, args.symbol, args.magic)
                        pos_side = "-"
                        pos_entry = ""
                        pos_profit = ""
                        if pos is not None:
                            pos_side = "BUY" if int(pos.type) == mt5.POSITION_TYPE_BUY else "SELL"
                            pos_entry = f"{float(pos.price_open):.10f}"
                            pos_profit = f"{float(getattr(pos, 'profit', 0.0) or 0.0):.6f}"
                        row = {
                            "wall_utc": datetime.now(timezone.utc).isoformat(),
                            "bucket_utc": datetime.fromtimestamp(current_bucket / 1_000_000_000, tz=timezone.utc).isoformat(),
                            "symbol": args.symbol,
                            "timeframe": args.timeframe,
                            "open": f"{exact_o:.10f}",
                            "high": f"{exact_h:.10f}",
                            "low": f"{exact_l:.10f}",
                            "close": f"{exact_c:.10f}",
                            "avg_spread_points": f"{avg_spread / ps:.3f}",
                            "raw_p": f"{float(raw_p):.9f}",
                            "trade_p": f"{trade_p:.9f}",
                            "p_ma": f"{p_ma:.9f}",
                            "signal": signal_name(sig),
                            "session_ok": int(session_allowed(mt5, args.symbol, int(args.session))),
                            "window_start_utc": datetime.fromtimestamp(
                                candles.times.astype("int64")[-int(ns.window)] / 1_000_000_000,
                                tz=timezone.utc,
                            ).isoformat() if len(candles.ohlc) >= int(ns.window) else "",
                            "window_end_utc": datetime.fromtimestamp(
                                candles.times.astype("int64")[-1] / 1_000_000_000,
                                tz=timezone.utc,
                            ).isoformat(),
                            "window_hash": window_hash(candles, int(ns.window)),
                            "position_side": pos_side,
                            "position_entry": pos_entry,
                            "position_profit": pos_profit,
                            "upper": args.upper,
                            "lower": args.lower,
                            "prob_ma": args.prob_ma,
                            "signal_mode": args.signal_mode,
                            "model_file": Path(args.model).name,
                        }
                        w.writerow(row)
                        if len(candles.ohlc) >= int(ns.window):
                            win = int(ns.window)
                            times_i64 = candles.times.astype("int64")
                            for off in range(-win + 1, 1):
                                j = len(candles.ohlc) + off - 1
                                ww.writerow({
                                    "bucket_utc": row["bucket_utc"],
                                    "offset": off,
                                    "time_utc": datetime.fromtimestamp(int(times_i64[j]) / 1_000_000_000, tz=timezone.utc).isoformat(),
                                    "open": f"{float(candles.ohlc[j, 0]):.10f}",
                                    "high": f"{float(candles.ohlc[j, 1]):.10f}",
                                    "low": f"{float(candles.ohlc[j, 2]):.10f}",
                                    "close": f"{float(candles.ohlc[j, 3]):.10f}",
                                    "spread_points": f"{float(candles.spread[j]) / ps:.6f}",
                                })
                        f.flush()
                        fw.flush()
                        print(
                            f"[tcn2-log] {row['bucket_utc'][5:16]} close={exact_c:.5f} "
                            f"raw={float(raw_p):.3f} trade={trade_p:.3f} ma={p_ma:.3f} "
                            f"sig={row['signal']} pos={pos_side} sess={row['session_ok']}",
                            flush=True,
                        )
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
                time.sleep(args.poll)
        finally:
            mt5.shutdown()
    print(f"[tcn2-log] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
