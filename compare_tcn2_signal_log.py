from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from forex_ml_tick_simulator import build_bid_candles, load_torch_model, predict_model
from forex_signal_paper_common import point_size
from forex_strategy_common import timeframe_to_ns
from forex_tcn2_nextbar_paper import apply_signal_mode, signal_from_prob, signal_name


def parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def window_hash(candles, end_i: int, window: int) -> str:
    if end_i - window + 1 < 0:
        return ""
    payload = np.column_stack([
        candles.ohlc[end_i - window + 1:end_i + 1].astype(np.float64),
        candles.spread[end_i - window + 1:end_i + 1].astype(np.float64),
    ])
    payload = np.round(payload, 10)
    return hashlib.sha256(np.ascontiguousarray(payload).tobytes()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare live TCN2 signal log against tick-built MT5 replay")
    ap.add_argument("--log", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--upper", type=float, default=None)
    ap.add_argument("--lower", type=float, default=None)
    ap.add_argument("--prob-ma", type=int, default=None)
    ap.add_argument("--signal-mode", choices=["normal", "invert"], default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.log, newline="", encoding="utf-8")))
    if not rows:
        raise SystemExit("empty log")
    log_path = Path(args.log)
    live_window_path = log_path.with_name(log_path.stem + "_windows.csv")
    live_windows = {}
    if live_window_path.exists():
        for wr in csv.DictReader(open(live_window_path, newline="", encoding="utf-8")):
            live_windows[(parse_dt(wr["bucket_utc"]).isoformat(), int(wr["offset"]))] = wr
    symbol = (args.symbol or rows[0]["symbol"]).upper()
    tf = rows[0]["timeframe"]
    upper = float(args.upper if args.upper is not None else rows[0]["upper"])
    lower = float(args.lower if args.lower is not None else rows[0]["lower"])
    prob_ma = int(args.prob_ma if args.prob_ma is not None else rows[0]["prob_ma"])
    signal_mode = args.signal_mode or rows[0]["signal_mode"]
    start = parse_dt(rows[0]["bucket_utc"]) - timedelta(minutes=180)
    end = parse_dt(rows[-1]["bucket_utc"]) + timedelta(minutes=5)

    import MetaTrader5 as mt5

    if not mt5.initialize():
        raise SystemExit(f"mt5.initialize failed: {mt5.last_error()}")
    try:
        if not mt5.symbol_select(symbol, True):
            raise SystemExit(f"symbol_select failed: {symbol}")
        ticks = mt5.copy_ticks_range(symbol, start, end, mt5.COPY_TICKS_INFO)
        if ticks is None or len(ticks) == 0:
            raise SystemExit(f"no ticks loaded: {mt5.last_error()}")
        bid = np.asarray([float(t["bid"]) for t in ticks], dtype=np.float64)
        ask = np.asarray([float(t["ask"]) for t in ticks], dtype=np.float64)
        ts_ns = np.asarray(
            [int(t["time_msc"]) * 1_000_000 if int(t["time_msc"]) > 0 else int(t["time"]) * 1_000_000_000 for t in ticks],
            dtype=np.int64,
        )
        ps = point_size(mt5, symbol)
    finally:
        mt5.shutdown()

    candles = build_bid_candles(bid, ask, ts_ns, tf)
    model, ns, _ = load_torch_model(Path(args.model))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds = predict_model(model, ns, candles, ps, device)
    raw = preds[:, 0].astype(np.float64)
    trade = raw.copy()
    if signal_mode == "invert":
        trade = 1.0 - trade
    if prob_ma <= 1:
        ma = trade
    else:
        ma = np.full_like(trade, np.nan)
        for i in range(len(trade)):
            lo = max(0, i - prob_ma + 1)
            vals = trade[lo:i + 1]
            vals = vals[np.isfinite(vals)]
            if len(vals):
                ma[i] = float(vals.mean())

    replay = {}
    times_ns = candles.times.astype("int64")
    for i, ns_time in enumerate(times_ns):
        replay[datetime.fromtimestamp(int(ns_time) / 1_000_000_000, tz=timezone.utc).isoformat()] = i

    out_path = Path(args.out) if args.out else Path(args.log).with_name(Path(args.log).stem + "_compare.csv")
    window_diff_path = out_path.with_name(out_path.stem + "_windowdiff.csv")
    fields = [
        "bucket_utc",
        "live_open", "replay_open", "open_diff_points",
        "live_high", "replay_high", "high_diff_points",
        "live_low", "replay_low", "low_diff_points",
        "live_close", "replay_close", "close_diff_points",
        "live_spread_points", "replay_spread_points", "spread_diff_points",
        "live_raw", "replay_raw", "raw_diff", "live_trade", "replay_trade",
        "live_ma", "replay_ma", "live_signal", "replay_signal",
        "live_window_hash", "replay_window_hash", "window_match", "match",
    ]
    diff_fields = [
        "bucket_utc", "offset", "live_time", "replay_time",
        "open_diff_points", "high_diff_points", "low_diff_points", "close_diff_points", "spread_diff_points",
    ]
    mismatches = 0
    compared = 0
    max_abs_ohlc_diff = 0.0
    max_abs_spread_diff = 0.0
    max_abs_raw_diff = 0.0
    window = int(getattr(ns, "window", 64))
    with out_path.open("w", newline="", encoding="utf-8") as f, window_diff_path.open("w", newline="", encoding="utf-8") as fd:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        wd = csv.DictWriter(fd, fieldnames=diff_fields)
        wd.writeheader()
        for r in rows:
            bucket = parse_dt(r["bucket_utc"]).isoformat()
            i = replay.get(bucket)
            if i is None:
                continue
            replay_sig = signal_name(signal_from_prob(float(ma[i]), upper, lower))
            live_sig = r["signal"]
            match = live_sig == replay_sig
            compared += 1
            if not match:
                mismatches += 1
            live_open = float(r["open"])
            live_high = float(r["high"])
            live_low = float(r["low"])
            live_close = float(r["close"])
            live_spread = float(r["avg_spread_points"])
            replay_open = float(candles.ohlc[i, 0])
            replay_high = float(candles.ohlc[i, 1])
            replay_low = float(candles.ohlc[i, 2])
            replay_close = float(candles.ohlc[i, 3])
            replay_spread = float(candles.spread[i]) / ps
            ohlc_diffs = [
                (live_open - replay_open) / ps,
                (live_high - replay_high) / ps,
                (live_low - replay_low) / ps,
                (live_close - replay_close) / ps,
            ]
            spread_diff = live_spread - replay_spread
            raw_diff = float(r["raw_p"]) - float(raw[i])
            live_hash = r.get("window_hash", "")
            replay_hash = window_hash(candles, i, window)
            window_match = bool(live_hash and live_hash == replay_hash)
            max_abs_ohlc_diff = max(max_abs_ohlc_diff, max(abs(x) for x in ohlc_diffs))
            max_abs_spread_diff = max(max_abs_spread_diff, abs(spread_diff))
            max_abs_raw_diff = max(max_abs_raw_diff, abs(raw_diff))
            if live_hash and live_hash != replay_hash and i - window + 1 >= 0:
                for off in range(-window + 1, 1):
                    j = i + off
                    lw = live_windows.get((bucket, off))
                    replay_time = datetime.fromtimestamp(int(candles.times.astype("int64")[j]) / 1_000_000_000, tz=timezone.utc).isoformat()
                    if lw:
                        lo = float(lw["open"])
                        lh = float(lw["high"])
                        ll = float(lw["low"])
                        lc = float(lw["close"])
                        ls = float(lw["spread_points"])
                        ro = float(candles.ohlc[j, 0])
                        rh = float(candles.ohlc[j, 1])
                        rl = float(candles.ohlc[j, 2])
                        rc = float(candles.ohlc[j, 3])
                        rs = float(candles.spread[j]) / ps
                        live_time = lw["time_utc"]
                        od = f"{(lo - ro) / ps:.3f}"
                        hd = f"{(lh - rh) / ps:.3f}"
                        ld = f"{(ll - rl) / ps:.3f}"
                        cd = f"{(lc - rc) / ps:.3f}"
                        sd = f"{ls - rs:.3f}"
                    else:
                        live_time = ""
                        od = hd = ld = cd = sd = ""
                    wd.writerow({
                        "bucket_utc": bucket,
                        "offset": off,
                        "live_time": live_time,
                        "replay_time": replay_time,
                        "open_diff_points": od,
                        "high_diff_points": hd,
                        "low_diff_points": ld,
                        "close_diff_points": cd,
                        "spread_diff_points": sd,
                    })
            w.writerow({
                "bucket_utc": bucket,
                "live_open": f"{live_open:.10f}",
                "replay_open": f"{replay_open:.10f}",
                "open_diff_points": f"{ohlc_diffs[0]:.3f}",
                "live_high": f"{live_high:.10f}",
                "replay_high": f"{replay_high:.10f}",
                "high_diff_points": f"{ohlc_diffs[1]:.3f}",
                "live_low": f"{live_low:.10f}",
                "replay_low": f"{replay_low:.10f}",
                "low_diff_points": f"{ohlc_diffs[2]:.3f}",
                "live_close": f"{live_close:.10f}",
                "replay_close": f"{replay_close:.10f}",
                "close_diff_points": f"{ohlc_diffs[3]:.3f}",
                "live_spread_points": f"{live_spread:.3f}",
                "replay_spread_points": f"{replay_spread:.3f}",
                "spread_diff_points": f"{spread_diff:.3f}",
                "live_raw": r["raw_p"],
                "replay_raw": f"{float(raw[i]):.9f}",
                "raw_diff": f"{raw_diff:.9f}",
                "live_trade": r["trade_p"],
                "replay_trade": f"{float(trade[i]):.9f}",
                "live_ma": r["p_ma"],
                "replay_ma": f"{float(ma[i]):.9f}",
                "live_signal": live_sig,
                "replay_signal": replay_sig,
                "live_window_hash": live_hash,
                "replay_window_hash": replay_hash,
                "window_match": int(window_match),
                "match": int(match),
            })

    print(f"compared={compared} mismatches={mismatches} wrote={out_path}")
    print(f"window_diff={window_diff_path}")
    if compared:
        print(f"mismatch_rate={mismatches / compared * 100:.1f}%")
        print(
            f"max_abs_ohlc_diff={max_abs_ohlc_diff:.3f}pt "
            f"max_abs_spread_diff={max_abs_spread_diff:.3f}pt "
            f"max_abs_raw_diff={max_abs_raw_diff:.9f}"
        )
    with out_path.open(newline="", encoding="utf-8") as f:
        sample = list(csv.DictReader(f))[:20]
    for r in sample:
        print(
            f"{r['bucket_utc'][5:16]} "
            f"O/H/L/Cd={r['open_diff_points']}/{r['high_diff_points']}/{r['low_diff_points']}/{r['close_diff_points']}pt "
            f"sprd={r['spread_diff_points']}pt "
            f"raw {float(r['live_raw']):.3f}/{float(r['replay_raw']):.3f} "
            f"sig {r['live_signal']}/{r['replay_signal']} match={r['match']}"
        )


if __name__ == "__main__":
    main()
