"""Replay selected MACD params on a separate tick window."""

from __future__ import annotations

import argparse
import csv
import os
import re

import numpy as np
import pandas as pd

from forex_backtest import FOREX_DIR
from forex_macd_backtest import ema
from forex_strategy_common import (
    DEFAULT_AMOUNT,
    DEFAULT_SL_POINTS,
    DEFAULT_TP_POINTS,
    build_parser,
    candle_state_to_ticks,
    closed_candle_series,
    day_ids_from_timestamps,
    default_point_size,
    live_candles,
    load_market,
    parse_num_list,
    simulate_triggers,
    write_results,
)


def parse_params(s: str) -> dict[str, str]:
    out = {}
    for part in str(s).split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def simulate_one(args, pair: str, g: pd.DataFrame, row: dict):
    bid = g["bid"].to_numpy(np.float64)
    ask = g["ask"].to_numpy(np.float64)
    # MT5 native XAUUSD candles match bid OHLC, not mid/ask.
    mid = bid
    ts = g["timestamp"].astype("int64").to_numpy()
    day_id, max_days = day_ids_from_timestamps(ts)
    point_size = args.point_size or default_point_size(pair)

    mode = row["mode"]
    fast = int(row["fast"])
    slow = int(row["slow"])
    macd_ema = int(row["macd_ema"])
    deadband = float(row.get("deadband", 0.1))
    timeframe = row["timeframe"]
    tp = float(row["tp_points"])
    sl = float(row["sl_points"])
    warmup = int(row.get("warmup", slow + macd_ema))
    reverse = bool(int(row.get("reverse", 0)))
    if args.reverse_on_flip is not None:
        reverse = bool(int(args.reverse_on_flip))

    if mode == "candle":
        close, close_tick_idx = closed_candle_series(mid, ts, timeframe)
        raw_macd = ema(close, fast) - ema(close, slow)
        line = ema(raw_macd, macd_ema)
        if warmup > 0:
            line[:warmup] = 0.0
        candle_state = np.where(line >= deadband, 1.0, np.where(line <= -deadband, -1.0, 0.0))
        tick_state = candle_state_to_ticks(len(bid), close_tick_idx, candle_state)
        prev_tick_state = np.roll(tick_state, 1)
        prev_tick_state[0] = 0.0
        long_trigger = (tick_state == 1) & (prev_tick_state != 1)
        short_trigger = (tick_state == -1) & (prev_tick_state != -1)
        long_exit = tick_state <= 0
        short_exit = tick_state >= 0
    else:
        _, _, _, close, _ = live_candles(mid, ts, timeframe)
        raw_macd = ema(close, fast) - ema(close, slow)
        line = ema(raw_macd, macd_ema)
        prev = np.roll(line, 1)
        prev[0] = np.nan
        long_trigger = (prev < deadband) & (line >= deadband)
        short_trigger = (prev > -deadband) & (line <= -deadband)
        if warmup > 0:
            long_trigger[:warmup] = False
            short_trigger[:warmup] = False
        long_exit = line <= -deadband
        short_exit = line >= deadband

    params = (
        f"mode={mode};fast={fast};slow={slow};macd_ema={macd_ema};"
        f"warmup={warmup};deadband={deadband};reverse={int(reverse)}"
    )
    return simulate_triggers(
        pair, "macd", params, timeframe, bid, ask, long_trigger, short_trigger,
        long_exit, short_exit, tp, sl, point_size, args.amount, args.compound,
        args.leverage, args.commission_per_million, args.side,
        reverse_on_flip=reverse,
        day_id=day_id,
        max_days=max_days,
        return_trades=bool(args.trades_out),
    )


def main() -> None:
    ap = build_parser("MACD selected-param simulator", "forex_macd_sim_results.csv")
    ap.add_argument("--params-csv", default=os.path.join(FOREX_DIR, "forex_macd_results.csv"))
    ap.add_argument("--row", type=int, default=0, help="0-based row from params CSV")
    ap.add_argument("--reverse-on-flip", choices=["0", "1"], default=None,
                    help="override selected row reverse param; default reads reverse= from params")
    ap.add_argument("--trades-out", default=os.path.join(FOREX_DIR, "forex_macd_sim_trades.csv"))
    args = ap.parse_args()

    params_df = pd.read_csv(args.params_csv)
    if params_df.empty:
        raise SystemExit("params CSV is empty")
    src = params_df.iloc[int(args.row)]
    p = parse_params(src["params"])
    row = {
        "mode": p.get("mode", "candle"),
        "fast": p["fast"],
        "slow": p["slow"],
        "macd_ema": p["macd_ema"],
        "warmup": p.get("warmup", ""),
        "deadband": p.get("deadband", "0.1"),
        "reverse": p.get("reverse", "0"),
        "timeframe": src["timeframe"],
        "tp_points": src["tp_points"],
        "sl_points": src["sl_points"],
    }

    ticks, _ = load_market(args)
    results = []
    trades = []
    for pair, g in ticks.groupby("pair", sort=False):
        print(f"[macd-sim] {pair} params={row}", flush=True)
        sim = simulate_one(args, pair, g.sort_values("timestamp").reset_index(drop=True), row)
        if args.trades_out:
            res, trade_logs = sim
            results.append(res)
            trades.extend(trade_logs)
        else:
            results.append(sim)

    write_results(args.out, results, args.top, args.sort_by)
    if args.trades_out:
        with open(args.trades_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "pair", "strategy", "params", "timeframe", "side", "entry_i",
                "exit_i", "entry_px", "exit_px", "pnl", "reason", "equity",
            ])
            for t in trades:
                w.writerow([
                    t.pair, t.strategy, t.params, t.timeframe, t.side,
                    t.entry_i, t.exit_i, round(t.entry_px, 6), round(t.exit_px, 6),
                    round(t.pnl, 6), t.reason, round(t.equity, 6),
                ])
        print(f"[macd-sim] wrote trades {args.trades_out}", flush=True)
    print(f"[macd-sim] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
