"""Visualize MACD sim trades against XAUUSD ticks/candles."""

from __future__ import annotations

import argparse
import os

import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forex_backtest import FOREX_DIR
from forex_macd_backtest import ema
from forex_strategy_common import (
    build_parser,
    closed_candle_series,
    default_point_size,
    load_market,
    parse_num_list,
)


def parse_params(s: str) -> dict[str, str]:
    out = {}
    for part in str(s).split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def main() -> None:
    ap = build_parser("MACD trade visualizer", "unused.csv")
    ap.add_argument("--params-csv", default=os.path.join(FOREX_DIR, "forex_macd_results.csv"))
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--trades-csv", default=os.path.join(FOREX_DIR, "forex_macd_sim_trades.csv"))
    ap.add_argument("--save", default=os.path.join(FOREX_DIR, "forex_macd_visual.png"))
    args = ap.parse_args()

    params_df = pd.read_csv(args.params_csv)
    src = params_df.iloc[int(args.row)]
    p = parse_params(src["params"])
    mode = p.get("mode", "candle")
    fast = int(p["fast"])
    slow = int(p["slow"])
    macd_ema = int(p["macd_ema"])
    warmup = int(p.get("warmup", slow + macd_ema))
    deadband = float(p.get("deadband", 0.1))
    timeframe = str(src["timeframe"])

    ticks, _ = load_market(args)
    pair, g = next(iter(ticks.groupby("pair", sort=False)))
    g = g.sort_values("timestamp").reset_index(drop=True)
    bid = g["bid"].to_numpy(np.float64)
    ask = g["ask"].to_numpy(np.float64)
    # MT5 native XAUUSD candles match bid OHLC, not mid/ask.
    mid = bid
    ts = pd.to_datetime(g["timestamp"], utc=True)
    ts_ns = ts.astype("int64").to_numpy()

    close, close_tick_idx = closed_candle_series(mid, ts_ns, timeframe)
    raw = ema(close, fast) - ema(close, slow)
    line = ema(raw, macd_ema)
    if warmup > 0:
        line[:warmup] = np.nan
    candle_ts = ts.iloc[close_tick_idx].to_numpy()

    trades = pd.read_csv(args.trades_csv) if os.path.exists(args.trades_csv) else pd.DataFrame()
    fig, (ax_price, ax_macd) = plt.subplots(
        2, 1, figsize=(16, 9), sharex=False, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_price.plot(ts, mid, color="#d8d8d8", linewidth=0.8, label="mid price")
    if not trades.empty:
        for _, t in trades.iterrows():
            ei = int(t["entry_i"])
            xi = int(t["exit_i"])
            side = str(t["side"])
            pnl = float(t["pnl"])
            reason = str(t["reason"])
            entry_color = "#1f77ff" if side == "long" else "#ff9f1a"
            exit_color = {
                "take_profit": "#2ecc71",
                "signal_exit": "#9b59b6",
                "stop_loss": "#e74c3c",
                "liquidation": "#000000",
            }.get(reason, "#7f8c8d")
            path_color = "#2ecc71" if pnl >= 0 else "#e74c3c"
            marker = "^" if side == "long" else "v"
            ax_price.scatter(ts.iloc[ei], float(t["entry_px"]), marker=marker, s=60,
                             color=entry_color, edgecolor="black", linewidth=0.5, zorder=5)
            ax_price.scatter(ts.iloc[xi], float(t["exit_px"]), marker="x", s=65,
                             color=exit_color, linewidth=1.8, zorder=6)
            ax_price.plot([ts.iloc[ei], ts.iloc[xi]], [float(t["entry_px"]), float(t["exit_px"])],
                          color=path_color, linewidth=0.9, alpha=0.65)
            for idx, style, color in [
                (ei, "--", entry_color),
                (xi, ":", exit_color),
            ]:
                ax_price.axvline(ts.iloc[idx], color=color, linestyle=style, linewidth=0.8, alpha=0.55)
                ax_macd.axvline(ts.iloc[idx], color=color, linestyle=style, linewidth=0.8, alpha=0.55)

    ax_price.set_title(
        f"{pair} MACD {mode} {timeframe} fast={fast} slow={slow} macd_ema={macd_ema}"
    )
    ax_price.grid(True, alpha=0.2)
    legend_items = [
        Line2D([0], [0], color="#d8d8d8", lw=1, label="mid price"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#1f77ff",
               markeredgecolor="black", label="long entry"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#ff9f1a",
               markeredgecolor="black", label="short entry"),
        Line2D([0], [0], marker="x", color="#2ecc71", lw=0, label="TP exit"),
        Line2D([0], [0], marker="x", color="#9b59b6", lw=0, label="signal exit"),
        Line2D([0], [0], marker="x", color="#e74c3c", lw=0, label="SL exit"),
        Line2D([0], [0], color="#2ecc71", lw=1, label="winning trade path"),
        Line2D([0], [0], color="#e74c3c", lw=1, label="losing trade path"),
    ]
    ax_price.legend(handles=legend_items, loc="upper left", fontsize=8)

    ax_macd.axhline(0, color="#888", linewidth=0.8)
    ax_macd.axhline(deadband, color="#2ecc71", linewidth=0.7, linestyle="--", alpha=0.6)
    ax_macd.axhline(-deadband, color="#e74c3c", linewidth=0.7, linestyle="--", alpha=0.6)
    ax_macd.plot(
        candle_ts, line, color="#f1c40f", linewidth=1.2,
        drawstyle="steps-post" if mode == "candle" else "default",
        label="MACD line/state",
    )
    ax_macd.fill_between(candle_ts, line, 0, where=line >= deadband, color="#2ecc71", alpha=0.25,
                         label="long region")
    ax_macd.fill_between(candle_ts, line, 0, where=line <= -deadband, color="#e74c3c", alpha=0.25,
                         label="short region")
    if not trades.empty:
        for _, t in trades.iterrows():
            ei = int(t["entry_i"])
            xi = int(t["exit_i"])
            side = str(t["side"])
            entry_color = "#1f77ff" if side == "long" else "#ff9f1a"
            entry_marker = "^" if side == "long" else "v"
            exit_color = {
                "take_profit": "#2ecc71",
                "signal_exit": "#9b59b6",
                "stop_loss": "#e74c3c",
                "liquidation": "#000000",
            }.get(str(t["reason"]), "#7f8c8d")
            ax_macd.scatter(ts.iloc[ei], 0, marker=entry_marker, s=55,
                            color=entry_color, edgecolor="black", linewidth=0.5, zorder=6)
            ax_macd.scatter(ts.iloc[xi], 0, marker="x", s=55,
                            color=exit_color, linewidth=1.5, zorder=6)
    ax_macd.grid(True, alpha=0.2)
    ax_macd.legend(loc="upper left")
    ax_macd.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

    fig.autofmt_xdate()
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    fig.savefig(args.save, dpi=140)
    print(f"[visual] wrote {args.save}", flush=True)


if __name__ == "__main__":
    main()
