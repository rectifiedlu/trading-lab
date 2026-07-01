"""Summarize per-trade CSVs from strategy backtests."""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def _session(hour: int) -> str:
    if 0 <= hour < 7:
        return "asia"
    if 7 <= hour < 12:
        return "london"
    if 12 <= hour < 17:
        return "ny_overlap"
    if 17 <= hour < 22:
        return "ny_late"
    return "rollover"


def write_equity_html(df: pd.DataFrame, equity: pd.Series, drawdown: pd.Series, out: str) -> None:
    x = df["time"].astype(str).tolist() if df["time"].notna().any() else [str(i + 1) for i in range(len(df))]
    pnl = df["pnl"].cumsum().tolist()
    eq = equity.tolist()
    dd = drawdown.tolist()
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Forex Trade Equity Stats</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ background:#101214; color:#e7e1d4; font-family:ui-monospace,Consolas,monospace; margin:20px; }}
    #chart {{ width:100%; height:760px; }}
  </style>
</head>
<body>
  <h3>Equity / PnL / Drawdown</h3>
  <div id="chart"></div>
  <script>
    const x = {x!r};
    const pnl = {pnl!r};
    const equity = {eq!r};
    const dd = {dd!r};
    Plotly.newPlot('chart', [
      {{x, y:pnl, name:'Cumulative PnL', type:'scatter', mode:'lines', line:{{color:'#2dd4bf', width:2}}}},
      {{x, y:equity, name:'Equity', type:'scatter', mode:'lines', line:{{color:'#a3e635', width:1.5}}, yaxis:'y2'}},
      {{x, y:dd, name:'Drawdown', type:'scatter', mode:'lines', fill:'tozeroy', line:{{color:'#fb7185', width:1.5}}, yaxis:'y3'}}
    ], {{
      paper_bgcolor:'#101214',
      plot_bgcolor:'#101214',
      font:{{color:'#e7e1d4'}},
      hovermode:'x unified',
      grid:{{rows:3, columns:1, pattern:'independent'}},
      yaxis:{{title:'PnL'}},
      yaxis2:{{title:'Equity'}},
      yaxis3:{{title:'Drawdown'}},
      margin:{{l:60,r:30,t:30,b:80}}
    }});
  </script>
</body>
</html>"""
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[stats] wrote graph {out}", flush=True)


def summarize(df: pd.DataFrame, start_balance: float, html_out: str | None) -> None:
    if df.empty:
        print("[stats] no trades")
        return
    if "pnl" not in df.columns:
        raise SystemExit("trade CSV must have pnl column")
    df = df.copy()
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)
    if "equity" in df.columns:
        eq = pd.to_numeric(df["equity"], errors="coerce")
        if eq.notna().any():
            # Backtest trade-log pnl excludes entry commission, but equity includes
            # all fees. Use equity deltas so stats match reported strategy totals.
            prev_eq = eq.shift(1)
            prev_eq.iloc[0] = start_balance
            df["pnl"] = (eq - prev_eq).fillna(df["pnl"])
    if "entry_time" in df.columns:
        df["time"] = pd.to_datetime(df["entry_time"], utc=True, format="mixed")
    elif "timestamp" in df.columns:
        df["time"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
    elif "date" in df.columns:
        df["time"] = pd.to_datetime(df["date"], utc=True, format="mixed")
    else:
        df["time"] = pd.NaT

    equity = start_balance + df["pnl"].cumsum()
    peak = equity.cummax()
    dd = peak - equity
    wins = int((df["pnl"] > 0).sum())
    losses = int((df["pnl"] < 0).sum())
    gross_win = float(df.loc[df["pnl"] > 0, "pnl"].sum())
    gross_loss = float(-df.loc[df["pnl"] < 0, "pnl"].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)

    print(
        f"[stats] trades={len(df)} total=${df['pnl'].sum():+.4f} "
        f"wins={wins} losses={losses} wr={wins / len(df) * 100:.1f}% "
        f"pf={pf:.3f} max_cum_dd=${float(dd.max()):.4f} "
        f"avg_cum_dd=${float(dd.mean()):.4f}",
        flush=True,
    )
    print(
        f"[stats] avg=${df['pnl'].mean():+.4f} median=${df['pnl'].median():+.4f} "
        f"best=${df['pnl'].max():+.4f} worst=${df['pnl'].min():+.4f}",
        flush=True,
    )

    if df["time"].notna().any():
        df["day"] = df["time"].dt.strftime("%Y-%m-%d")
        df["session"] = df["time"].dt.hour.map(_session)
        daily = df.groupby("day")["pnl"].agg(["count", "sum", "mean"])
        daily["wins"] = df.groupby("day")["pnl"].apply(lambda x: int((x > 0).sum()))
        daily["losses"] = df.groupby("day")["pnl"].apply(lambda x: int((x < 0).sum()))
        print("\n[stats] daily:")
        print(daily.sort_index().to_string(float_format=lambda x: f"{x:+.4f}"))
        print("\n[stats] worst days:")
        print(daily.sort_values("sum").head(5).to_string(float_format=lambda x: f"{x:+.4f}"))
        sessions = df.groupby("session")["pnl"].agg(["count", "sum", "mean"])
        sessions["wins"] = df.groupby("session")["pnl"].apply(lambda x: int((x > 0).sum()))
        sessions["losses"] = df.groupby("session")["pnl"].apply(lambda x: int((x < 0).sum()))
        print("\n[stats] sessions:")
        print(sessions.sort_values("sum", ascending=False).to_string(float_format=lambda x: f"{x:+.4f}"))

    if html_out:
        write_equity_html(df, equity, dd, html_out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--start-balance", type=float, default=50.0)
    ap.add_argument("--html", default=None, help="write interactive equity/drawdown HTML")
    args = ap.parse_args()
    if not os.path.exists(args.csv):
        raise SystemExit(f"missing CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    summarize(df, args.start_balance, args.html)


if __name__ == "__main__":
    main()
