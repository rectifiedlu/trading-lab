from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


MONEY_COLS = {
    "total", "realised", "open_unrealized", "open", "max_drawdown", "acct_dd",
    "cum_max_drawdown", "cum_dd", "trade_max_drawdown", "tr_max",
    "trade_avg_drawdown", "worst_trade_pnl", "worst_loss", "median_loss",
    "med_loss", "avg_day", "median_day", "med_day",
}
INT_COLS = {
    "trades", "tr", "wins", "losses", "long_trades", "short_trades",
    "stop_losses", "stops", "signal_exits", "sig", "liquidations", "liq",
    "account_dead", "dead", "active_days",
}
FLOAT_COLS = {"win_rate", "wr", "profit_factor", "pf", "pnl_dd", "pnl/dd", "open_bps"}


ALIASES = {
    "tr": "trades",
    "wr": "win_rate",
    "pf": "profit_factor",
    "acct_dd": "max_drawdown",
    "tr_max": "trade_max_drawdown",
    "cum_dd": "cum_max_drawdown",
    "worst_loss": "worst_trade_pnl",
    "med_loss": "median_loss",
    "med_day": "median_day",
    "sig": "signal_exits",
    "stops": "stop_losses",
    "liq": "liquidations",
    "dead": "account_dead",
}


DEFAULT_COLS = [
    "pair", "strategy", "timeframe", "tp_points", "sl_points",
    "total", "trades", "win_rate", "profit_factor",
    "max_drawdown", "cum_max_drawdown", "trade_max_drawdown",
    "worst_trade_pnl", "median_loss", "median_day", "params",
]


def canonical(col: str, df: pd.DataFrame) -> str:
    c = col.strip()
    if c in df.columns:
        return c
    mapped = ALIASES.get(c, c)
    if mapped in df.columns:
        return mapped
    raise SystemExit(f"unknown column: {col}")


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def fmt_cell(col: str, value) -> str:
    if pd.isna(value):
        return ""
    label = col
    if label in {"tp_points", "sl_points", "point_size"}:
        try:
            return f"{float(value):g}"
        except Exception:
            return str(value)
    if label in MONEY_COLS:
        try:
            return f"${float(value):+.4f}" if label in {"total", "realised", "open_unrealized", "open", "avg_day", "median_day", "med_day", "worst_trade_pnl", "worst_loss", "median_loss", "med_loss"} else f"${float(value):.2f}"
        except Exception:
            return str(value)
    if label in INT_COLS:
        try:
            return str(int(float(value)))
        except Exception:
            return str(value)
    if label in FLOAT_COLS:
        try:
            v = float(value)
            if label in {"win_rate", "wr"}:
                return f"{v:.1f}"
            return f"{v:.4g}"
        except Exception:
            return str(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.6g}"
    return str(value)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "pnl_dd" not in out.columns and "total" in out.columns and "max_drawdown" in out.columns:
        dd = pd.to_numeric(out["max_drawdown"], errors="coerce").replace(0, pd.NA)
        out["pnl_dd"] = pd.to_numeric(out["total"], errors="coerce") / dd
    return out


def apply_filter(df: pd.DataFrame, expr: str) -> pd.DataFrame:
    ops = ["<=", ">=", "!=", "==", "<", ">"]
    for op in ops:
        if op in expr:
            left, right = expr.split(op, 1)
            col = canonical(left.strip(), df)
            raw = right.strip()
            series = df[col]
            num = pd.to_numeric(series, errors="coerce")
            try:
                val = float(raw)
                target = num
            except ValueError:
                val = raw
                target = series.astype(str)
            if op == "<=":
                return df[target <= val]
            if op == ">=":
                return df[target >= val]
            if op == "<":
                return df[target < val]
            if op == ">":
                return df[target > val]
            if op == "==":
                return df[target == val]
            if op == "!=":
                return df[target != val]
    raise SystemExit(f"bad filter expression: {expr}")


def print_table(df: pd.DataFrame, cols: list[str], title: str) -> None:
    rows = []
    headers = []
    for c in cols:
        if c not in df.columns:
            continue
        headers.append(c)
    for _, row in df.iterrows():
        rows.append([fmt_cell(c, row[c]) for c in headers])
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    print()
    print(f"  {title}")
    if not headers:
        print("  no columns")
        return
    print("  " + " ".join(headers[i].rjust(widths[i]) for i in range(len(headers))))
    print("  " + "-" * (sum(widths) + len(widths) - 1))
    for row in rows:
        print("  " + " ".join(row[i].rjust(widths[i]) for i in range(len(headers))))


def main() -> None:
    ap = argparse.ArgumentParser(description="Pretty-print strategy CSV results.")
    ap.add_argument("csv", help="CSV path")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--sort", default="", help="column to sort by; aliases: pnl/dd, acct_dd, tr_max, cum_dd")
    ap.add_argument("--asc", action="store_true", help="sort ascending")
    ap.add_argument("--cols", default="", help="comma-separated columns to show")
    ap.add_argument("--filter", action="append", default=[], help="filter expression, e.g. 'trades>=50' or 'acct_dd<=40'")
    ap.add_argument("--min-trades", type=float, default=None)
    ap.add_argument("--max-dd", type=float, default=None, help="alias for acct_dd<=N")
    ap.add_argument("--max-cum-dd", type=float, default=None)
    ap.add_argument("--min-total", type=float, default=None)
    ap.add_argument("--pair", default="")
    ap.add_argument("--strategy", default="")
    args = ap.parse_args()

    path = Path(args.csv)
    df = pd.read_csv(path)
    df = add_derived(df)

    if args.pair:
        df = df[df["pair"].astype(str).str.upper() == args.pair.upper()]
    if args.strategy:
        df = df[df["strategy"].astype(str) == args.strategy]
    if args.min_trades is not None and "trades" in df.columns:
        df = df[pd.to_numeric(df["trades"], errors="coerce") >= args.min_trades]
    if args.max_dd is not None and "max_drawdown" in df.columns:
        df = df[pd.to_numeric(df["max_drawdown"], errors="coerce") <= args.max_dd]
    if args.max_cum_dd is not None and "cum_max_drawdown" in df.columns:
        df = df[pd.to_numeric(df["cum_max_drawdown"], errors="coerce") <= args.max_cum_dd]
    if args.min_total is not None and "total" in df.columns:
        df = df[pd.to_numeric(df["total"], errors="coerce") >= args.min_total]
    for expr in args.filter:
        df = apply_filter(df, expr)

    if args.sort:
        sort_col = canonical(args.sort, df)
        df = df.sort_values(sort_col, ascending=args.asc)

    cols = [canonical(c, df) for c in split_csv(args.cols)] if args.cols else [c for c in DEFAULT_COLS if c in df.columns]
    title = f"{path.name} rows={len(df)} top={args.top}"
    print_table(df.head(args.top), cols, title)


if __name__ == "__main__":
    main()
