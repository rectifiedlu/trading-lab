from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_FILES = [
    Path("data/forex/unified_xauusd_21d_rerun.csv"),
    Path("data/forex/unified_fx_21d_rerun.csv"),
]

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
    "ratio": "realised_cumdd_ratio",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank unified strategy rows by realised PnL / realised cumulative drawdown."
    )
    parser.add_argument(
        "--files",
        nargs="*",
        type=Path,
        default=DEFAULT_FILES,
        help="CSV files to combine. Defaults to the 21d XAUUSD and FX reruns.",
    )
    parser.add_argument("--top", type=int, default=75, help="Rows to print.")
    parser.add_argument("--min-realised", type=float, default=0.0)
    parser.add_argument("--min-total", type=float, default=None)
    parser.add_argument("--min-trades", type=int, default=1)
    parser.add_argument("--max-dd", type=float, default=None, help="Alias for acct_dd/max_drawdown <= N.")
    parser.add_argument("--max-cum-dd", type=float, default=None)
    parser.add_argument("--max-tr-max", type=float, default=None)
    parser.add_argument("--min-med-day", type=float, default=None)
    parser.add_argument("--pair", default="", help="Filter to one pair, e.g. XAUUSD or USDJPY.")
    parser.add_argument("--strategy", default="", help="Filter to one strategy, e.g. bb_rsi.")
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        help="Filter expression, e.g. 'trades>=50', 'acct_dd<=40', 'params==foo'. Can repeat.",
    )
    parser.add_argument(
        "--sort",
        default="realised_cumdd_ratio",
        help="Sort column/alias. Default: realised_cumdd_ratio. Aliases from tableprint work.",
    )
    parser.add_argument("--asc", action="store_true", help="Sort ascending.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/forex/analysis/top_realised_cumdd.csv"),
        help="Where to write the ranked CSV.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["source_file"] = path.name
    return df


def to_number(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def canonical(col: str, df: pd.DataFrame) -> str:
    c = col.strip()
    if c in df.columns:
        return c
    mapped = ALIASES.get(c, c)
    if mapped in df.columns:
        return mapped
    raise SystemExit(f"unknown column: {col}")


def apply_filter(df: pd.DataFrame, expr: str) -> pd.DataFrame:
    ops = ["<=", ">=", "!=", "==", "<", ">"]
    for op in ops:
        if op not in expr:
            continue
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


def fmt_money(value: float, signed: bool = True) -> str:
    if pd.isna(value):
        return "na"
    return f"${value:+.2f}" if signed else f"${value:.2f}"


def fmt_row(i: int, row: pd.Series) -> str:
    return (
        f"{i:>3}. {row['pair']:<6} {row['strategy']:<10} {str(row['timeframe']):>3} "
        f"tp={row['tp_points']:g} sl={row['sl_points']:g} "
        f"real={fmt_money(row['realised'])} total={fmt_money(row['total'])} "
        f"tr={int(row['trades']):>4} wr={row['win_rate']:>5.1f}% pf={row['profit_factor']:>6.2f} "
        f"cum_dd={fmt_money(row['cum_max_drawdown'], signed=False)} "
        f"acct_dd={fmt_money(row['max_drawdown'], signed=False)} "
        f"tr_max={fmt_money(row['trade_max_drawdown'], signed=False)} "
        f"worst={fmt_money(row['worst_trade_pnl'])} "
        f"med_day={fmt_money(row['median_day'])} "
        f"ratio={row['realised_cumdd_ratio']:>7.2f} | {row['params']}"
    )


def main() -> None:
    args = parse_args()
    frames = []
    for path in args.files:
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(load_csv(path))

    df = pd.concat(frames, ignore_index=True)
    numeric_cols = [
        "tp_points",
        "sl_points",
        "realised",
        "open_unrealized",
        "total",
        "trades",
        "win_rate",
        "profit_factor",
        "max_drawdown",
        "avg_day",
        "median_day",
        "trade_max_drawdown",
        "cum_max_drawdown",
        "worst_trade_pnl",
        "median_loss",
    ]
    to_number(df, numeric_cols)
    # If realised DD is zero/near-zero, treat the ratio as extremely strong but finite.
    # This prevents division-by-zero from hiding rows with no closed-equity drawdown.
    # denom = df["cum_max_drawdown"].clip(lower=0.01)
    # df["realised_cumdd_ratio"] = df["realised"] / denom

    denom = df["max_drawdown"].clip(lower=0.01) 
    df["realised_cumdd_ratio"] = df["realised"] / denom

    ranked = df[(df["realised"] >= args.min_realised) & (df["trades"] >= args.min_trades)]
    if args.min_total is not None:
        ranked = ranked[ranked["total"] >= args.min_total]
    if args.max_dd is not None:
        ranked = ranked[ranked["max_drawdown"] <= args.max_dd]
    if args.max_cum_dd is not None:
        ranked = ranked[ranked["cum_max_drawdown"] <= args.max_cum_dd]
    if args.max_tr_max is not None:
        ranked = ranked[ranked["trade_max_drawdown"] <= args.max_tr_max]
    if args.min_med_day is not None:
        ranked = ranked[ranked["median_day"] >= args.min_med_day]
    if args.pair:
        ranked = ranked[ranked["pair"].astype(str).str.upper() == args.pair.upper()]
    if args.strategy:
        ranked = ranked[ranked["strategy"].astype(str) == args.strategy]
    for expr in args.filter:
        ranked = apply_filter(ranked, expr)

    sort_col = canonical(args.sort, ranked)
    ranked = ranked[ranked["cum_max_drawdown"].notna()].sort_values(
        [sort_col, "realised", "median_day"],
        ascending=[args.asc, False, False],
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.out, index=False)

    print(f"[rank] combined_rows={len(df):,} ranked_rows={len(ranked):,}")
    print(f"[rank] wrote {args.out}")
    print()
    print(f"Top {min(args.top, len(ranked))} by {sort_col}")
    print("-" * 180)
    for i, (_, row) in enumerate(ranked.head(args.top).iterrows(), start=1):
        print(fmt_row(i, row))


if __name__ == "__main__":
    main()
