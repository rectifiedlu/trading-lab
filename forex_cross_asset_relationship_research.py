"""Non-trading AUDUSD/XAUUSD relationship discovery.

Stage one measures lead/lag and conditional responses directly from aligned
prices.  Stage two uses symbolic regression on both markets' lagged price
movements, reporting only held-out prediction metrics.  No execution logic
exists in this file.
"""
from __future__ import annotations

import csv
import os

import numpy as np
import pandas as pd

from forex_signal_sweep_common import build_bid_ohlc
from forex_strategy_common import build_parser, load_market, timeframe_to_ns


LOOKBACKS = (1, 3, 6, 12)
HORIZONS = (1, 3, 6, 12)
LAGS = (0, 1, 2, 3, 6, 12)
PYSR_BINARY_OPERATORS = ["+", "-", "*", "/", "^", "max", "min"]
PYSR_UNARY_OPERATORS = ["abs", "sign", "square", "cube", "sqrtabs(x) = sqrt(abs(x))", "logabs(x) = log1p(abs(x))", "tanh", "atan", "sin", "cos"]


def _sympy_sqrtabs(x):
    import sympy
    return sympy.sqrt(sympy.Abs(x))


def _sympy_logabs(x):
    import sympy
    return sympy.log(1 + sympy.Abs(x))

def canonical_candles(group: pd.DataFrame, timeframe: str, prefix: str) -> pd.DataFrame:
    bid = group.bid.to_numpy(np.float64)
    ts_ns = group.timestamp.to_numpy(dtype="datetime64[ns]").astype("int64")
    open_, high, low, close, close_idx = build_bid_ohlc(bid, ts_ns, timeframe)
    tf_ns = timeframe_to_ns(timeframe)
    return pd.DataFrame({
        "timestamp": pd.to_datetime((ts_ns[close_idx] // tf_ns) * tf_ns, utc=True),
        f"{prefix}_open": open_, f"{prefix}_high": high, f"{prefix}_low": low, f"{prefix}_close": close,
    })


def bps_return(close: np.ndarray, steps: int) -> np.ndarray:
    out = np.full(len(close), np.nan, dtype=np.float64)
    out[steps:] = (close[steps:] / close[:-steps] - 1.0) * 10_000.0
    return out


def future_bps(close: np.ndarray, steps: int) -> np.ndarray:
    out = np.full(len(close), np.nan, dtype=np.float64)
    out[:-steps] = (close[steps:] / close[:-steps] - 1.0) * 10_000.0
    return out


def corr_stats(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    valid = np.isfinite(x) & np.isfinite(y)
    n = int(valid.sum())
    if n < 30 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return 0.0, 0.0, n
    corr = float(np.corrcoef(x[valid], y[valid])[0, 1])
    hit = float(np.mean(np.sign(x[valid]) == np.sign(y[valid])) * 100.0)
    return corr, hit, n


def pure_rows(frame: pd.DataFrame, timeframe: str) -> list[dict]:
    rows = []
    thirds = np.array_split(np.arange(len(frame)), 3)
    for leader, follower in (("xau", "aud"), ("aud", "xau"), ("xaua", "aud"), ("aud", "xaua")):
        for lookback in LOOKBACKS:
            x = bps_return(frame[f"{leader}_close"].to_numpy(), lookback)
            for horizon in HORIZONS:
                y = future_bps(frame[f"{follower}_close"].to_numpy(), horizon)
                for lag in LAGS:
                    xx = x[:-lag] if lag else x
                    yy = y[lag:] if lag else y
                    corr, hit, n = corr_stats(xx, yy)
                    segment_corr = []
                    for segment in thirds:
                        seg_x = xx[segment[segment < len(xx)]]
                        seg_y = yy[segment[segment < len(yy)]]
                        segment_corr.append(corr_stats(seg_x, seg_y)[0])
                    valid = np.isfinite(xx) & np.isfinite(yy)
                    cutoff = float(np.nanquantile(np.abs(xx[valid]), 0.90)) if valid.any() else np.nan
                    event = valid & (np.abs(xx) >= cutoff)
                    signed_mean = float(np.mean(np.sign(xx[event]) * yy[event])) if event.any() else np.nan
                    rows.append({
                        "kind": "pure", "timeframe": timeframe, "leader": leader, "follower": follower,
                        "lookback": lookback, "horizon": horizon, "lag": lag, "samples": n,
                        "corr": corr, "sign_hit_pct": hit, "segment_corr_min": min(segment_corr),
                        "segment_corr_max": max(segment_corr), "segment_corr_mean": float(np.mean(segment_corr)),
                        "stable_abs_corr": min(abs(value) for value in segment_corr) if (all(value >= 0 for value in segment_corr) or all(value <= 0 for value in segment_corr)) else 0.0,
                        "event_cutoff_bps": cutoff,
                        "event_signed_future_mean_bps": signed_mean,
                    })
    return rows


def symbolic_rows(frame: pd.DataFrame, timeframe: str, generations: int, population: int, seed: int,
                  backend: str, binary_operators: list[str], unary_operators: list[str], maxsize: int,
                  pysr_workers: int, depot_dir: str) -> list[dict]:
    if backend == "gplearn":
        from gplearn.genetic import SymbolicRegressor
    else:
        os.environ.setdefault("JULIA_DEPOT_PATH", depot_dir)
        from pysr import PySRRegressor

    rows = []
    base = {}
    names = []
    for pair in ("aud", "xau", "xaua"):
        close = frame[f"{pair}_close"].to_numpy()
        for lag in range(6):
            value = np.full(len(close), np.nan)
            if lag == 0:
                value[1:] = (close[1:] / close[:-1] - 1.0) * 10_000.0
            else:
                value[lag + 1:] = (close[lag + 1:] / close[lag:-1] - 1.0) * 10_000.0
            name = f"{pair}_r{lag}"
            base[name] = value
            names.append(name)
    x_all = np.column_stack([base[name] for name in names])
    for target_pair in ("aud", "xau", "xaua"):
        close = frame[f"{target_pair}_close"].to_numpy()
        for horizon in (1, 3, 6):
            y_all = future_bps(close, horizon)
            valid = np.isfinite(x_all).all(axis=1) & np.isfinite(y_all)
            x = x_all[valid]
            y = y_all[valid]
            split = int(len(y) * 0.70)
            if split < 300 or len(y) - split < 100:
                continue
            if backend == "gplearn":
                model = SymbolicRegressor(
                    population_size=population, generations=generations, tournament_size=12,
                    function_set=("add", "sub", "mul", "div"), metric="mean absolute error",
                    parsimony_coefficient=0.01, max_samples=0.80, random_state=seed,
                    feature_names=names, n_jobs=1, verbose=0,
                )
                model.fit(x[:split], y[:split])
                expression = str(model._program)
            else:
                model = PySRRegressor(
                    niterations=generations, populations=max(1, pysr_workers),
                    population_size=min(max((population + pysr_workers - 1) // pysr_workers, 20), 200), binary_operators=binary_operators,
                    unary_operators=unary_operators, model_selection="best", maxsize=maxsize,
                    parallelism="multiprocessing", procs=max(1, pysr_workers),
                    extra_sympy_mappings={"sqrtabs": _sympy_sqrtabs, "logabs": _sympy_logabs},
                    random_state=seed if pysr_workers <= 1 else None, progress=False, verbosity=0,
                )
                model.fit(x[:split], y[:split], variable_names=names)
                expression = str(model.equations_.iloc[-1]["equation"])
            pred = np.asarray(model.predict(x[split:]), dtype=np.float64)
            actual = y[split:]
            corr, hit, _ = corr_stats(pred, actual)
            mae = float(np.mean(np.abs(pred - actual)))
            baseline_mae = float(np.mean(np.abs(actual)))
            rows.append({
                "kind": "symbolic", "backend": backend, "timeframe": timeframe, "target": target_pair,
                "horizon": horizon, "train_samples": split, "test_samples": len(actual), "test_corr": corr,
                "test_sign_hit_pct": hit, "test_mae_bps": mae, "zero_baseline_mae_bps": baseline_mae,
                "mae_improvement_pct": (1.0 - mae / max(baseline_mae, 1e-9)) * 100.0,
                "expression": expression,
            })
    return rows


def main() -> None:
    ap = build_parser("AUDUSD/XAUUSD relationship discovery", "forex_cross_asset_relationship.csv")
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--population", type=int, default=1200)
    ap.add_argument("--backend", choices=["gplearn", "pysr"], default="pysr")
    ap.add_argument("--maxsize", type=int, default=48)
    ap.add_argument("--pysr-binary-operators", default=",".join(PYSR_BINARY_OPERATORS))
    ap.add_argument("--pysr-unary-operators", default=",".join(PYSR_UNARY_OPERATORS))
    ap.add_argument("--pysr-workers", type=int, default=15)
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--skip-symbolic", action="store_true", help="run pure lead/lag analysis only")
    args = ap.parse_args()
    args.timeframes = args.timeframes or "5m,15m"
    args.pairs = ["AUDUSD", "XAUUSD"]
    ticks, _ = load_market(args)
    groups = {p: g.sort_values("timestamp").reset_index(drop=True) for p, g in ticks.groupby("pair")}
    if set(groups) != {"AUDUSD", "XAUUSD"}:
        raise SystemExit(f"need AUDUSD and XAUUSD; got {sorted(groups)}")
    rows = []
    for timeframe in [x.strip() for x in args.timeframes.split(",") if x.strip()]:
        aud = canonical_candles(groups["AUDUSD"], timeframe, "aud")
        xau = canonical_candles(groups["XAUUSD"], timeframe, "xau")
        frame = aud.merge(xau, on="timestamp", how="inner")
        # Gold priced in Australian dollars removes the shared USD denomination.
        frame["xaua_close"] = frame["xau_close"] / frame["aud_close"]
        print(f"[cross] tf={timeframe} aligned_candles={len(frame):,}", flush=True)
        rows.extend(pure_rows(frame, timeframe))
        if not args.skip_symbolic:
            rows.extend(symbolic_rows(
                frame, timeframe, args.generations, args.population, args.seed, args.backend,
                [x.strip() for x in args.pysr_binary_operators.split(",") if x.strip()],
                [x.strip() for x in args.pysr_unary_operators.split(",") if x.strip()], args.maxsize, args.pysr_workers,
                os.path.abspath(os.path.join(os.path.dirname(args.out) or ".", ".julia")),
            ))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        fields = sorted({key for row in rows for key in row})
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    pure = [r for r in rows if r["kind"] == "pure"]
    symbolic = [r for r in rows if r["kind"] == "symbolic"]
    pure.sort(key=lambda r: (r["stable_abs_corr"], abs(r["segment_corr_mean"])), reverse=True)
    symbolic.sort(key=lambda r: (r["test_corr"], r["mae_improvement_pct"]), reverse=True)
    print("[cross] strongest stable pure relationships", flush=True)
    for row in pure[:12]: print(row, flush=True)
    print("[cross] symbolic held-out results", flush=True)
    for row in symbolic: print(row, flush=True)
    print(f"[cross] wrote {args.out} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
