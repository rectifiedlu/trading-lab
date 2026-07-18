"""Train symbolic future-return formulas from raw lagged OHLC.

Artifact scope is one pair/session/timeframe/window/horizon. Inputs are raw
OHLC lags only: o0,h0,l0,c0,o1,h1,l1,c1,... Target is the signed dominant
future excursion in points over candles t+1 through t+horizon. It is positive
when the largest excursion is upward and negative when it is downward.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import time
from pathlib import Path

import numpy as np

from forex_signal_sweep_common import build_bid_ohlc
from forex_strategy_common import active_session_allowed, build_parser, default_point_size, load_market, parse_num_list, parse_str_list

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None

DEFAULT_TIMEFRAMES = ["1m", "3m", "5m"]
DEFAULT_SESSIONS = [-1, 0, 1, 2]
DEFAULT_WINDOWS = [64]
DEFAULT_HORIZONS = [3]
DEFAULT_PYSR_BINARY_OPERATORS = ["+", "-", "*", "/", "^", "max", "min"]
DEFAULT_PYSR_UNARY_OPERATORS = ["abs", "sign", "square", "cube", "sqrtabs(x) = sqrt(abs(x))", "logabs(x) = log1p(abs(x))", "tanh", "atan", "sin", "cos"]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value))


def feature_names(window: int) -> list[str]:
    names: list[str] = []
    for lag in range(window):
        names.extend([f"o{lag}", f"h{lag}", f"l{lag}", f"c{lag}"])
    return names


if njit is not None:
    @njit(cache=True)
    def _build_lagged(open_, high, low, close, valid, window, horizon, point_size):
        start = window - 1
        stop = len(close) - horizon
        n = 0
        for i in range(start, stop):
            if valid[i]:
                n += 1
        x = np.empty((n, window * 4), dtype=np.float64)
        y = np.empty(n, dtype=np.float64)
        row = 0
        for i in range(start, stop):
            if not valid[i]:
                continue
            col = 0
            for lag in range(window):
                j = i - lag
                x[row, col] = open_[j]
                x[row, col + 1] = high[j]
                x[row, col + 2] = low[j]
                x[row, col + 3] = close[j]
                col += 4
            future_high = high[i + 1]
            future_low = low[i + 1]
            for j in range(i + 2, i + horizon + 1):
                if high[j] > future_high:
                    future_high = high[j]
                if low[j] < future_low:
                    future_low = low[j]
            up_points = (future_high - close[i]) / point_size
            down_points = (close[i] - future_low) / point_size
            y[row] = up_points if up_points >= down_points else -down_points
            row += 1
        return x, y


def build_lagged(open_, high, low, close, valid, window: int, horizon: int, point_size: float):
    if njit is not None:
        return _build_lagged(open_, high, low, close, valid, int(window), int(horizon), float(point_size))
    rows = []
    targets = []
    for i in range(window - 1, len(close) - horizon):
        if not valid[i]:
            continue
        row = []
        for lag in range(window):
            j = i - lag
            row.extend([open_[j], high[j], low[j], close[j]])
        rows.append(row)
        future_high = np.max(high[i + 1:i + horizon + 1])
        future_low = np.min(low[i + 1:i + horizon + 1])
        up_points = (future_high - close[i]) / point_size
        down_points = (close[i] - future_low) / point_size
        targets.append(up_points if up_points >= down_points else -down_points)
    return np.asarray(rows, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def _sympy_sqrtabs(x):
    import sympy
    return sympy.sqrt(sympy.Abs(x))


def _sympy_logabs(x):
    import sympy
    return sympy.log(1 + sympy.Abs(x))

def fit_gplearn(args, x_train: np.ndarray, y_train: np.ndarray, names: list[str]):
    from gplearn.genetic import SymbolicRegressor

    model = SymbolicRegressor(
        population_size=args.population,
        generations=args.generations,
        tournament_size=args.tournament_size,
        stopping_criteria=args.stopping_criteria,
        const_range=(-args.const_range, args.const_range),
        init_depth=(2, args.init_depth),
        function_set=tuple(parse_str_list(args.functions, ["add", "sub", "mul", "div"])),
        metric=args.metric,
        parsimony_coefficient=args.parsimony,
        p_crossover=0.70,
        p_subtree_mutation=0.10,
        p_hoist_mutation=0.05,
        p_point_mutation=0.10,
        max_samples=args.fit_sample_fraction,
        verbose=1 if args.verbose else 0,
        random_state=args.seed,
        n_jobs=args.jobs,
        feature_names=names,
    )
    model.fit(x_train, y_train)
    return model, str(model._program)


def fit_pysr(args, x_train: np.ndarray, y_train: np.ndarray, names: list[str]):
    os.environ.setdefault("JULIA_DEPOT_PATH", str((Path(args.out_dir) / ".julia").resolve()))
    from pysr import PySRRegressor

    model = PySRRegressor(
        niterations=args.generations,
        populations=max(1, args.pysr_workers),
        population_size=min(max((args.population + args.pysr_workers - 1) // args.pysr_workers, 20), 200),
        binary_operators=parse_str_list(args.pysr_binary_operators, DEFAULT_PYSR_BINARY_OPERATORS),
        unary_operators=parse_str_list(args.pysr_unary_operators, DEFAULT_PYSR_UNARY_OPERATORS),
        model_selection="best",
        maxsize=args.maxsize,
        parallelism="multiprocessing",
        procs=max(1, args.pysr_workers),
        extra_sympy_mappings={"sqrtabs": _sympy_sqrtabs, "logabs": _sympy_logabs},
        random_state=args.seed if args.pysr_workers <= 1 else None,
        progress=bool(args.verbose),
        verbosity=1 if args.verbose else 0,
    )
    model.fit(x_train, y_train, variable_names=names)
    # Native text avoids SymPy recursion for deeply nested piecewise expressions.
    equation = model.equations_.iloc[-1]["equation"] if getattr(model, "equations_", None) is not None else "<unavailable>"
    return model, str(equation)


def fmt_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m{sec:04.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h{int(minutes):02d}m"


def progress_text(done: int, total: int, start: float) -> str:
    elapsed = time.time() - start
    rate = done / max(elapsed, 1e-9)
    eta = (total - done) / max(rate, 1e-9) if done > 0 else 0.0
    pct = done / max(total, 1) * 100.0
    return f"{done:,}/{total:,} ({pct:.1f}%) elapsed={fmt_elapsed(elapsed)} rate={rate:.2f}/s eta={fmt_elapsed(eta)}"


def split_stats(model, x: np.ndarray, y: np.ndarray) -> dict:
    if len(y) == 0:
        return {"n": 0, "mae": None, "rmse": None, "corr": None}
    pred = np.asarray(model.predict(x), dtype=np.float64)
    err = pred - y
    corr = float(np.corrcoef(pred, y)[0, 1]) if len(y) > 2 and np.std(pred) > 0 and np.std(y) > 0 else 0.0
    return {"n": int(len(y)), "mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err * err))), "corr": corr}


def rank_corr(meta: dict) -> float:
    """Use held-out correlation whenever the model has a held-out split."""
    test = meta.get("test", {})
    train = meta.get("train", {})
    value = test.get("corr") if test.get("n", 0) else train.get("corr")
    return float(value) if value is not None and np.isfinite(value) else float("-inf")


def print_ranked_summary(summary: list[dict], top: int = 30) -> None:
    print("[symtrain] ranked by test corr; train corr when test split is empty", flush=True)
    for rank, meta in enumerate(summary[:top], 1):
        source = "test" if meta["test"].get("n", 0) else "train"
        print(
            f"[symtrain] rank={rank} corr={rank_corr(meta):.6f} source={source} "
            f"pair={meta['pair']} tf={meta['timeframe']} sess={meta['session']} "
            f"w={meta['window']} h={meta['horizon']} expr={meta['expression']}",
            flush=True,
        )

def main() -> None:
    ap = build_parser("Symbolic raw-OHLC future-return trainer", "unused.csv")
    ap.add_argument("--sessions", default=",".join(str(x) for x in DEFAULT_SESSIONS))
    ap.add_argument("--windows", default=",".join(str(x) for x in DEFAULT_WINDOWS))
    ap.add_argument("--horizons", default=",".join(str(x) for x in DEFAULT_HORIZONS))
    ap.add_argument("--backend", choices=["gplearn", "pysr"], default="pysr")
    ap.add_argument("--out-dir", default=os.path.join("data", "forex", "symbolic_models"))
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--train-only", action="store_true", help="fit on all available samples; no held-out test split")
    ap.add_argument("--max-samples", type=int, default=50000)
    ap.add_argument("--min-samples", type=int, default=1000)
    ap.add_argument("--population", type=int, default=1200)
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--tournament-size", type=int, default=20)
    ap.add_argument("--stopping-criteria", type=float, default=0.0)
    ap.add_argument("--const-range", type=float, default=10.0)
    ap.add_argument("--init-depth", type=int, default=4)
    ap.add_argument("--functions", default="add,sub,mul,div")
    ap.add_argument("--metric", choices=["mean absolute error", "mse", "rmse", "pearson", "spearman"], default="mean absolute error")
    ap.add_argument("--parsimony", type=float, default=0.001)
    ap.add_argument("--fit-sample-fraction", type=float, default=0.80)
    ap.add_argument("--maxsize", type=int, default=48, help="PySR max expression size")
    ap.add_argument("--pysr-binary-operators", default=",".join(DEFAULT_PYSR_BINARY_OPERATORS))
    ap.add_argument("--pysr-unary-operators", default=",".join(DEFAULT_PYSR_UNARY_OPERATORS))
    ap.add_argument("--pysr-workers", type=int, default=15, help="Julia worker processes per PySR fit")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--jobs", type=int, default=15, help="gplearn worker processes; PySR uses --pysr-workers")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    timeframes = parse_str_list(args.timeframes, DEFAULT_TIMEFRAMES)
    sessions = [int(x) for x in parse_num_list(args.sessions, DEFAULT_SESSIONS)]
    windows = [int(x) for x in parse_num_list(args.windows, DEFAULT_WINDOWS)]
    horizons = [int(x) for x in parse_num_list(args.horizons, DEFAULT_HORIZONS)]
    os.makedirs(args.out_dir, exist_ok=True)
    ticks, _ = load_market(args)
    summary = []
    total_jobs = len(ticks["pair"].unique()) * len(timeframes) * len(sessions) * len(windows) * len(horizons)
    done_jobs = 0
    trained_jobs = 0
    skipped_jobs = 0
    print(f"[symtrain] plan pairs={ticks['pair'].nunique()} timeframes={timeframes} sessions={sessions} windows={windows} horizons={horizons} total={total_jobs:,} backend={args.backend} pysr_workers={args.pysr_workers} train_only={int(args.train_only)}", flush=True)

    for pair, g in ticks.groupby("pair", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        bid = g["bid"].to_numpy(np.float64)
        ts_ns = g["timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64").astype(np.int64)
        point = float(args.point_size or default_point_size(pair))
        for tf in timeframes:
            open_, high, low, close, close_idx = build_bid_ohlc(bid, ts_ns, tf)
            if len(close) < max(windows) + max(horizons) + 10:
                skipped = len(sessions) * len(windows) * len(horizons)
                done_jobs += skipped
                skipped_jobs += skipped
                print(f"[symtrain] skip pair={pair} tf={tf} candles={len(close)} too_few progress={progress_text(done_jobs, total_jobs, t0)}", flush=True)
                continue
            candle_ts = ts_ns[close_idx]
            session_masks = {s: active_session_allowed(candle_ts, s) for s in sessions}
            for sess in sessions:
                valid = session_masks[sess].astype(np.bool_)
                for window in windows:
                    names = feature_names(window)
                    for horizon in horizons:
                        x, y = build_lagged(open_, high, low, close, valid, window, horizon, point)
                        if len(y) < args.min_samples:
                            done_jobs += 1
                            skipped_jobs += 1
                            print(f"[symtrain] skip {pair} tf={tf} sess={sess} w={window} h={horizon} samples={len(y):,}/{args.min_samples:,} progress={progress_text(done_jobs, total_jobs, t0)}", flush=True)
                            continue
                        if args.max_samples > 0 and len(y) > args.max_samples:
                            take = np.linspace(0, len(y) - 1, args.max_samples).astype(np.int64)
                            x = x[take]
                            y = y[take]
                        if args.train_only:
                            split = len(y)
                        else:
                            split = int(len(y) * max(0.05, min(0.95, args.train_frac)))
                        x_train, y_train = x[:split], y[:split]
                        x_test, y_test = x[split:], y[split:]
                        print(f"[symtrain] fit start {done_jobs + 1:,}/{total_jobs:,} pair={pair} tf={tf} sess={sess} w={window} h={horizon} backend={args.backend} train={len(y_train):,} test={len(y_test):,}", flush=True)
                        fit_start = time.time()
                        model, expr = fit_pysr(args, x_train, y_train, names) if args.backend == "pysr" else fit_gplearn(args, x_train, y_train, names)
                        train_stats = split_stats(model, x_train, y_train)
                        test_stats = split_stats(model, x_test, y_test)
                        stem = f"symbolic_{safe_name(pair)}_s{sess}_tf{safe_name(tf)}_w{window}_h{horizon}_{args.backend}_seed{args.seed}"
                        model_path = Path(args.out_dir) / f"{stem}.pkl"
                        meta_path = Path(args.out_dir) / f"{stem}.json"
                        if model_path.exists() or meta_path.exists():
                            print(f"[symtrain] overwrite model={model_path.name} meta={meta_path.name}", flush=True)
                        with open(model_path, "wb") as f:
                            pickle.dump(model, f)
                        meta = {
                            "pair": pair, "session": int(sess), "timeframe": tf, "window": int(window),
                            "horizon": int(horizon), "point_size": point, "backend": args.backend,
                            "model_path": str(model_path), "feature_names": names,
                            "target": "signed_dominant_future_excursion_points",
                            "expression": expr, "samples": int(len(y)), "train": train_stats, "test": test_stats,
                            "fit_seconds": round(time.time() - fit_start, 3),
                        }
                        with open(meta_path, "w", encoding="utf-8") as f:
                            json.dump(meta, f, indent=2)
                        summary.append(meta)
                        done_jobs += 1
                        trained_jobs += 1
                        print(f"[symtrain] fit done {progress_text(done_jobs, total_jobs, t0)} trained={trained_jobs:,} skipped={skipped_jobs:,} fit={fmt_elapsed(time.time() - fit_start)} test_mae={test_stats['mae']} test_corr={test_stats['corr']} expr={expr}", flush=True)
                        print(f"[symtrain] wrote {meta_path}", flush=True)

    summary.sort(key=rank_corr, reverse=True)
    for meta in summary:
        meta["rank_corr"] = rank_corr(meta)
        meta["rank_source"] = "test" if meta["test"].get("n", 0) else "train"
    summary_path = Path(args.out_dir) / "symbolic_training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print_ranked_summary(summary)
    print(f"[symtrain] done {progress_text(done_jobs, total_jobs, t0)} models={len(summary):,} trained={trained_jobs:,} skipped={skipped_jobs:,} summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()





