"""Evaluate saved TCN3 excursion models against realised future scores.

This is a model evaluator, not a trade simulator. For every eligible candle it
compares the predicted net-excursion score with the score realised over the
model's own horizon, then aggregates and ranks model-level accuracy metrics.
"""
from __future__ import annotations

import csv
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from forex_ml_barrier_cnn import (
    BarrierData,
    EXCURSION_LABEL,
    label_excursion,
    make_time_features,
    make_window_feature_batch,
)
from forex_ml_tick_simulator import build_bid_candles, load_torch_model, model_files, parse_model_name
from forex_strategy_common import (
    active_session_allowed,
    build_parser,
    load_market,
    parse_num_list,
    parse_str_list,
)


FIELDS = [
    "rank",
    "pair",
    "timeframe",
    "label_session",
    "window",
    "horizon",
    "samples",
    "score_accuracy_mean_pct",
    "score_accuracy_median_pct",
    "score_accuracy_p10_pct",
    "samples_ge_50pct",
    "samples_ge_75pct",
    "samples_ge_90pct",
    "direction_accuracy_pct",
    "mae_points",
    "median_ae_points",
    "rmse_points",
    "mean_actual_abs_points",
    "normalized_mae_pct",
    "bias_points",
    "correlation",
    "r2",
    "pred_mean_points",
    "actual_mean_points",
    "model_file",
]


def requested_ints(value: str | None) -> set[int] | None:
    return None if value is None else {int(x) for x in parse_num_list(value, [])}


def finite_or(value: float, fallback: float = 0.0) -> float:
    return float(value) if np.isfinite(value) else float(fallback)


def score_metrics(prediction: np.ndarray, actual: np.ndarray) -> dict[str, float | int]:
    prediction = np.asarray(prediction, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    mask = np.isfinite(prediction) & np.isfinite(actual)
    prediction = prediction[mask]
    actual = actual[mask]
    if not len(actual):
        raise ValueError("no finite prediction/actual pairs")

    error = prediction - actual
    abs_error = np.abs(error)
    denominator = np.abs(prediction) + np.abs(actual)
    relative_error = np.divide(
        abs_error,
        denominator,
        out=np.zeros_like(abs_error),
        where=denominator > 1e-12,
    )
    similarity = np.maximum(0.0, 1.0 - relative_error)
    actual_mean = float(np.mean(actual))
    residual_sum = float(np.sum(error * error))
    total_sum = float(np.sum((actual - actual_mean) ** 2))
    correlation = (
        float(np.corrcoef(prediction, actual)[0, 1])
        if len(actual) > 1 and np.std(prediction) > 0.0 and np.std(actual) > 0.0
        else 0.0
    )
    mean_actual_abs = float(np.mean(np.abs(actual)))
    mae = float(np.mean(abs_error))
    return {
        "samples": int(len(actual)),
        "score_accuracy_mean_pct": float(np.mean(similarity) * 100.0),
        "score_accuracy_median_pct": float(np.median(similarity) * 100.0),
        "score_accuracy_p10_pct": float(np.percentile(similarity, 10) * 100.0),
        "samples_ge_50pct": float(np.mean(similarity >= 0.50) * 100.0),
        "samples_ge_75pct": float(np.mean(similarity >= 0.75) * 100.0),
        "samples_ge_90pct": float(np.mean(similarity >= 0.90) * 100.0),
        "direction_accuracy_pct": float(np.mean((prediction >= 0.0) == (actual >= 0.0)) * 100.0),
        "mae_points": mae,
        "median_ae_points": float(np.median(abs_error)),
        "rmse_points": float(math.sqrt(np.mean(error * error))),
        "mean_actual_abs_points": mean_actual_abs,
        "normalized_mae_pct": float(mae / max(mean_actual_abs, 1e-12) * 100.0),
        "bias_points": float(np.mean(error)),
        "correlation": finite_or(correlation),
        "r2": finite_or(1.0 - residual_sum / total_sum) if total_sum > 0.0 else 0.0,
        "pred_mean_points": float(np.mean(prediction)),
        "actual_mean_points": actual_mean,
    }


def predict_indices(
    model,
    namespace,
    data: BarrierData,
    indices: np.ndarray,
    device: torch.device,
    feature_batch_size: int,
    inference_batch_size: int,
    progress_prefix: str,
) -> np.ndarray:
    predictions = np.empty(len(indices), dtype=np.float32)
    scale_points = float(getattr(namespace, "barrier_points", getattr(namespace, "move_scale_points", 100.0)))
    output_scale = float(getattr(namespace, "move_scale_points", 100.0))
    time_features = make_time_features(data.times)
    feature_batch_size = max(1, int(feature_batch_size))
    inference_batch_size = max(1, int(inference_batch_size))
    amp_enabled = device.type == "cuda" and not bool(getattr(namespace, "no_amp", False))
    model = model.to(device).eval()
    started = time.time()

    with torch.inference_mode():
        for feature_start in range(0, len(indices), feature_batch_size):
            feature_stop = min(feature_start + feature_batch_size, len(indices))
            chunk_indices = indices[feature_start:feature_stop]
            features = make_window_feature_batch(
                data,
                chunk_indices,
                int(namespace.window),
                scale_points,
                str(namespace.feature_set),
            )
            spread = (data.spread[chunk_indices] / max(data.point_size * scale_points, 1e-12)).reshape(-1, 1)
            if getattr(namespace, "session_feature", False):
                session = data.session[chunk_indices].reshape(-1, 1)
                extras = np.concatenate([spread, session, time_features[chunk_indices]], axis=1).astype(np.float32)
            else:
                extras = np.concatenate([spread, time_features[chunk_indices]], axis=1).astype(np.float32)

            for batch_start in range(0, len(chunk_indices), inference_batch_size):
                batch_stop = min(batch_start + inference_batch_size, len(chunk_indices))
                x = torch.from_numpy(features[batch_start:batch_stop]).to(device, non_blocking=device.type == "cuda")
                extra = torch.from_numpy(extras[batch_start:batch_stop]).to(device, non_blocking=device.type == "cuda")
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    raw = model(x, extra)
                predictions[feature_start + batch_start:feature_start + batch_stop] = (
                    raw.reshape(-1).float().cpu().numpy() * output_scale
                )

            done = feature_stop
            elapsed = time.time() - started
            rate = done / max(elapsed, 1e-9)
            eta = (len(indices) - done) / max(rate, 1e-9)
            print(
                f"[tcn3-eval] {progress_prefix} samples={done:,}/{len(indices):,} "
                f"({done / len(indices) * 100.0:.1f}%) rate={rate:,.0f}/s eta={eta:.1f}s",
                flush=True,
            )
    return predictions


def write_results(path: str, rows: list[dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in FIELDS} for row in rows)


def rank_rows(rows: list[dict[str, object]]) -> None:
    rows.sort(key=lambda row: (
        float(row["score_accuracy_mean_pct"]),
        float(row["direction_accuracy_pct"]),
        float(row["correlation"]),
        -float(row["mae_points"]),
    ), reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank


def print_rankings(rows: list[dict[str, object]], top: int) -> None:
    print("\n[tcn3-eval] most accurate models", flush=True)
    for row in rows[:max(1, top)]:
        print(
            f"#{int(row['rank']):02d} score_acc={float(row['score_accuracy_mean_pct']):6.2f}% "
            f"direction={float(row['direction_accuracy_pct']):6.2f}% "
            f"corr={float(row['correlation']):+.4f} mae={float(row['mae_points']):8.2f}pt "
            f"n={int(row['samples']):,} {row['pair']} {row['timeframe']} "
            f"s={row['label_session']} w={row['window']} h={row['horizon']} {row['model_file']}",
            flush=True,
        )


def main() -> None:
    parser = build_parser("TCN3 excursion score evaluator", "tcn3_excursion_evaluation.csv")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-glob", default="*excursion*_*tcn3_*.pt")
    parser.add_argument("--sessions", default=None, help="filter model label sessions")
    parser.add_argument("--windows", default=None)
    parser.add_argument("--horizons", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--feature-batch-size", type=int, default=8192)
    parser.add_argument("--inference-batch-size", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=0, help="0 evaluates every eligible sample")
    args = parser.parse_args()

    pairs = {str(pair).upper() for pair in args.pairs}
    timeframes = None if not args.timeframes else {str(tf).lower() for tf in parse_str_list(args.timeframes, [])}
    sessions = requested_ints(args.sessions)
    windows = requested_ints(args.windows)
    horizons = requested_ints(args.horizons)
    paths = model_files(
        Path(args.model_dir),
        parse_str_list(args.model_glob, ["*.pt"]),
        pairs,
        timeframes,
        windows,
        sessions,
    )
    paths = [
        path for path in paths
        if str(parse_model_name(path).get("target", "")) == "excursion"
        and str(parse_model_name(path).get("model", "")) == "tcn3"
        and (horizons is None or int(parse_model_name(path).get("horizon", -1)) in horizons)
    ]
    paths.sort(key=lambda path: (
        str(parse_model_name(path).get("pair", "")),
        str(parse_model_name(path).get("tf", "")),
        int(parse_model_name(path).get("label_session", 0)),
        int(parse_model_name(path).get("window", 0)),
        int(parse_model_name(path).get("horizon", 0)),
        path.name,
    ))
    if not paths:
        raise SystemExit("no matching TCN3 excursion models found")
    if os.path.exists(args.out):
        os.remove(args.out)
        print(f"[tcn3-eval] overwrite out={args.out}", flush=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    ticks, _ = load_market(args)
    sample_text = "all" if args.max_samples <= 0 else f"{args.max_samples:,}"
    print(
        f"[tcn3-eval] models={len(paths):,} device={device} days={args.days} "
        f"max_samples={sample_text} "
        "metric=bounded_symmetric_score_similarity evaluation_overlap=may_include_training_data",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    active_pair = None
    pair_ticks = None
    candle_cache: dict[str, object] = {}
    model_total = len(paths)
    for model_number, path in enumerate(paths, 1):
        meta = parse_model_name(path)
        pair = str(meta["pair"]).upper()
        timeframe = str(meta["tf"]).lower()
        if pair != active_pair:
            pair_ticks = ticks[ticks["pair"].str.upper() == pair].sort_values("timestamp").reset_index(drop=True)
            if pair_ticks.empty:
                raise SystemExit(f"no market data loaded for model pair={pair}")
            active_pair = pair
            candle_cache = {}
        if timeframe not in candle_cache:
            bid = pair_ticks["bid"].to_numpy(np.float64)
            ask = pair_ticks["ask"].to_numpy(np.float64)
            ts_ns = pair_ticks["timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64").astype(np.int64)
            candle_cache[timeframe] = build_bid_candles(bid, ask, ts_ns, timeframe)
        candles = candle_cache[timeframe]

        model, namespace, point = load_torch_model(path)
        if str(getattr(namespace, "excursion_label", "")) != EXCURSION_LABEL:
            raise SystemExit(f"incompatible excursion label in {path.name}; retrain this model")
        horizon = int(meta["horizon"])
        label_session = int(meta.get("label_session", 0))
        actual_scaled, valid = label_excursion(
            candles.ohlc[:, 3].astype(np.float64),
            candles.ohlc[:, 1].astype(np.float64),
            candles.ohlc[:, 2].astype(np.float64),
            horizon,
            float(point),
            1.0,
        )
        session_mask = active_session_allowed(candles.times.astype("int64"), label_session)
        valid &= session_mask
        indices = np.flatnonzero(valid)
        indices = indices[indices >= int(namespace.window) - 1]
        if args.max_samples > 0 and len(indices) > args.max_samples:
            positions = np.linspace(0, len(indices) - 1, args.max_samples, dtype=np.int64)
            indices = indices[positions]
        if not len(indices):
            print(f"[tcn3-eval] skip {path.name}: no eligible samples", flush=True)
            continue

        data = BarrierData(
            times=candles.times,
            ohlc=candles.ohlc,
            spread=candles.spread,
            labels=np.zeros(len(candles.ohlc), dtype=np.float32),
            valid=valid,
            session=active_session_allowed(candles.times.astype("int64"), 1).astype(np.float32),
            point_size=float(point),
        )
        print(
            f"[tcn3-eval] model {model_number}/{model_total} {path.name} "
            f"candles={len(candles.ohlc):,} eligible={len(indices):,}", flush=True,
        )
        prediction = predict_indices(
            model,
            namespace,
            data,
            indices,
            device,
            args.feature_batch_size,
            args.inference_batch_size,
            f"model={model_number}/{model_total}",
        )
        actual = actual_scaled[indices].astype(np.float64)
        metrics = score_metrics(prediction, actual)
        row: dict[str, object] = {
            "pair": pair,
            "timeframe": timeframe,
            "label_session": label_session,
            "window": int(meta["window"]),
            "horizon": horizon,
            "model_file": path.name,
            **metrics,
        }
        rows.append(row)
        rank_rows(rows)
        write_results(args.out, rows)
        print(
            f"[tcn3-eval] done score_acc={metrics['score_accuracy_mean_pct']:.2f}% "
            f"direction={metrics['direction_accuracy_pct']:.2f}% corr={metrics['correlation']:+.4f} "
            f"mae={metrics['mae_points']:.2f}pt checkpoint={args.out}",
            flush=True,
        )

    if not rows:
        raise SystemExit("no models produced evaluation results")
    rank_rows(rows)
    write_results(args.out, rows)
    print_rankings(rows, args.top)
    print(f"[tcn3-eval] wrote {args.out} rows={len(rows):,}", flush=True)


if __name__ == "__main__":
    main()
