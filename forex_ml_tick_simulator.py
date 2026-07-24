"""Clean tick simulator for trained move4 ML models.

The model predicts on closed bid candles. Entries happen on the next tick after
the signal candle closes. Fixed TP/SL exits are tick-based; signal exits are
evaluated on closed candles. There is no implicit horizon exit.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from forex_ml_barrier_cnn import (
    BarrierData,
    build_model,
    feature_names,
    make_time_features,
    make_window_feature_batch,
)
from forex_strategy_common import (
    TradeResult,
    active_session_allowed,
    build_parser,
    day_ids_from_timestamps,
    default_point_size,
    load_market,
    parse_num_list,
    parse_str_list,
    timeframe_to_ns,
    write_results,
)

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


DEFAULT_MODEL_DIR = Path("data") / "forex" / "ml_models"
THRESHOLDS_DEFAULT = "0.55,0.60,0.65,0.70,0.75,0.80"
RISK_VALUES_DEFAULT = "1.5,2.0,2.5,3,3.5"
RR_FLOOR_VALUES_DEFAULT = "1.2,1.4,1.6"
MODES = {"fixed": 0, "prob_flip": 1, "signal_raw": 2, "signal_exit": 3, "rr_floor": 4}
RISK_MODES = {"none": 0, "rr": 1, "edge": 2, "prob_edge": 3}
GOLD_TP = [50, 100, 200, 300, 400]
GOLD_SL = [50, 100, 200, 300, 400]
FX_TP = [15, 30, 45, 60, 75, 90]
FX_SL = [15, 30, 45, 60, 75, 90]

@dataclass
class CandleData:
    times: np.ndarray
    ohlc: np.ndarray
    spread: np.ndarray
    close_tick_idx: np.ndarray
    tick_to_candle: np.ndarray


@dataclass
class PreparedModelInputs:
    features: np.ndarray
    extras: np.ndarray
    candle_indices: np.ndarray
    candle_count: int


def parse_model_name(path: Path) -> dict[str, str | int]:
    name = path.name
    out: dict[str, str | int] = {"file": name}
    m = re.match(r"forex_ml_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)_tf([^_]+)_", name)
    if m:
        out.update({
            "pair": m.group(1),
            "target": m.group(2),
            "side": m.group(3),
            "model": m.group(4),
            "feature_set": m.group(5),
            "tf": m.group(6),
        })
    patterns = {
        "scale": r"_scale([0-9.]+)_",
        "label_session": r"_s(-?\d+)_",
        "window": r"_w(\d+)_",
        "horizon": r"_h(\d+)_",
        "channels": r"_c(\d+)_",
        "kernel": r"_k(\d+)_",
        "layers": r"_l(\d+)",
    }
    for key, pat in patterns.items():
        mm = re.search(pat, name)
        if not mm:
            continue
        val = mm.group(1)
        out[key] = float(val) if key == "scale" else int(val)
    return out


def load_torch_model(path: Path):
    payload = torch.load(path, map_location="cpu")
    args_dict = dict(payload.get("args", {}))
    meta = parse_model_name(path)
    args_dict.setdefault("model", meta.get("model", "tcn"))
    args_dict.setdefault("target", meta.get("target", "move4"))
    args_dict.setdefault("feature_set", meta.get("feature_set", "ohlc12"))
    args_dict.setdefault("window", int(meta.get("window", 128)))
    args_dict.setdefault("channels", int(meta.get("channels", 64)))
    args_dict.setdefault("kernel_size", int(meta.get("kernel", 3)))
    args_dict.setdefault("layers", int(meta.get("layers", 5)))
    args_dict.setdefault("hidden", 128)
    args_dict.setdefault("dropout", 0.15)
    args_dict.setdefault("heads", 4)
    args_dict.setdefault("move_scale_points", float(meta.get("scale", 100.0)))
    args_dict.setdefault("barrier_points", float(args_dict.get("move_scale_points", 100.0)))
    args_dict.setdefault("session_feature", False)
    ns = SimpleNamespace(**args_dict)
    input_dim = len(feature_names(ns.feature_set))
    extra_dim = 6 if getattr(ns, "session_feature", False) else 5
    model = build_model(ns.model, int(ns.window), input_dim, extra_dim, ns)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    point_size = float(payload.get("point_size") or default_point_size(str(meta.get("pair", ""))))
    return model, ns, point_size


def build_bid_candles(bid: np.ndarray, ask: np.ndarray, ts_ns: np.ndarray, timeframe: str) -> CandleData:
    tf_ns = timeframe_to_ns(timeframe)
    buckets = (ts_ns // tf_ns) * tf_ns
    df = pd.DataFrame({"bucket": buckets, "bid": bid, "ask": ask, "tick_i": np.arange(len(bid), dtype=np.int64)})
    grouped = df.groupby("bucket", sort=True)
    ohlc_df = grouped["bid"].agg(["first", "max", "min", "last"])
    close_tick_idx = grouped["tick_i"].last().to_numpy(np.int64)
    spread = grouped.apply(lambda x: float(np.mean(x["ask"].to_numpy() - x["bid"].to_numpy()))).to_numpy(np.float32)
    unique_buckets = ohlc_df.index.to_numpy(np.int64)
    tick_to_candle = np.searchsorted(unique_buckets, buckets, side="right") - 1
    tick_to_candle = np.clip(tick_to_candle, 0, len(unique_buckets) - 1).astype(np.int64)
    return CandleData(
        times=unique_buckets.astype("datetime64[ns]"),
        ohlc=ohlc_df[["first", "max", "min", "last"]].to_numpy(np.float32),
        spread=spread,
        close_tick_idx=close_tick_idx,
        tick_to_candle=tick_to_candle,
    )


def prepare_model_inputs(ns, candles: CandleData, point_size: float) -> PreparedModelInputs:
    window = int(ns.window)
    barrier = float(getattr(ns, "barrier_points", getattr(ns, "move_scale_points", 100.0)))
    session = active_session_allowed(candles.times.astype("int64"), 1).astype(np.float32)
    data = BarrierData(
        times=candles.times,
        ohlc=candles.ohlc,
        spread=candles.spread,
        labels=np.zeros(len(candles.ohlc), dtype=np.float32),
        valid=np.ones(len(candles.ohlc), dtype=np.bool_),
        session=session,
        point_size=point_size,
    )
    candle_indices = np.arange(window - 1, len(candles.ohlc), dtype=np.int64)
    features = make_window_feature_batch(
        data,
        candle_indices,
        window,
        barrier,
        ns.feature_set,
    )
    time_features = make_time_features(data.times)[candle_indices]
    scale = max(data.point_size * barrier, 1e-12)
    spread_feature = (data.spread[candle_indices] / scale).reshape(-1, 1)
    if getattr(ns, "session_feature", False):
        session_feature = data.session[candle_indices].reshape(-1, 1)
        extras = np.concatenate([spread_feature, session_feature, time_features], axis=1)
    else:
        extras = np.concatenate([spread_feature, time_features], axis=1)
    return PreparedModelInputs(
        features=np.ascontiguousarray(features, dtype=np.float32),
        extras=np.ascontiguousarray(extras, dtype=np.float32),
        candle_indices=candle_indices,
        candle_count=len(candles.ohlc),
    )


def _format_model_output(raw: torch.Tensor, ns) -> np.ndarray:
    if getattr(ns, "target", "") == "move4":
        probability = torch.sigmoid(raw[:, :2])
        moves = torch.relu(raw[:, 2:]) * float(getattr(ns, "move_scale_points", 100.0))
        output = torch.cat([probability, moves], dim=1)
    elif getattr(ns, "target", "") == "excursion":
        score = raw.reshape(-1, 1) * float(getattr(ns, "move_scale_points", 100.0))
        empty = torch.full_like(score, float("nan"))
        output = torch.cat([score, empty, empty, empty], dim=1)
    else:
        probability_up = torch.sigmoid(raw).reshape(-1, 1)
        huge = torch.full_like(probability_up, 1.0e9)
        output = torch.cat([probability_up, 1.0 - probability_up, huge, huge], dim=1)
    return output.float().cpu().numpy()


def predict_prepared_model(
    model,
    ns,
    prepared: PreparedModelInputs,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    predictions = np.full((prepared.candle_count, 4), np.nan, dtype=np.float32)
    model = model.to(device)
    amp_enabled = device.type == "cuda" and not bool(getattr(ns, "no_amp", False))
    with torch.inference_mode():
        for start in range(0, len(prepared.candle_indices), int(batch_size)):
            stop = min(start + int(batch_size), len(prepared.candle_indices))
            feature_tensor = torch.from_numpy(prepared.features[start:stop])
            extra_tensor = torch.from_numpy(prepared.extras[start:stop])
            if device.type == "cuda":
                feature_tensor = feature_tensor.pin_memory()
                extra_tensor = extra_tensor.pin_memory()
            feature_tensor = feature_tensor.to(device, non_blocking=device.type == "cuda")
            extra_tensor = extra_tensor.to(device, non_blocking=device.type == "cuda")
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                raw = model(feature_tensor, extra_tensor)
            output = _format_model_output(raw, ns)
            predictions[prepared.candle_indices[start:stop]] = output.astype(np.float32)
    return predictions


def predict_model(model, ns, candles: CandleData, point_size: float, device: torch.device) -> np.ndarray:
    prepared = prepare_model_inputs(ns, candles, point_size)
    return predict_prepared_model(model, ns, prepared, device)

def smooth_predictions(preds: np.ndarray, ma: int) -> np.ndarray:
    ma = int(ma)
    if ma <= 1:
        return preds
    out = np.full_like(preds, np.nan)
    valid = np.isfinite(preds[:, 0])
    for col in range(preds.shape[1]):
        x = np.where(valid & np.isfinite(preds[:, col]), preds[:, col], 0.0).astype(np.float64)
        v = (valid & np.isfinite(preds[:, col])).astype(np.float64)
        cs = np.cumsum(np.insert(x, 0, 0.0))
        cv = np.cumsum(np.insert(v, 0, 0.0))
        sums = cs[ma:] - cs[:-ma]
        counts = cv[ma:] - cv[:-ma]
        vals = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
        out[ma - 1:, col] = vals.astype(np.float32)
    return out


def profit_factor(gross_win: float, gross_loss: float) -> float:
    if gross_loss > 0:
        return gross_win / gross_loss
    return 999.0 if gross_win > 0 else 0.0


def default_tp_grid(pair: str) -> list[float]:
    return [float(x) for x in (GOLD_TP if pair.upper().startswith("XAU") else FX_TP)]


def default_sl_grid(pair: str) -> list[float]:
    return [float(x) for x in (GOLD_SL if pair.upper().startswith("XAU") else FX_SL)]


if njit is not None:
    @njit(cache=True)
    def _risk_ok(side, p_long, p_short, exp_up, exp_down, threshold, risk_code, risk_value, min_tp):
        if side == 1:
            if p_long < threshold or exp_up < min_tp:
                return False
            if risk_code == 0:
                return True
            if risk_code == 1:
                return exp_up / max(exp_down, 1e-9) >= risk_value
            if risk_code == 2:
                return exp_up - exp_down >= risk_value
            return p_long - p_short >= risk_value
        if p_short < threshold or exp_down < min_tp:
            return False
        if risk_code == 0:
            return True
        if risk_code == 1:
            return exp_down / max(exp_up, 1e-9) >= risk_value
        if risk_code == 2:
            return exp_down - exp_up >= risk_value
        return p_short - p_long >= risk_value


    @njit(cache=True)
    def _mode_exit(mode_code, side, p_long, p_short, exp_up, exp_down, threshold, risk_code, risk_value, min_tp, rr_floor):
        opp = -side
        if mode_code == 1:  # prob_flip
            return p_short > p_long if side == 1 else p_long > p_short
        if mode_code == 2:  # signal_raw
            return p_short >= threshold if side == 1 else p_long >= threshold
        if mode_code == 3:  # signal_exit
            return _risk_ok(opp, p_long, p_short, exp_up, exp_down, threshold, risk_code, risk_value, min_tp)
        if mode_code == 4:  # rr_floor
            return exp_up / max(exp_down, 1e-9) <= rr_floor if side == 1 else exp_down / max(exp_up, 1e-9) <= rr_floor
        return False


    @njit(cache=True)
    def _simulate_core(
        bid, ask, close_tick_idx, tick_to_candle, preds, allowed, day_id, max_days,
        point_size, threshold, risk_code, risk_value, tp_mode, sl_mode, rr_floor, fixed_tp, fixed_sl,
        min_tp, amount, leverage, commission_per_million, side_filter_code, start_candle,
    ):
        daily = np.zeros(max_days, dtype=np.float64)
        daily_active = np.zeros(max_days, dtype=np.bool_)
        trade_pnls = np.zeros(len(close_tick_idx) + 1, dtype=np.float64)
        trade_dds = np.zeros(len(close_tick_idx) + 1, dtype=np.float64)
        notional = amount * leverage
        fee = notional / 1000000.0 * commission_per_million * 2.0
        cash = amount
        equity_peak = amount
        cum_pnl = 0.0
        cum_peak = 0.0
        max_dd = 0.0
        cum_dd = 0.0
        gross_win = 0.0
        gross_loss = 0.0
        trades = wins = losses = stops = sig = long_trades = short_trades = 0
        worst = 0.0
        max_trade_dd = 0.0
        sum_trade_dd = 0.0

        candle_i = start_candle
        tick_floor = 0
        pending_side = 0
        while candle_i < len(close_tick_idx):
            if pending_side != 0:
                side = pending_side
                pending_side = 0
                entry_tick = tick_floor
                if entry_tick >= len(bid):
                    break
                candle_i = int(tick_to_candle[entry_tick])
            else:
                p0 = preds[candle_i, 0]
                p1 = preds[candle_i, 1]
                eu = preds[candle_i, 2]
                ed = preds[candle_i, 3]
                if not (np.isfinite(p0) and np.isfinite(p1) and np.isfinite(eu) and np.isfinite(ed)) or not allowed[candle_i]:
                    candle_i += 1
                    continue
                side = 0
                if side_filter_code != -1 and _risk_ok(1, p0, p1, eu, ed, threshold, risk_code, risk_value, min_tp):
                    side = 1
                elif side_filter_code != 1 and _risk_ok(-1, p0, p1, eu, ed, threshold, risk_code, risk_value, min_tp):
                    side = -1
                if side == 0:
                    candle_i += 1
                    continue
                entry_tick = int(close_tick_idx[candle_i]) + 1
                if entry_tick < tick_floor:
                    entry_tick = tick_floor
                if entry_tick >= len(bid):
                    break

            entry = ask[entry_tick] if side == 1 else bid[entry_tick]
            trade_max = 0.0
            exit_tick = len(bid) - 1
            result_points = ((bid[exit_tick] if side == 1 else ask[exit_tick]) - entry) / point_size * side
            reason = 4  # end
            next_signal_candle = int(tick_to_candle[entry_tick]) + 1
            signal_exit_side = 0

            for ti in range(entry_tick + 1, len(bid)):
                if side == 1:
                    live_points = (bid[ti] - entry) / point_size
                    adverse = -live_points if live_points < 0.0 else 0.0
                else:
                    live_points = (entry - ask[ti]) / point_size
                    adverse = -live_points if live_points < 0.0 else 0.0
                denom = entry if entry > 1e-12 else 1e-12
                tdd = adverse * point_size * (notional / denom)
                if tdd > trade_max:
                    trade_max = tdd
                live_dd = equity_peak - (cash - tdd)
                if live_dd > max_dd:
                    max_dd = live_dd

                if tp_mode == 0 and fixed_tp > 0.0 and live_points >= fixed_tp:
                    result_points = fixed_tp
                    exit_tick = ti
                    reason = 3
                    break
                if sl_mode == 0 and fixed_sl > 0.0 and live_points <= -fixed_sl:
                    result_points = -fixed_sl
                    exit_tick = ti
                    reason = 2
                    break

                current_candle = int(tick_to_candle[ti])
                while next_signal_candle <= current_candle and next_signal_candle < len(close_tick_idx):
                    q0 = preds[next_signal_candle, 0]
                    q1 = preds[next_signal_candle, 1]
                    qu = preds[next_signal_candle, 2]
                    qd = preds[next_signal_candle, 3]
                    if np.isfinite(q0) and np.isfinite(q1) and np.isfinite(qu) and np.isfinite(qd):
                        exit_now = False
                        mode_used = 0
                        if live_points >= 0.0 and tp_mode != 0:
                            exit_now = _mode_exit(tp_mode, side, q0, q1, qu, qd, threshold, risk_code, risk_value, min_tp, rr_floor)
                            mode_used = tp_mode
                        if (not exit_now) and live_points <= 0.0 and sl_mode != 0:
                            exit_now = _mode_exit(sl_mode, side, q0, q1, qu, qd, threshold, risk_code, risk_value, min_tp, rr_floor)
                            mode_used = sl_mode
                        if exit_now:
                            result_points = live_points
                            exit_tick = ti
                            reason = 1
                            if _risk_ok(-side, q0, q1, qu, qd, threshold, risk_code, risk_value, min_tp):
                                signal_exit_side = -side
                            break
                    next_signal_candle += 1
                if reason == 1:
                    break

            pnl = result_points * point_size * (notional / (entry if entry > 1e-12 else 1e-12))
            trade_pnl = pnl - fee
            cash += trade_pnl
            if cash > equity_peak:
                equity_peak = cash
            closed_dd = equity_peak - cash
            if closed_dd > max_dd:
                max_dd = closed_dd
            cum_pnl += trade_pnl
            if cum_pnl > cum_peak:
                cum_peak = cum_pnl
            local_cum_dd = cum_peak - cum_pnl
            if local_cum_dd > cum_dd:
                cum_dd = local_cum_dd
            d = int(day_id[exit_tick])
            if 0 <= d < max_days:
                daily[d] += trade_pnl
                daily_active[d] = True
            trades += 1
            if side == 1:
                long_trades += 1
            else:
                short_trades += 1
            if reason == 2:
                stops += 1
            elif reason == 1:
                sig += 1
            if trade_pnl >= 0.0:
                wins += 1
                gross_win += trade_pnl
            else:
                losses += 1
                gross_loss += -trade_pnl
            if trade_pnl < worst:
                worst = trade_pnl
            trade_pnls[trades - 1] = trade_pnl
            trade_dds[trades - 1] = trade_max
            if trade_max > max_trade_dd:
                max_trade_dd = trade_max
            sum_trade_dd += trade_max

            tick_floor = exit_tick + 1
            if tick_floor >= len(bid):
                break
            if reason == 1 and signal_exit_side != 0:
                pending_side = signal_exit_side
                continue
            next_candle = int(tick_to_candle[tick_floor])
            candle_i = next_candle if next_candle > candle_i else candle_i + 1

        return (
            cash - amount, trades, wins, losses, gross_win, gross_loss, max_dd, cum_dd,
            long_trades, short_trades, stops, sig, worst, max_trade_dd,
            sum_trade_dd / trades if trades else 0.0, daily, daily_active, trade_pnls[:trades]
        )


def mode_list_arg(value: str | None) -> list[str]:
    return parse_str_list(value, ["fixed", "prob_flip", "signal_raw", "rr_floor"])


def parse_session_modes(value: str, fallback: int) -> list[int] | None:
    if str(value).lower() == "label":
        return None
    try:
        return [int(x) for x in parse_num_list(value, [fallback])]
    except ValueError as exc:
        raise SystemExit("--sessions expects session IDs like -1,0,1,2,3 or 'label'") from exc


def model_files(model_dir: Path, patterns: list[str], pairs: set[str] | None, tfs: set[str] | None, windows: set[int] | None, label_sessions: set[int] | None) -> list[Path]:
    files: list[Path] = []
    for pat in patterns:
        files.extend(model_dir.glob(pat))
    out = []
    for path in sorted(set(files)):
        if path.suffix.lower() != ".pt":
            continue
        meta = parse_model_name(path)
        if pairs and str(meta.get("pair", "")).upper() not in pairs:
            continue
        if tfs and str(meta.get("tf", "")) not in tfs:
            continue
        if windows and int(meta.get("window", 0)) not in windows:
            continue
        if label_sessions and int(meta.get("label_session", 0)) not in label_sessions:
            continue
        out.append(path)
    return out


def simulate_one(pair, meta, bid, ask, ts_ns, candles, preds, threshold, risk_mode, risk_value, tp_mode, sl_mode, rr_floor, prob_ma, fixed_tp, fixed_sl, session, min_tp, amount, leverage, commission, side_filter) -> TradeResult:
    point = default_point_size(pair)
    allowed = active_session_allowed(candles.times.astype("int64"), session)
    day_id, max_days = day_ids_from_timestamps(ts_ns)
    side_code = 0 if side_filter == "both" else (1 if side_filter == "long" else -1)
    out = _simulate_core(
        bid, ask, candles.close_tick_idx, candles.tick_to_candle, preds, allowed, day_id, max_days,
        point, threshold, RISK_MODES[risk_mode], risk_value, MODES[tp_mode], MODES[sl_mode], rr_floor,
        fixed_tp, fixed_sl, min_tp, amount, leverage, commission, side_code,
        int(meta.get("window", 128)) - 1,
    )
    total, trades, wins, losses, gw, gl, max_dd, cum_dd, longs, shorts, stops, sig, worst, tr_max, tr_avg, daily, daily_active, pnls = out
    params = (
        f"threshold={threshold:g};risk_mode={risk_mode};risk_value={risk_value:g};"
        f"tp_mode={tp_mode};tp={fixed_tp:g};sl_mode={sl_mode};sl={fixed_sl:g};rr_floor={rr_floor:g};"
        f"prob_ma={prob_ma};eval_session={session};label_session={meta.get('label_session','')};"
        f"window={meta.get('window','')};horizon={meta.get('horizon','')};file={meta.get('file','')}"
    )
    r = TradeResult(
        pair=pair, strategy="ml_tick", params=params, timeframe=str(meta.get("tf", "?")),
        tp_points=float(fixed_tp), sl_points=float(fixed_sl), point_size=point,
        realised=float(total), open_unrealized=0.0, total=float(total), trades=int(trades),
        wins=int(wins), losses=int(losses), win_rate=float(wins / trades * 100.0 if trades else 0.0),
        profit_factor=profit_factor(float(gw), float(gl)), max_drawdown=float(max_dd),
        long_trades=int(longs), short_trades=int(shorts), stop_losses=int(stops),
        signal_exits=int(sig), liquidations=0, account_dead=False, open_side="-", open_bps=0.0,
    )
    for name, val in {
        "threshold": threshold, "risk_mode": risk_mode, "risk_value": risk_value,
        "tp_mode": tp_mode, "sl_mode": sl_mode, "rr_floor": rr_floor, "prob_ma": prob_ma, "eval_session": session,
        "label_session": meta.get("label_session", 0), "window": meta.get("window", 0),
        "horizon": meta.get("horizon", 0), "model_file": meta.get("file", ""),
        "cum_max_drawdown": cum_dd, "trade_max_drawdown": tr_max, "trade_avg_drawdown": tr_avg,
        "worst_trade_pnl": worst, "median_loss": float(np.median(pnls[pnls < 0.0]) if np.any(pnls < 0.0) else 0.0),
        "avg_day": float(np.mean(daily) if len(daily) else 0.0),
        "median_day": float(np.median(daily) if len(daily) else 0.0),
        "active_days": int(np.count_nonzero(daily_active)),
    }.items():
        setattr(r, name, val)
    return r


def append_results(path: str, rows: list[TradeResult], header_written: bool) -> bool:
    if not rows:
        return header_written
    import csv
    exists = header_written or os.path.exists(path)
    fields = [
        "pair", "strategy", "timeframe", "threshold", "risk_mode", "risk_value",
        "tp_mode", "tp_points", "sl_mode", "sl_points", "rr_floor", "prob_ma", "total", "trades",
        "win_rate", "profit_factor", "max_drawdown", "cum_max_drawdown",
        "trade_max_drawdown", "worst_trade_pnl", "avg_day", "median_day",
        "long_trades", "short_trades", "stop_losses", "signal_exits",
        "label_session", "eval_session", "window", "horizon", "model_file", "params",
    ]
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(fields)
        for r in rows:
            w.writerow([getattr(r, k, "") if hasattr(r, k) else getattr(r, k, "") for k in fields])
    return True


def main() -> None:
    ap = build_parser("Clean ML tick simulator", "forex_ml_tick_clean_results.csv")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--model-glob", default="forex_ml_*.pt")
    ap.add_argument("--thresholds", default=THRESHOLDS_DEFAULT)
    ap.add_argument("--risk-modes", default="rr", help="comma list: rr,edge,prob_edge,none")
    ap.add_argument("--risk-values", default=RISK_VALUES_DEFAULT)
    ap.add_argument("--tp-modes", default="fixed,prob_flip,signal_raw,rr_floor")
    ap.add_argument("--sl-modes", default="fixed,prob_flip,signal_raw,rr_floor")
    ap.add_argument("--rr-floor-values", default=RR_FLOOR_VALUES_DEFAULT)
    ap.add_argument("--prob-ma-values", default="1", help="moving-average lengths applied to model output probabilities")
    ap.add_argument("--sessions", default="label")
    ap.add_argument("--label-sessions", default=None, help="filter trained model label sessions, e.g. 0,1,2")
    ap.add_argument("--windows", default=None)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()
    pairs = {p.upper() for p in args.pairs} if args.pairs else None
    tfs = set(parse_str_list(args.timeframes, [])) if args.timeframes else None
    windows = {int(x) for x in parse_num_list(args.windows, [])} if args.windows else None
    label_sessions = {int(x) for x in parse_num_list(args.label_sessions, [])} if args.label_sessions else None
    paths = model_files(Path(args.model_dir), parse_str_list(args.model_glob, ["*.pt"]), pairs, tfs, windows, label_sessions)
    if not paths:
        raise SystemExit("no matching .pt models found")
    ticks, _ = load_market(args)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    thresholds = parse_num_list(args.thresholds, [0.55])
    risk_modes = parse_str_list(args.risk_modes, ["rr"])
    risk_values = parse_num_list(args.risk_values, [2.0])
    rr_floor_values = parse_num_list(args.rr_floor_values, [1.0, 1.2, 1.4])
    prob_ma_values = [int(x) for x in parse_num_list(args.prob_ma_values, [1])]
    tp_modes = mode_list_arg(args.tp_modes)
    sl_modes = mode_list_arg(args.sl_modes)
    out_path = Path(args.out)
    if out_path.exists():
        out_path.unlink()
    results: list[TradeResult] = []
    header_written = False
    print(f"[ml-clean] models={len(paths)} device={device} thresholds={len(thresholds)} risk={risk_modes}/{len(risk_values)} tp_modes={tp_modes} sl_modes={sl_modes} rr_floor={rr_floor_values} prob_ma={prob_ma_values}", flush=True)
    for idx, path in enumerate(paths, 1):
        t0 = time.time()
        meta = parse_model_name(path)
        pair = str(meta.get("pair", "")).upper()
        g = ticks[ticks["pair"].str.upper() == pair].sort_values("timestamp").reset_index(drop=True)
        if g.empty:
            print(f"[ml-clean] skip {path.name} no ticks", flush=True)
            continue
        bid = g["bid"].to_numpy(np.float64)
        ask = g["ask"].to_numpy(np.float64)
        ts_ns = g["timestamp"].astype("int64").to_numpy()
        tf = str(meta.get("tf", "1m"))
        model, ns, point = load_torch_model(path)
        candles = build_bid_candles(bid, ask, ts_ns, tf)
        print(f"[ml-clean] {idx}/{len(paths)} {path.name} ticks={len(g):,} candles={len(candles.ohlc):,} predicting...", flush=True)
        preds_raw = predict_model(model, ns, candles, point, device)
        cap_tps = parse_num_list(args.tp_points, default_tp_grid(pair))
        cap_sls = parse_num_list(args.sl_points, default_sl_grid(pair))
        min_tp = 25.0 if pair.startswith("XAU") else 5.0
        sessions = [int(meta.get("label_session", 0))] if parse_session_modes(args.sessions, 0) is None else parse_session_modes(args.sessions, 0)
        mode_combo_count = 0
        for tm in tp_modes:
            for sm in sl_modes:
                tp_n = len(cap_tps) if tm == "fixed" else 1
                sl_n = len(cap_sls) if sm == "fixed" else 1
                rr_n = len(rr_floor_values) if (tm == "rr_floor" or sm == "rr_floor") else 1
                mode_combo_count += tp_n * sl_n * rr_n
        combo_total = len(prob_ma_values) * len(thresholds) * len(risk_modes) * len(risk_values) * len(sessions) * mode_combo_count
        print(f"[ml-clean] {idx}/{len(paths)} grid={combo_total:,} sessions={sessions}", flush=True)
        model_rows: list[TradeResult] = []
        done = 0
        for prob_ma in prob_ma_values:
            preds = smooth_predictions(preds_raw, prob_ma)
            for th in thresholds:
                for risk_mode in risk_modes:
                    if risk_mode not in RISK_MODES:
                        raise SystemExit(f"bad risk mode: {risk_mode}")
                    for rv in risk_values:
                        for sess in sessions:
                            for tm in tp_modes:
                                tp_vals = cap_tps if tm == "fixed" else [0.0]
                                for sm in sl_modes:
                                    sl_vals = cap_sls if sm == "fixed" else [0.0]
                                    rr_vals = rr_floor_values if (tm == "rr_floor" or sm == "rr_floor") else [0.0]
                                    for rr_floor in rr_vals:
                                        for tp in tp_vals:
                                            for sl in sl_vals:
                                                r = simulate_one(pair, meta, bid, ask, ts_ns, candles, preds, th, risk_mode, rv, tm, sm, float(rr_floor), int(prob_ma), float(tp), float(sl), int(sess), min_tp, args.amount, args.leverage, args.commission_per_million, args.side)
                                                if r.trades >= args.min_trades:
                                                    model_rows.append(r)
                                                done += 1
                                                if done % 1000 == 0 or done == combo_total:
                                                    print(f"[ml-clean] {idx}/{len(paths)} sim {done:,}/{combo_total:,} rows={len(model_rows):,}", flush=True)
        results.extend(model_rows)
        header_written = append_results(args.out, model_rows, header_written)
        best = max((r.total for r in model_rows), default=0.0)
        print(f"[ml-clean] {idx}/{len(paths)} done rows={len(model_rows):,} best=${best:+.2f} elapsed={time.time()-t0:.1f}s", flush=True)
    if not results:
        raise SystemExit("no results survived --min-trades")
    write_results(args.out, results, args.top, args.sort_by)
    print(f"[ml-clean] wrote {args.out} rows={len(results):,}", flush=True)


if __name__ == "__main__":
    main()
