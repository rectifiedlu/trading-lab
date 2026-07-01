"""Raw OHLC-window CNN for XAUUSD barrier direction.

Target:
    1 = +barrier points is hit before -barrier points after the window.
    0 = -barrier points is hit before +barrier points after the window.

The model gets normalized OHLC windows plus small time context. It does not
receive hand-built indicators like RSI/MACD/EMA. Session is exported for
analysis, but is not used as a model input unless --session-feature is passed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import pickle
import random
import time
from copy import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from forex_strategy_common import active_session_allowed, default_point_size, load_market, parse_num_list

try:
    from numba import njit
except Exception:  # pragma: no cover - optional speed dependency
    njit = None


DEFAULT_THRESHOLDS = [0.55, 0.58, 0.60, 0.62, 0.65]
DEFAULT_TIMEFRAMES = ["1m", "3m", "5m", "10m", "15m","30m"]
GOLD_BARRIERS = [200, 300, 400]
FX_BARRIERS = [30, 50, 80]
GOLD_EVAL_TP = [0, 50, 100, 150, 200, 250, 300, 400]
GOLD_EVAL_SL = [0, 50, 100, 150, 200, 250, 300, 400]
FX_EVAL_TP = [0, 10, 20, 30, 40, 50, 65, 80, 95]
FX_EVAL_SL = [0, 10, 20, 30, 40, 50, 65, 80, 95]

MT5_TIMEFRAMES = {
    "1m": "TIMEFRAME_M1",
    "2m": "TIMEFRAME_M2",
    "3m": "TIMEFRAME_M3",
    "4m": "TIMEFRAME_M4",
    "5m": "TIMEFRAME_M5",
    "6m": "TIMEFRAME_M6",
    "10m": "TIMEFRAME_M10",
    "12m": "TIMEFRAME_M12",
    "15m": "TIMEFRAME_M15",
    "20m": "TIMEFRAME_M20",
    "30m": "TIMEFRAME_M30",
    "1h": "TIMEFRAME_H1",
    "2h": "TIMEFRAME_H2",
    "3h": "TIMEFRAME_H3",
    "4h": "TIMEFRAME_H4",
    "1d": "TIMEFRAME_D1",
}


@dataclass
class BarrierData:
    times: np.ndarray
    ohlc: np.ndarray
    spread: np.ndarray
    labels: np.ndarray
    valid: np.ndarray
    session: np.ndarray
    point_size: float


def model_output_dim(args: argparse.Namespace) -> int:
    return 4 if getattr(args, "target", "") == "move4" else 1


FEATURE_SETS = {
    "ohlc4": ["open", "high", "low", "close"],
    "ohlc12": [
        "open",
        "high",
        "low",
        "close",
        "body",
        "range",
        "upper_wick",
        "lower_wick",
        "close_change",
        "hl_position",
        "spread",
        "session",
    ],
}


def is_gold_pair(pair: str) -> bool:
    return pair.upper().startswith("XAU")


def default_barriers_for_pair(pair: str) -> list[float]:
    return [float(x) for x in (GOLD_BARRIERS if is_gold_pair(pair) else FX_BARRIERS)]


def default_eval_tp_for_pair(pair: str) -> list[float]:
    return [float(x) for x in (GOLD_EVAL_TP if is_gold_pair(pair) else FX_EVAL_TP)]


def default_eval_sl_for_pair(pair: str) -> list[float]:
    return [float(x) for x in (GOLD_EVAL_SL if is_gold_pair(pair) else FX_EVAL_SL)]


def apply_date_window(args: argparse.Namespace) -> None:
    if args.start and args.to:
        return
    end = pd.Timestamp.utcnow().floor("D")
    days_value = str(args.days).strip().lower()
    if days_value in {"max", "all"}:
        # MT5 returns only locally/broker-available history; this is just a broad request.
        start = pd.Timestamp("2000-01-01", tz="UTC")
    else:
        days = max(1.0, float(args.days))
        start = end - pd.Timedelta(days=days)
    args.start = args.start or start.date().isoformat()
    args.to = args.to or end.date().isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pandas_freq(timeframe: str) -> str:
    tf = timeframe.strip().lower()
    if tf.endswith("m"):
        return f"{float(tf[:-1]):g}min"
    if tf.endswith("s"):
        return f"{float(tf[:-1]):g}s"
    if tf.endswith("h"):
        return f"{float(tf[:-1]):g}h"
    return timeframe


def build_bid_ask_ohlc(ticks: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    ts = pd.to_datetime(ticks["timestamp"], utc=True)
    bucket = pd.DatetimeIndex(ts).floor(pandas_freq(timeframe))
    df = pd.DataFrame({"bid": ticks["bid"].to_numpy(), "ask": ticks["ask"].to_numpy()}, index=bucket)
    out = df.groupby(level=0).agg(
        open=("bid", "first"),
        high=("bid", "max"),
        low=("bid", "min"),
        close=("bid", "last"),
        ask_close=("ask", "last"),
        spread=("ask", lambda x: float(np.nanmean(x.to_numpy()))),
    )
    # spread currently holds average ask; convert to average ask-bid with a second groupby.
    spread = df.assign(spread=df["ask"] - df["bid"]).groupby(level=0)["spread"].mean()
    out["spread"] = spread
    return out.dropna().reset_index(names="time")


def load_native_mt5_ohlc(args: argparse.Namespace, point_size: float) -> pd.DataFrame:
    tf_name = args.timeframe.strip().lower()
    if tf_name not in MT5_TIMEFRAMES:
        raise SystemExit(
            f"--ohlc-source native does not support timeframe={args.timeframe}; "
            "use MT5-native frames like 1m,3m,5m,15m,1h or use --ohlc-source ticks"
        )
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise SystemExit("MetaTrader5 package missing. Run: pip install MetaTrader5") from exc

    start = pd.to_datetime(args.start, utc=True, format="mixed").to_pydatetime()
    end = pd.to_datetime(args.to, utc=True, format="mixed").to_pydatetime()
    if not mt5.initialize():
        raise SystemExit(f"mt5.initialize failed: {mt5.last_error()}")
    try:
        info = mt5.symbol_info(args.pair)
        if info is None:
            raise SystemExit(f"symbol not found in MT5: {args.pair}")
        if not info.visible and not mt5.symbol_select(args.pair, True):
            raise SystemExit(f"could not select symbol: {args.pair}")
        tf = getattr(mt5, MT5_TIMEFRAMES[tf_name])
        print(
            f"[ml] native_ohlc symbol={args.pair} tf={tf_name} "
            f"from={start.isoformat()} to={end.isoformat()} "
            f"digits={info.digits} point={info.point} spread={info.spread}",
            flush=True,
        )
        chunks = []
        chunk_days = max(float(getattr(args, "native_chunk_days", 30.0)), 1.0)
        cur = pd.Timestamp(start).to_pydatetime()
        while cur < end:
            nxt = min((pd.Timestamp(cur) + pd.Timedelta(days=chunk_days)).to_pydatetime(), end)
            rates_part = mt5.copy_rates_range(args.pair, tf, cur, nxt)
            if rates_part is None:
                err = mt5.last_error()
                print(f"[ml] native_ohlc chunk failed {cur.isoformat()} -> {nxt.isoformat()} err={err}", flush=True)
            elif len(rates_part):
                chunks.append(pd.DataFrame(rates_part))
            cur = nxt
        if not chunks:
            raise SystemExit(f"copy_rates_range failed/no chunks: {mt5.last_error()}")
    finally:
        mt5.shutdown()

    raw = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time")
    if raw.empty:
        raise SystemExit("[ml] native_ohlc returned no bars")
    out = pd.DataFrame({
        "time": pd.to_datetime(raw["time"], unit="s", utc=True),
        "open": raw["open"].astype(float),
        "high": raw["high"].astype(float),
        "low": raw["low"].astype(float),
        "close": raw["close"].astype(float),
        "spread": raw["spread"].astype(float) * point_size,
    })
    return out.dropna().sort_values("time").reset_index(drop=True)


def label_barriers(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    barrier_points: float,
    horizon: int,
    point_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    if njit is not None:
        return _label_barriers_numba(close, high, low, barrier_points, horizon, point_size)
    labels = np.full(len(close), -1, dtype=np.int8)
    valid = np.zeros(len(close), dtype=np.bool_)
    dist = barrier_points * point_size
    for i in range(len(close) - horizon - 1):
        base = close[i]
        up = base + dist
        down = base - dist
        hit = -1
        for j in range(i + 1, i + horizon + 1):
            up_hit = high[j] >= up
            down_hit = low[j] <= down
            if up_hit and down_hit:
                # Intrabar order is unknowable from OHLC; skip ambiguous samples.
                hit = -1
                break
            if up_hit:
                hit = 1
                break
            if down_hit:
                hit = 0
                break
        if hit >= 0:
            labels[i] = hit
            valid[i] = True
    return labels, valid


def label_trade_outcome(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    tp_points: float,
    sl_points: float,
    horizon: int,
    point_size: float,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    if njit is not None:
        side_code = 1 if side == "long" else -1
        return _label_trade_outcome_numba(close, high, low, tp_points, sl_points, horizon, point_size, side_code)
    labels = np.full(len(close), -1, dtype=np.int8)
    valid = np.zeros(len(close), dtype=np.bool_)
    tp = tp_points * point_size
    sl = sl_points * point_size
    for i in range(len(close) - horizon - 1):
        base = close[i]
        if side == "long":
            win_px = base + tp
            loss_px = base - sl
        else:
            win_px = base - tp
            loss_px = base + sl
        hit = -1
        for j in range(i + 1, i + horizon + 1):
            if side == "long":
                win_hit = high[j] >= win_px
                loss_hit = low[j] <= loss_px
            else:
                win_hit = low[j] <= win_px
                loss_hit = high[j] >= loss_px
            if win_hit and loss_hit:
                hit = -1
                break
            if win_hit:
                hit = 1
                break
            if loss_hit:
                hit = 0
                break
        if hit >= 0:
            labels[i] = hit
            valid[i] = True
    return labels, valid


def label_tpsl_direction(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    tp_points: float,
    sl_points: float,
    horizon: int,
    point_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    if njit is not None:
        return _label_tpsl_direction_numba(close, high, low, tp_points, sl_points, horizon, point_size)
    labels = np.full(len(close), -1, dtype=np.int8)
    valid = np.zeros(len(close), dtype=np.bool_)
    tp = tp_points * point_size
    sl = sl_points * point_size
    for i in range(len(close) - horizon - 1):
        base = close[i]
        long_tp = base + tp
        long_sl = base - sl
        short_tp = base - tp
        short_sl = base + sl
        hit = -1
        for j in range(i + 1, i + horizon + 1):
            long_win = high[j] >= long_tp
            long_loss = low[j] <= long_sl
            short_win = low[j] <= short_tp
            short_loss = high[j] >= short_sl
            if (long_win and long_loss) or (short_win and short_loss) or (long_win and short_win):
                hit = -1
                break
            if long_win:
                hit = 1
                break
            if short_win:
                hit = 0
                break
        if hit >= 0:
            labels[i] = hit
            valid[i] = True
    return labels, valid


def label_move4(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    horizon: int,
    point_size: float,
    scale_points: float,
) -> tuple[np.ndarray, np.ndarray]:
    if njit is not None:
        return _label_move4_numba(close, high, low, horizon, point_size, scale_points)
    labels = np.zeros((len(close), 4), dtype=np.float32)
    valid = np.zeros(len(close), dtype=np.bool_)
    scale = max(float(scale_points), 1.0)
    stop = len(close) - horizon - 1
    for i in range(stop):
        base = close[i]
        fut_high = np.max(high[i + 1:i + horizon + 1])
        fut_low = np.min(low[i + 1:i + horizon + 1])
        max_up = max((fut_high - base) / point_size, 0.0)
        max_down = max((base - fut_low) / point_size, 0.0)
        long_label = 1.0 if max_up > max_down else 0.0
        short_label = 1.0 if max_down > max_up else 0.0
        labels[i, 0] = long_label
        labels[i, 1] = short_label
        labels[i, 2] = np.float32(min(max_up / scale, 10.0))
        labels[i, 3] = np.float32(min(max_down / scale, 10.0))
        valid[i] = max_up > 0.0 or max_down > 0.0
    return labels, valid


def label_nextbar(close: np.ndarray, horizon: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Binary target: future close is above current close."""
    h = max(1, int(horizon))
    labels = np.zeros(len(close), dtype=np.float32)
    valid = np.zeros(len(close), dtype=np.bool_)
    stop = len(close) - h
    if stop > 0:
        labels[:stop] = (close[h:] > close[:stop]).astype(np.float32)
        valid[:stop] = True
    return labels, valid


if njit is not None:
    @njit(cache=True)
    def _label_move4_numba(
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        horizon: int,
        point_size: float,
        scale_points: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        labels = np.zeros((len(close), 4), dtype=np.float32)
        valid = np.zeros(len(close), dtype=np.bool_)
        scale = scale_points
        if scale < 1.0:
            scale = 1.0
        stop = len(close) - horizon - 1
        for i in range(stop):
            base = close[i]
            fut_high = high[i + 1]
            fut_low = low[i + 1]
            for j in range(i + 2, i + horizon + 1):
                if high[j] > fut_high:
                    fut_high = high[j]
                if low[j] < fut_low:
                    fut_low = low[j]
            max_up = (fut_high - base) / point_size
            if max_up < 0.0:
                max_up = 0.0
            max_down = (base - fut_low) / point_size
            if max_down < 0.0:
                max_down = 0.0
            labels[i, 0] = 1.0 if max_up > max_down else 0.0
            labels[i, 1] = 1.0 if max_down > max_up else 0.0
            up_scaled = max_up / scale
            down_scaled = max_down / scale
            labels[i, 2] = np.float32(10.0 if up_scaled > 10.0 else up_scaled)
            labels[i, 3] = np.float32(10.0 if down_scaled > 10.0 else down_scaled)
            valid[i] = max_up > 0.0 or max_down > 0.0
        return labels, valid

    @njit(cache=True)
    def _label_barriers_numba(
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        barrier_points: float,
        horizon: int,
        point_size: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        labels = np.full(len(close), -1, dtype=np.int8)
        valid = np.zeros(len(close), dtype=np.bool_)
        dist = barrier_points * point_size
        stop = len(close) - horizon - 1
        for i in range(stop):
            base = close[i]
            up = base + dist
            down = base - dist
            hit = -1
            for j in range(i + 1, i + horizon + 1):
                up_hit = high[j] >= up
                down_hit = low[j] <= down
                if up_hit and down_hit:
                    hit = -1
                    break
                if up_hit:
                    hit = 1
                    break
                if down_hit:
                    hit = 0
                    break
            if hit >= 0:
                labels[i] = hit
                valid[i] = True
        return labels, valid

    @njit(cache=True)
    def _label_trade_outcome_numba(
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        tp_points: float,
        sl_points: float,
        horizon: int,
        point_size: float,
        side_code: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        labels = np.full(len(close), -1, dtype=np.int8)
        valid = np.zeros(len(close), dtype=np.bool_)
        tp = tp_points * point_size
        sl = sl_points * point_size
        stop = len(close) - horizon - 1
        for i in range(stop):
            base = close[i]
            if side_code == 1:
                win_px = base + tp
                loss_px = base - sl
            else:
                win_px = base - tp
                loss_px = base + sl
            hit = -1
            for j in range(i + 1, i + horizon + 1):
                if side_code == 1:
                    win_hit = high[j] >= win_px
                    loss_hit = low[j] <= loss_px
                else:
                    win_hit = low[j] <= win_px
                    loss_hit = high[j] >= loss_px
                if win_hit and loss_hit:
                    hit = -1
                    break
                if win_hit:
                    hit = 1
                    break
                if loss_hit:
                    hit = 0
                    break
            if hit >= 0:
                labels[i] = hit
                valid[i] = True
        return labels, valid

    @njit(cache=True)
    def _label_tpsl_direction_numba(
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        tp_points: float,
        sl_points: float,
        horizon: int,
        point_size: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        labels = np.full(len(close), -1, dtype=np.int8)
        valid = np.zeros(len(close), dtype=np.bool_)
        tp = tp_points * point_size
        sl = sl_points * point_size
        stop = len(close) - horizon - 1
        for i in range(stop):
            base = close[i]
            long_tp = base + tp
            long_sl = base - sl
            short_tp = base - tp
            short_sl = base + sl
            hit = -1
            for j in range(i + 1, i + horizon + 1):
                long_win = high[j] >= long_tp
                long_loss = low[j] <= long_sl
                short_win = low[j] <= short_tp
                short_loss = high[j] >= short_sl
                if (long_win and long_loss) or (short_win and short_loss) or (long_win and short_win):
                    hit = -1
                    break
                if long_win:
                    hit = 1
                    break
                if short_win:
                    hit = 0
                    break
            if hit >= 0:
                labels[i] = hit
                valid[i] = True
        return labels, valid


def make_time_features(times: np.ndarray) -> np.ndarray:
    dt = pd.to_datetime(times, utc=True)
    mins = dt.hour.to_numpy(np.float32) * 60.0 + dt.minute.to_numpy(np.float32)
    dow = dt.dayofweek.to_numpy(np.float32)
    tod = 2.0 * np.pi * mins / 1440.0
    week = 2.0 * np.pi * dow / 7.0
    return np.column_stack([
        np.sin(tod),
        np.cos(tod),
        np.sin(week),
        np.cos(week),
    ]).astype(np.float32)


def feature_names(feature_set: str) -> list[str]:
    key = feature_set.lower()
    if key not in FEATURE_SETS:
        raise ValueError(f"unknown feature set: {feature_set}")
    return FEATURE_SETS[key]


def make_window_features(data: BarrierData, idx: int, window: int, barrier_points: float, feature_set: str) -> np.ndarray:
    w = data.ohlc[idx - window + 1:idx + 1].astype(np.float32)
    scale = np.float32(max(data.point_size * barrier_points, 1e-12))
    ref = np.float32(w[-1, 3])
    open_ = (w[:, 0] - ref) / scale
    high = (w[:, 1] - ref) / scale
    low = (w[:, 2] - ref) / scale
    close = (w[:, 3] - ref) / scale
    if feature_set == "ohlc4":
        return np.clip(np.column_stack([open_, high, low, close]), -10.0, 10.0).T.astype(np.float32)
    body = (w[:, 3] - w[:, 0]) / scale
    rng_raw = np.maximum(w[:, 1] - w[:, 2], data.point_size)
    rng = rng_raw / scale
    upper_wick = (w[:, 1] - np.maximum(w[:, 0], w[:, 3])) / scale
    lower_wick = (np.minimum(w[:, 0], w[:, 3]) - w[:, 2]) / scale
    prev_close = np.concatenate(([w[0, 3]], w[:-1, 3]))
    close_change = (w[:, 3] - prev_close) / scale
    hl_position = (w[:, 3] - w[:, 2]) / rng_raw
    spread = data.spread[idx - window + 1:idx + 1].astype(np.float32) / scale
    session = data.session[idx - window + 1:idx + 1].astype(np.float32)
    features = np.column_stack([
        open_,
        high,
        low,
        close,
        body,
        rng,
        upper_wick,
        lower_wick,
        close_change,
        hl_position,
        spread,
        session,
    ])
    return np.clip(features, -10.0, 10.0).T.astype(np.float32)


def load_barrier_data(args: argparse.Namespace) -> BarrierData:
    point_size = float(args.point_size or default_point_size(args.pair))
    if args.ohlc_source == "native":
        candles = load_native_mt5_ohlc(args, point_size)
        print(
            f"[ml] native_range={candles['time'].iloc[0]} -> {candles['time'].iloc[-1]} "
            f"candles={len(candles):,}",
            flush=True,
        )
    else:
        ticks, _ = load_market(args)
        if ticks["pair"].nunique() > 1:
            ticks = ticks[ticks["pair"] == args.pair]
        ticks = ticks.sort_values("timestamp").reset_index(drop=True)
        print(
            f"[ml] tick_range={ticks['timestamp'].iloc[0]} -> {ticks['timestamp'].iloc[-1]} "
            f"ticks={len(ticks):,}",
            flush=True,
        )
        candles = build_bid_ask_ohlc(ticks, args.timeframe)
    ohlc = candles[["open", "high", "low", "close"]].to_numpy(np.float32)
    labels, valid = label_barriers(
        candles["close"].to_numpy(np.float64),
        candles["high"].to_numpy(np.float64),
        candles["low"].to_numpy(np.float64),
        float(args.barrier_points),
        args.horizon_bars,
        point_size,
    )
    ts_ns = candles["time"].astype("int64").to_numpy(np.int64)
    session = active_session_allowed(ts_ns, 1).astype(np.float32)
    return BarrierData(
        times=candles["time"].to_numpy(),
        ohlc=ohlc,
        spread=candles["spread"].to_numpy(np.float32),
        labels=labels,
        valid=valid,
        session=session,
        point_size=point_size,
    )


class BarrierDataset(Dataset):
    def __init__(
        self,
        data: BarrierData,
        indices: np.ndarray,
        window: int,
        barrier_points: float,
        feature_set: str = "ohlc4",
        use_session_feature: bool = False,
    ):
        self.data = data
        self.indices = indices.astype(np.int64)
        self.window = int(window)
        self.barrier_points = float(barrier_points)
        self.feature_set = feature_set.lower()
        self.use_session_feature = bool(use_session_feature)
        self.time_feat = make_time_features(data.times)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, n: int):
        i = int(self.indices[n])
        x = make_window_features(self.data, i, self.window, self.barrier_points, self.feature_set)
        extra_values = [
            self.data.spread[i] / (self.data.point_size * self.barrier_points),
            *self.time_feat[i],
        ]
        if self.use_session_feature:
            extra_values.insert(1, self.data.session[i])
        extra = np.array(extra_values, dtype=np.float32)
        y = np.float32(self.data.labels[i])
        return torch.from_numpy(x), torch.from_numpy(extra), torch.tensor(y)


class PrecomputedBarrierDataset(Dataset):
    def __init__(
        self,
        data: BarrierData,
        indices: np.ndarray,
        window: int,
        barrier_points: float,
        feature_set: str = "ohlc4",
        use_session_feature: bool = False,
    ):
        self.indices = indices.astype(np.int64)
        self.window = int(window)
        self.input_dim = len(feature_names(feature_set))
        self.x = np.empty((len(self.indices), self.input_dim, self.window), dtype=np.float32)
        extra_dim = 6 if use_session_feature else 5
        self.extra = np.empty((len(self.indices), extra_dim), dtype=np.float32)
        self.y = np.empty((len(self.indices),) + np.shape(data.labels[0]), dtype=np.float32)
        time_feat = make_time_features(data.times)
        scale = max(data.point_size * float(barrier_points), 1e-12)
        for row, i_raw in enumerate(self.indices):
            i = int(i_raw)
            self.x[row] = make_window_features(data, i, self.window, barrier_points, feature_set)
            extra_values = [
                data.spread[i] / scale,
                *time_feat[i],
            ]
            if use_session_feature:
                extra_values.insert(1, data.session[i])
            self.extra[row] = np.asarray(extra_values, dtype=np.float32)
            self.y[row] = np.asarray(data.labels[i], dtype=np.float32)
        self.x = torch.from_numpy(self.x)
        self.extra = torch.from_numpy(self.extra)
        self.y = torch.from_numpy(self.y)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, n: int):
        return self.x[n], self.extra[n], self.y[n]


class CandleCNN(nn.Module):
    def __init__(self, window: int, input_dim: int, extra_dim: int = 6, channels: int = 64, kernel: int = 5, dropout: float = 0.15, output_dim: int = 1):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, channels // 2, kernel_size=kernel, padding=pad),
            nn.ReLU(),
            nn.BatchNorm1d(channels // 2),
            nn.Conv1d(channels // 2, channels, kernel_size=kernel, padding=pad),
            nn.ReLU(),
            nn.BatchNorm1d(channels),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(channels + extra_dim, channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels, output_dim),
        )

    def forward(self, x: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        z = self.conv(x).squeeze(-1)
        z = torch.cat([z, extra], dim=1)
        out = self.head(z)
        return out.squeeze(1) if out.shape[1] == 1 else out


class CausalChomp1d(nn.Module):
    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = int(chomp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp] if self.chomp > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=pad, dilation=dilation),
            CausalChomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel, padding=pad, dilation=dilation),
            CausalChomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.down(x)


class TemporalConvNet(nn.Module):
    def __init__(self, window: int, input_dim: int, extra_dim: int, channels: int = 64, kernel: int = 3, layers: int = 5, dropout: float = 0.15, output_dim: int = 1):
        super().__init__()
        del window
        blocks = []
        in_ch = input_dim
        for i in range(layers):
            blocks.append(TemporalBlock(in_ch, channels, kernel, 2 ** i, dropout))
            in_ch = channels
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(channels + extra_dim, channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels, output_dim),
        )

    def forward(self, x: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        z = self.tcn(x)[:, :, -1]
        out = self.head(torch.cat([z, extra], dim=1))
        return out.squeeze(1) if out.shape[1] == 1 else out


class FlatMLP(nn.Module):
    def __init__(self, window: int, input_dim: int, extra_dim: int, hidden: int = 256, dropout: float = 0.2, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window * input_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden // 2 + extra_dim, output_dim)

    def forward(self, x: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        out = self.head(torch.cat([z, extra], dim=1))
        return out.squeeze(1) if out.shape[1] == 1 else out


class LinearBaseline(nn.Module):
    def __init__(self, window: int, input_dim: int, extra_dim: int, output_dim: int = 1):
        super().__init__()
        self.head = nn.Linear(window * input_dim + extra_dim, output_dim)

    def forward(self, x: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x.flatten(1), extra], dim=1)
        out = self.head(z)
        return out.squeeze(1) if out.shape[1] == 1 else out


class RecurrentModel(nn.Module):
    def __init__(self, window: int, input_dim: int, extra_dim: int, kind: str, hidden: int = 96, layers: int = 2, dropout: float = 0.15, output_dim: int = 1):
        super().__init__()
        del window
        rnn_cls = nn.GRU if kind == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=input_dim, hidden_size=hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden + extra_dim, 96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, output_dim),
        )

    def forward(self, x: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        seq = x.transpose(1, 2)
        out, _ = self.rnn(seq)
        z = out[:, -1, :]
        out = self.head(torch.cat([z, extra], dim=1))
        return out.squeeze(1) if out.shape[1] == 1 else out


class TransformerModel(nn.Module):
    def __init__(self, window: int, input_dim: int, extra_dim: int, d_model: int = 64, heads: int = 4, layers: int = 2, dropout: float = 0.15, output_dim: int = 1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, window, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=160,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.Linear(d_model + extra_dim, 96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, output_dim),
        )

    def forward(self, x: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        seq = x.transpose(1, 2)
        z = self.input_proj(seq) + self.pos[:, :seq.shape[1], :]
        z = self.encoder(z).mean(dim=1)
        out = self.head(torch.cat([z, extra], dim=1))
        return out.squeeze(1) if out.shape[1] == 1 else out


def build_model(name: str, window: int, input_dim: int, extra_dim: int, args: argparse.Namespace) -> nn.Module:
    model_name = name.lower()
    out_dim = model_output_dim(args)
    if model_name == "cnn":
        return CandleCNN(window, input_dim=input_dim, extra_dim=extra_dim, channels=args.channels, kernel=args.kernel_size, dropout=args.dropout, output_dim=out_dim)
    if model_name in {"tcn", "tcn2"}:
        return TemporalConvNet(window, input_dim=input_dim, extra_dim=extra_dim, channels=args.channels, kernel=args.kernel_size, layers=args.layers, dropout=args.dropout, output_dim=out_dim)
    if model_name == "mlp":
        return FlatMLP(window, input_dim=input_dim, extra_dim=extra_dim, hidden=args.hidden, dropout=args.dropout, output_dim=out_dim)
    if model_name == "linear":
        return LinearBaseline(window, input_dim=input_dim, extra_dim=extra_dim, output_dim=out_dim)
    if model_name in {"gru", "lstm"}:
        return RecurrentModel(window, input_dim=input_dim, extra_dim=extra_dim, kind=model_name, hidden=args.hidden, layers=args.layers, dropout=args.dropout, output_dim=out_dim)
    if model_name == "transformer":
        return TransformerModel(window, input_dim=input_dim, extra_dim=extra_dim, d_model=args.channels, heads=args.heads, layers=args.layers, dropout=args.dropout, output_dim=out_dim)
    raise ValueError(f"unknown model: {name}")


def split_indices(data: BarrierData, window: int, train_frac: float, max_samples: int, seed: int):
    idx = np.flatnonzero(data.valid)
    idx = idx[idx >= window - 1]
    if max_samples > 0 and len(idx) > max_samples:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=max_samples, replace=False))
    cut_time_idx = int(len(data.ohlc) * train_frac)
    train_idx = idx[idx < cut_time_idx]
    test_idx = idx[idx >= cut_time_idx]
    return train_idx, test_idx


def train_model(args: argparse.Namespace, data: BarrierData, train_idx: np.ndarray, test_idx: np.ndarray):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    if args.verbose:
        print(f"[ml] device={device} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}", flush=True)
    ds_cls = BarrierDataset if args.no_precompute_features else PrecomputedBarrierDataset
    prep_t0 = time.time()
    train_ds = ds_cls(data, train_idx, args.window, args.barrier_points, args.feature_set, args.session_feature)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    eval_idx = test_idx if len(test_idx) else train_idx
    eval_ds = train_ds if len(test_idx) == 0 and not args.no_precompute_features else ds_cls(data, eval_idx, args.window, args.barrier_points, args.feature_set, args.session_feature)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    if args.verbose:
        mode = "lazy" if args.no_precompute_features else "precomputed"
        print(f"[ml] dataset={mode} train={len(train_idx):,} eval={len(eval_idx):,} prep={time.time() - prep_t0:.1f}s", flush=True)
    extra_dim = 6 if args.session_feature else 5
    input_dim = len(feature_names(args.feature_set))
    model = build_model(args.model, args.window, input_dim, extra_dim, args).to(device)
    if args.verbose:
        print(
            f"[ml] model={args.model} features={args.feature_set}/{input_dim} "
            f"channels={args.channels} kernel={args.kernel_size} layers={args.layers} "
            f"dropout={args.dropout:g} params={sum(p.numel() for p in model.parameters()):,}",
            flush=True,
        )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    reg_loss_fn = nn.SmoothL1Loss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for x, extra, y in train_loader:
            x = x.to(device)
            extra = extra.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x, extra)
            if args.target == "move4":
                loss = loss_fn(logits[:, :2], y[:, :2]) + args.move_reg_weight * reg_loss_fn(torch.relu(logits[:, 2:]), y[:, 2:])
            else:
                loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(y)
            seen += len(y)
        train_loss = total_loss / max(seen, 1)
        acc, bce = evaluate_model(model, eval_loader, device, args)
        if args.verbose or epoch == args.epochs:
            eval_name = "test" if len(test_idx) else "train"
            print(f"[ml] epoch {epoch:02d} train_bce={train_loss:.4f} {eval_name}_bce={bce:.4f} {eval_name}_acc={acc*100:.2f}%", flush=True)
    return model, device


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad():
        for x, extra, y in loader:
            x = x.to(device)
            extra = extra.to(device)
            y = y.to(device)
            logits = model(x, extra)
            if args.target == "move4":
                loss += float(loss_fn(logits[:, :2], y[:, :2]).item())
                pred = torch.argmax(logits[:, :2], dim=1)
                truth = torch.argmax(y[:, :2], dim=1)
                correct += int((pred == truth).sum().item())
            else:
                loss += float(loss_fn(logits, y).item())
                pred = (torch.sigmoid(logits) >= 0.5).float()
                correct += int((pred == y).sum().item())
            total += len(y)
    return correct / max(total, 1), loss / max(total, 1)


def predict_probs(model: nn.Module, device: torch.device, data: BarrierData, idx: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    ds = BarrierDataset(data, idx, args.window, args.barrier_points, args.feature_set, args.session_feature)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    out = []
    model.eval()
    with torch.no_grad():
        for x, extra, _ in loader:
            raw = model(x.to(device), extra.to(device))
            if args.target == "move4":
                p_dir = torch.sigmoid(raw[:, :2])
                moves = torch.relu(raw[:, 2:]) * float(args.move_scale_points)
                p = torch.cat([p_dir, moves], dim=1).cpu().numpy()
            else:
                p = torch.sigmoid(raw).cpu().numpy()
            out.append(p)
    return np.concatenate(out) if out else np.array([], dtype=np.float32)


def make_flat_features(data: BarrierData, idx: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    time_feat = make_time_features(data.times)
    extra_dim = 6 if args.session_feature else 5
    input_dim = len(feature_names(args.feature_set))
    out = np.empty((len(idx), args.window * input_dim + extra_dim), dtype=np.float32)
    scale = data.point_size * args.barrier_points
    for row, i_raw in enumerate(idx):
        i = int(i_raw)
        x = make_window_features(data, i, args.window, args.barrier_points, args.feature_set).reshape(-1)
        extra_values = [
            data.spread[i] / scale,
            *time_feat[i],
        ]
        if args.session_feature:
            extra_values.insert(1, data.session[i])
        out[row, :args.window * input_dim] = x
        out[row, args.window * input_dim:] = np.asarray(extra_values, dtype=np.float32)
    return out


def train_tree_model(args: argparse.Namespace, data: BarrierData, train_idx: np.ndarray, test_idx: np.ndarray):
    x_train = make_flat_features(data, train_idx, args)
    y_train = data.labels[train_idx].astype(np.int32)
    eval_idx = test_idx if len(test_idx) else train_idx
    x_test = make_flat_features(data, eval_idx, args)
    if args.model == "rf":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=args.trees,
            max_depth=args.tree_depth if args.tree_depth > 0 else None,
            min_samples_leaf=args.min_leaf,
            n_jobs=args.tree_jobs,
            random_state=args.seed,
            class_weight="balanced_subsample",
        )
    elif args.model == "xgb":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=args.trees,
            max_depth=args.tree_depth if args.tree_depth > 0 else 5,
            learning_rate=args.xgb_lr,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=args.tree_jobs,
            random_state=args.seed,
        )
    else:
        raise ValueError(f"not a tree model: {args.model}")
    print(f"[ml] model={args.model} train_matrix={x_train.shape} trees={args.trees}", flush=True)
    model.fit(x_train, y_train)
    probs = model.predict_proba(x_test)[:, 1].astype(np.float32)
    acc = float(np.mean((probs >= 0.5).astype(np.int8) == data.labels[eval_idx]))
    eval_name = "test" if len(test_idx) else "train"
    print(f"[ml] model={args.model} {eval_name}_acc={acc*100:.2f}%", flush=True)
    return model, probs


def backtest_thresholds(data: BarrierData, idx: np.ndarray, probs: np.ndarray, args: argparse.Namespace) -> list[dict]:
    if args.target == "move4":
        return []
    rows = []
    # This evaluates trades only on labelled samples. It is a model-quality test,
    # not yet a full live simulator over every candle.
    labels = data.labels[idx]
    for th in parse_num_list(args.thresholds, DEFAULT_THRESHOLDS):
        cash = args.amount
        peak = args.amount
        trades = wins = losses = 0
        gross_win = gross_loss = 0.0
        for sample_idx, p, y in zip(idx, probs, labels):
            side = 0
            if args.target == "trade":
                if p >= th:
                    side = 1 if args.trade_side == "long" else -1
            else:
                if p >= th:
                    side = 1
                elif p <= 1.0 - th:
                    side = -1
            if side == 0:
                continue
            margin = cash if args.compound else args.amount
            entry_px = float(data.ohlc[int(sample_idx), 3])
            pnl = args.barrier_points * data.point_size * ((margin * args.leverage) / max(entry_px, 1e-9))
            fee = (margin * args.leverage) / 1_000_000.0 * args.commission_per_million * 2.0
            if args.target == "trade":
                correct = bool(y == 1)
            else:
                correct = (side == 1 and y == 1) or (side == -1 and y == 0)
            trade_pnl = pnl - fee if correct else -pnl - fee
            cash += trade_pnl
            peak = max(peak, cash)
            trades += 1
            if trade_pnl >= 0:
                wins += 1
                gross_win += trade_pnl
            else:
                losses += 1
                gross_loss += -trade_pnl
        rows.append({
            "threshold": th,
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "wr": wins / trades * 100.0 if trades else 0.0,
            "pf": gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0),
            "pnl": cash - args.amount,
            "dd": peak - cash,
        })
    return rows


def moving_average(values: np.ndarray, length: int) -> np.ndarray:
    if length <= 1:
        return values.astype(np.float32, copy=False)
    s = pd.Series(values.astype(np.float64))
    return s.rolling(length, min_periods=1).mean().to_numpy(np.float32)


def backtest_external_probability_signals(
    data: BarrierData,
    idx: np.ndarray,
    probs: np.ndarray,
    args: argparse.Namespace,
) -> list[dict]:
    rows = []
    if len(idx) == 0:
        return rows
    tp_values = parse_num_list(args.eval_tp_points, default_eval_tp_for_pair(args.pair))
    sl_values = parse_num_list(args.eval_sl_points, default_eval_sl_for_pair(args.pair))
    sessions = [int(x) for x in parse_num_list(args.eval_sessions, [-1, 0, 1, 2])]
    ma_values = parse_int_list(args.prob_ma, [1])
    thresholds = parse_num_list(args.thresholds, DEFAULT_THRESHOLDS)
    ts_ns = pd.to_datetime(data.times, utc=True).astype("int64").to_numpy(np.int64)
    session_cache = {s: active_session_allowed(ts_ns, s) for s in sessions}
    high = data.ohlc[:, 1].astype(np.float64)
    low = data.ohlc[:, 2].astype(np.float64)
    close = data.ohlc[:, 3].astype(np.float64)
    for ma_len in ma_values:
        if probs.ndim == 2:
            p_smooth = np.column_stack([moving_average(probs[:, col], ma_len) for col in range(probs.shape[1])])
        else:
            p_smooth = moving_average(probs, ma_len)
        for th in thresholds:
            for tp_points in tp_values:
                for sl_points in sl_values:
                    tp = float(tp_points) * data.point_size
                    sl = float(sl_points) * data.point_size
                    if tp <= 0.0 or sl <= 0.0:
                        continue
                    for sess in sessions:
                        allowed = session_cache[sess]
                        cash = args.amount
                        peak = args.amount
                        trades = wins = losses = 0
                        gross_win = gross_loss = 0.0
                        worst = 0.0
                        k = 0
                        while k < len(idx):
                            bar = int(idx[k])
                            if not allowed[bar]:
                                k += 1
                                continue
                            p = p_smooth[k]
                            side = 0
                            if probs.ndim == 2:
                                long_prob, short_prob, exp_up, exp_down = map(float, p[:4])
                                if long_prob >= th and exp_up >= tp_points and exp_down <= sl_points:
                                    side = 1
                                elif short_prob >= th and exp_down >= tp_points and exp_up <= sl_points:
                                    side = -1
                            else:
                                p = float(p)
                                if p >= th:
                                    side = 1
                                elif p <= 1.0 - th:
                                    side = -1
                            if side == 0:
                                k += 1
                                continue
                            entry = close[bar]
                            exit_bar = min(bar + args.horizon_bars, len(close) - 1)
                            result_points = 0.0
                            for j in range(bar + 1, exit_bar + 1):
                                if side == 1:
                                    win_hit = high[j] >= entry + tp
                                    loss_hit = low[j] <= entry - sl
                                else:
                                    win_hit = low[j] <= entry - tp
                                    loss_hit = high[j] >= entry + sl
                                if win_hit and loss_hit:
                                    result_points = -sl_points
                                    exit_bar = j
                                    break
                                if win_hit:
                                    result_points = tp_points
                                    exit_bar = j
                                    break
                                if loss_hit:
                                    result_points = -sl_points
                                    exit_bar = j
                                    break
                            else:
                                result_points = (close[exit_bar] - entry) / data.point_size * side
                            margin = cash if args.compound else args.amount
                            notional = margin * args.leverage
                            pnl = result_points * data.point_size * (notional / max(entry, 1e-9))
                            fee = notional / 1_000_000.0 * args.commission_per_million * 2.0
                            trade_pnl = pnl - fee
                            cash += trade_pnl
                            peak = max(peak, cash)
                            trades += 1
                            worst = min(worst, trade_pnl)
                            if trade_pnl >= 0:
                                wins += 1
                                gross_win += trade_pnl
                            else:
                                losses += 1
                                gross_loss += -trade_pnl
                            while k < len(idx) and int(idx[k]) <= exit_bar:
                                k += 1
                        rows.append({
                            "threshold": th,
                            "tp": tp_points,
                            "sl": sl_points,
                            "session": sess,
                            "prob_ma": ma_len,
                            "trades": trades,
                            "wins": wins,
                            "losses": losses,
                            "wr": wins / trades * 100.0 if trades else 0.0,
                            "pf": gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0),
                            "pnl": cash - args.amount,
                            "dd": peak - cash,
                            "worst": worst,
                        })
    rows.sort(key=lambda r: (r["pnl"] / max(r["dd"], 1.0), r["pnl"], r["wr"]), reverse=True)
    return rows


def export_predictions(
    data: BarrierData,
    idx: np.ndarray,
    probs: np.ndarray,
    path: str,
    args: argparse.Namespace,
) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    o = data.ohlc[idx, 0].astype(np.float64)
    h = data.ohlc[idx, 1].astype(np.float64)
    l = data.ohlc[idx, 2].astype(np.float64)
    c = data.ohlc[idx, 3].astype(np.float64)
    prev = data.ohlc[np.maximum(idx - 1, 0), 3].astype(np.float64)
    point = data.point_size
    if args.target == "move4":
        side = np.where(probs[:, 0] >= probs[:, 1], "long", "short")
    elif args.target == "trade":
        side = np.full(len(probs), args.trade_side)
    else:
        side = np.where(probs >= 0.5, "long", "short")
    labels = data.labels[idx]
    if args.target == "move4":
        correct = np.argmax(probs[:, :2], axis=1) == np.argmax(labels[:, :2], axis=1)
    elif args.target == "trade":
        correct = labels == 1
    else:
        correct = ((probs >= 0.5) & (labels == 1)) | ((probs < 0.5) & (labels == 0))
    confidence = np.abs((probs[:, 0] if args.target == "move4" else probs) - 0.5) * 2.0
    window_start = idx - args.window + 1
    window_open = data.ohlc[window_start, 0].astype(np.float64)
    window_high = np.array([np.max(data.ohlc[s:i + 1, 1]) for s, i in zip(window_start, idx)], dtype=np.float64)
    window_low = np.array([np.min(data.ohlc[s:i + 1, 2]) for s, i in zip(window_start, idx)], dtype=np.float64)
    dt = pd.to_datetime(data.times[idx], utc=True)
    df = pd.DataFrame({
        "time": dt.astype(str),
        "prob_up": probs[:, 0] if args.target == "move4" else probs,
        "prob_short": probs[:, 1] if args.target == "move4" else 1.0 - probs,
        "expected_max_up_points": probs[:, 2] if args.target == "move4" else np.nan,
        "expected_max_down_points": probs[:, 3] if args.target == "move4" else np.nan,
        "prob_win": probs[:, 0] if args.target == "move4" else probs,
        "confidence": confidence,
        "model_side": side,
        "label": np.argmax(labels[:, :2], axis=1) if args.target == "move4" else labels,
        "actual_max_up_points": labels[:, 2] * float(args.move_scale_points) if args.target == "move4" else np.nan,
        "actual_max_down_points": labels[:, 3] * float(args.move_scale_points) if args.target == "move4" else np.nan,
        "correct": correct.astype(np.int8),
        "inside_session": data.session[idx].astype(np.int8),
        "hour_utc": dt.hour,
        "dow": dt.dayofweek,
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "spread_points": data.spread[idx] / point,
        "bar_body_points": (c - o) / point,
        "bar_range_points": (h - l) / point,
        "prev_close_move_points": (c - prev) / point,
        "window_return_points": (c - window_open) / point,
        "window_range_points": (window_high - window_low) / point,
        "window_pos": (c - window_low) / np.maximum(window_high - window_low, point),
    })
    df.to_csv(path, index=False)
    print(f"[ml] wrote predictions {path}", flush=True)


def print_diagnostics(data: BarrierData, idx: np.ndarray, probs: np.ndarray) -> None:
    if len(idx) == 0:
        return
    labels = data.labels[idx]
    if probs.ndim == 2:
        pred = np.argmax(probs[:, :2], axis=1)
        label_dir = np.argmax(labels[:, :2], axis=1)
        correct = pred == label_dir
        conf = np.abs(probs[:, 0] - probs[:, 1])
        label_up_rate = (label_dir == 0).astype(np.float32)
        avg_prob_series = probs[:, 0]
    else:
        pred = (probs >= 0.5).astype(np.int8)
        correct = pred == labels
        conf = np.abs(probs - 0.5) * 2.0
        label_up_rate = labels
        avg_prob_series = probs
    print("\n[ml] confidence diagnostics", flush=True)
    for lo, hi in [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.0)]:
        mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        print(
            f"conf {lo:.1f}-{hi:.1f} n={int(mask.sum()):6} "
            f"acc={float(np.mean(correct[mask]))*100:5.1f}% "
            f"up%={float(np.mean(label_up_rate[mask]))*100:5.1f}%",
            flush=True,
        )

    session = data.session[idx].astype(np.bool_)
    for name, mask in [("inside", session), ("outside", ~session)]:
        if np.any(mask):
            print(
                f"session {name:7} n={int(mask.sum()):6} "
                f"acc={float(np.mean(correct[mask]))*100:5.1f}% "
                f"avg_prob={float(np.mean(avg_prob_series[mask])):.3f} "
                f"up%={float(np.mean(label_up_rate[mask]))*100:5.1f}%",
                flush=True,
            )

    dt = pd.to_datetime(data.times[idx], utc=True)
    hours = dt.hour.to_numpy()
    hour_rows = []
    for hour in range(24):
        mask = hours == hour
        if int(mask.sum()) < 20:
            continue
        hour_rows.append((
            float(np.mean(correct[mask])),
            hour,
            int(mask.sum()),
            float(np.mean(avg_prob_series[mask])),
            float(np.mean(label_up_rate[mask])),
        ))
    hour_rows.sort(reverse=True)
    print("[ml] best UTC hours by raw accuracy", flush=True)
    for acc, hour, n, avg_prob, up_rate in hour_rows[:8]:
        print(f"hour={hour:02d} n={n:6} acc={acc*100:5.1f}% avg_prob={avg_prob:.3f} up%={up_rate*100:5.1f}%", flush=True)


def save_model(
    model: nn.Module,
    path: str,
    args: argparse.Namespace,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    data: BarrierData,
) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "args": vars(args),
        "point_size": data.point_size,
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "saved_at": pd.Timestamp.utcnow().isoformat(),
    }
    torch.save(payload, path)
    meta_path = os.path.splitext(path)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in payload.items() if k != "state_dict"}, f, indent=2, default=str)
    print(f"[ml] wrote model {path}", flush=True)
    print(f"[ml] wrote model meta {meta_path}", flush=True)


def save_any_model(model, path: str, args: argparse.Namespace, train_idx: np.ndarray, test_idx: np.ndarray, data: BarrierData) -> None:
    if args.model in {"rf", "xgb"}:
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "model": model,
            "args": vars(args),
            "point_size": data.point_size,
            "train_samples": int(len(train_idx)),
            "test_samples": int(len(test_idx)),
            "saved_at": pd.Timestamp.utcnow().isoformat(),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        meta_path = os.path.splitext(path)[0] + ".json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in payload.items() if k != "model"}, f, indent=2, default=str)
        print(f"[ml] wrote model {path}", flush=True)
        print(f"[ml] wrote model meta {meta_path}", flush=True)
    else:
        save_model(model, path, args, train_idx, test_idx, data)


def append_notes(path: str, args: argparse.Namespace, rows: list[dict], train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    best_pnl = max(rows, key=lambda r: (r["pnl"], r["pf"], r["wr"])) if rows else {}
    best_wr = max(rows, key=lambda r: (r["wr"], r["trades"], r["pnl"])) if rows else {}
    row = {
        "time": pd.Timestamp.utcnow().isoformat(),
        "pair": args.pair,
        "model": args.model,
        "target": args.target,
        "trade_side": args.trade_side,
        "timeframe": args.timeframe,
        "window": args.window,
        "feature_set": args.feature_set,
        "horizon": args.horizon_bars,
        "barrier_points": args.barrier_points,
        "label_tp_points": getattr(args, "label_tp_points", ""),
        "label_sl_points": getattr(args, "label_sl_points", ""),
        "label_session": getattr(args, "label_session", ""),
        "channels": args.channels,
        "kernel": args.kernel_size,
        "layers": args.layers,
        "dropout": args.dropout,
        "session_feature": int(args.session_feature),
        "train": len(train_idx),
        "test": len(test_idx),
        "best_pnl_th": best_pnl.get("threshold", ""),
        "best_pnl": best_pnl.get("pnl", ""),
        "best_pnl_wr": best_pnl.get("wr", ""),
        "best_pnl_pf": best_pnl.get("pf", ""),
        "best_pnl_trades": best_pnl.get("trades", ""),
        "best_wr_th": best_wr.get("threshold", ""),
        "best_wr": best_wr.get("wr", ""),
        "best_wr_pnl": best_wr.get("pnl", ""),
        "best_wr_trades": best_wr.get("trades", ""),
    }
    exists = os.path.exists(path)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not exists, index=False)
    print(f"[ml] appended notes {path}", flush=True)


def parse_int_list(value: str | None, default: list[int]) -> list[int]:
    if not value:
        return list(default)
    return [int(float(x.strip())) for x in value.split(",") if x.strip()]


def parse_str_list(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return list(default)
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return [x.strip() for x in str(value).split(",") if x.strip()]


def arch_combos_for_model(
    model_name: str,
    channels_values: list[int],
    kernel_values: list[int],
    layer_values: list[int],
    hidden_values: list[int],
    dropout_values: list[float],
    args: argparse.Namespace,
) -> list[tuple[int, int, int, int, float]]:
    name = model_name.lower()
    if name in {"rf", "xgb", "linear"}:
        return [(args.channels, args.kernel_size, args.layers, args.hidden, args.dropout)]
    if name == "mlp":
        return [(args.channels, args.kernel_size, args.layers, h, d) for h in hidden_values for d in dropout_values]
    if name in {"gru", "lstm"}:
        return [(args.channels, args.kernel_size, l, h, d) for l in layer_values for h in hidden_values for d in dropout_values]
    if name == "transformer":
        return [(c, args.kernel_size, l, args.hidden, d) for c in channels_values for l in layer_values for d in dropout_values]
    return [(c, k, l, args.hidden, d) for c in channels_values for k in kernel_values for l in layer_values for d in dropout_values]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Raw OHLC CNN barrier model")
    ap.add_argument("--source", choices=["mt5", "local", "dukascopy"], default="mt5")
    ap.add_argument("--ohlc-source", choices=["ticks", "native"], default="ticks",
                    help="ticks = build bid candles from ticks; native = use MT5 copy_rates_range candles")
    ap.add_argument("--native-chunk-days", type=float, default=30.0,
                    help="chunk size for MT5 native OHLC downloads")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--pair", default="XAUUSD")
    ap.add_argument("--pairs", nargs="+", default=None)
    ap.add_argument("--days", default="90", help="lookback days or 'max' for all MT5-available native OHLC")
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--to", default=None)
    ap.add_argument("--timeframe", default=None,
                    help="single timeframe override")
    ap.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES),
                    help="comma list of timeframes to sweep")
    ap.add_argument("--target", choices=["barrier", "trade", "tpsl_direction", "move4", "nextbar"], default="move4",
                    help="barrier = equal up/down first; trade = side-specific win label; tpsl_direction = fixed TP/SL direction; move4 = long/short prob + max up/down forecast; nextbar = future close up probability")
    ap.add_argument("--trade-side", choices=["long", "short"], default="long",
                    help="side to label when --target trade")
    ap.add_argument("--model", choices=["cnn", "tcn", "tcn2", "mlp", "linear", "gru", "lstm", "transformer", "rf", "xgb"], default="tcn")
    ap.add_argument("--models", default=None,
                    help="comma list of models to sweep; overrides --model")
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--windows", default=None,
                    help="comma list of past window lengths to sweep")
    ap.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="ohlc12",
                    help="ohlc4 = old raw OHLC channels; ohlc12 = OHLC plus bar shape/spread/session channels")
    ap.add_argument("--barrier-points", type=float, default=None,
                    help="single label barrier; default auto-detects XAU vs FX")
    ap.add_argument("--barrier-points-list", default=None,
                    help="comma list of label barriers to sweep; default auto-detects XAU vs FX")
    ap.add_argument("--label-tp-points-list", default=None,
                    help="TP list for --target tpsl_direction; default auto-detects XAU vs FX")
    ap.add_argument("--label-sl-points-list", default=None,
                    help="SL list for --target tpsl_direction; default auto-detects XAU vs FX")
    ap.add_argument("--label-sessions", default="-1,0,1,2",
                    help="training label session filter sweep; samples outside this session are excluded")
    ap.add_argument("--move-scale-points", type=float, default=None,
                    help="normalization scale for move4 max-up/max-down targets; default auto-detects XAU vs FX")
    ap.add_argument("--move-reg-weight", type=float, default=0.35,
                    help="loss weight for move4 max-up/max-down regression heads")
    ap.add_argument("--sl-points", type=float, default=None,
                    help="SL points for --target trade labels; default follows --barrier-points")
    ap.add_argument("--horizon-bars", type=int, default=200)
    ap.add_argument("--horizons", default=None,
                    help="comma list of future barrier horizon lengths to sweep")
    ap.add_argument("--train-frac", type=float, default=0.95)
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--max-samples", type=int, default=0,
                    help="0 = use all labelled samples; set a cap for quick experiments")
    ap.add_argument("--min-train-samples", type=int, default=1000)
    ap.add_argument("--min-test-samples", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--no-precompute-features", action="store_true",
                    help="disable cached window tensors; slower but lower peak RAM")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--channels-list", default=None,
                    help="comma list of channel/d_model sizes to sweep")
    ap.add_argument("--kernel-size", type=int, default=3)
    ap.add_argument("--kernel-sizes", default=None,
                    help="comma list of CNN/TCN kernel sizes to sweep")
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--layers-list", default=None,
                    help="comma list of recurrent/TCN/transformer layer counts to sweep")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--hidden-list", default=None,
                    help="comma list of MLP/RNN hidden sizes to sweep")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--dropouts", default=None,
                    help="comma list of dropout values to sweep")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--session-feature", action="store_true",
                    help="include inside-session flag as a model input; off by default")
    ap.add_argument("--point-size", type=float, default=None)
    ap.add_argument("--amount", type=float, default=50.0)
    ap.add_argument("--compound", action="store_true")
    ap.add_argument("--leverage", type=float, default=100.0)
    ap.add_argument("--commission-per-million", type=float, default=30.0)
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--tree-depth", type=int, default=5)
    ap.add_argument("--tree-jobs", type=int, default=1,
                    help="parallel jobs for sklearn/xgboost tree models; 1 is safest on Windows")
    ap.add_argument("--min-leaf", type=int, default=20)
    ap.add_argument("--xgb-lr", type=float, default=0.035)
    ap.add_argument("--eval-tp-points", default=None,
                    help="external TP sweep for probability signals; default uses --barrier-points")
    ap.add_argument("--eval-sl-points", default=None,
                    help="external SL sweep for probability signals; default uses --barrier-points")
    ap.add_argument("--eval-sessions", default="-1,0,1,2",
                    help="external session filter sweep for probability signals")
    ap.add_argument("--prob-ma", default="1,3,5,10",
                    help="comma list of probability moving-average lengths for external signal eval")
    ap.add_argument("--model-out", default=None)
    ap.add_argument("--pred-out", default="",
                    help="optional prediction CSV path; empty disables prediction CSV export")
    ap.add_argument("--notes-out", default=os.path.join("data", "forex", "ml_summaries", "forex_ml_barrier_notes.csv"))
    ap.add_argument("--verbose", action="store_true",
                    help="print per-epoch diagnostics and raw threshold/debug sections")
    ap.add_argument("--train-only", action="store_true",
                    help="save trained models without internal probability/backtest diagnostics")
    ap.add_argument("--predict-split", choices=["test", "train", "all"], default="test",
                    help="which split to export/evaluate predictions on; use all for in-sample simulator sweeps")
    return ap


def run_one(args: argparse.Namespace, data: BarrierData) -> None:
    if args.target == "move4":
        label_tag = f"scale{float(args.move_scale_points):g}"
        side_tag = "both"
    elif args.target == "nextbar":
        label_tag = "nextbar"
        side_tag = "up"
    else:
        label_tag = (
            f"tp{getattr(args, 'label_tp_points', args.barrier_points):g}_"
            f"sl{getattr(args, 'label_sl_points', args.sl_points if args.sl_points is not None else args.barrier_points):g}"
        )
        side_tag = args.trade_side
    if args.model_out is None:
        ext = "pkl" if args.model in {"rf", "xgb"} else "pt"
        args.model_out = os.path.join(
            "data", "forex", "ml_models",
            f"forex_ml_{args.pair}_{args.target}_{side_tag}_{args.model}_{args.feature_set}_"
            f"tf{args.timeframe}_"
            f"{label_tag}_"
            f"s{getattr(args, 'label_session', -1)}_"
            f"w{args.window}_h{args.horizon_bars}_c{args.channels}_k{args.kernel_size}_l{args.layers}.{ext}",
        )
    if args.pred_out is None:
        args.pred_out = os.path.join(
            "data", "forex", "ml_predictions",
            f"forex_ml_{args.pair}_{args.target}_{side_tag}_{args.model}_{args.feature_set}_"
            f"tf{args.timeframe}_"
            f"{label_tag}_"
            f"s{getattr(args, 'label_session', -1)}_"
            f"w{args.window}_h{args.horizon_bars}_c{args.channels}_k{args.kernel_size}_l{args.layers}_predictions.csv",
        )

    close = data.ohlc[:, 3].astype(np.float64)
    high = data.ohlc[:, 1].astype(np.float64)
    low = data.ohlc[:, 2].astype(np.float64)
    if args.target == "move4":
        labels, valid = label_move4(
            close, high, low,
            args.horizon_bars,
            data.point_size,
            float(args.move_scale_points),
        )
        label_session = int(getattr(args, "label_session", -1))
        if label_session != -1:
            ts_ns = pd.to_datetime(data.times, utc=True).astype("int64").to_numpy(np.int64)
            valid &= active_session_allowed(ts_ns, label_session)
    elif args.target == "nextbar":
        labels, valid = label_nextbar(close, args.horizon_bars)
        label_session = int(getattr(args, "label_session", -1))
        if label_session != -1:
            ts_ns = pd.to_datetime(data.times, utc=True).astype("int64").to_numpy(np.int64)
            valid &= active_session_allowed(ts_ns, label_session)
    elif args.target == "tpsl_direction":
        label_tp = float(getattr(args, "label_tp_points", args.barrier_points))
        label_sl = float(getattr(args, "label_sl_points", args.sl_points if args.sl_points is not None else label_tp))
        labels, valid = label_tpsl_direction(
            close, high, low,
            label_tp,
            label_sl,
            args.horizon_bars,
            data.point_size,
        )
        label_session = int(getattr(args, "label_session", -1))
        if label_session != -1:
            ts_ns = pd.to_datetime(data.times, utc=True).astype("int64").to_numpy(np.int64)
            valid &= active_session_allowed(ts_ns, label_session)
    elif args.target == "trade":
        labels, valid = label_trade_outcome(
            close, high, low,
            float(args.barrier_points),
            float(args.sl_points if args.sl_points is not None else args.barrier_points),
            args.horizon_bars,
            data.point_size,
            args.trade_side,
        )
    else:
        labels, valid = label_barriers(
            close, high, low,
            float(args.barrier_points),
            args.horizon_bars,
            data.point_size,
        )
    data.labels = labels
    data.valid = valid
    train_idx, test_idx = split_indices(data, args.window, args.train_frac, args.max_samples, args.seed)
    if args.target == "move4":
        train_up = float(np.mean(np.argmax(data.labels[train_idx, :2], axis=1))) * 100.0 if len(train_idx) else 0.0
        test_up = float(np.mean(np.argmax(data.labels[test_idx, :2], axis=1))) * 100.0 if len(test_idx) else 0.0
    else:
        train_up = float(np.mean(data.labels[train_idx])) * 100.0 if len(train_idx) else 0.0
        test_up = float(np.mean(data.labels[test_idx])) * 100.0 if len(test_idx) else 0.0
    print(
        f"[ml] candles={len(data.ohlc):,} labelled={int(data.valid.sum()):,} "
        f"train={len(train_idx):,} test={len(test_idx):,} target={args.target}/{args.trade_side} "
        f"model={args.model} tf={args.timeframe} "
        f"window={args.window} features={args.feature_set} barrier={args.barrier_points:g} "
        f"label_tp={getattr(args, 'label_tp_points', '')} label_sl={getattr(args, 'label_sl_points', '')} "
        f"label_session={getattr(args, 'label_session', '')} horizon={args.horizon_bars} "
        f"up% train/test={train_up:.1f}/{test_up:.1f}",
        flush=True,
    )
    require_test_samples = args.train_frac < 1.0
    if len(train_idx) < args.min_train_samples or (require_test_samples and len(test_idx) < args.min_test_samples):
        print(
            f"[ml] SKIP not enough labelled samples "
            f"train={len(train_idx):,}/{args.min_train_samples:,} "
            f"test={len(test_idx):,}/{args.min_test_samples if require_test_samples else 0:,}; "
            "increase days, reduce window/horizon, or lower min sample thresholds",
            flush=True,
        )
        return
    if args.target == "move4" and args.model in {"rf", "xgb"}:
        raise SystemExit("--target move4 currently supports torch models only: tcn,cnn,gru,lstm,transformer,mlp,linear")
    if args.model in {"rf", "xgb"}:
        model, probs = train_tree_model(args, data, train_idx, test_idx)
    else:
        model, device = train_model(args, data, train_idx, test_idx)
    if args.train_only:
        save_any_model(model, args.model_out, args, train_idx, test_idx, data)
        append_notes(args.notes_out, args, [], train_idx, test_idx)
        return
    if args.model in {"rf", "xgb"}:
        # tree models already produced probabilities above
        pass
    else:
        eval_idx = test_idx if len(test_idx) else train_idx
        probs = predict_probs(model, device, data, eval_idx, args)
    if args.predict_split == "train":
        pred_idx = train_idx
    elif args.predict_split == "all":
        pred_idx = np.sort(np.concatenate([train_idx, test_idx]))
    else:
        pred_idx = test_idx if len(test_idx) else train_idx
    if args.predict_split == "test" and len(test_idx):
        pred_probs = probs
    else:
        pred_probs = predict_probs(model, device, data, pred_idx, args) if args.model not in {"rf", "xgb"} else probs

    rows = backtest_thresholds(data, pred_idx, pred_probs, args)
    if args.verbose:
        print("\n[ml] threshold backtest on labelled test samples")
        for r in rows:
            print(
                f"th={r['threshold']:.2f} trades={r['trades']:5} wr={r['wr']:5.1f}% "
                f"pf={r['pf']:6.2f} pnl=${r['pnl']:+8.2f} dd=${r['dd']:7.2f}",
                flush=True,
            )
        print_diagnostics(data, test_idx if len(test_idx) else train_idx, probs)
    external_rows = backtest_external_probability_signals(data, pred_idx, pred_probs, args)
    if external_rows:
        print("[ml] external best", flush=True)
        for r in external_rows[:3]:
            print(
                f"th={r['threshold']:.2f} ma={r['prob_ma']:2} tp={r['tp']:g} sl={r['sl']:g} sess={r['session']:2} "
                f"trades={r['trades']:5} wr={r['wr']:5.1f}% pf={r['pf']:6.2f} "
                f"pnl=${r['pnl']:+8.2f} dd=${r['dd']:7.2f} worst=${r['worst']:+7.2f}",
                flush=True,
            )
    export_predictions(data, pred_idx, pred_probs, args.pred_out, args)
    save_any_model(model, args.model_out, args, train_idx, test_idx, data)
    append_notes(args.notes_out, args, rows if rows else external_rows, train_idx, test_idx)


def main() -> None:
    args = build_parser().parse_args()
    if isinstance(args.timeframe, list):
        args.timeframe = str(args.timeframe[0])
    set_seed(args.seed)
    from forex_strategy_common import prepare_args
    apply_date_window(args)
    args.days = None
    prepare_args(args)
    t0 = time.time()
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else [args.model]
    timeframes = [args.timeframe] if args.timeframe else parse_str_list(args.timeframes, DEFAULT_TIMEFRAMES)
    windows = parse_int_list(args.windows, [args.window])
    horizons = parse_int_list(args.horizons, [args.horizon_bars])
    channels_values = parse_int_list(args.channels_list, [args.channels])
    kernel_values = parse_int_list(args.kernel_sizes, [args.kernel_size])
    layer_values = parse_int_list(args.layers_list, [args.layers])
    hidden_values = parse_int_list(args.hidden_list, [args.hidden])
    dropout_values = parse_num_list(args.dropouts, [args.dropout])
    arch_by_model = {
        model_name: arch_combos_for_model(
            model_name,
            channels_values,
            kernel_values,
            layer_values,
            hidden_values,
            dropout_values,
            args,
        )
        for model_name in models
    }
    pairs = args.pairs if args.pairs else [args.pair]
    total_arch = sum(len(arch_by_model[m]) for m in models)
    grand_total = 0
    pair_label_grids: dict[str, list[tuple[float, float, int]]] = {}
    for pair in pairs:
        if args.target == "move4":
            scale = float(args.move_scale_points) if args.move_scale_points is not None else max(default_eval_tp_for_pair(pair))
            label_sessions = [int(x) for x in parse_num_list(args.label_sessions, [-1, 0, 1, 2])]
            label_grid = [(scale, scale, int(sess)) for sess in label_sessions]
        elif args.target == "nextbar":
            scale = float(args.move_scale_points) if args.move_scale_points is not None else max(default_eval_tp_for_pair(pair))
            label_sessions = [int(x) for x in parse_num_list(args.label_sessions, [-1, 0, 1, 2])]
            label_grid = [(scale, scale, int(sess)) for sess in label_sessions]
        elif args.target == "tpsl_direction":
            label_tps = parse_num_list(args.label_tp_points_list, default_eval_tp_for_pair(pair))
            label_sls = parse_num_list(args.label_sl_points_list, default_eval_sl_for_pair(pair))
            label_sessions = [int(x) for x in parse_num_list(args.label_sessions, [-1, 0, 1, 2])]
            label_grid = [(float(tp), float(sl), int(sess)) for tp in label_tps for sl in label_sls for sess in label_sessions]
        else:
            if args.barrier_points_list:
                barriers = parse_num_list(args.barrier_points_list, default_barriers_for_pair(pair))
            elif args.barrier_points is not None:
                barriers = [float(args.barrier_points)]
            else:
                barriers = default_barriers_for_pair(pair)
            label_grid = [(float(barrier), float(args.sl_points) if args.sl_points is not None else float(barrier), -1) for barrier in barriers]
        pair_label_grids[pair] = label_grid
        grand_total += len(timeframes) * len(label_grid) * len(windows) * len(horizons) * total_arch
    done = 0
    for pair in pairs:
        pair_args = copy(args)
        pair_args.pair = pair
        pair_args.pairs = [pair]
        pair_args.barrier_points = pair_label_grids[pair][0][0]
        pair_args.move_scale_points = float(args.move_scale_points) if args.move_scale_points is not None else max(default_eval_tp_for_pair(pair))
        print(
            f"\n[ml] pair={pair} point={'auto' if pair_args.point_size is None else pair_args.point_size} "
            f"timeframes={','.join(timeframes)} "
            f"label_grid={len(pair_label_grids[pair])} eval_tp={parse_num_list(pair_args.eval_tp_points, default_eval_tp_for_pair(pair))} "
            f"eval_sl={parse_num_list(pair_args.eval_sl_points, default_eval_sl_for_pair(pair))}",
            flush=True,
        )
        for tf in timeframes:
            tf_args = copy(pair_args)
            tf_args.timeframe = tf
            data = load_barrier_data(tf_args)
            for label_tp, label_sl, label_session in pair_label_grids[pair]:
                for model_name in models:
                    for window in windows:
                        for horizon in horizons:
                            for channels, kernel, layers, hidden, dropout in arch_by_model[model_name]:
                                done += 1
                                run_args = copy(tf_args)
                                run_args.model = model_name
                                run_args.window = window
                                run_args.horizon_bars = horizon
                                run_args.barrier_points = float(label_tp)
                                run_args.sl_points = float(label_sl)
                                run_args.label_tp_points = float(label_tp)
                                run_args.label_sl_points = float(label_sl)
                                run_args.label_session = int(label_session)
                                run_args.move_scale_points = float(pair_args.move_scale_points)
                                run_args.channels = channels
                                run_args.kernel_size = kernel
                                run_args.layers = layers
                                run_args.hidden = hidden
                                run_args.dropout = float(dropout)
                                if args.model_out is None:
                                    run_args.model_out = None
                                if args.pred_out is None:
                                    run_args.pred_out = None
                                print(
                                    f"\n[ml] combo {done}/{grand_total} pair={pair} tf={tf} model={model_name} "
                                    f"window={window} horizon={horizon} label_tp={label_tp:g} label_sl={label_sl:g} label_session={label_session} "
                                    f"channels={channels} kernel={kernel} layers={layers} hidden={hidden} dropout={float(dropout):g}",
                                    flush=True,
                                )
                                run_one(run_args, data)
    print(f"[ml] elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
