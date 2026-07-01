"""Trade-level analysis for selected TCN2 nextbar configs."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from forex_ml_tick_simulator import build_bid_candles, load_torch_model, predict_model, smooth_predictions
from forex_strategy_common import active_session_allowed, build_parser, default_point_size, load_market
from forex_tcn2_nextbar_simulator import parse_reentry_mode


@dataclass(frozen=True)
class Config:
    name: str
    pair: str
    timeframe: str
    model_file: str
    upper: float
    lower: float
    prob_ma: int
    tp_mode: str
    sl_mode: str
    tp_points: float
    sl_points: float
    session: int
    label_session: int
    reentry_mode: str = "ma_reset"


CONFIGS = [
    Config(
        name="xau_1m_h10_fixed_400_400_ma3",
        pair="XAUUSD",
        timeframe="1m",
        model_file="forex_ml_XAUUSD_nextbar_up_tcn2_ohlc12_tf1m_nextbar_s2_w128_h10_c64_k3_l5.pt",
        upper=0.6,
        lower=0.4,
        prob_ma=3,
        tp_mode="fixed",
        sl_mode="fixed",
        tp_points=400,
        sl_points=400,
        session=2,
        label_session=2,
    ),
    Config(
        name="aud_5m_h10_neutral_sl30_ma14",
        pair="AUDUSD",
        timeframe="5m",
        model_file="forex_ml_AUDUSD_nextbar_up_tcn2_ohlc12_tf5m_nextbar_s2_w128_h10_c64_k3_l5.pt",
        upper=0.6,
        lower=0.4,
        prob_ma=14,
        tp_mode="neutral",
        sl_mode="fixed",
        tp_points=0,
        sl_points=30,
        session=2,
        label_session=2,
    ),
    Config(
        name="aud_5m_h10_fixed30_opposite_ma3",
        pair="AUDUSD",
        timeframe="5m",
        model_file="forex_ml_AUDUSD_nextbar_up_tcn2_ohlc12_tf5m_nextbar_s2_w128_h10_c64_k3_l5.pt",
        upper=0.7,
        lower=0.3,
        prob_ma=3,
        tp_mode="fixed",
        sl_mode="opposite",
        tp_points=30,
        sl_points=0,
        session=2,
        label_session=2,
    ),
    Config(
        name="aud_5m_h10_opposite_sl75_ma3",
        pair="AUDUSD",
        timeframe="5m",
        model_file="forex_ml_AUDUSD_nextbar_up_tcn2_ohlc12_tf5m_nextbar_s2_w128_h10_c64_k3_l5.pt",
        upper=0.6,
        lower=0.4,
        prob_ma=3,
        tp_mode="opposite",
        sl_mode="fixed",
        tp_points=0,
        sl_points=75,
        session=2,
        label_session=2,
    ),
    Config(
        name="aud_5m_h10_opposite_sl60_ma3",
        pair="AUDUSD",
        timeframe="5m",
        model_file="forex_ml_AUDUSD_nextbar_up_tcn2_ohlc12_tf5m_nextbar_s2_w128_h10_c64_k3_l5.pt",
        upper=0.6,
        lower=0.4,
        prob_ma=3,
        tp_mode="opposite",
        sl_mode="fixed",
        tp_points=0,
        sl_points=60,
        session=2,
        label_session=2,
    ),
    Config(
        name="aud_5m_h10_fixed45_opposite_ma5",
        pair="AUDUSD",
        timeframe="5m",
        model_file="forex_ml_AUDUSD_nextbar_up_tcn2_ohlc12_tf5m_nextbar_s2_w128_h10_c64_k3_l5.pt",
        upper=0.6,
        lower=0.4,
        prob_ma=5,
        tp_mode="fixed",
        sl_mode="opposite",
        tp_points=45,
        sl_points=0,
        session=2,
        label_session=2,
    ),
]


def signal_from_prob(p_up: float, upper: float, lower: float) -> int:
    if not np.isfinite(p_up):
        return 0
    if p_up >= upper:
        return 1
    if p_up <= lower:
        return -1
    return 0


def signal_name(side: int) -> str:
    return "long" if side == 1 else "short" if side == -1 else "flat"


def mode_exit(mode: str, side: int, sig: int) -> bool:
    if mode == "fixed":
        return False
    if mode == "opposite":
        return sig == -side
    if mode == "neutral":
        return sig != side
    raise ValueError(f"bad mode: {mode}")


def simulate_trades(
    cfg: Config,
    bid: np.ndarray,
    ask: np.ndarray,
    ts_ns: np.ndarray,
    candles,
    p_up: np.ndarray,
    amount: float,
    leverage: float,
    commission_per_million: float,
) -> pd.DataFrame:
    point = default_point_size(cfg.pair)
    allowed = active_session_allowed(candles.times.astype("int64"), int(cfg.session))
    notional = amount * leverage
    fee = notional / 1_000_000.0 * commission_per_million * 2.0
    trades: list[dict] = []

    candle_i = int(128) - 1
    tick_floor = 0
    pending_side = 0
    block_long = False
    block_short = False
    cash = amount
    equity_peak = amount
    cum_pnl = 0.0
    cum_peak = 0.0

    while candle_i < len(candles.close_tick_idx):
        if pending_side:
            side = pending_side
            pending_side = 0
            entry_tick = tick_floor
            if entry_tick >= len(bid):
                break
            candle_i = int(candles.tick_to_candle[entry_tick])
        else:
            if not allowed[candle_i]:
                candle_i += 1
                continue
            p_now = float(p_up[candle_i])
            if cfg.reentry_mode == "ma_reset":
                if block_long and (not np.isfinite(p_now) or p_now < cfg.upper):
                    block_long = False
                if block_short and (not np.isfinite(p_now) or p_now > cfg.lower):
                    block_short = False
            side = signal_from_prob(p_now, cfg.upper, cfg.lower)
            if side == 0:
                candle_i += 1
                continue
            if cfg.reentry_mode == "ma_reset" and ((side == 1 and block_long) or (side == -1 and block_short)):
                candle_i += 1
                continue
            entry_tick = int(candles.close_tick_idx[candle_i]) + 1
            if entry_tick < tick_floor:
                entry_tick = tick_floor
            if entry_tick >= len(bid):
                break

        entry = float(ask[entry_tick] if side == 1 else bid[entry_tick])
        entry_candle = int(candles.tick_to_candle[entry_tick])
        entry_time = pd.Timestamp(ts_ns[entry_tick], unit="ns", tz="UTC")
        entry_prob = float(p_up[entry_candle]) if 0 <= entry_candle < len(p_up) else np.nan
        exit_tick = len(bid) - 1
        result_points = ((bid[exit_tick] if side == 1 else ask[exit_tick]) - entry) / point * side
        reason = "open_end"
        reverse_side = 0
        next_signal_candle = entry_candle + 1
        max_adverse_points = 0.0
        max_favour_points = 0.0
        max_trade_dd = 0.0
        exit_prob = np.nan

        for ti in range(entry_tick + 1, len(bid)):
            live_points = (float(bid[ti]) - entry) / point if side == 1 else (entry - float(ask[ti])) / point
            max_favour_points = max(max_favour_points, live_points)
            max_adverse_points = max(max_adverse_points, -live_points)
            denom = entry if entry > 1e-12 else 1e-12
            trade_dd = max(0.0, -live_points) * point * (notional / denom)
            max_trade_dd = max(max_trade_dd, trade_dd)
            live_equity = cash - trade_dd
            equity_peak = max(equity_peak, live_equity)

            if cfg.tp_mode == "fixed" and cfg.tp_points > 0 and live_points >= cfg.tp_points:
                result_points = cfg.tp_points
                exit_tick = ti
                reason = "fixed_tp"
                break
            if cfg.sl_mode == "fixed" and cfg.sl_points > 0 and live_points <= -cfg.sl_points:
                result_points = -cfg.sl_points
                exit_tick = ti
                reason = "fixed_sl"
                break

            current_candle = int(candles.tick_to_candle[ti])
            while next_signal_candle <= current_candle and next_signal_candle < len(candles.close_tick_idx):
                sig = signal_from_prob(float(p_up[next_signal_candle]), cfg.upper, cfg.lower)
                mode = cfg.tp_mode if live_points >= 0 else cfg.sl_mode
                if mode_exit(mode, side, sig):
                    result_points = live_points
                    exit_tick = ti
                    reason = f"{mode}_exit"
                    exit_prob = float(p_up[next_signal_candle])
                    if sig == -side:
                        reverse_side = sig
                    break
                next_signal_candle += 1
            if reason.endswith("_exit"):
                break

        pnl = result_points * point * (notional / (entry if entry > 1e-12 else 1e-12)) - fee
        cash += pnl
        cum_pnl += pnl
        cum_peak = max(cum_peak, cum_pnl)
        equity_peak = max(equity_peak, cash)
        exit_time = pd.Timestamp(ts_ns[exit_tick], unit="ns", tz="UTC")
        trades.append({
            "config": cfg.name,
            "pair": cfg.pair,
            "timeframe": cfg.timeframe,
            "side": signal_name(side),
            "entry_time": entry_time,
            "exit_time": exit_time,
            "hold_min": (exit_time - entry_time).total_seconds() / 60.0,
            "entry_price": entry,
            "exit_price": float(bid[exit_tick] if side == 1 else ask[exit_tick]),
            "points": result_points,
            "pnl": pnl,
            "reason": reason,
            "entry_prob": entry_prob,
            "exit_prob": exit_prob,
            "max_adverse_points": max_adverse_points,
            "max_favour_points": max_favour_points,
            "trade_max_dd": max_trade_dd,
            "cum_pnl": cum_pnl,
            "cum_dd": cum_peak - cum_pnl,
            "entry_hour_utc": entry_time.hour,
            "exit_hour_utc": exit_time.hour,
            "entry_weekday": entry_time.day_name(),
        })

        if reason == "fixed_sl" and cfg.reentry_mode == "ma_reset":
            if side == 1:
                block_long = True
            else:
                block_short = True

        tick_floor = exit_tick + 1
        if tick_floor >= len(bid):
            break
        if reason.endswith("_exit") and reverse_side != 0:
            pending_side = reverse_side
            continue
        next_candle = int(candles.tick_to_candle[tick_floor])
        candle_i = next_candle if next_candle > candle_i else candle_i + 1

    return pd.DataFrame(trades)


def summarize(name: str, trades: pd.DataFrame) -> None:
    if trades.empty:
        print(f"\n{name}: no trades")
        return
    wins = trades[trades.pnl >= 0]
    losses = trades[trades.pnl < 0]
    gross_win = float(wins.pnl.sum())
    gross_loss = float(-losses.pnl.sum())
    pf = gross_win / gross_loss if gross_loss else 999.0
    print(f"\n{name}")
    print(
        f"trades={len(trades)} pnl=${trades.pnl.sum():+.2f} wr={len(wins)/len(trades)*100:.1f}% "
        f"pf={pf:.2f} worst=${trades.pnl.min():+.2f} med_loss=${losses.pnl.median() if len(losses) else 0:+.2f} "
        f"max_trade_dd=${trades.trade_max_dd.max():.2f} med_trade_dd=${trades.trade_max_dd.median():.2f} "
        f"med_hold={trades.hold_min.median():.1f}m p90_hold={trades.hold_min.quantile(.9):.1f}m"
    )
    print("reason counts:")
    print(trades.reason.value_counts().to_string())
    print("pnl by side:")
    print(trades.groupby("side").pnl.agg(["count", "sum", "mean", "min"]).to_string())
    print("worst 8 trades:")
    cols = ["entry_time", "side", "pnl", "reason", "hold_min", "trade_max_dd", "max_adverse_points", "entry_prob", "exit_prob"]
    print(trades.sort_values("pnl").head(8)[cols].to_string(index=False))
    print("entry hour pnl/trades:")
    hour = trades.groupby("entry_hour_utc").pnl.agg(["count", "sum", "mean"]).sort_values("sum")
    print(hour.to_string())


def main() -> None:
    ap = build_parser("TCN2 selected trade analysis", "unused.csv")
    ap.add_argument("--model-dir", default=str(Path("data") / "forex" / "ml_models"))
    ap.add_argument("--only", default="", help="comma list of config names/substrings")
    ap.add_argument("--out-dir", default=str(Path("data") / "forex" / "analysis" / "tcn2_trades"))
    ap.add_argument("--reentry-mode", default=None, help="override config reentry mode: immediate or ma_reset")
    args = ap.parse_args()
    selected = CONFIGS
    if args.only.strip():
        keys = [x.strip().lower() for x in args.only.split(",") if x.strip()]
        selected = [c for c in CONFIGS if any(k in c.name.lower() for k in keys)]
    if not selected:
        raise SystemExit("no configs selected")
    if args.reentry_mode is not None:
        mode = parse_reentry_mode(args.reentry_mode)
        selected = [
            Config(
                name=f"{c.name}_{mode}",
                pair=c.pair,
                timeframe=c.timeframe,
                model_file=c.model_file,
                upper=c.upper,
                lower=c.lower,
                prob_ma=c.prob_ma,
                tp_mode=c.tp_mode,
                sl_mode=c.sl_mode,
                tp_points=c.tp_points,
                sl_points=c.sl_points,
                session=c.session,
                label_session=c.label_session,
                reentry_mode=mode,
            )
            for c in selected
        ]
    args.pairs = sorted({c.pair for c in selected})

    ticks, t0 = load_market(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[tcn2-analysis] configs={len(selected)} device={device}", flush=True)

    all_trades = []
    pred_cache = {}
    for cfg in selected:
        model_path = Path(args.model_dir) / cfg.model_file
        if not model_path.exists():
            print(f"[tcn2-analysis] missing model {model_path}", flush=True)
            continue
        g = ticks[ticks["pair"].str.upper() == cfg.pair].sort_values("timestamp").reset_index(drop=True)
        bid = g["bid"].to_numpy(np.float64)
        ask = g["ask"].to_numpy(np.float64)
        ts_ns = g["timestamp"].astype("int64").to_numpy()
        cache_key = (cfg.model_file, cfg.pair, cfg.timeframe, cfg.prob_ma)
        if cache_key not in pred_cache:
            model, ns, point = load_torch_model(model_path)
            model = model.to(device)
            candles = build_bid_candles(bid, ask, ts_ns, cfg.timeframe)
            print(f"[tcn2-analysis] {cfg.name} predicting candles={len(candles.ohlc):,}", flush=True)
            preds = predict_model(model, ns, candles, point, device)
            preds = smooth_predictions(preds, cfg.prob_ma)
            pred_cache[cache_key] = (candles, preds[:, 0].astype(np.float64))
        candles, p_up = pred_cache[cache_key]
        trades = simulate_trades(
            cfg,
            bid,
            ask,
            ts_ns,
            candles,
            p_up,
            float(args.amount),
            float(args.leverage),
            float(args.commission_per_million),
        )
        path = out_dir / f"{cfg.name}_trades.csv"
        trades.to_csv(path, index=False)
        print(f"[tcn2-analysis] wrote {path} rows={len(trades)}", flush=True)
        summarize(cfg.name, trades)
        all_trades.append(trades)

    if all_trades:
        merged = pd.concat(all_trades, ignore_index=True)
        merged_path = out_dir / "all_selected_trades.csv"
        merged.to_csv(merged_path, index=False)
        print(f"\n[tcn2-analysis] wrote merged {merged_path} rows={len(merged)} elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
