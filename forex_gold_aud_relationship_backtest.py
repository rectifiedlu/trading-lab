"""Walk-forward AUDUSD/XAUUSD lead-lag and divergence research.

For each training fold, discover whether a large move in one market is followed
by continuation or reversal in the other.  The selected leader, follower,
lookback, horizon, direction, and event cutoff are frozen for the next fold.
Only the frozen signal receives tick-level bid/ask execution.
"""
from __future__ import annotations

import csv
import os

import numpy as np
import pandas as pd

from forex_signal_sweep_common import build_bid_ohlc, map_state_to_ticks, simulate_state_strategy
from forex_strategy_common import active_session_allowed, build_parser, default_point_size, load_market, timeframe_to_ns


LOOKBACKS = (1, 3, 6, 12)
HORIZONS = (1, 3, 6)
QUANTILES = (0.80, 0.90)


def candles(group: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    bid = group.bid.to_numpy(np.float64)
    ts = group.timestamp.to_numpy(dtype="datetime64[ns]").astype("int64")
    _, high, low, close, close_idx = build_bid_ohlc(bid, ts, timeframe)
    return pd.DataFrame({
        "timestamp": pd.to_datetime((ts[close_idx] // timeframe_to_ns(timeframe)) * timeframe_to_ns(timeframe), utc=True), "close": close,
        "range": high - low, "close_idx": close_idx,
    })


def train_candidate(frame: pd.DataFrame, train: np.ndarray):
    best = None
    for leader in ("xau", "aud"):
        follower = "aud" if leader == "xau" else "xau"
        lead_close = frame[f"{leader}_close"].to_numpy(np.float64)
        follow_close = frame[f"{follower}_close"].to_numpy(np.float64)
        for lookback in LOOKBACKS:
            lead_move = np.full(len(frame), np.nan)
            lead_move[lookback:] = (lead_close[lookback:] / lead_close[:-lookback] - 1.0) * 10_000.0
            for horizon in HORIZONS:
                future = np.full(len(frame), np.nan)
                future[:-horizon] = (follow_close[horizon:] / follow_close[:-horizon] - 1.0) * 10_000.0
                eligible = train & np.isfinite(lead_move) & np.isfinite(future)
                if eligible.sum() < 300:
                    continue
                for quantile in QUANTILES:
                    cutoff = float(np.quantile(np.abs(lead_move[eligible]), quantile))
                    event = eligible & (np.abs(lead_move) >= cutoff)
                    signed = np.sign(lead_move[event]) * future[event]
                    if len(signed) < 30:
                        continue
                    mean = float(np.mean(signed))
                    std = float(np.std(signed, ddof=1))
                    t_stat = mean / max(std / np.sqrt(len(signed)), 1e-9)
                    # Positive means follower historically continued leader's move;
                    # negative means it historically corrected it.
                    direction = 1 if mean >= 0 else -1
                    score = abs(t_stat) * np.sqrt(len(signed))
                    item = (score, leader, follower, lookback, horizon, quantile, cutoff, direction, mean, t_stat, len(signed))
                    if best is None or item[0] > best[0]:
                        best = item
    return best


def test_state(frame: pd.DataFrame, selected, follower: str) -> np.ndarray:
    _, leader, _, lookback, _, _, cutoff, direction, *_ = selected
    close = frame[f"{leader}_close"].to_numpy(np.float64)
    move = np.full(len(frame), np.nan)
    move[lookback:] = (close[lookback:] / close[:-lookback] - 1.0) * 10_000.0
    state = np.zeros(len(frame), dtype=np.float64)
    state[move >= cutoff] = float(direction)
    state[move <= -cutoff] = float(-direction)
    return state


def run_test(pair_ticks: pd.DataFrame, pair_candles: pd.DataFrame, state: np.ndarray, start, end, stop_points: float, args):
    mask = (pair_ticks.timestamp >= start) & (pair_ticks.timestamp < end)
    ticks = pair_ticks.loc[mask].reset_index(drop=True)
    first = np.flatnonzero(mask.to_numpy())[0]
    last = np.flatnonzero(mask.to_numpy())[-1]
    c0 = np.searchsorted(pair_candles.close_idx.to_numpy(), first, side="left")
    c1 = np.searchsorted(pair_candles.close_idx.to_numpy(), last, side="right")
    close_idx = pair_candles.close_idx.to_numpy()[c0:c1] - first
    local_state = state[c0:c1]
    ts_ns = ticks.timestamp.to_numpy(dtype="datetime64[ns]").astype("int64")
    return simulate_state_strategy(
        str(ticks.pair.iloc[0]), "gold_aud_relationship", "", args.timeframes, ticks.bid.to_numpy(np.float64),
        ticks.ask.to_numpy(np.float64), ts_ns, map_state_to_ticks(len(ticks), close_idx, local_state),
        map_state_to_ticks(len(ticks), close_idx, local_state), active_session_allowed(ts_ns, 0), 0.0, 0.0,
        stop_points, 0.0, default_point_size(str(ticks.pair.iloc[0])), args.amount, args.compound, args.leverage,
        args.commission_per_million, "both", signal_exit_always=True,
    )


def main() -> None:
    ap = build_parser("AUDUSD/XAUUSD relationship walk-forward", "forex_gold_aud_relationship_results.csv")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--train-days", type=int, default=14)
    ap.add_argument("--test-days", type=int, default=7)
    args = ap.parse_args()
    args.timeframes = args.timeframes or "5m"
    if {p.upper() for p in args.pairs} != {"AUDUSD", "XAUUSD"}:
        args.pairs = ["AUDUSD", "XAUUSD"]
    ticks, _ = load_market(args)
    grouped = {pair: g.sort_values("timestamp").reset_index(drop=True) for pair, g in ticks.groupby("pair")}
    if set(grouped) != {"AUDUSD", "XAUUSD"}:
        raise SystemExit(f"need both AUDUSD and XAUUSD ticks; got {sorted(grouped)}")
    aud = candles(grouped["AUDUSD"], args.timeframes).rename(columns={"close": "aud_close", "range": "aud_range", "close_idx": "aud_close_idx"})
    xau = candles(grouped["XAUUSD"], args.timeframes).rename(columns={"close": "xau_close", "range": "xau_range", "close_idx": "xau_close_idx"})
    aligned = aud.merge(xau[["timestamp", "xau_close", "xau_range", "xau_close_idx"]], on="timestamp", how="inner")
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.to, tz="UTC")
    rows = []
    for fold in range(args.folds):
        test_end = end - pd.Timedelta(days=(args.folds - 1 - fold) * args.test_days)
        train_end = test_end - pd.Timedelta(days=args.test_days)
        train_start = train_end - pd.Timedelta(days=args.train_days)
        train = ((aligned.timestamp >= train_start) & (aligned.timestamp < train_end)).to_numpy()
        selected = train_candidate(aligned, train)
        if selected is None:
            continue
        _, leader, follower, lookback, horizon, quantile, cutoff, direction, mean_bps, t_stat, events = selected
        state = test_state(aligned, selected, follower)
        pair_candles = aligned[["timestamp", f"{follower}_close_idx"]].rename(columns={f"{follower}_close_idx": "close_idx"})
        # A 90th-percentile historical follower candle range is a pair-specific hard risk cap.
        range_col = f"{follower}_range"
        stop_points = float(np.quantile(aligned.loc[train, range_col], 0.90) / default_point_size("AUDUSD" if follower == "aud" else "XAUUSD"))
        r = run_test(grouped["AUDUSD" if follower == "aud" else "XAUUSD"], pair_candles, state, train_end, test_end, stop_points, args)
        rows.append({
            "fold": fold + 1, "train_start": train_start.isoformat(), "test_start": train_end.isoformat(),
            "test_end": test_end.isoformat(), "leader": leader, "follower": follower, "lookback": lookback,
            "horizon": horizon, "quantile": quantile, "cutoff_bps": cutoff, "direction": "continue" if direction == 1 else "reverse",
            "train_mean_bps": mean_bps, "train_t_stat": t_stat, "train_events": events, "stop_points": stop_points,
            "total": r.total, "realised": r.realised, "trades": r.trades, "win_rate": r.win_rate,
            "profit_factor": r.profit_factor, "max_drawdown": r.max_drawdown,
            "dd_pct_amount": r.max_drawdown / args.amount * 100.0, "pnl_dd": r.total / max(r.max_drawdown, 1e-9),
            "median_day": r.median_day,
        })
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["fold"])
        w.writeheader(); w.writerows(rows)
    print(f"[gold-aud] wrote {args.out} folds={len(rows)}", flush=True)
    for row in rows:
        print(row, flush=True)


if __name__ == "__main__":
    main()
