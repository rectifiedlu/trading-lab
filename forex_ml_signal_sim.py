from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SimResult:
    mode: str
    threshold: float
    exit_mode: str
    reverse: int
    cooldown: int
    trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    realised: float
    max_drawdown: float
    median_day: float
    avg_day: float
    signal_exits: int
    barrier_exits: int


def parse_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(float(x.strip())) for x in value.split(",") if x.strip()]


def signal_from_prob(prob: np.ndarray, threshold: float) -> np.ndarray:
    sig = np.zeros(len(prob), dtype=np.int8)
    sig[prob >= threshold] = 1
    sig[prob <= 1.0 - threshold] = -1
    return sig


def signal_from_two_probs(long_prob: np.ndarray, short_prob: np.ndarray, threshold: float, min_edge: float) -> np.ndarray:
    sig = np.zeros(len(long_prob), dtype=np.int8)
    long_ok = (long_prob >= threshold) & (long_prob >= short_prob + min_edge)
    short_ok = (short_prob >= threshold) & (short_prob >= long_prob + min_edge)
    sig[long_ok] = 1
    sig[short_ok] = -1
    return sig


def trade_pnl(side: int, entry: float, exit_px: float, amount: float, leverage: float, commission_per_million: float) -> float:
    units = (amount * leverage) / max(entry, 1e-9)
    gross = (exit_px - entry) * units if side == 1 else (entry - exit_px) * units
    fee = (amount * leverage) / 1_000_000.0 * commission_per_million * 2.0
    return gross - fee


def simulate(
    df: pd.DataFrame,
    threshold: float,
    mode: str,
    exit_mode: str,
    reverse: int,
    cooldown: int,
    tp_points: float,
    sl_points: float,
    point_size: float,
    amount: float,
    leverage: float,
    commission_per_million: float,
    return_trades: bool = False,
    precomputed_signal: np.ndarray | None = None,
) -> SimResult | tuple[SimResult, pd.DataFrame]:
    close = df["close"].to_numpy(np.float64)
    high = df["high"].to_numpy(np.float64)
    low = df["low"].to_numpy(np.float64)
    prob = df["prob_up"].to_numpy(np.float64) if "prob_up" in df.columns else df["prob_win"].to_numpy(np.float64)
    times = pd.to_datetime(df["time"], utc=True)
    sig = precomputed_signal if precomputed_signal is not None else signal_from_prob(prob, threshold)

    pos = 0
    entry = 0.0
    entry_time = None
    cool = 0
    realised = 0.0
    peak = 0.0
    max_dd = 0.0
    wins = losses = signal_exits = barrier_exits = 0
    gross_win = gross_loss = 0.0
    daily: dict[object, float] = {}
    trades = []
    last_sig = 0

    def close_trade(i: int, px: float, reason: str) -> None:
        nonlocal pos, entry, entry_time, realised, peak, max_dd
        nonlocal wins, losses, signal_exits, barrier_exits, gross_win, gross_loss, cool
        pnl = trade_pnl(pos, entry, px, amount, leverage, commission_per_million)
        realised += pnl
        peak = max(peak, realised)
        max_dd = max(max_dd, peak - realised)
        day = times.iloc[i].date()
        daily[day] = daily.get(day, 0.0) + pnl
        if pnl >= 0:
            wins += 1
            gross_win += pnl
        else:
            losses += 1
            gross_loss += -pnl
        if reason == "signal":
            signal_exits += 1
        else:
            barrier_exits += 1
        if return_trades:
            trades.append({
                "entry_time": entry_time,
                "exit_time": times.iloc[i],
                "side": "long" if pos == 1 else "short",
                "entry": entry,
                "exit": px,
                "pnl": pnl,
                "reason": reason,
            })
        pos = 0
        entry = 0.0
        entry_time = None
        cool = cooldown

    for i in range(len(df)):
        cur_sig = int(sig[i])
        px = float(close[i])
        if pos != 0:
            exited = False
            if tp_points > 0:
                if pos == 1 and high[i] >= entry + tp_points * point_size:
                    close_trade(i, entry + tp_points * point_size, "barrier")
                    exited = True
                elif pos == -1 and low[i] <= entry - tp_points * point_size:
                    close_trade(i, entry - tp_points * point_size, "barrier")
                    exited = True
            if not exited and sl_points > 0:
                if pos == 1 and low[i] <= entry - sl_points * point_size:
                    close_trade(i, entry - sl_points * point_size, "barrier")
                    exited = True
                elif pos == -1 and high[i] >= entry + sl_points * point_size:
                    close_trade(i, entry + sl_points * point_size, "barrier")
                    exited = True
            if not exited and exit_mode in {"flip", "any"} and cur_sig == -pos:
                close_trade(i, px, "signal")
                exited = True
                if reverse and cur_sig != 0:
                    pos = cur_sig
                    entry = px
                    entry_time = times.iloc[i]
                    cool = 0
            if not exited and exit_mode == "flat" and cur_sig == 0:
                close_trade(i, px, "signal")
            last_sig = cur_sig
            continue

        if cool > 0:
            cool -= 1
            last_sig = cur_sig
            continue

        enter = False
        if cur_sig != 0:
            if mode == "level":
                enter = True
            elif mode == "change":
                enter = cur_sig != last_sig
        if enter:
            pos = cur_sig
            entry = px
            entry_time = times.iloc[i]
        last_sig = cur_sig

    if pos != 0:
        close_trade(len(df) - 1, float(close[-1]), "signal")

    trade_count = wins + losses
    pf = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
    day_vals = np.array(list(daily.values()), dtype=np.float64) if daily else np.array([0.0])
    result = SimResult(
        mode=mode,
        threshold=threshold,
        exit_mode=exit_mode,
        reverse=reverse,
        cooldown=cooldown,
        trades=trade_count,
        wins=wins,
        losses=losses,
        win_rate=wins / trade_count * 100.0 if trade_count else 0.0,
        profit_factor=pf,
        realised=realised,
        max_drawdown=max_dd,
        median_day=float(np.median(day_vals)),
        avg_day=float(np.mean(day_vals)),
        signal_exits=signal_exits,
        barrier_exits=barrier_exits,
    )
    if return_trades:
        return result, pd.DataFrame(trades)
    return result


def print_results(results: list[SimResult], top: int) -> None:
    rows = sorted(results, key=lambda r: (r.realised, r.median_day, r.profit_factor), reverse=True)[:top]
    print("\n  top by realised PnL")
    print("   # mode   th exit  rev cool trades   wr%    pf   realised    dd  med/day avg/day sig bar")
    print("  --------------------------------------------------------------------------------------------")
    for n, r in enumerate(rows, 1):
        print(
            f"{n:4d} {r.mode:6} {r.threshold:4.2f} {r.exit_mode:5} {r.reverse:3d} {r.cooldown:4d} "
            f"{r.trades:6d} {r.win_rate:5.1f} {r.profit_factor:5.2f} "
            f"${r.realised:+9.2f} ${r.max_drawdown:6.2f} ${r.median_day:+7.2f} ${r.avg_day:+7.2f} "
            f"{r.signal_exits:3d} {r.barrier_exits:3d}"
        )
    rows = sorted(results, key=lambda r: (r.median_day, r.realised, -r.max_drawdown), reverse=True)[:top]
    print("\n  top by median daily PnL")
    print("   # mode   th exit  rev cool trades   wr%    pf   realised    dd  med/day avg/day sig bar")
    print("  --------------------------------------------------------------------------------------------")
    for n, r in enumerate(rows, 1):
        print(
            f"{n:4d} {r.mode:6} {r.threshold:4.2f} {r.exit_mode:5} {r.reverse:3d} {r.cooldown:4d} "
            f"{r.trades:6d} {r.win_rate:5.1f} {r.profit_factor:5.2f} "
            f"${r.realised:+9.2f} ${r.max_drawdown:6.2f} ${r.median_day:+7.2f} ${r.avg_day:+7.2f} "
            f"{r.signal_exits:3d} {r.barrier_exits:3d}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Sequential simulator for ML prediction CSVs")
    ap.add_argument("--pred", default=None)
    ap.add_argument("--long-pred", default=None)
    ap.add_argument("--short-pred", default=None)
    ap.add_argument("--min-edge", type=float, default=0.0,
                    help="combined mode: long_prob must exceed short_prob by this much, and vice versa")
    ap.add_argument("--thresholds", default="0.55,0.58,0.60,0.62,0.65")
    ap.add_argument("--modes", default="level,change")
    ap.add_argument("--exit-modes", default="barrier,flip,flat")
    ap.add_argument("--reverse", default="0,1")
    ap.add_argument("--cooldowns", default="0,1,3,5")
    ap.add_argument("--tp-points", type=float, default=300.0)
    ap.add_argument("--sl-points", type=float, default=300.0)
    ap.add_argument("--point-size", type=float, default=0.01)
    ap.add_argument("--amount", type=float, default=50.0)
    ap.add_argument("--leverage", type=float, default=100.0)
    ap.add_argument("--commission-per-million", type=float, default=30.0)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--trades-out", default=None)
    args = ap.parse_args()

    if args.long_pred and args.short_pred:
        long_df = pd.read_csv(args.long_pred)
        short_df = pd.read_csv(args.short_pred)
        df = long_df.copy()
        long_prob = long_df["prob_win"].to_numpy(np.float64)
        short_prob = short_df["prob_win"].to_numpy(np.float64)
        if len(long_prob) != len(short_prob) or not np.all(long_df["time"].to_numpy() == short_df["time"].to_numpy()):
            raise SystemExit("--long-pred and --short-pred must have matching timestamps")
    else:
        if not args.pred:
            raise SystemExit("provide --pred or both --long-pred and --short-pred")
        df = pd.read_csv(args.pred)
        long_prob = short_prob = None
    results: list[SimResult] = []
    best_key = None
    for mode in [x.strip() for x in args.modes.split(",") if x.strip()]:
        for exit_mode in [x.strip() for x in args.exit_modes.split(",") if x.strip()]:
            for reverse in parse_ints(args.reverse):
                if exit_mode == "barrier" and reverse:
                    continue
                for cooldown in parse_ints(args.cooldowns):
                    for th in parse_floats(args.thresholds):
                        res = simulate(
                            df, th, mode, exit_mode, reverse, cooldown,
                            args.tp_points, args.sl_points, args.point_size,
                            args.amount, args.leverage, args.commission_per_million,
                            precomputed_signal=signal_from_two_probs(long_prob, short_prob, th, args.min_edge) if long_prob is not None else None,
                        )
                        results.append(res)
                        key = (res.realised, res.median_day, res.profit_factor)
                        if best_key is None or key > best_key[0]:
                            best_key = (key, mode, exit_mode, reverse, cooldown, th)
    print_results(results, args.top)

    if args.trades_out and best_key is not None:
        _, mode, exit_mode, reverse, cooldown, th = best_key
        _, trades = simulate(
            df, th, mode, exit_mode, reverse, cooldown,
            args.tp_points, args.sl_points, args.point_size,
            args.amount, args.leverage, args.commission_per_million,
            return_trades=True,
            precomputed_signal=signal_from_two_probs(long_prob, short_prob, th, args.min_edge) if long_prob is not None else None,
        )
        os.makedirs(os.path.dirname(args.trades_out) or ".", exist_ok=True)
        trades.to_csv(args.trades_out, index=False)
        print(f"[ml-sim] wrote trades {args.trades_out}")


if __name__ == "__main__":
    main()
