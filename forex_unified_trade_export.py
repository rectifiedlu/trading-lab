"""Export per-trade logs for selected rows from unified signal results.

This replays the same base execution model used by forex_unified_signal_backtest:
indicator state -> transition entry -> fixed TP/SL or conditional signal exit.
It is intentionally separate from the sweeper so we can inspect candidate rows.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from forex_signal_sweep_common import build_bid_ohlc, map_state_to_ticks
from forex_strategy_common import (
    active_session_allowed,
    build_parser,
    day_ids_from_timestamps,
    default_point_size,
    load_market,
    njit,
)
from forex_unified_signal_backtest import (
    bb_rsi_state,
    bollinger_state,
    cci_state,
    dmi_state,
    donchian_state,
    ema_pair_state,
    ema_price_state,
    keltner_state,
    macd_state,
    psar_state,
    rsi_state,
    stochastic_state,
    supertrend_state,
    volty_tick_state,
)


DEFAULT_CANDIDATES = os.path.join("data", "forex", "unified_trade_export_candidates.csv")
DEFAULT_TRADES_OUT = os.path.join("data", "forex", "unified_candidate_trades.csv")
DEFAULT_SUMMARY_OUT = os.path.join("data", "forex", "unified_candidate_trade_summary.csv")


REASON_SIGNAL = 1
REASON_STOP = 2
REASON_TP = 3
REASON_LIQ = 4


def parse_params(params: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in str(params or "").split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def fparam(params: dict[str, str], key: str, default: float) -> float:
    return float(params.get(key, default))


def iparam(params: dict[str, str], key: str, default: int) -> int:
    return int(float(params.get(key, default)))


def sparam(params: dict[str, str], key: str, default: str) -> str:
    return str(params.get(key, default))


def reason_name(code: int) -> str:
    return {
        REASON_SIGNAL: "signal",
        REASON_STOP: "stop",
        REASON_TP: "tp",
        REASON_LIQ: "liquidation",
    }.get(int(code), "unknown")


@dataclass
class Candidate:
    candidate_id: int
    row: dict[str, str]
    pair: str
    strategy: str
    timeframe: str
    params: dict[str, str]
    params_raw: str
    tp_points: float
    sl_points: float
    point_size: float
    session: int


if njit is None:
    raise RuntimeError("numba is required for unified trade export")


@njit(cache=True)
def _simulate_with_logs_numba(
    bid: np.ndarray,
    ask: np.ndarray,
    ts_ns: np.ndarray,
    day_id: np.ndarray,
    max_days: int,
    state: np.ndarray,
    entry_allowed: np.ndarray,
    tp_points: float,
    sl_points: float,
    point_size: float,
    amount: float,
    compound: bool,
    leverage: float,
    commission_per_million: float,
    ignore_signal_exit_when_bracket: bool,
    max_logs: int,
):
    tp_dist = tp_points * point_size if tp_points > 0.0 else 0.0
    sl_dist = sl_points * point_size if sl_points > 0.0 else 0.0

    entry_i = np.empty(max_logs, dtype=np.int64)
    exit_i = np.empty(max_logs, dtype=np.int64)
    side = np.empty(max_logs, dtype=np.int64)
    reason = np.empty(max_logs, dtype=np.int64)
    entry_px = np.empty(max_logs, dtype=np.float64)
    exit_px = np.empty(max_logs, dtype=np.float64)
    pnl_arr = np.empty(max_logs, dtype=np.float64)
    equity_arr = np.empty(max_logs, dtype=np.float64)
    trade_dd_arr = np.empty(max_logs, dtype=np.float64)
    daily_pnl = np.zeros(max_days, dtype=np.float64)

    prev_state = np.empty(len(state), dtype=np.float64)
    prev_state[0] = 0.0
    for k in range(1, len(state)):
        prev_state[k] = state[k - 1]

    cash = amount
    equity_peak = amount
    max_dd = 0.0
    gross_win = 0.0
    gross_loss = 0.0
    wins = 0
    losses = 0
    long_trades = 0
    short_trades = 0
    stop_losses = 0
    signal_exits = 0
    liquidations = 0
    pos = 0
    entry = 0.0
    units = 0.0
    entry_tick = 0
    cur_trade_drawdown = 0.0
    max_trade_drawdown = 0.0
    worst_trade_pnl = 0.0
    n_logs = 0

    for i in range(len(bid) - 1):
        j = i + 1
        b = bid[j]
        a = ask[j]

        if pos != 0:
            live_u = (b - entry) * units if pos == 1 else (entry - a) * units
            if -live_u > cur_trade_drawdown:
                cur_trade_drawdown = -live_u
            if cur_trade_drawdown < 0.0:
                cur_trade_drawdown = 0.0
            eq = cash + live_u
            if eq > equity_peak:
                equity_peak = eq
            dd = equity_peak - eq
            if dd > max_dd:
                max_dd = dd
            if eq <= 0.0:
                liquidations += 1
                if n_logs < max_logs:
                    entry_i[n_logs] = entry_tick
                    exit_i[n_logs] = j
                    side[n_logs] = pos
                    reason[n_logs] = REASON_LIQ
                    entry_px[n_logs] = entry
                    exit_px[n_logs] = b if pos == 1 else a
                    pnl_arr[n_logs] = -cash
                    equity_arr[n_logs] = 0.0
                    trade_dd_arr[n_logs] = cur_trade_drawdown
                    n_logs += 1
                cash = 0.0
                pos = 0
                break

        close = False
        exit = 0.0
        close_reason = 0
        if pos == 1:
            open_pnl = (b - entry) * units
            open_points = (b - entry) / point_size
            if tp_dist > 0.0 and b >= entry + tp_dist:
                close = True
                exit = b
                close_reason = REASON_TP
            elif sl_dist > 0.0 and open_points <= -sl_points:
                close = True
                exit = b
                close_reason = REASON_STOP
            elif (
                (not (ignore_signal_exit_when_bracket and tp_dist > 0.0 and sl_dist > 0.0))
                and state[i] == -1.0
                and (tp_points <= 0.0 or open_pnl < 0.0)
            ):
                close = True
                exit = b
                close_reason = REASON_SIGNAL
            if close:
                pnl = (exit - entry) * units
                pnl -= abs(exit * units) / 1_000_000.0 * commission_per_million
                cash += pnl
                d = day_id[j]
                if d >= 0 and d < max_days:
                    daily_pnl[d] += pnl
                long_trades += 1
                if close_reason == REASON_STOP:
                    stop_losses += 1
                else:
                    signal_exits += 1
                if cur_trade_drawdown > max_trade_drawdown:
                    max_trade_drawdown = cur_trade_drawdown
                if pnl >= 0.0:
                    wins += 1
                    gross_win += pnl
                else:
                    losses += 1
                    gross_loss += -pnl
                    if pnl < worst_trade_pnl:
                        worst_trade_pnl = pnl
                if n_logs < max_logs:
                    entry_i[n_logs] = entry_tick
                    exit_i[n_logs] = j
                    side[n_logs] = 1
                    reason[n_logs] = close_reason
                    entry_px[n_logs] = entry
                    exit_px[n_logs] = exit
                    pnl_arr[n_logs] = pnl
                    equity_arr[n_logs] = cash
                    trade_dd_arr[n_logs] = cur_trade_drawdown
                    n_logs += 1
                pos = 0
                cur_trade_drawdown = 0.0
                continue
        elif pos == -1:
            open_pnl = (entry - a) * units
            open_points = (entry - a) / point_size
            if tp_dist > 0.0 and a <= entry - tp_dist:
                close = True
                exit = a
                close_reason = REASON_TP
            elif sl_dist > 0.0 and open_points <= -sl_points:
                close = True
                exit = a
                close_reason = REASON_STOP
            elif (
                (not (ignore_signal_exit_when_bracket and tp_dist > 0.0 and sl_dist > 0.0))
                and state[i] == 1.0
                and (tp_points <= 0.0 or open_pnl < 0.0)
            ):
                close = True
                exit = a
                close_reason = REASON_SIGNAL
            if close:
                pnl = (entry - exit) * units
                pnl -= abs(exit * units) / 1_000_000.0 * commission_per_million
                cash += pnl
                d = day_id[j]
                if d >= 0 and d < max_days:
                    daily_pnl[d] += pnl
                short_trades += 1
                if close_reason == REASON_STOP:
                    stop_losses += 1
                else:
                    signal_exits += 1
                if cur_trade_drawdown > max_trade_drawdown:
                    max_trade_drawdown = cur_trade_drawdown
                if pnl >= 0.0:
                    wins += 1
                    gross_win += pnl
                else:
                    losses += 1
                    gross_loss += -pnl
                    if pnl < worst_trade_pnl:
                        worst_trade_pnl = pnl
                if n_logs < max_logs:
                    entry_i[n_logs] = entry_tick
                    exit_i[n_logs] = j
                    side[n_logs] = -1
                    reason[n_logs] = close_reason
                    entry_px[n_logs] = entry
                    exit_px[n_logs] = exit
                    pnl_arr[n_logs] = pnl
                    equity_arr[n_logs] = cash
                    trade_dd_arr[n_logs] = cur_trade_drawdown
                    n_logs += 1
                pos = 0
                cur_trade_drawdown = 0.0
                continue

        if pos == 0 and entry_allowed[i]:
            margin = cash if compound else amount
            if margin > 0.0:
                if state[i] == 1.0 and prev_state[i] != 1.0:
                    entry = a
                    units = (margin * leverage) / entry
                    fee = abs(entry * units) / 1_000_000.0 * commission_per_million
                    cash -= fee
                    d = day_id[j]
                    if d >= 0 and d < max_days:
                        daily_pnl[d] -= fee
                    pos = 1
                    entry_tick = j
                    cur_trade_drawdown = 0.0
                elif state[i] == -1.0 and prev_state[i] != -1.0:
                    entry = b
                    units = (margin * leverage) / entry
                    fee = abs(entry * units) / 1_000_000.0 * commission_per_million
                    cash -= fee
                    d = day_id[j]
                    if d >= 0 and d < max_days:
                        daily_pnl[d] -= fee
                    pos = -1
                    entry_tick = j
                    cur_trade_drawdown = 0.0

        if cash > equity_peak:
            equity_peak = cash
        dd_cash = equity_peak - cash
        if dd_cash > max_dd:
            max_dd = dd_cash

    open_u = 0.0
    open_side = 0
    if pos == 1:
        open_side = 1
        open_u = (bid[-1] - entry) * units
    elif pos == -1:
        open_side = -1
        open_u = (entry - ask[-1]) * units

    realised = cash - amount
    total = realised + open_u
    pf = gross_win / gross_loss if gross_loss > 0.0 else (999.0 if gross_win > 0.0 else 0.0)
    daily_sorted = daily_pnl.copy()
    daily_sorted.sort()
    avg_day = 0.0
    for d in range(max_days):
        avg_day += daily_pnl[d]
    avg_day /= max(max_days, 1)
    if max_days == 0:
        median_day = 0.0
    elif max_days % 2 == 1:
        median_day = daily_sorted[max_days // 2]
    else:
        median_day = 0.5 * (daily_sorted[max_days // 2 - 1] + daily_sorted[max_days // 2])

    return (
        entry_i, exit_i, side, reason, entry_px, exit_px, pnl_arr, equity_arr,
        trade_dd_arr, n_logs, realised, open_u, total, wins, losses, pf, max_dd,
        long_trades, short_trades, stop_losses, signal_exits, liquidations,
        open_side, max_trade_drawdown, avg_day, median_day, worst_trade_pnl,
        daily_pnl,
    )


def read_candidates(path: str, max_candidates: int | None) -> list[Candidate]:
    rows = pd.read_csv(path)
    if max_candidates is not None and max_candidates > 0:
        rows = rows.head(max_candidates)
    out: list[Candidate] = []
    for idx, row in rows.reset_index(drop=True).iterrows():
        raw = row.to_dict()
        params_raw = str(raw.get("params", ""))
        params = parse_params(params_raw)
        pair = str(raw["pair"])
        point = float(raw.get("point_size") or default_point_size(pair))
        out.append(Candidate(
            candidate_id=idx + 1,
            row={k: "" if pd.isna(v) else str(v) for k, v in raw.items()},
            pair=pair,
            strategy=str(raw["strategy"]),
            timeframe=str(raw["timeframe"]),
            params=params,
            params_raw=params_raw,
            tp_points=float(raw["tp_points"]),
            sl_points=float(raw["sl_points"]),
            point_size=point,
            session=int(float(params.get("session", 0))),
        ))
    return out


def candidate_state(c: Candidate, bid: np.ndarray, ask: np.ndarray, ts_ns: np.ndarray, ohlc_cache: dict[str, tuple]) -> np.ndarray:
    tf = c.timeframe
    if tf not in ohlc_cache:
        ohlc_cache[tf] = build_bid_ohlc(bid, ts_ns, tf)
    _, highs, lows, closes, close_idx = ohlc_cache[tf]
    p = c.params
    mode = sparam(p, "mode", "normal")
    st: np.ndarray
    if c.strategy == "keltner":
        st, _ = keltner_state(highs, lows, closes, iparam(p, "length", 20), fparam(p, "mult", 1.5), mode)
        return map_state_to_ticks(len(bid), close_idx, st)
    if c.strategy == "donchian":
        st, _ = donchian_state(highs, lows, closes, iparam(p, "length", 20), mode)
        return map_state_to_ticks(len(bid), close_idx, st)
    if c.strategy == "bollinger":
        st, _ = bollinger_state(highs, lows, closes, iparam(p, "length", 20), fparam(p, "mult", 2.0), mode)
        return map_state_to_ticks(len(bid), close_idx, st)
    if c.strategy == "bb_rsi":
        st, _ = bb_rsi_state(
            highs, lows, closes,
            iparam(p, "bb", 20), fparam(p, "mult", 2.0), iparam(p, "rsi", 14),
            fparam(p, "os", 30), fparam(p, "ob", 70), mode,
        )
        return map_state_to_ticks(len(bid), close_idx, st)
    if c.strategy == "rsi":
        st, _ = rsi_state(closes, iparam(p, "period", 14), sparam(p, "kind", "rsi50"), mode)
        return map_state_to_ticks(len(bid), close_idx, st)
    if c.strategy == "stoch":
        st, _ = stochastic_state(highs, lows, closes, iparam(p, "length", 14), fparam(p, "low", 20), fparam(p, "high", 80), mode)
        return map_state_to_ticks(len(bid), close_idx, st)
    if c.strategy == "macd":
        st, _ = macd_state(closes, iparam(p, "fast", 12), iparam(p, "slow", 26), iparam(p, "signal", 1), fparam(p, "deadband", 0.0), mode)
        return map_state_to_ticks(len(bid), close_idx, st)
    if c.strategy == "ema":
        st, _ = ema_price_state(closes, iparam(p, "length", 21), mode)
        return map_state_to_ticks(len(bid), close_idx, st)
    if c.strategy == "ema_pair":
        st, _ = ema_pair_state(closes, iparam(p, "fast", 6), iparam(p, "slow", 150), mode)
        return map_state_to_ticks(len(bid), close_idx, st)
    if c.strategy == "cci":
        st, _ = cci_state(highs, lows, closes, iparam(p, "length", 20), fparam(p, "threshold", 100), mode)
        return map_state_to_ticks(len(bid), close_idx, st)
    if c.strategy == "dmi":
        st, _ = dmi_state(highs, lows, closes, iparam(p, "di", 14), iparam(p, "adx", 14), fparam(p, "adx_min", 20), mode)
        return map_state_to_ticks(len(bid), close_idx, st)
    if c.strategy == "supertrend":
        st, _, _ = supertrend_state(highs, lows, closes, iparam(p, "length", 10), fparam(p, "mult", 2.0), mode)
        return map_state_to_ticks(len(bid), close_idx, st)
    if c.strategy == "psar":
        st, _ = psar_state(highs, lows, closes, fparam(p, "start", 0.02), fparam(p, "inc", 0.02), fparam(p, "max", 0.2), mode)
        return map_state_to_ticks(len(bid), close_idx, st)
    if c.strategy == "volty":
        st, _ = volty_tick_state(
            len(bid), close_idx, bid, ask, highs, lows, closes,
            iparam(p, "length", 5), fparam(p, "mult", 0.75), mode,
        )
        return st
    raise ValueError(f"unsupported strategy in candidate row: {c.strategy}")


def write_outputs(trades_path: str, summary_path: str, trades: list[dict], summaries: list[dict]) -> None:
    os.makedirs(os.path.dirname(trades_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    trade_fields = [
        "candidate_id", "pair", "strategy", "timeframe", "tp_points", "sl_points",
        "params", "side", "entry_time", "exit_time", "hold_minutes", "entry_px",
        "exit_px", "pnl", "equity", "trade_drawdown", "reason",
    ]
    summary_fields = [
        "candidate_id", "pair", "strategy", "timeframe", "tp_points", "sl_points",
        "params", "total", "realised", "open", "trades", "wins", "losses",
        "win_rate", "profit_factor", "account_dd", "trade_max_dd", "worst_trade",
        "median_trade", "mean_trade", "median_win", "median_loss", "avg_hold_min",
        "median_hold_min", "max_hold_min", "avg_day", "median_day", "positive_days",
        "total_days", "positive_day_rate", "long_trades", "short_trades",
        "stop_losses", "signal_exits", "liquidations",
    ]
    with open(trades_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=trade_fields)
        writer.writeheader()
        writer.writerows(trades)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)


def summarize_candidate(c: Candidate, sim_out, ts_ns: np.ndarray) -> tuple[list[dict], dict]:
    (
        entry_i, exit_i, side, reason, entry_px, exit_px, pnl_arr, equity_arr,
        trade_dd_arr, n_logs, realised, open_u, total, wins, losses, pf, max_dd,
        long_trades, short_trades, stop_losses, signal_exits, liquidations,
        _open_side, max_trade_drawdown, avg_day, median_day, worst_trade_pnl,
        daily_pnl,
    ) = sim_out
    n = int(n_logs)
    trades: list[dict] = []
    pnls = pnl_arr[:n].copy()
    holds = np.empty(n, dtype=np.float64)
    for k in range(n):
        ei = int(entry_i[k])
        xi = int(exit_i[k])
        hold = float(ts_ns[xi] - ts_ns[ei]) / 60_000_000_000.0
        holds[k] = hold
        trades.append({
            "candidate_id": c.candidate_id,
            "pair": c.pair,
            "strategy": c.strategy,
            "timeframe": c.timeframe,
            "tp_points": c.tp_points,
            "sl_points": c.sl_points,
            "params": c.params_raw,
            "side": "long" if int(side[k]) == 1 else "short",
            "entry_time": pd.to_datetime(int(ts_ns[ei]), utc=True).isoformat(),
            "exit_time": pd.to_datetime(int(ts_ns[xi]), utc=True).isoformat(),
            "hold_minutes": round(hold, 4),
            "entry_px": round(float(entry_px[k]), 8),
            "exit_px": round(float(exit_px[k]), 8),
            "pnl": round(float(pnl_arr[k]), 6),
            "equity": round(float(equity_arr[k]), 6),
            "trade_drawdown": round(float(trade_dd_arr[k]), 6),
            "reason": reason_name(int(reason[k])),
        })
    win_rate = float(wins) / n * 100.0 if n else 0.0
    losing = pnls[pnls < 0.0]
    winning = pnls[pnls >= 0.0]
    summary = {
        "candidate_id": c.candidate_id,
        "pair": c.pair,
        "strategy": c.strategy,
        "timeframe": c.timeframe,
        "tp_points": c.tp_points,
        "sl_points": c.sl_points,
        "params": c.params_raw,
        "total": round(float(total), 6),
        "realised": round(float(realised), 6),
        "open": round(float(open_u), 6),
        "trades": n,
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": round(win_rate, 2),
        "profit_factor": round(float(pf), 4),
        "account_dd": round(float(max_dd), 6),
        "trade_max_dd": round(float(max_trade_drawdown), 6),
        "worst_trade": round(float(np.min(pnls)) if n else 0.0, 6),
        "median_trade": round(float(np.median(pnls)) if n else 0.0, 6),
        "mean_trade": round(float(np.mean(pnls)) if n else 0.0, 6),
        "median_win": round(float(np.median(winning)) if len(winning) else 0.0, 6),
        "median_loss": round(float(np.median(losing)) if len(losing) else 0.0, 6),
        "avg_hold_min": round(float(np.mean(holds)) if n else 0.0, 4),
        "median_hold_min": round(float(np.median(holds)) if n else 0.0, 4),
        "max_hold_min": round(float(np.max(holds)) if n else 0.0, 4),
        "avg_day": round(float(avg_day), 6),
        "median_day": round(float(median_day), 6),
        "positive_days": int(np.sum(daily_pnl > 0.0)),
        "total_days": int(len(daily_pnl)),
        "positive_day_rate": round(float(np.mean(daily_pnl > 0.0) * 100.0) if len(daily_pnl) else 0.0, 2),
        "long_trades": int(long_trades),
        "short_trades": int(short_trades),
        "stop_losses": int(stop_losses),
        "signal_exits": int(signal_exits),
        "liquidations": int(liquidations),
    }
    return trades, summary


def print_summary(summaries: list[dict], top: int) -> None:
    if not summaries:
        print("[export] no summaries", flush=True)
        return
    ranked = sorted(
        summaries,
        key=lambda r: (
            float(r["total"]) / max(float(r["account_dd"]), 1e-9),
            float(r["median_day"]) / max(float(r["account_dd"]), 1e-9),
            float(r["total"]),
        ),
        reverse=True,
    )
    headers = ["#", "id", "pair", "strat", "tf", "tp", "sl", "total", "dd", "pnl/dd", "med/day", "wr", "tr", "worst", "med_loss", "posd", "params"]
    rows = []
    for i, r in enumerate(ranked[:top], 1):
        dd = float(r["account_dd"])
        rows.append([
            str(i), str(r["candidate_id"]), r["pair"], r["strategy"], r["timeframe"],
            f"{float(r['tp_points']):g}", f"{float(r['sl_points']):g}",
            f"${float(r['total']):+.2f}", f"${dd:.2f}",
            f"{float(r['total']) / max(dd, 1e-9):.2f}",
            f"${float(r['median_day']):+.2f}", f"{float(r['win_rate']):.1f}",
            str(r["trades"]), f"${float(r['worst_trade']):+.2f}",
            f"${float(r['median_loss']):+.2f}",
            f"{r['positive_days']}/{r['total_days']}",
            r["params"],
        ])
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    print("\n  exported candidates ranked by total/account DD", flush=True)
    print("  " + " ".join(headers[i].rjust(widths[i]) for i in range(len(headers))), flush=True)
    print("  " + "-" * (sum(widths) + len(widths) - 1), flush=True)
    for row in rows:
        print("  " + " ".join(str(row[i]).rjust(widths[i]) for i in range(len(headers))), flush=True)


def main() -> None:
    ap = build_parser("Export trade logs for unified candidate rows", "unused.csv")
    ap.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    ap.add_argument("--out-trades", default=DEFAULT_TRADES_OUT)
    ap.add_argument("--out-summary", default=DEFAULT_SUMMARY_OUT)
    ap.add_argument("--max-candidates", type=int, default=0, help="0 = all rows")
    ap.add_argument("--max-logs-per-candidate", type=int, default=200000)
    args = ap.parse_args()

    candidates = read_candidates(args.candidates, None if args.max_candidates <= 0 else args.max_candidates)
    needed_pairs = sorted({c.pair for c in candidates})
    if not args.pairs or args.pairs == ["XAUUSD"]:
        args.pairs = needed_pairs

    ticks, t0 = load_market(args)
    all_trades: list[dict] = []
    summaries: list[dict] = []

    by_pair: dict[str, list[Candidate]] = {}
    for c in candidates:
        by_pair.setdefault(c.pair, []).append(c)

    for pair, pair_candidates in by_pair.items():
        g = ticks[ticks["pair"] == pair].sort_values("timestamp").reset_index(drop=True)
        if g.empty:
            print(f"[export] skip {pair}: no ticks", flush=True)
            continue
        bid = g["bid"].to_numpy(np.float64)
        ask = g["ask"].to_numpy(np.float64)
        ts_ns = g["timestamp"].astype("int64").to_numpy(np.int64)
        day_id, max_days = day_ids_from_timestamps(ts_ns)
        session_cache: dict[int, np.ndarray] = {}
        ohlc_cache: dict[str, tuple] = {}
        print(f"[export] {pair} ticks={len(g):,} candidates={len(pair_candidates)}", flush=True)

        for n, c in enumerate(pair_candidates, 1):
            state = candidate_state(c, bid, ask, ts_ns, ohlc_cache)
            allowed = session_cache.get(c.session)
            if allowed is None:
                allowed = active_session_allowed(ts_ns, c.session)
                session_cache[c.session] = allowed
            sim_out = _simulate_with_logs_numba(
                bid, ask, ts_ns, day_id, max_days, state, allowed,
                float(c.tp_points), float(c.sl_points), float(c.point_size),
                float(args.amount), bool(args.compound), float(args.leverage),
                float(args.commission_per_million),
                bool(c.tp_points > 0.0 and c.sl_points > 0.0),
                int(args.max_logs_per_candidate),
            )
            trades, summary = summarize_candidate(c, sim_out, ts_ns)
            all_trades.extend(trades)
            summaries.append(summary)
            print(
                f"[export] {pair} {n}/{len(pair_candidates)} id={c.candidate_id} "
                f"{c.strategy} {c.timeframe} tr={summary['trades']} "
                f"total=${summary['total']:+.2f} dd=${summary['account_dd']:.2f}",
                flush=True,
            )

    write_outputs(args.out_trades, args.out_summary, all_trades, summaries)
    print_summary(summaries, args.top)
    print(f"[export] wrote trades {args.out_trades}", flush=True)
    print(f"[export] wrote summary {args.out_summary}", flush=True)
    print(f"[export] elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
