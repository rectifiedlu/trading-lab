"""Backtest symbolic future-return formulas with threshold and TP/SL sweeps."""
from __future__ import annotations

import csv
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from forex_signal_sweep_common import build_bid_ohlc
from forex_strategy_common import TradeResult, active_session_allowed, build_parser, default_point_size, load_market, parse_num_list, parse_str_list, print_ranked_sections

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None

EXIT_MODES = {"opposite": 0, "neutral": 1, "fixed": 2, "fixed_signal": 3}
DEFAULT_THRESHOLDS = [20, 40, 60, 100, 150, 200, 300]
GOLD_TP = [0, 100, 200, 300, 400, 600]
GOLD_SL = [0, 100, 200, 300, 400, 600]
FX_TP = [0, 15, 30, 45, 60, 90]
FX_SL = [0, 15, 30, 45, 60, 90]


if njit is not None:
    @njit(cache=True)
    def _build_tick_to_candle(n_ticks, close_idx):
        out = np.empty(n_ticks, dtype=np.int64)
        start = 0
        for c in range(len(close_idx)):
            end = int(close_idx[c])
            if end >= n_ticks:
                end = n_ticks - 1
            for i in range(start, end + 1):
                out[i] = c
            start = end + 1
        for i in range(start, n_ticks):
            out[i] = len(close_idx) - 1
        return out

    @njit(cache=True)
    def _build_lagged(open_, high, low, close, window):
        n = len(close)
        x = np.empty((n, window * 4), dtype=np.float64)
        valid = np.zeros(n, dtype=np.bool_)
        for i in range(window - 1, n):
            col = 0
            for lag in range(window):
                j = i - lag
                x[i, col] = open_[j]
                x[i, col + 1] = high[j]
                x[i, col + 2] = low[j]
                x[i, col + 3] = close[j]
                col += 4
            valid[i] = True
        return x, valid

    @njit(cache=True)
    def _sig(score, threshold, invert):
        if not np.isfinite(score):
            return 0
        if score >= threshold:
            return -1 if invert else 1
        if score <= -threshold:
            return 1 if invert else -1
        return 0

    @njit(cache=True)
    def _simulate(bid, ask, close_idx, tick_to_candle, scores, entry_allowed, day_id, max_days,
                  threshold, invert, tp_mode, sl_mode, tp_points, sl_points, point_size,
                  amount, compound, leverage, commission_per_million, side_filter_code):
        daily = np.zeros(max_days, dtype=np.float64)
        trade_pnls = np.zeros(len(close_idx) + 1, dtype=np.float64)
        cash = amount
        equity_peak = amount
        cum_pnl = 0.0
        cum_peak = 0.0
        max_dd = 0.0
        cum_dd = 0.0
        gross_win = 0.0
        gross_loss = 0.0
        trades = wins = losses = longs = shorts = stops = sig_exits = 0
        worst = 0.0
        max_trade_dd = 0.0
        open_unrealized = 0.0
        candle_i = 0
        tick_floor = 0
        pending_side = 0

        while candle_i < len(close_idx):
            if pending_side != 0:
                side = pending_side
                pending_side = 0
                entry_tick = tick_floor
                if entry_tick >= len(bid):
                    break
                candle_i = int(tick_to_candle[entry_tick])
            else:
                side = _sig(scores[candle_i], threshold, invert)
                if side == 0 or side_filter_code == -side or not entry_allowed[candle_i]:
                    candle_i += 1
                    continue
                entry_tick = int(close_idx[candle_i]) + 1
                if entry_tick < tick_floor:
                    entry_tick = tick_floor
                if entry_tick >= len(bid):
                    break
                if not entry_allowed[int(tick_to_candle[entry_tick])]:
                    candle_i += 1
                    continue

            margin = cash if compound else amount
            if margin <= 0.0:
                break
            entry = ask[entry_tick] if side == 1 else bid[entry_tick]
            units = (margin * leverage) / max(entry, 1e-12)
            fee_in = abs(entry * units) / 1_000_000.0 * commission_per_million
            cash -= fee_in
            d0 = int(day_id[entry_tick])
            if 0 <= d0 < max_days:
                daily[d0] -= fee_in
            trade_start_cash = cash
            exit_tick = len(bid) - 1
            result_points = ((bid[exit_tick] if side == 1 else ask[exit_tick]) - entry) / point_size * side
            reason = 4
            reverse_side = 0
            trade_max = 0.0
            next_signal_candle = int(tick_to_candle[entry_tick])

            for ti in range(entry_tick + 1, len(bid)):
                live_points = (bid[ti] - entry) / point_size if side == 1 else (entry - ask[ti]) / point_size
                adverse = -live_points if live_points < 0.0 else 0.0
                tdd = adverse * point_size * units
                if tdd > trade_max:
                    trade_max = tdd
                live_eq = cash + live_points * point_size * units
                if live_eq > equity_peak:
                    equity_peak = live_eq
                live_dd = equity_peak - live_eq
                if live_dd > max_dd:
                    max_dd = live_dd

                if (tp_mode == 2 or tp_mode == 3) and tp_points > 0.0 and live_points >= tp_points:
                    result_points = tp_points
                    exit_tick = ti
                    reason = 3
                    break
                if (sl_mode == 2 or sl_mode == 3) and sl_points > 0.0 and live_points <= -sl_points:
                    result_points = -sl_points
                    exit_tick = ti
                    reason = 2
                    break

                current_candle = int(tick_to_candle[ti])
                while next_signal_candle < current_candle and next_signal_candle < len(close_idx):
                    sig = _sig(scores[next_signal_candle], threshold, invert)
                    mode = tp_mode if live_points >= 0.0 else sl_mode
                    exit_now = False
                    if mode == 0:
                        exit_now = sig == -side
                    elif mode == 1:
                        exit_now = sig != side
                    elif mode == 3:
                        exit_now = sig == -side
                    if exit_now:
                        result_points = live_points
                        exit_tick = ti
                        reason = 1
                        if sig == -side:
                            reverse_side = sig
                        break
                    next_signal_candle += 1
                if reason == 1:
                    break

            if reason == 4:
                open_unrealized = result_points * point_size * units
                break
            exit_px = entry + result_points * point_size if side == 1 else entry - result_points * point_size
            fee_out = abs(exit_px * units) / 1_000_000.0 * commission_per_million
            pnl = result_points * point_size * units - fee_out
            cash += pnl
            full_trade_pnl = cash - trade_start_cash
            if cash > equity_peak:
                equity_peak = cash
            closed_dd = equity_peak - cash
            if closed_dd > max_dd:
                max_dd = closed_dd
            cum_pnl += full_trade_pnl
            if cum_pnl > cum_peak:
                cum_peak = cum_pnl
            local_cum_dd = cum_peak - cum_pnl
            if local_cum_dd > cum_dd:
                cum_dd = local_cum_dd
            d = int(day_id[exit_tick])
            if 0 <= d < max_days:
                daily[d] += pnl
            trades += 1
            if side == 1:
                longs += 1
            else:
                shorts += 1
            if reason == 2:
                stops += 1
            elif reason == 1:
                sig_exits += 1
            if full_trade_pnl >= 0.0:
                wins += 1
                gross_win += full_trade_pnl
            else:
                losses += 1
                gross_loss += -full_trade_pnl
                if full_trade_pnl < worst:
                    worst = full_trade_pnl
            if trade_max > max_trade_dd:
                max_trade_dd = trade_max
            trade_pnls[trades - 1] = full_trade_pnl
            tick_floor = exit_tick + 1
            if tick_floor >= len(bid):
                break
            if reason == 1 and reverse_side != 0 and side_filter_code != -reverse_side:
                pending_side = reverse_side
                continue
            candle_i = int(tick_to_candle[tick_floor])

        return (cash - amount, open_unrealized, trades, wins, losses, gross_win, gross_loss, max_dd, cum_dd,
                longs, shorts, stops, sig_exits, worst, max_trade_dd, daily, trade_pnls[:trades])


def build_tick_to_candle(n_ticks: int, close_idx: np.ndarray) -> np.ndarray:
    if njit is not None:
        return _build_tick_to_candle(n_ticks, close_idx.astype(np.int64))
    out = np.empty(n_ticks, dtype=np.int64)
    start = 0
    for c, end in enumerate(close_idx):
        out[start:int(end) + 1] = c
        start = int(end) + 1
    out[start:] = len(close_idx) - 1
    return out


def build_lagged(open_, high, low, close, window: int):
    if njit is not None:
        return _build_lagged(open_, high, low, close, int(window))
    x = np.empty((len(close), window * 4), dtype=np.float64)
    valid = np.zeros(len(close), dtype=np.bool_)
    for i in range(window - 1, len(close)):
        row = []
        for lag in range(window):
            j = i - lag
            row.extend([open_[j], high[j], low[j], close[j]])
        x[i] = row
        valid[i] = True
    return x, valid



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
def default_tp_sl_for_pair(pair: str) -> tuple[list[float], list[float]]:
    return (GOLD_TP, GOLD_SL) if pair.upper().startswith("XAU") else (FX_TP, FX_SL)


def parse_exit_modes(value: str | None, default: list[str]) -> list[str]:
    modes = parse_str_list(value, default)
    bad = [m for m in modes if m not in EXIT_MODES]
    if bad:
        raise SystemExit(f"bad exit mode(s): {bad}; valid={sorted(EXIT_MODES)}")
    return modes


def effective_modes(modes: list[str], points: float) -> list[str]:
    if points > 0:
        return modes
    return [m for m in modes if m in {"opposite", "neutral"}] or ["opposite", "neutral"]


def profit_factor(gross_win: float, gross_loss: float) -> float:
    return gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)


def day_ids(ts_ns: np.ndarray) -> tuple[np.ndarray, int]:
    day_ns = np.int64(86_400_000_000_000)
    days = (ts_ns // day_ns).astype(np.int64)
    first = int(days[0]) if len(days) else 0
    ids = (days - first).astype(np.int64)
    return ids, int(ids[-1]) + 1 if len(ids) else 1


def load_artifact(meta_path: Path):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    model_path = Path(meta.get("model_path", ""))
    if not model_path.is_absolute():
        model_path = meta_path.parent / model_path.name
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return meta, model


def simulate_one(pair, meta, bid, ask, ts_ns, close_idx, scores, threshold, mode_name, tp_mode, sl_mode, tp, sl, args) -> TradeResult:
    point = float(meta.get("point_size") or default_point_size(pair))
    candle_ts = ts_ns[close_idx]
    entry_allowed = active_session_allowed(candle_ts, int(meta["session"]))
    tick_to_candle = build_tick_to_candle(len(bid), close_idx)
    did, max_days = day_ids(ts_ns)
    side_code = 0 if args.side == "both" else (1 if args.side == "long" else -1)
    out = _simulate(
        bid, ask, close_idx.astype(np.int64), tick_to_candle, scores.astype(np.float64), entry_allowed.astype(np.bool_),
        did.astype(np.int64), int(max_days), float(threshold), 1 if mode_name == "invert" else 0,
        EXIT_MODES[tp_mode], EXIT_MODES[sl_mode], float(tp), float(sl), point,
        args.amount, bool(args.compound), args.leverage, args.commission_per_million, side_code,
    )
    realised, open_u, trades, wins, losses, gw, gl, max_dd, cum_dd, longs, shorts, stops, sig, worst, tr_max, daily, pnls = out
    r = TradeResult(
        pair=pair, strategy="symbolic_return",
        params=(f"expr={meta.get('expression','')};threshold={threshold:g};mode={mode_name};tp_mode={tp_mode};"
                f"sl_mode={sl_mode};session={meta['session']};window={meta['window']};horizon={meta['horizon']};"
                f"backend={meta.get('backend','')};file={Path(meta.get('model_path','')).name}"),
        timeframe=str(meta["timeframe"]), tp_points=float(tp), sl_points=float(sl), point_size=point,
        realised=float(realised), open_unrealized=float(open_u), total=float(realised + open_u), trades=int(trades),
        wins=int(wins), losses=int(losses), win_rate=float(wins / trades * 100.0 if trades else 0.0),
        profit_factor=profit_factor(float(gw), float(gl)), max_drawdown=float(max_dd), long_trades=int(longs),
        short_trades=int(shorts), stop_losses=int(stops), signal_exits=int(sig), liquidations=0,
        account_dead=False, open_side="-", open_bps=0.0,
    )
    extras = {
        "threshold": threshold, "mode": mode_name, "tp_mode": tp_mode, "sl_mode": sl_mode,
        "eval_session": int(meta["session"]), "window": int(meta["window"]), "horizon": int(meta["horizon"]),
        "backend": meta.get("backend", ""), "model_file": Path(meta.get("model_path", "")).name,
        "cum_max_drawdown": float(cum_dd), "trade_max_drawdown": float(tr_max), "worst_trade_pnl": float(worst),
        "median_loss": float(np.median(pnls[pnls < 0.0]) if len(pnls) and np.any(pnls < 0.0) else 0.0),
        "avg_day": float(np.mean(daily) if len(daily) else 0.0),
        "median_day": float(np.median(daily) if len(daily) else 0.0),
    }
    for name, value in extras.items():
        setattr(r, name, value)
    return r


def write_symbolic_csv(path: str, rows: list[TradeResult]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows.sort(key=lambda r: (r.total, r.profit_factor, -r.max_drawdown), reverse=True)
    fields = ["pair", "strategy", "timeframe", "threshold", "mode", "tp_mode", "tp_points", "sl_mode", "sl_points",
              "realised", "open_unrealized", "total", "trades", "win_rate", "profit_factor", "max_drawdown",
              "cum_max_drawdown", "trade_max_drawdown", "worst_trade_pnl", "avg_day", "median_day", "long_trades",
              "short_trades", "stop_losses", "signal_exits", "eval_session", "window", "horizon", "backend", "model_file", "params"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            w.writerow([getattr(r, field, "") for field in fields])


def main() -> None:
    ap = build_parser("Symbolic future-return formula backtest", "forex_symbolic_return_results.csv")
    ap.add_argument("--model-dir", default=os.path.join("data", "forex", "symbolic_models"))
    ap.add_argument("--model-glob", default="symbolic_*.json")
    ap.add_argument("--thresholds", default=",".join(str(x) for x in DEFAULT_THRESHOLDS))
    ap.add_argument("--modes", default="normal,invert")
    ap.add_argument("--tp-modes", default="fixed,fixed_signal,opposite,neutral")
    ap.add_argument("--sl-modes", default="fixed,fixed_signal,opposite,neutral")
    args = ap.parse_args()

    paths = sorted(Path(args.model_dir).glob(args.model_glob))
    if not paths:
        raise SystemExit(f"no model json files found: {args.model_dir}/{args.model_glob}")
    ticks, _ = load_market(args)
    thresholds = parse_num_list(args.thresholds, DEFAULT_THRESHOLDS)
    modes = parse_str_list(args.modes, ["normal", "invert"])
    tp_modes = parse_exit_modes(args.tp_modes, ["fixed", "fixed_signal", "opposite", "neutral"])
    sl_modes = parse_exit_modes(args.sl_modes, ["fixed", "fixed_signal", "opposite", "neutral"])
    results: list[TradeResult] = []
    t0 = time.time()
    if os.path.exists(args.out):
        os.remove(args.out)
        print(f"[symbt] overwrite out={args.out}", flush=True)
    print(
        f"[symbt] plan models={len(paths):,} thresholds={thresholds} modes={modes} "
        f"tp_modes={tp_modes} sl_modes={sl_modes} out={args.out}",
        flush=True,
    )

    for n, path in enumerate(paths, 1):
        meta, model = load_artifact(path)
        pair = str(meta["pair"]).upper()
        g = ticks[ticks["pair"].str.upper() == pair].sort_values("timestamp").reset_index(drop=True)
        if g.empty:
            print(f"[symbt] skip {path.name}: no ticks for {pair}", flush=True)
            continue
        bid = g["bid"].to_numpy(np.float64)
        ask = g["ask"].to_numpy(np.float64)
        ts_ns = g["timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64").astype(np.int64)
        open_, high, low, close, close_idx = build_bid_ohlc(bid, ts_ns, str(meta["timeframe"]))
        x, valid = build_lagged(open_, high, low, close, int(meta["window"]))
        scores = np.full(len(close), np.nan, dtype=np.float64)
        if np.any(valid):
            scores[valid] = np.asarray(model.predict(x[valid]), dtype=np.float64)
        tp_values = parse_num_list(args.tp_points, default_tp_sl_for_pair(pair)[0])
        sl_values = parse_num_list(args.sl_points, default_tp_sl_for_pair(pair)[1])
        model_rows = 0
        combo_total = 0
        for _tp in tp_values:
            for _sl in sl_values:
                combo_total += len(effective_modes(tp_modes, float(_tp))) * len(effective_modes(sl_modes, float(_sl)))
        combo_total *= len(thresholds) * len(modes)
        combo_done = 0
        model_start = time.time()
        print(
            f"[symbt] model start {n:,}/{len(paths):,} {path.name} "
            f"candles={len(close):,} ticks={len(g):,} combos={combo_total:,}",
            flush=True,
        )
        for threshold in thresholds:
            for mode_name in modes:
                if mode_name not in {"normal", "invert"}:
                    raise SystemExit(f"bad mode: {mode_name}")
                for tp in tp_values:
                    for sl in sl_values:
                        for tp_mode in effective_modes(tp_modes, float(tp)):
                            for sl_mode in effective_modes(sl_modes, float(sl)):
                                r = simulate_one(pair, meta, bid, ask, ts_ns, close_idx, scores, threshold, mode_name, tp_mode, sl_mode, tp, sl, args)
                                if r.trades >= args.min_trades:
                                    results.append(r)
                                    model_rows += 1
                                combo_done += 1
                                if combo_done == combo_total or combo_done % 500 == 0:
                                    print(
                                        f"[symbt] model {n:,}/{len(paths):,} "
                                        f"combos={progress_text(combo_done, combo_total, model_start)} "
                                        f"rows={model_rows:,} total_rows={len(results):,}",
                                        flush=True,
                                    )
        print(f"[symbt] model done {n:,}/{len(paths):,} {path.name} rows={model_rows:,} total_rows={len(results):,} overall={progress_text(n, len(paths), t0)}", flush=True)
        write_symbolic_csv(args.out, results)

    filtered = [r for r in results if r.trades >= args.min_trades]
    write_symbolic_csv(args.out, filtered)
    print_ranked_sections(filtered, args.top)
    print(f"[symbt] wrote {args.out} rows={len(filtered):,} elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()



