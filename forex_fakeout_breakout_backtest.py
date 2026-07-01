"""Rolling high/low fakeout and breakout-continuation backtest.

Signals are candle-confirmed from bid OHLC:
    continue: candle sweeps previous rolling high/low and closes outside it
    fade:     candle sweeps previous rolling high/low and closes back inside it

Execution modes:
    tick:   sparse trigger enters on the next tick after the signal candle;
            TP/SL are checked on bid/ask ticks.
    close:  sparse trigger enters on the next candle close, using median spread;
            TP/SL are checked only on candle closes. If a candle closes past TP/SL,
            the exit records that close price, not the exact threshold.
    candle: alias for close, kept for old commands.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

import numpy as np

from forex_signal_sweep_common import build_bid_ohlc
from forex_strategy_common import (
    active_session_allowed,
    build_parser,
    day_ids_from_timestamps,
    default_point_size,
    load_market,
    parse_num_list,
    parse_str_list,
    simulate_triggers,
    write_results,
)


DEFAULT_TIMEFRAMES = ["1m", "3m", "5m"]
DEFAULT_LOOKBACKS = [8, 13, 21, 34, 55]
DEFAULT_SWEEPS = [0, 50, 100, 150]
DEFAULT_INSIDE = [0, 25, 50]
DEFAULT_WICKS = [0.0, 0.35, 0.5]
DEFAULT_BODY_REQ = [0, 1]
DEFAULT_MODES = ["continue", "fade"]
DEFAULT_EXECUTION = ["tick", "close"]
DEFAULT_SESSIONS = [-1, 0, 1]
DEFAULT_TP = [0, 200, 300, 400]
DEFAULT_SL = [0, 200, 300, 400]


def fakeout_state(
    highs: np.ndarray,
    lows: np.ndarray,
    opens: np.ndarray,
    closes: np.ndarray,
    lookback: int,
    sweep_points: float,
    inside_points: float,
    wick_ratio: float,
    body_required: int,
    mode: str,
    point_size: float,
) -> np.ndarray:
    out = np.zeros(len(closes), dtype=np.int8)
    sweep = sweep_points * point_size
    inside = inside_points * point_size
    rng = np.maximum(highs - lows, 1e-12)
    for i in range(lookback, len(closes)):
        prev_high = float(np.max(highs[i - lookback:i]))
        prev_low = float(np.min(lows[i - lookback:i]))
        upper_wick = (highs[i] - max(opens[i], closes[i])) / rng[i]
        lower_wick = (min(opens[i], closes[i]) - lows[i]) / rng[i]

        if highs[i] >= prev_high + sweep:
            if mode == "fade":
                ok = closes[i] <= prev_high - inside
                ok = ok and upper_wick >= wick_ratio
                ok = ok and (not body_required or closes[i] < opens[i])
                if ok:
                    out[i] = -1
            else:
                ok = closes[i] >= prev_high + inside
                ok = ok and (not body_required or closes[i] > opens[i])
                if ok:
                    out[i] = 1

        if out[i] == 0 and lows[i] <= prev_low - sweep:
            if mode == "fade":
                ok = closes[i] >= prev_low + inside
                ok = ok and lower_wick >= wick_ratio
                ok = ok and (not body_required or closes[i] > opens[i])
                if ok:
                    out[i] = 1
            else:
                ok = closes[i] <= prev_low - inside
                ok = ok and (not body_required or closes[i] < opens[i])
                if ok:
                    out[i] = -1
    return out


def candle_triggers_to_ticks(
    n_ticks: int,
    close_idx: np.ndarray,
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    long_trigger = np.zeros(n_ticks, dtype=np.bool_)
    short_trigger = np.zeros(n_ticks, dtype=np.bool_)
    for i, st in enumerate(state):
        if st == 0:
            continue
        tick_i = int(close_idx[i]) + 1
        if tick_i >= n_ticks:
            continue
        if st > 0:
            long_trigger[tick_i] = True
        else:
            short_trigger[tick_i] = True
    return long_trigger, short_trigger


def candle_triggers_to_candles(
    n_bars: int,
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    long_trigger = np.zeros(n_bars, dtype=np.bool_)
    short_trigger = np.zeros(n_bars, dtype=np.bool_)
    for i, st in enumerate(state):
        bar_i = i + 1
        if st == 0 or bar_i >= n_bars:
            continue
        if st > 0:
            long_trigger[bar_i] = True
        else:
            short_trigger[bar_i] = True
    return long_trigger, short_trigger


def main() -> None:
    ap = build_parser("Rolling fakeout/breakout-continuation backtest", "forex_fakeout_breakout_results.csv")
    ap.add_argument("--lookbacks", default=None)
    ap.add_argument("--sweep-points", default=None)
    ap.add_argument("--inside-points", default=None)
    ap.add_argument("--wick-ratios", default=None)
    ap.add_argument("--body-required", default=None, help="0,1")
    ap.add_argument("--modes", default=None, help="continue,fade")
    ap.add_argument("--execution-mode", default=None, help="tick,close,candle")
    ap.add_argument("--sessions", default=None, help="-1 outside, 0 all, 1 inside major sessions")
    ap.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    args = ap.parse_args()

    timeframes = parse_str_list(args.timeframes, DEFAULT_TIMEFRAMES)
    lookbacks = [int(x) for x in parse_num_list(args.lookbacks, DEFAULT_LOOKBACKS)]
    sweeps = parse_num_list(args.sweep_points, DEFAULT_SWEEPS)
    insides = parse_num_list(args.inside_points, DEFAULT_INSIDE)
    wicks = parse_num_list(args.wick_ratios, DEFAULT_WICKS)
    body_reqs = [int(x) for x in parse_num_list(args.body_required, DEFAULT_BODY_REQ)]
    modes = parse_str_list(args.modes, DEFAULT_MODES)
    executions = parse_str_list(args.execution_mode, DEFAULT_EXECUTION)
    sessions = [int(x) for x in parse_num_list(args.sessions, DEFAULT_SESSIONS)]
    tps = parse_num_list(args.tp_points, DEFAULT_TP)
    sls = parse_num_list(args.sl_points, DEFAULT_SL)

    ticks, t0 = load_market(args)
    results = []
    combo_count = (
        len(timeframes) * len(lookbacks) * len(sweeps) * len(insides) *
        len(wicks) * len(body_reqs) * len(modes) * len(executions) *
        len(sessions) * len(tps) * len(sls)
    )
    print(f"[fakeout] combos_per_pair={combo_count:,} workers={args.workers}", flush=True)

    for pair, g in ticks.groupby("pair", sort=False):
        g = g.sort_values("timestamp")
        bid = g["bid"].to_numpy(np.float64)
        ask = g["ask"].to_numpy(np.float64)
        ts_ns = g["timestamp"].astype("int64").to_numpy(np.int64)
        point_size = float(args.point_size or default_point_size(pair))
        median_spread = float(np.nanmedian(ask - bid)) if len(ask) else 0.0
        session_tick = {s: active_session_allowed(ts_ns, s) for s in sessions}
        day_tick, max_day_tick = day_ids_from_timestamps(ts_ns)
        print(f"[fakeout] {pair} ticks={len(g):,} point={point_size:g} spread_med={median_spread:g}", flush=True)

        candle_cache = {}
        state_cache = {}
        for tf in timeframes:
            opens, highs, lows, closes, close_idx = build_bid_ohlc(bid, ts_ns, tf)
            candle_ts = ts_ns[close_idx]
            candle_cache[tf] = (opens, highs, lows, closes, close_idx, candle_ts)
            for lb, sw, inside, wick, body, mode in product(lookbacks, sweeps, insides, wicks, body_reqs, modes):
                state_cache[(tf, lb, sw, inside, wick, body, mode)] = fakeout_state(
                    highs, lows, opens, closes, lb, sw, inside, wick, body, mode, point_size,
                )

        combos = list(product(
            timeframes, lookbacks, sweeps, insides, wicks, body_reqs,
            modes, executions, sessions, tps, sls,
        ))

        def run_combo(combo):
            tf, lb, sw, inside, wick, body, mode, exe, sess, tp, sl = combo
            opens, highs, lows, closes, close_idx, candle_ts = candle_cache[tf]
            state = state_cache[(tf, lb, sw, inside, wick, body, mode)]
            params = (
                f"lookback={lb};sweep={sw:g};inside={inside:g};wick={wick:g};"
                f"body={body};mode={mode};exec={exe};session={sess}"
            )
            if exe == "tick":
                long_trig, short_trig = candle_triggers_to_ticks(len(bid), close_idx, state)
                allowed = session_tick[int(sess)]
                long_trig &= allowed
                short_trig &= allowed
                return simulate_triggers(
                    pair, "fakeout", params, tf, bid, ask,
                    long_trig, short_trig, short_trig, long_trig,
                    tp, sl, point_size, args.amount, args.compound, args.leverage,
                    args.commission_per_million, args.side,
                    day_id=day_tick, max_days=max_day_tick,
                )
            if exe in ("close", "candle"):
                long_trig, short_trig = candle_triggers_to_candles(len(closes), state)
                allowed = active_session_allowed(candle_ts, int(sess))
                long_trig &= allowed
                short_trig &= allowed
                candle_ask = closes + median_spread
                day_candle, max_day_candle = day_ids_from_timestamps(candle_ts)
                exec_label = "close" if exe == "candle" else exe
                close_params = (
                    f"lookback={lb};sweep={sw:g};inside={inside:g};wick={wick:g};"
                    f"body={body};mode={mode};exec={exec_label};session={sess}"
                )
                return simulate_triggers(
                    pair, "fakeout", close_params, tf, closes, candle_ask,
                    long_trig, short_trig, short_trig, long_trig,
                    tp, sl, point_size, args.amount, args.compound, args.leverage,
                    args.commission_per_million, args.side,
                    day_id=day_candle, max_days=max_day_candle,
                )
            raise ValueError(f"unsupported execution mode: {exe}")

        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_combo, c) for c in combos]
            for fut in as_completed(futs):
                results.append(fut.result())
                done += 1
                if done % max(1, len(combos) // 10) == 0:
                    print(f"[fakeout] {pair} progress {done}/{len(combos)}", flush=True)

    filtered = [r for r in results if r.trades >= args.min_trades]
    write_results(args.out, filtered, args.top, args.sort_by)
    print(f"[fakeout] wrote {args.out} elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
