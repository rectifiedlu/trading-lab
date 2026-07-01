"""Keltner regime + oscillator re-entry backtest.

Signal model:
    - Keltner channel defines regime from closed candles.
    - Outside Keltner: trade Keltner continuation.
    - Inside Keltner: trade RSI/Stoch re-entry mean reversion.
    - Entries are candle-close signals mapped to ticks.
    - TP/SL exits are tick based via simulate_state_strategy.
"""

from __future__ import annotations

import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import product

import numpy as np

from forex_signal_sweep_common import build_bid_ohlc, ema, map_state_to_ticks, rma, simulate_state_strategy
from forex_strategy_common import (
    active_session_allowed,
    build_parser,
    default_point_size,
    load_market,
    parse_num_list,
    parse_str_list,
    write_results,
)


DEFAULT_PAIRS = ["XAUUSD"]
DEFAULT_KELTNER_TIMEFRAMES = ["10m","15m"]
DEFAULT_MEANREV_TIMEFRAMES = ["5m", "10m", "15m"]
DEFAULT_KELTNER_LENGTHS = [20, 34, 48, 55, 64, 89, 144]
DEFAULT_KELTNER_MULTS = [1.0, 1.5, 1.75, 2.0, 2.5]
DEFAULT_KELTNER_EXIT_MODES = ["signal", "neutral", "inside"]
DEFAULT_MEANREV_TYPES = ["rsi","stoch"]
DEFAULT_RSI_PERIODS = [7, 10, 14]
DEFAULT_RSI_OVERSOLD = [25,30,35,40]
DEFAULT_RSI_OVERBOUGHT = [75,70,65,60]
DEFAULT_STOCH_LENGTHS = [5, 7, 10, 14]
DEFAULT_STOCH_LOWS = [15, 20, 25, 30]
DEFAULT_STOCH_HIGHS = [85, 80, 75, 70]
GOLD_TP = [0, 100, 200, 300, 400]
GOLD_SL = [0, 200,400,600,800]
FX_TP = [0, 20, 30, 50, 80, 100]
FX_SL = [0, 20, 30, 50, 80, 100]
DEFAULT_SESSIONS = [-1, 0, 1]


def true_range(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    prev = np.empty_like(closes)
    prev[0] = closes[0]
    prev[1:] = closes[:-1]
    return np.maximum(highs - lows, np.maximum(np.abs(highs - prev), np.abs(lows - prev)))


def keltner_regime(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    length: int,
    mult: float,
) -> tuple[np.ndarray, np.ndarray]:
    tr = true_range(highs, lows, closes)
    center = ema(closes, length)
    atr = rma(tr, length)
    upper = center + mult * atr
    lower = center - mult * atr
    trend = np.zeros(len(closes), dtype=np.float64)
    trend[closes > upper] = 1.0
    trend[closes < lower] = -1.0
    trend[~np.isfinite(atr)] = 0.0
    inside = np.isfinite(atr) & (closes <= upper) & (closes >= lower)
    return trend, inside.astype(np.float64)


def keltner_neutral_exit_state(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    length: int,
    mult: float,
) -> np.ndarray:
    del highs, lows, mult
    center = ema(closes, length)
    out = np.zeros(len(closes), dtype=np.float64)
    out[closes <= center] = -1.0
    out[closes >= center] = 1.0
    out[~np.isfinite(center)] = 0.0
    return out


def keltner_inside_exit_state(trend: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Exit Keltner continuation trades when the next closed candle returns inside."""
    out = np.zeros(len(trend), dtype=np.float64)
    for i in range(1, len(trend)):
        if inside[i] < 0.5:
            continue
        if trend[i - 1] > 0.0:
            out[i] = -1.0
        elif trend[i - 1] < 0.0:
            out[i] = 1.0
    return out


def rsi_reentry_state(closes: np.ndarray, period: int, oversold: float, overbought: float) -> np.ndarray:
    delta = np.empty(len(closes), dtype=np.float64)
    delta[0] = 0.0
    delta[1:] = closes[1:] - closes[:-1]
    gains = np.maximum(delta, 0.0)
    losses = np.maximum(-delta, 0.0)
    avg_gain = rma(gains, period)
    avg_loss = rma(losses, period)
    rs = avg_gain / np.maximum(avg_loss, 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    state = np.zeros(len(closes), dtype=np.float64)
    long_armed = False
    short_armed = False
    for i in range(len(closes)):
        if not np.isfinite(rsi[i]):
            continue
        if rsi[i] <= oversold:
            long_armed = True
            short_armed = False
        elif long_armed and rsi[i] > oversold:
            state[i] = 1.0
            long_armed = False
        if rsi[i] >= overbought:
            short_armed = True
            long_armed = False
        elif short_armed and rsi[i] < overbought:
            state[i] = -1.0
            short_armed = False
    return state


def stoch_reentry_state(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    length: int,
    low_level: float,
    high_level: float,
) -> np.ndarray:
    k = np.full(len(closes), np.nan, dtype=np.float64)
    for i in range(length - 1, len(closes)):
        lo = float(np.min(lows[i - length + 1:i + 1]))
        hi = float(np.max(highs[i - length + 1:i + 1]))
        k[i] = 100.0 * (closes[i] - lo) / max(hi - lo, 1e-12)
    state = np.zeros(len(closes), dtype=np.float64)
    long_armed = False
    short_armed = False
    for i in range(len(closes)):
        if not np.isfinite(k[i]):
            continue
        if k[i] <= low_level:
            long_armed = True
            short_armed = False
        elif long_armed and k[i] > low_level:
            state[i] = 1.0
            long_armed = False
        if k[i] >= high_level:
            short_armed = True
            long_armed = False
        elif short_armed and k[i] < high_level:
            state[i] = -1.0
            short_armed = False
    return state


def pair_thresholds(lows: list[float], highs: list[float]) -> list[tuple[float, float]]:
    if len(lows) == len(highs):
        return list(zip(lows, highs))
    if len(lows) == 1:
        return [(lows[0], high) for high in highs]
    if len(highs) == 1:
        return [(low, highs[0]) for low in lows]
    raise SystemExit("threshold lists must have same length, or one side must contain one value")


def cross_thresholds(lows: list[float], highs: list[float]) -> list[tuple[float, float]]:
    return [(low, high) for low in lows for high in highs if low < high]


def default_tp_sl_for_pair(pair: str) -> tuple[list[float], list[float]]:
    if pair.upper() == "XAUUSD":
        return GOLD_TP, GOLD_SL
    return FX_TP, FX_SL


def fmt_list(values) -> str:
    return ",".join(f"{v:g}" if isinstance(v, float) else str(v) for v in values)


def main() -> None:
    ap = build_parser("Keltner regime + oscillator re-entry sweep", "forex_keltner_regime_results.csv")
    ap.set_defaults(pairs=DEFAULT_PAIRS)
    ap.add_argument("--keltner-timeframes", default=",".join(DEFAULT_KELTNER_TIMEFRAMES))
    ap.add_argument("--meanrev-timeframes", default=",".join(DEFAULT_MEANREV_TIMEFRAMES))
    ap.add_argument("--keltner-lengths", default=None)
    ap.add_argument("--keltner-mults", default=None)
    ap.add_argument("--keltner-exit-modes", default=",".join(DEFAULT_KELTNER_EXIT_MODES),
                    help="sl=0 exit behavior: signal, neutral, inside")
    ap.add_argument("--meanrev-types", default=",".join(DEFAULT_MEANREV_TYPES), help="rsi,stoch")
    ap.add_argument("--rsi-periods", default=None)
    ap.add_argument("--rsi-oversold", default=None)
    ap.add_argument("--rsi-overbought", default=None)
    ap.add_argument("--stoch-lengths", default=None)
    ap.add_argument("--stoch-lows", default=None)
    ap.add_argument("--stoch-highs", default=None)
    ap.add_argument("--sessions", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--progress-every", type=int, default=1000,
                    help="Print progress every N completed combos.")
    args = ap.parse_args()

    k_tfs = parse_str_list(args.keltner_timeframes, DEFAULT_KELTNER_TIMEFRAMES)
    m_tfs = parse_str_list(args.meanrev_timeframes, DEFAULT_MEANREV_TIMEFRAMES)
    k_lengths = [int(x) for x in parse_num_list(args.keltner_lengths, DEFAULT_KELTNER_LENGTHS)]
    k_mults = parse_num_list(args.keltner_mults, DEFAULT_KELTNER_MULTS)
    k_exit_modes = parse_str_list(args.keltner_exit_modes, DEFAULT_KELTNER_EXIT_MODES)
    bad_exit_modes = [m for m in k_exit_modes if m not in {"signal", "neutral", "inside"}]
    if bad_exit_modes:
        raise SystemExit(f"unknown --keltner-exit-modes values: {','.join(bad_exit_modes)}")
    meanrev_types = parse_str_list(args.meanrev_types, DEFAULT_MEANREV_TYPES)
    rsi_periods = [int(x) for x in parse_num_list(args.rsi_periods, DEFAULT_RSI_PERIODS)]
    rsi_pairs = pair_thresholds(
        parse_num_list(args.rsi_oversold, DEFAULT_RSI_OVERSOLD),
        parse_num_list(args.rsi_overbought, DEFAULT_RSI_OVERBOUGHT),
    )
    stoch_lengths = [int(x) for x in parse_num_list(args.stoch_lengths, DEFAULT_STOCH_LENGTHS)]
    stoch_pairs = pair_thresholds(
        parse_num_list(args.stoch_lows, DEFAULT_STOCH_LOWS),
        parse_num_list(args.stoch_highs, DEFAULT_STOCH_HIGHS),
    )
    sessions = [int(x) for x in parse_num_list(args.sessions, DEFAULT_SESSIONS)]

    print(
        "[keltner-regime] grid "
        f"k_tf={','.join(k_tfs)} k_len={fmt_list(k_lengths)} k_mult={fmt_list(k_mults)} "
        f"k_exit={','.join(k_exit_modes)} mean_tf={','.join(m_tfs)} mean={','.join(meanrev_types)} "
        f"rsi_periods={fmt_list(rsi_periods)} rsi_pairs={','.join(f'{a:g}/{b:g}' for a,b in rsi_pairs)} "
        f"stoch_lengths={fmt_list(stoch_lengths)} stoch_pairs={','.join(f'{a:g}/{b:g}' for a,b in stoch_pairs)} "
        f"sessions={fmt_list(sessions)} workers={args.workers}",
        flush=True,
    )

    ticks, t0 = load_market(args)
    results = []

    for pair, g in ticks.groupby("pair", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        bid = g["bid"].to_numpy(np.float64)
        ask = g["ask"].to_numpy(np.float64)
        ts_ns = g["timestamp"].astype("int64").to_numpy(np.int64)
        point_size = float(args.point_size or default_point_size(pair))
        default_tp, default_sl = default_tp_sl_for_pair(pair)
        tp_values = parse_num_list(args.tp_points, default_tp)
        sl_values = parse_num_list(args.sl_points, default_sl)
        session_cache = {s: active_session_allowed(ts_ns, s) for s in sessions}
        print(
            f"[keltner-regime] {pair} ticks={len(g):,} point={point_size:g} "
            f"tp={tp_values} sl={sl_values}",
            flush=True,
        )

        k_cache = {}
        for tf in k_tfs:
            _, highs, lows, closes, close_idx = build_bid_ohlc(bid, ts_ns, tf)
            print(
                f"[keltner-regime] {pair} build keltner tf={tf} bars={len(closes):,} "
                f"states={len(k_lengths) * len(k_mults):,}",
                flush=True,
            )
            for length, mult in product(k_lengths, k_mults):
                trend_c, inside_c = keltner_regime(highs, lows, closes, length, mult)
                inside_exit_c = keltner_inside_exit_state(trend_c, inside_c)
                neutral_exit_c = keltner_neutral_exit_state(highs, lows, closes, length, mult)
                k_cache[(tf, length, mult)] = (
                    map_state_to_ticks(len(bid), close_idx, trend_c),
                    map_state_to_ticks(len(bid), close_idx, inside_c),
                    map_state_to_ticks(len(bid), close_idx, inside_exit_c),
                    map_state_to_ticks(len(bid), close_idx, neutral_exit_c),
                )

        meanrev_cache = {}
        for tf in m_tfs:
            _, highs, lows, closes, close_idx = build_bid_ohlc(bid, ts_ns, tf)
            tf_state_count = 0
            if "rsi" in meanrev_types:
                for period, (os_, ob) in product(rsi_periods, rsi_pairs):
                    params = f"mean=rsi;mtf={tf};rsi={period};os={os_:g};ob={ob:g}"
                    st = rsi_reentry_state(closes, period, os_, ob)
                    meanrev_cache[("rsi", tf, period, os_, ob)] = (
                        params,
                        map_state_to_ticks(len(bid), close_idx, st),
                    )
                    tf_state_count += 1
            if "stoch" in meanrev_types:
                for length, (lo, hi) in product(stoch_lengths, stoch_pairs):
                    params = f"mean=stoch;mtf={tf};stoch={length};low={lo:g};high={hi:g}"
                    st = stoch_reentry_state(highs, lows, closes, length, lo, hi)
                    meanrev_cache[("stoch", tf, length, lo, hi)] = (
                        params,
                        map_state_to_ticks(len(bid), close_idx, st),
                    )
                    tf_state_count += 1
            print(
                f"[keltner-regime] {pair} build meanrev tf={tf} bars={len(closes):,} states={tf_state_count:,}",
                flush=True,
            )

        combos = []
        exit_counts = {m: 0 for m in k_exit_modes}
        for k_tf, k_len, k_mult, k_exit_mode in product(k_tfs, k_lengths, k_mults, k_exit_modes):
            for mean_key in meanrev_cache:
                for tp, sl, sess in product(tp_values, sl_values, sessions):
                    if k_exit_mode != "signal" and float(sl) > 0.0:
                        continue
                    combos.append((k_tf, k_len, k_mult, k_exit_mode, mean_key, tp, sl, sess))
                    exit_counts[k_exit_mode] = exit_counts.get(k_exit_mode, 0) + 1
        total_combos = len(combos)
        print(
            f"[keltner-regime] {pair} meanrev_states={len(meanrev_cache):,} "
            f"k_states={len(k_cache):,} combos={total_combos:,} "
            f"exit_counts={exit_counts} workers={args.workers}",
            flush=True,
        )

        def make_entry_state(trend_state, inside, mean_state):
            state = np.zeros(len(bid), dtype=np.float64)
            outside = inside < 0.5
            state[outside] = trend_state[outside]
            state[~outside] = mean_state[~outside]
            return state

        def make_exit_state(state, inside_exit_state, neutral_exit_state, k_exit_mode):
            if k_exit_mode == "signal":
                return state
            if k_exit_mode == "inside":
                exit_state = state.copy()
                use_inside_exit = inside_exit_state != 0.0
                exit_state[use_inside_exit] = inside_exit_state[use_inside_exit]
                return exit_state
            if k_exit_mode == "neutral":
                exit_state = state.copy()
                use_neutral_exit = neutral_exit_state != 0.0
                exit_state[use_neutral_exit] = neutral_exit_state[use_neutral_exit]
                return exit_state
            raise ValueError(f"unknown keltner exit mode: {k_exit_mode}")

        def run_combo(combo):
            k_tf, k_len, k_mult, k_exit_mode, mean_key, tp, sl, sess = combo
            trend_state, inside, inside_exit_state, neutral_exit_state = k_cache[(k_tf, k_len, k_mult)]
            mean_params, mean_state = meanrev_cache[mean_key]
            state = make_entry_state(trend_state, inside, mean_state)
            exit_state = make_exit_state(state, inside_exit_state, neutral_exit_state, k_exit_mode)
            params = f"k_tf={k_tf};k_len={k_len};k_mult={k_mult:g};{mean_params};k_exit={k_exit_mode};session={sess}"
            mtf = "-"
            for part in mean_params.split(";"):
                if part.startswith("mtf="):
                    mtf = part.split("=", 1)[1]
                    break
            return simulate_state_strategy(
                pair, "keltner_regime", params, f"{k_tf}/{mtf}",
                bid, ask, ts_ns, state, exit_state, session_cache[int(sess)],
                tp, 0.0, sl, 0.0, point_size,
                args.amount, args.compound, args.leverage, args.commission_per_million,
                args.side, reverse_on_signal=False,
                ignore_signal_exit_when_bracket=(tp > 0 and sl > 0),
            )

        done = 0
        last_progress_t = time.time()
        max_pending = max(args.workers * 4, args.workers)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            pending = deque()
            combo_iter = iter(combos)

            def submit_next() -> bool:
                try:
                    combo = next(combo_iter)
                except StopIteration:
                    return False
                pending.append(ex.submit(run_combo, combo))
                return True

            for _ in range(min(max_pending, total_combos)):
                submit_next()

            while pending:
                fut = pending.popleft()
                results.append(fut.result())
                done += 1
                submit_next()

                now = time.time()
                if done % max(1, args.progress_every) == 0 or now - last_progress_t >= 30.0:
                    last_progress_t = now
                    pct = done / max(1, total_combos) * 100.0
                    elapsed = now - t0
                    rate = done / max(elapsed, 1e-9)
                    remaining = (total_combos - done) / max(rate, 1e-9)
                    print(
                        f"[keltner-regime] {pair} progress {done:,}/{total_combos:,} "
                        f"({pct:.1f}%) pending={len(pending):,} "
                        f"rate={rate:.1f}/s eta={remaining/60.0:.1f}m",
                        flush=True,
                    )
        pair_results = sum(1 for r in results if r.pair == pair and r.trades >= args.min_trades)
        print(
            f"[keltner-regime] {pair} complete combos={done:,} "
            f"kept_rows_for_pair={pair_results:,} total_rows_so_far={len(results):,}",
            flush=True,
        )

    filtered = [r for r in results if r.trades >= args.min_trades]
    write_results(args.out, filtered, args.top, args.sort_by)
    print(f"[keltner-regime] wrote {args.out} elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
