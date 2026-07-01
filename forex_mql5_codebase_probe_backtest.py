"""MQL5 CodeBase strategy probes on MT5 tick data.

This file only ports clear, deterministic CodeBase strategy descriptions.
It is intentionally a sandbox: if a rule looks promising here, move it into
the unified tester/paper trader after validating the exact mechanics.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from forex_signal_sweep_common import (
    build_bid_ohlc,
    map_state_to_ticks,
    rma,
    rolling_high_prev,
    rolling_low_prev,
    simulate_state_strategy,
)
from forex_strategy_common import (
    active_session_allowed,
    build_parser,
    default_point_size,
    load_market,
    parse_num_list,
    parse_str_list,
)
from forex_unified_signal_backtest import (
    apply_mode,
    print_unified_sections,
    rolling_max,
    rolling_min,
    rolling_sma,
    true_range,
    write_unified_csv,
)


DEFAULT_STRATEGIES: list[str] = [
    "inside_bar_filtered",
    "universal_breakout_box",
    "quantum_xau_silver",
    "indiana_mean",
    "xander_keltner_ema",
]
DEFAULT_TIMEFRAMES = ["1m", "3m", "5m", "15m", "1h"]
DEFAULT_MODES = ["normal", "invert"]
DEFAULT_SESSIONS = [-1, 0, 1, 2]
GOLD_TP = [0, 50, 100, 150, 200, 300, 400]
GOLD_SL = [0, 50, 100, 150, 200, 300, 400]
FX_TP = [0, 10, 20, 30, 50, 80, 100]
FX_SL = [0, 10, 20, 30, 50, 80, 100]


def pair_tp_sl(pair: str) -> tuple[list[float], list[float]]:
    if pair.upper().startswith("XAU"):
        return GOLD_TP, GOLD_SL
    return FX_TP, FX_SL


def one_bar_events(signal: np.ndarray) -> np.ndarray:
    out = np.zeros(len(signal), dtype=np.float64)
    prev = 0.0
    for i, v in enumerate(signal):
        if v != 0.0 and v != prev:
            out[i] = v
        prev = v
    return out


def rsi_values(closes: np.ndarray, period: int) -> np.ndarray:
    delta = np.diff(closes, prepend=closes[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = rma(gain, period)
    avg_loss = rma(loss, period)
    rs = avg_gain / np.where(avg_loss == 0.0, np.nan, avg_loss)
    out = 100.0 - (100.0 / (1.0 + rs))
    out[(avg_loss == 0.0) & np.isfinite(avg_gain)] = 100.0
    return out


def adx_values(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
    up = highs - np.roll(highs, 1)
    down = np.roll(lows, 1) - lows
    up[0] = 0.0
    down[0] = 0.0
    plus_dm = np.where((up > down) & (up > 0.0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0.0), down, 0.0)
    tr = true_range(highs, lows, closes)
    atr = rma(tr, period)
    plus_di = 100.0 * rma(plus_dm, period) / np.maximum(atr, 1e-12)
    minus_di = 100.0 * rma(minus_dm, period) / np.maximum(atr, 1e-12)
    dx = 100.0 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-12)
    return rma(dx, period)


def inside_bar_filtered_tick_state(
    bid: np.ndarray,
    close_idx: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    mode: str,
    min_body_pct: float = 80.0,
    max_inside_pct: float = 50.0,
    expiry_bars: int = 3,
) -> tuple[np.ndarray, str]:
    state = np.zeros(len(bid), dtype=np.float64)
    pending_side = 0
    pending_level = 0.0
    expiry_tick = -1
    bar_for_tick = np.zeros(len(bid), dtype=np.int64)
    prev = 0
    for b, idx in enumerate(close_idx):
        bar_for_tick[prev:idx + 1] = b
        prev = idx + 1
    bar_for_tick[prev:] = len(close_idx) - 1
    for b in range(2, len(closes)):
        start = int(close_idx[b - 1]) + 1
        end = int(close_idx[b]) if b < len(close_idx) else len(bid) - 1
        if pending_side != 0:
            for i in range(start, min(end + 1, len(bid))):
                if i > expiry_tick:
                    pending_side = 0
                    break
                px = float(bid[i])
                if pending_side == 1 and px >= pending_level:
                    state[i] = 1.0
                    pending_side = 0
                    break
                if pending_side == -1 and px <= pending_level:
                    state[i] = -1.0
                    pending_side = 0
                    break
        main = b - 2
        inside = b - 1
        main_range = highs[main] - lows[main]
        if main_range <= 0.0:
            continue
        if highs[inside] >= highs[main] or lows[inside] <= lows[main]:
            continue
        body_pct = abs(closes[main] - opens[main]) / main_range * 100.0
        inside_pct = (highs[inside] - lows[inside]) / main_range * 100.0
        if body_pct < min_body_pct or inside_pct > max_inside_pct:
            continue
        pending_side = 1 if closes[main] > opens[main] else -1
        pending_level = highs[main] if pending_side == 1 else lows[main]
        expiry_bar = min(b + expiry_bars, len(close_idx) - 1)
        expiry_tick = int(close_idx[expiry_bar])
    return apply_mode(state, mode), (
        f"main_body>={min_body_pct:g};inside<={max_inside_pct:g};expiry={expiry_bars};mode={mode}"
    )


def universal_breakout_box_tick_state(
    bid: np.ndarray,
    ts_ns: np.ndarray,
    mode: str,
    start_hour: int = 0,
    box_h1_bars: int = 48,
    expiry_minutes: int = 1110,
) -> tuple[np.ndarray, str]:
    state = np.zeros(len(bid), dtype=np.float64)
    tf_ns = 3_600_000_000_000
    day_ns = 86_400_000_000_000
    bucket = ts_ns // tf_ns
    h1_open_idx = []
    h1_high = []
    h1_low = []
    cur = int(bucket[0])
    hi = lo = float(bid[0])
    start_i = 0
    for i, px0 in enumerate(bid):
        b = int(bucket[i])
        px = float(px0)
        if b != cur:
            h1_open_idx.append(start_i)
            h1_high.append(hi)
            h1_low.append(lo)
            cur = b
            start_i = i
            hi = lo = px
        else:
            hi = max(hi, px)
            lo = min(lo, px)
    h1_open_idx.append(start_i)
    h1_high.append(hi)
    h1_low.append(lo)
    h1_open_idx = np.array(h1_open_idx, dtype=np.int64)
    h1_high = np.array(h1_high, dtype=np.float64)
    h1_low = np.array(h1_low, dtype=np.float64)
    h1_ts = ts_ns[h1_open_idx]
    h1_hour = ((h1_ts % day_ns) // tf_ns).astype(np.int64)
    active = False
    upper = lower = 0.0
    expiry_ns = 0
    used = False
    for j in range(box_h1_bars, len(h1_open_idx)):
        if h1_hour[j - box_h1_bars] == start_hour:
            upper = float(np.max(h1_high[j - box_h1_bars:j]))
            lower = float(np.min(h1_low[j - box_h1_bars:j]))
            expiry_ns = int(h1_ts[j] + expiry_minutes * 60 * 1_000_000_000)
            active = True
            used = False
        start = int(h1_open_idx[j])
        end = int(h1_open_idx[j + 1]) if j + 1 < len(h1_open_idx) else len(bid)
        if not active or used:
            continue
        for i in range(start, end):
            if ts_ns[i] > expiry_ns:
                active = False
                break
            px = float(bid[i])
            if px >= upper:
                state[i] = 1.0
                used = True
                break
            if px <= lower:
                state[i] = -1.0
                used = True
                break
    return apply_mode(state, mode), f"start_h={start_hour};h1bars={box_h1_bars};expiry_min={expiry_minutes};mode={mode}"


def quantum_xau_silver_state(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    mode: str,
    rsi_period: int = 14,
    adx_period: int = 14,
    fast_ma: int = 50,
    slow_ma: int = 200,
    chaos_period: int = 20,
    threshold: float = 0.65,
    chaos_threshold: float = 0.35,
) -> tuple[np.ndarray, str]:
    rsi = rsi_values(closes, rsi_period)
    adx = adx_values(highs, lows, closes, adx_period)
    fast = rolling_sma(closes, fast_ma)
    slow = rolling_sma(closes, slow_ma)
    returns = np.zeros(len(closes), dtype=np.float64)
    returns[1:] = np.log(np.maximum(closes[1:], 1e-12)) - np.log(np.maximum(closes[:-1], 1e-12))
    chaos = np.full(len(closes), np.nan, dtype=np.float64)
    for i in range(chaos_period, len(closes)):
        w = returns[i - chaos_period + 1:i + 1]
        sd = np.std(w)
        chaos[i] = abs(float(np.mean(w)) / sd) if sd > 0 else 0.5
    state = np.zeros(len(closes), dtype=np.float64)
    point_norm = np.maximum(np.nanmedian(np.abs(np.diff(closes))), 1e-12)
    for i in range(len(closes)):
        if not all(np.isfinite(x) for x in (rsi[i], adx[i], fast[i], slow[i], chaos[i])):
            continue
        if chaos[i] < chaos_threshold:
            continue
        # Source has random adaptive state term; excluded for deterministic probe.
        strength = 2.0 * 0.5 * (rsi[i] - 50.0) / 50.0
        strength += 1.0 * 0.5 * (adx[i] - 20.0) / 30.0
        strength += 1.0 * 0.5 * (fast[i] - slow[i]) / point_norm
        strength += 1.0 * 0.5 * chaos[i]
        if strength > threshold:
            state[i] = 1.0
        elif strength < -threshold:
            state[i] = -1.0
    return apply_mode(one_bar_events(state), mode), (
        f"rsi={rsi_period};adx={adx_period};ma={fast_ma}/{slow_ma};chaos={chaos_period};"
        f"thr={threshold:g};chaos_thr={chaos_threshold:g};deterministic=1;mode={mode}"
    )


def indiana_state(
    bid: np.ndarray,
    close_idx: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    length: int,
    mode: str,
) -> tuple[np.ndarray, str]:
    upper = rolling_high_prev(highs, length)
    lower = rolling_low_prev(lows, length)
    upper_tick = map_state_to_ticks(len(bid), close_idx, upper)
    lower_tick = map_state_to_ticks(len(bid), close_idx, lower)
    raw = np.zeros(len(bid), dtype=np.float64)
    prev = float(bid[0])
    for i in range(1, len(bid)):
        px = float(bid[i])
        # Indiana Jones Mean Reversion: fade n-lookback high/low touches.
        if np.isfinite(upper_tick[i]) and prev < upper_tick[i] and px >= upper_tick[i]:
            raw[i] = -1.0
        elif np.isfinite(lower_tick[i]) and prev > lower_tick[i] and px <= lower_tick[i]:
            raw[i] = 1.0
        prev = px
    return apply_mode(raw, mode), f"length={length};mode={mode}"


def rma_values(values: np.ndarray, length: int) -> np.ndarray:
    return rma(values.astype(np.float64), length)


def ema_values(values: np.ndarray, length: int) -> np.ndarray:
    out = np.empty(len(values), dtype=np.float64)
    alpha = 2.0 / (length + 1.0)
    val = float(values[0])
    for i, px in enumerate(values):
        val = alpha * float(px) + (1.0 - alpha) * val
        out[i] = val
    return out


def rolling_avg_range(highs: np.ndarray, lows: np.ndarray, length: int) -> np.ndarray:
    return rolling_sma(highs - lows, length)


def xander_keltner_ema_state(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    length: int,
    fast: int,
    slow: int,
    mode: str,
) -> tuple[np.ndarray, str]:
    center = ema_values(closes, length)
    channel_range = rolling_avg_range(highs, lows, length)
    upper = center + channel_range
    lower = center - channel_range
    fast_ema = ema_values(closes, fast)
    slow_ema = ema_values(closes, slow)
    state = np.zeros(len(closes), dtype=np.float64)
    for i in range(2, len(closes)):
        if not (np.isfinite(upper[i - 1]) and np.isfinite(lower[i - 1])):
            continue
        # Source GetSignal(): close2 outside previous channel, close1 back inside, EMA trend agrees.
        if closes[i - 2] < lower[i - 1] and closes[i - 1] > lower[i - 1] and fast_ema[i - 1] > slow_ema[i - 1]:
            state[i] = 1.0
        elif closes[i - 2] > upper[i - 1] and closes[i - 1] < upper[i - 1] and fast_ema[i - 1] < slow_ema[i - 1]:
            state[i] = -1.0
    return apply_mode(state, mode), f"keltn={length};range=sma_hl;fast={fast};slow={slow};mode={mode}"


def build_states(strategy: str, pair: str, tf: str, bid: np.ndarray, ts_ns: np.ndarray, mode: str):
    del pair
    opens, highs, lows, closes, close_idx = build_bid_ohlc(bid, ts_ns, tf)
    if strategy == "inside_bar_filtered":
        # Source: 002-Inside-Bar.mq5. The separate "inside bar.mq5" file is an indicator, not an EA.
        yield inside_bar_filtered_tick_state(bid, close_idx, opens, highs, lows, closes, mode)
        return
    if strategy == "universal_breakout_box":
        # Source: Universal Breakout Study.mq5. Box logic is H1 based even when tf is swept.
        yield universal_breakout_box_tick_state(bid, ts_ns, mode)
        return
    if strategy == "quantum_xau_silver":
        # Source: Quantum_XAUUSD_Silver_Trader.mq5. Random/learning state is intentionally omitted.
        state, params = quantum_xau_silver_state(highs, lows, closes, mode)
        yield map_state_to_ticks(len(bid), close_idx, state), params
        return
    if strategy == "indiana_mean":
        for length in (20, 50, 100):
            yield indiana_state(bid, close_idx, highs, lows, length, mode)
        return
    if strategy == "xander_keltner_ema":
        state, params = xander_keltner_ema_state(highs, lows, closes, 50, 10, 200, mode)
        yield map_state_to_ticks(len(bid), close_idx, state), params
        return
    raise ValueError(f"unknown or not-yet-ported MQL5 strategy: {strategy}")


def main() -> None:
    ap = build_parser("MQL5 CodeBase strategy probe backtest", "forex_mql5_codebase_probe_results.csv")
    ap.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES))
    ap.add_argument("--modes", default=",".join(DEFAULT_MODES))
    ap.add_argument("--sessions", default=",".join(str(x) for x in DEFAULT_SESSIONS))
    ap.add_argument("--workers", type=int, default=1, help="reserved; state count is small enough to run sequentially")
    args = ap.parse_args()

    strategies = parse_str_list(args.strategies, DEFAULT_STRATEGIES)
    if not strategies:
        raise SystemExit("no MQL5 probe strategies are currently ported")
    timeframes = parse_str_list(args.timeframes, DEFAULT_TIMEFRAMES)
    modes = parse_str_list(args.modes, DEFAULT_MODES)
    sessions = [int(x) for x in parse_num_list(args.sessions, DEFAULT_SESSIONS)]
    ticks, t0 = load_market(args)
    del t0

    all_results = []
    for pair, g in ticks.groupby("pair", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        bid = g["bid"].to_numpy(np.float64)
        ask = g["ask"].to_numpy(np.float64)
        ts_ns = g["timestamp"].astype("int64").to_numpy()
        point_size = args.point_size or default_point_size(pair)
        tp_default, sl_default = pair_tp_sl(pair)
        tps = parse_num_list(args.tp_points, tp_default)
        sls = parse_num_list(args.sl_points, sl_default)
        session_cache = {s: active_session_allowed(ts_ns, s) for s in sessions}
        print(
            f"[mql5-probe] {pair} ticks={len(g):,} point={point_size:g} "
            f"strategies={','.join(strategies)} tf={','.join(timeframes)} "
            f"tp={tps} sl={sls} sessions={sessions}",
            flush=True,
        )
        pair_results = []
        for strategy, tf, mode in product(strategies, timeframes, modes):
            built = list(build_states(strategy, pair, tf, bid, ts_ns, mode))
            print(f"[mql5-probe] {pair} {strategy} {tf} {mode} states={len(built)}", flush=True)
            for state, params in built:
                exit_state = -state
                for tp, sl, sess in product(tps, sls, sessions):
                    result = simulate_state_strategy(
                        pair,
                        strategy,
                        f"{params};session={sess}",
                        tf,
                        bid,
                        ask,
                        ts_ns,
                        state,
                        exit_state,
                        session_cache[int(sess)],
                        float(tp),
                        0.0,
                        float(sl),
                        0.0,
                        point_size,
                        args.amount,
                        args.compound,
                        args.leverage,
                        args.commission_per_million,
                        args.side,
                        ignore_signal_exit_when_bracket=True,
                    )
                    pair_results.append(result)
        all_results.extend(pair_results)
        if args.out:
            write_unified_csv(args.out, [r for r in all_results if r.trades >= args.min_trades])
            print(f"[mql5-probe] partial wrote {args.out}", flush=True)

    results = [r for r in all_results if r.trades >= args.min_trades]
    write_unified_csv(args.out, results)
    print_unified_sections(sorted(results, key=lambda r: r.total, reverse=True), args.top)
    print(f"[mql5-probe] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
