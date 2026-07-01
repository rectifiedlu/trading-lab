"""Asia range -> London push -> New York reversal backtest.

Idea:
    - Build an Asia range.
    - London window must push outside Asia high/low.
    - New York window looks for reversal confirmation against that London push.
    - Execute on the next tick after confirmation, then exit tick-by-tick.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from forex_asia_sweep_backtest import build_bid_candles
from forex_strategy_common import (
    TradeResult,
    build_parser,
    commission,
    default_point_size,
    load_market,
    njit,
    parse_num_list,
    parse_str_list,
    timeframe_to_ns,
    units_for_margin,
    write_results,
)


GOLD_TP = [100, 150, 200, 300, 400, 500]
GOLD_SL = [100, 150, 200, 300, 400, 500]
FX_TP = [20, 30, 50, 65, 80, 100]
FX_SL = [20, 30, 50, 65, 80, 100]


@dataclass
class NyTrade:
    pair: str
    params: str
    timeframe: str
    date: str
    side: str
    push: str
    entry_i: int
    exit_i: int
    entry_px: float
    exit_px: float
    pnl: float
    reason: str
    asia_high: float
    asia_low: float
    london_high: float
    london_low: float
    confirm_close: float


if njit is not None:
    @njit(cache=True)
    def _exit_trade_numba(
        side: int,
        entry_i: int,
        end_i: int,
        bid: np.ndarray,
        ask: np.ndarray,
        has_tp: bool,
        tp_px: float,
        has_sl: bool,
        sl_px: float,
    ) -> tuple[int, float, int]:
        if end_i >= len(bid):
            end_i = len(bid) - 1
        for i in range(entry_i + 1, end_i + 1):
            if side == 1:
                if has_sl and bid[i] <= sl_px:
                    return i, float(bid[i]), 1
                if has_tp and bid[i] >= tp_px:
                    return i, float(bid[i]), 2
            else:
                if has_sl and ask[i] >= sl_px:
                    return i, float(ask[i]), 1
                if has_tp and ask[i] <= tp_px:
                    return i, float(ask[i]), 2
        return end_i, float(bid[end_i] if side == 1 else ask[end_i]), 3
else:
    _exit_trade_numba = None


def _parse_hhmm(s: str) -> int:
    h, m = s.split(":", 1)
    return int(h) * 60 + int(m)


def _time_mask(ts: pd.Series, start_hhmm: str, end_hhmm: str) -> pd.Series:
    start = _parse_hhmm(start_hhmm)
    end = _parse_hhmm(end_hhmm)
    minute = ts.dt.hour * 60 + ts.dt.minute
    if start <= end:
        return (minute >= start) & (minute < end)
    return (minute >= start) | (minute < end)


def _entry_next_tick(close_i: int, n_ticks: int) -> int | None:
    idx = close_i + 1
    return idx if idx < n_ticks else None


def _exit_trade(
    side: int,
    entry_i: int,
    end_i: int,
    entry_px: float,
    bid: np.ndarray,
    ask: np.ndarray,
    tp_px: float | None,
    sl_px: float | None,
) -> tuple[int, float, str]:
    if _exit_trade_numba is not None:
        idx, px, code = _exit_trade_numba(
            int(side), int(entry_i), int(end_i), bid, ask,
            tp_px is not None, float(tp_px or 0.0),
            sl_px is not None, float(sl_px or 0.0),
        )
        return int(idx), float(px), "sl" if code == 1 else ("tp" if code == 2 else "ny_close")
    end_i = min(end_i, len(bid) - 1)
    for i in range(entry_i + 1, end_i + 1):
        if side == 1:
            if sl_px is not None and bid[i] <= sl_px:
                return i, float(bid[i]), "sl"
            if tp_px is not None and bid[i] >= tp_px:
                return i, float(bid[i]), "tp"
        else:
            if sl_px is not None and ask[i] >= sl_px:
                return i, float(ask[i]), "sl"
            if tp_px is not None and ask[i] <= tp_px:
                return i, float(ask[i]), "tp"
    return end_i, float(bid[end_i] if side == 1 else ask[end_i]), "ny_close"


def _targets(
    side: int,
    entry_px: float,
    tp_points: float,
    sl_points: float,
    point_size: float,
    target_mode: str,
    stop_mode: str,
    rr: float,
    asia_high: float,
    asia_low: float,
    london_high: float,
    london_low: float,
    stop_buffer_points: float,
) -> tuple[float | None, float | None]:
    buffer = stop_buffer_points * point_size
    if stop_mode == "fixed":
        sl_px = entry_px - sl_points * point_size if side == 1 else entry_px + sl_points * point_size
    elif stop_mode == "london_extreme":
        sl_px = london_low - buffer if side == 1 else london_high + buffer
    elif stop_mode == "asia_edge":
        sl_px = asia_low - buffer if side == 1 else asia_high + buffer
    else:
        raise ValueError(f"bad stop_mode {stop_mode}")

    if target_mode == "fixed":
        tp_px = entry_px + tp_points * point_size if side == 1 else entry_px - tp_points * point_size
    elif target_mode == "asia_mid":
        tp_px = (asia_high + asia_low) / 2.0
    elif target_mode == "asia_opposite":
        tp_px = asia_high if side == 1 else asia_low
    elif target_mode == "rr":
        risk = abs(entry_px - sl_px)
        tp_px = entry_px + rr * risk if side == 1 else entry_px - rr * risk
    else:
        raise ValueError(f"bad target_mode {target_mode}")
    return tp_px, sl_px


def simulate_combo(
    pair: str,
    g: pd.DataFrame,
    candles: pd.DataFrame,
    timeframe: str,
    confirm_mode: str,
    entry_mode: str,
    asia_start: str,
    asia_end: str,
    london_start: str,
    london_end: str,
    ny_start: str,
    ny_end: str,
    push_buffer_points: float,
    min_body_points: float,
    target_mode: str,
    stop_mode: str,
    rr: float,
    stop_buffer_points: float,
    tp_points: float,
    sl_points: float,
    point_size: float,
    amount: float,
    compound: bool,
    leverage: float,
    commission_per_million: float,
    side_filter: str,
) -> tuple[TradeResult, list[NyTrade]]:
    bid = g["bid"].to_numpy(np.float64)
    ask = g["ask"].to_numpy(np.float64)
    c = candles.copy()
    c["date"] = c["timestamp"].dt.strftime("%Y-%m-%d")
    c["asia"] = _time_mask(c["timestamp"], asia_start, asia_end)
    c["london"] = _time_mask(c["timestamp"], london_start, london_end)
    c["ny"] = _time_mask(c["timestamp"], ny_start, ny_end)
    push_buffer = push_buffer_points * point_size
    min_body = min_body_points * point_size

    cash = amount
    equity_peak = amount
    max_dd = 0.0
    gross_win = gross_loss = 0.0
    trades = wins = losses = long_trades = short_trades = stop_losses = signal_exits = 0
    daily: dict[str, float] = {}
    out: list[NyTrade] = []

    for date, day in c.groupby("date", sort=True):
        asia = day[day["asia"]]
        london = day[day["london"]]
        ny = day[day["ny"]]
        if len(asia) < 2 or len(london) < 1 or len(ny) < 1:
            continue
        asia_high = float(asia["high"].max())
        asia_low = float(asia["low"].min())
        london_high = float(london["high"].max())
        london_low = float(london["low"].min())
        pushed_up = london_high >= asia_high + push_buffer
        pushed_down = london_low <= asia_low - push_buffer
        if pushed_up == pushed_down:
            continue

        side = -1 if pushed_up else 1
        if side_filter == "long" and side != 1:
            continue
        if side_filter == "short" and side != -1:
            continue

        confirm = None
        for _, row in ny.iterrows():
            open_px = float(row["open"])
            close_px = float(row["close"])
            body = abs(close_px - open_px)
            if body < min_body:
                continue
            if pushed_up:
                if confirm_mode == "reclaim":
                    ok = close_px < asia_high and close_px < open_px
                elif confirm_mode == "color":
                    ok = close_px < open_px
                elif confirm_mode == "break_london_mid":
                    ok = close_px < (london_high + london_low) / 2.0 and close_px < open_px
                else:
                    raise ValueError(confirm_mode)
            else:
                if confirm_mode == "reclaim":
                    ok = close_px > asia_low and close_px > open_px
                elif confirm_mode == "color":
                    ok = close_px > open_px
                elif confirm_mode == "break_london_mid":
                    ok = close_px > (london_high + london_low) / 2.0 and close_px > open_px
                else:
                    raise ValueError(confirm_mode)
            if ok:
                confirm = row
                break
        if confirm is None:
            continue

        confirm_i = int(confirm["close_i"])
        if entry_mode == "close":
            entry_i = _entry_next_tick(confirm_i, len(bid))
        elif entry_mode == "retrace50":
            level = (float(confirm["open"]) + float(confirm["close"])) / 2.0
            end_i = int(ny["close_i"].iloc[-1])
            entry_i = None
            for i in range(confirm_i + 1, min(end_i, len(bid) - 1) + 1):
                if side == 1 and ask[i] <= level:
                    entry_i = i
                    break
                if side == -1 and bid[i] >= level:
                    entry_i = i
                    break
        else:
            raise ValueError(entry_mode)
        if entry_i is None:
            continue

        entry_px = float(ask[entry_i] if side == 1 else bid[entry_i])
        margin = cash if compound else amount
        if margin <= 0:
            break
        units = units_for_margin(margin, leverage, entry_px)
        entry_fee = commission(entry_px, units, commission_per_million)
        cash -= entry_fee
        tp_px, sl_px = _targets(
            side, entry_px, tp_points, sl_points, point_size, target_mode, stop_mode, rr,
            asia_high, asia_low, london_high, london_low, stop_buffer_points,
        )
        exit_i, exit_px, reason = _exit_trade(side, entry_i, int(ny["close_i"].iloc[-1]), entry_px, bid, ask, tp_px, sl_px)
        exit_fee = commission(exit_px, units, commission_per_million)
        pnl_raw = (exit_px - entry_px) * units if side == 1 else (entry_px - exit_px) * units
        pnl = pnl_raw - entry_fee - exit_fee
        cash += pnl_raw - exit_fee
        daily[date] = daily.get(date, 0.0) + pnl
        trades += 1
        if side == 1:
            long_trades += 1
            side_name = "long"
        else:
            short_trades += 1
            side_name = "short"
        if pnl >= 0:
            wins += 1
            gross_win += pnl
        else:
            losses += 1
            gross_loss += -pnl
        if reason == "sl":
            stop_losses += 1
        elif reason == "ny_close":
            signal_exits += 1
        equity_peak = max(equity_peak, cash)
        max_dd = max(max_dd, equity_peak - cash)
        out.append(NyTrade(
            pair, "", timeframe, date, side_name, "up" if pushed_up else "down",
            entry_i, exit_i, entry_px, exit_px, pnl, reason,
            asia_high, asia_low, london_high, london_low, float(confirm["close"]),
        ))

    realised = cash - amount
    pf = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
    wr = wins / trades * 100.0 if trades else 0.0
    days = np.array(list(daily.values()), dtype=np.float64) if daily else np.array([0.0])
    params = (
        f"confirm={confirm_mode};entry={entry_mode};asia={asia_start}-{asia_end};"
        f"london={london_start}-{london_end};ny={ny_start}-{ny_end};"
        f"push={push_buffer_points:g};body={min_body_points:g};target={target_mode};"
        f"stop={stop_mode};rr={rr:g};stop_buf={stop_buffer_points:g}"
    )
    for t in out:
        t.params = params
    res = TradeResult(
        pair, "asia_london_ny", params, timeframe, tp_points, sl_points, point_size,
        realised, 0.0, realised, trades, wins, losses, wr, pf, max_dd,
        long_trades, short_trades, stop_losses, signal_exits, 0, False, "-", 0.0,
    )
    res.avg_day = float(np.mean(days))
    res.median_day = float(np.median(days))
    res.active_days = int(len(daily))
    return res, out


def main() -> None:
    ap = build_parser("Asia London push New York reversal backtest", "forex_asia_london_ny_reversal_results.csv")
    ap.set_defaults(timeframes="5m", pairs=["XAUUSD"], tp_points="200,300,400", sl_points="200,300,400")
    ap.add_argument("--confirm-mode", default="reclaim,color")
    ap.add_argument("--entry-mode", default="close")
    ap.add_argument("--target-mode", default="fixed,rr")
    ap.add_argument("--stop-mode", default="fixed,london_extreme")
    ap.add_argument("--rr", default="1,1.5")
    ap.add_argument("--stop-buffer-points", default="0,50")
    ap.add_argument("--asia-start", default="00:00")
    ap.add_argument("--asia-end", default="06:00")
    ap.add_argument("--london-start", default="07:00")
    ap.add_argument("--london-end", default="11:00")
    ap.add_argument("--ny-start", default="13:30")
    ap.add_argument("--ny-end", default="17:00")
    ap.add_argument("--push-buffer-points", default="50,100")
    ap.add_argument("--min-body-points", default="0,50")
    ap.add_argument("--trades-out", default=None)
    args = ap.parse_args()

    ticks, t0 = load_market(args)
    results: list[TradeResult] = []
    trades_out: list[NyTrade] = []
    confirm_modes = parse_str_list(args.confirm_mode, ["reclaim"])
    entry_modes = parse_str_list(args.entry_mode, ["close"])
    target_modes = parse_str_list(args.target_mode, ["fixed"])
    stop_modes = parse_str_list(args.stop_mode, ["fixed"])
    timeframes = parse_str_list(args.timeframes, ["3m"])
    rrs = parse_num_list(args.rr, [1.0])
    stop_buffers = parse_num_list(args.stop_buffer_points, [0.0])
    push_buffers = parse_num_list(args.push_buffer_points, [0.0])
    min_bodies = parse_num_list(args.min_body_points, [0.0])

    for pair, g0 in ticks.groupby("pair", sort=False):
        g = g0.sort_values("timestamp").reset_index(drop=True)
        point_size = float(args.point_size or default_point_size(pair))
        default_tp = GOLD_TP if pair.upper() == "XAUUSD" else FX_TP
        default_sl = GOLD_SL if pair.upper() == "XAUUSD" else FX_SL
        tps = parse_num_list(args.tp_points, default_tp)
        sls = parse_num_list(args.sl_points, default_sl)
        candle_cache = {tf: build_bid_candles(g, tf) for tf in timeframes}
        combos = list(product(
            timeframes, confirm_modes, entry_modes, target_modes, stop_modes,
            rrs, stop_buffers, push_buffers, min_bodies, tps, sls,
        ))
        print(f"[asia-ny] {pair} ticks={len(g):,} combos={len(combos):,}", flush=True)
        step = max(1, len(combos) // 10)
        for i, combo in enumerate(combos, 1):
            tf, confirm, entry, target, stop, rr, stop_buf, push, body, tp, sl = combo
            res, tr = simulate_combo(
                pair, g, candle_cache[tf], tf, confirm, entry,
                args.asia_start, args.asia_end, args.london_start, args.london_end,
                args.ny_start, args.ny_end, push, body, target, stop, rr, stop_buf,
                tp, sl, point_size, args.amount, args.compound, args.leverage,
                args.commission_per_million, args.side,
            )
            results.append(res)
            if args.trades_out:
                trades_out.extend(tr)
            if i % step == 0 or i == len(combos):
                print(f"[asia-ny] {pair} progress {i:,}/{len(combos):,}", flush=True)

    filtered = [r for r in results if r.trades >= args.min_trades]
    write_results(args.out, filtered, args.top, args.sort_by)
    if args.trades_out:
        os.makedirs(os.path.dirname(args.trades_out) or ".", exist_ok=True)
        with open(args.trades_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "pair", "params", "timeframe", "date", "side", "push", "entry_i", "exit_i",
                "entry_px", "exit_px", "pnl", "reason", "asia_high", "asia_low",
                "london_high", "london_low", "confirm_close",
            ])
            for t in trades_out:
                w.writerow([
                    t.pair, t.params, t.timeframe, t.date, t.side, t.push, t.entry_i, t.exit_i,
                    t.entry_px, t.exit_px, t.pnl, t.reason, t.asia_high, t.asia_low,
                    t.london_high, t.london_low, t.confirm_close,
                ])
        print(f"[asia-ny] wrote trades {os.path.abspath(args.trades_out)}", flush=True)
    print(f"[asia-ny] wrote {os.path.abspath(args.out)} elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
