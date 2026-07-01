from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from forex_ml_tick_simulator import build_bid_candles, load_torch_model, predict_model
from forex_strategy_common import active_session_allowed, default_point_size, timeframe_to_ns
from forex_tcn2_nextbar_paper import signal_from_prob


SYMBOL = "AUDUSD"
MODEL = Path("data/forex/ml_models/forex_ml_AUDUSD_nextbar_up_tcn2_ohlc12_tf1m_nextbar_s2_w64_h1_c64_k3_l5.pt")
OUT_DIR = Path("data/forex/analysis/live_compare")

UPPER = 0.5
LOWER = 0.5
PROB_MA = 1
SIGNAL_MODE = "invert"
TP_MODE = "opposite"
SL_MODE = "fixed_signal"
TP_POINTS = 0.0
SL_POINTS = 75.0
SESSION = 2
REENTRY = "ma_reset"
MAGIC = 936601


@dataclass
class ExpectedTrade:
    n: int
    entry_i: int
    exit_i: int
    entry_time: datetime
    exit_time: datetime
    side: int
    entry: float
    exit: float
    points: float
    reason: str
    p_entry: float
    p_exit: float


def riyadh_to_utc_today(hour: int, minute: int = 0) -> datetime:
    now_utc = datetime.now(timezone.utc)
    # Riyadh is UTC+3 and has no DST.
    riyadh_now = now_utc + timedelta(hours=3)
    riyadh_start = riyadh_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return riyadh_start - timedelta(hours=3)


def ns_to_dt(ns: int) -> datetime:
    return datetime.fromtimestamp(int(ns) / 1_000_000_000, tz=timezone.utc)


def signal_from_p(p: float) -> int:
    return signal_from_prob(float(p), UPPER, LOWER)


def simulate_expected(bid, ask, ts_ns, candles, p_up, start_utc: datetime) -> list[ExpectedTrade]:
    ps = default_point_size(SYMBOL)
    allowed = active_session_allowed(candles.times.astype("int64"), SESSION)
    start_ns = int(start_utc.timestamp() * 1_000_000_000)
    start_candle = int(np.searchsorted(candles.times.astype("int64"), start_ns, side="left"))
    start_candle = max(int(getattr(load_torch_model(MODEL)[1], "window", 64)) - 1, start_candle)
    close_tick_idx = candles.close_tick_idx
    tick_to_candle = candles.tick_to_candle

    trades: list[ExpectedTrade] = []
    candle_i = start_candle
    tick_floor = 0
    pending_side = 0
    block_long = False
    block_short = False
    n = 0

    while candle_i < len(close_tick_idx):
        if pending_side:
            side = pending_side
            pending_side = 0
            entry_tick = tick_floor
            if entry_tick >= len(bid):
                break
            candle_i = int(tick_to_candle[entry_tick])
            if not allowed[candle_i]:
                candle_i += 1
                continue
            if REENTRY == "ma_reset" and ((side == 1 and block_long) or (side == -1 and block_short)):
                candle_i += 1
                continue
        else:
            p_now = float(p_up[candle_i])
            if REENTRY == "ma_reset":
                if block_long and (not np.isfinite(p_now) or p_now < UPPER):
                    block_long = False
                if block_short and (not np.isfinite(p_now) or p_now > LOWER):
                    block_short = False
            side = signal_from_p(p_now)
            if side == 0:
                candle_i += 1
                continue
            if REENTRY == "ma_reset" and ((side == 1 and block_long) or (side == -1 and block_short)):
                candle_i += 1
                continue
            entry_tick = int(close_tick_idx[candle_i]) + 1
            if entry_tick < tick_floor:
                entry_tick = tick_floor
            if entry_tick >= len(bid):
                break
            entry_candle = int(tick_to_candle[entry_tick])
            if not allowed[entry_candle]:
                candle_i += 1
                continue

        entry = float(ask[entry_tick] if side == 1 else bid[entry_tick])
        exit_tick = len(bid) - 1
        exit_px = float(bid[exit_tick] if side == 1 else ask[exit_tick])
        result_points = ((exit_px - entry) / ps) * side
        reason = "end"
        reverse_side = 0
        next_signal_candle = int(tick_to_candle[entry_tick])
        p_entry = float(p_up[int(tick_to_candle[entry_tick])])
        p_exit = np.nan

        for ti in range(entry_tick + 1, len(bid)):
            live_points = ((float(bid[ti]) - entry) / ps) if side == 1 else ((entry - float(ask[ti])) / ps)
            if SL_MODE in ("fixed", "fixed_signal") and SL_POINTS > 0 and live_points <= -SL_POINTS:
                exit_tick = ti
                exit_px = float(bid[ti] if side == 1 else ask[ti])
                result_points = -SL_POINTS
                reason = "fixed_sl"
                break

            current_candle = int(tick_to_candle[ti])
            while next_signal_candle < current_candle and next_signal_candle < len(close_tick_idx):
                sig = signal_from_p(float(p_up[next_signal_candle]))
                mode = TP_MODE if live_points >= 0 else SL_MODE
                exit_now = False
                if mode == "opposite":
                    exit_now = sig == -side
                elif mode == "neutral":
                    exit_now = sig != side
                elif mode == "fixed_signal":
                    exit_now = sig == -side
                if exit_now:
                    exit_tick = ti
                    exit_px = float(bid[ti] if side == 1 else ask[ti])
                    result_points = live_points
                    reason = "signal"
                    p_exit = float(p_up[next_signal_candle])
                    if sig == -side:
                        reverse_side = sig
                    break
                next_signal_candle += 1
            if reason == "signal":
                break

        if ts_ns[entry_tick] >= start_ns or ts_ns[exit_tick] >= start_ns:
            n += 1
            trades.append(ExpectedTrade(
                n=n,
                entry_i=int(entry_tick),
                exit_i=int(exit_tick),
                entry_time=ns_to_dt(int(ts_ns[entry_tick])),
                exit_time=ns_to_dt(int(ts_ns[exit_tick])),
                side=side,
                entry=entry,
                exit=exit_px,
                points=float(result_points),
                reason=reason,
                p_entry=p_entry,
                p_exit=float(p_exit) if np.isfinite(p_exit) else float(p_up[int(tick_to_candle[exit_tick])]),
            ))

        if reason == "fixed_sl" and REENTRY == "ma_reset":
            if side == 1:
                block_long = True
            else:
                block_short = True

        tick_floor = exit_tick + 1
        if tick_floor >= len(bid):
            break
        if reason == "fixed_sl":
            candle_i = int(tick_to_candle[exit_tick])
            continue
        if reason == "signal" and reverse_side != 0:
            pending_side = reverse_side
            continue
        next_candle = int(tick_to_candle[tick_floor])
        candle_i = next_candle if next_candle > candle_i else candle_i + 1

    return trades


def write_expected(path: Path, trades: list[ExpectedTrade]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n", "entry_utc", "exit_utc", "side", "entry", "exit", "points", "reason", "p_entry", "p_exit"])
        for t in trades:
            w.writerow([
                t.n, t.entry_time.isoformat(), t.exit_time.isoformat(),
                "BUY" if t.side == 1 else "SELL", f"{t.entry:.5f}", f"{t.exit:.5f}",
                f"{t.points:.2f}", t.reason, f"{t.p_entry:.6f}", f"{t.p_exit:.6f}",
            ])


def export_mt5_history(mt5, start_utc: datetime, end_utc: datetime, path: Path) -> list[dict]:
    deals = mt5.history_deals_get(start_utc, end_utc, group=f"*{SYMBOL}*")
    rows: list[dict] = []
    if deals is None:
        print(f"history_deals_get failed: {mt5.last_error()}")
        deals = []
    for d in deals:
        if getattr(d, "symbol", "").upper() != SYMBOL:
            continue
        rows.append({
            "time_utc": datetime.fromtimestamp(int(d.time), tz=timezone.utc).isoformat(),
            "ticket": int(d.ticket),
            "order": int(d.order),
            "position_id": int(d.position_id),
            "symbol": d.symbol,
            "type": int(d.type),
            "entry": int(d.entry),
            "volume": float(d.volume),
            "price": float(d.price),
            "profit": float(d.profit),
            "commission": float(d.commission),
            "swap": float(d.swap),
            "magic": int(d.magic),
            "comment": str(d.comment),
        })
    with path.open("w", newline="", encoding="utf-8") as f:
        fields = ["time_utc", "ticket", "order", "position_id", "symbol", "type", "entry", "volume", "price", "profit", "commission", "swap", "magic", "comment"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return rows


def build_mt5_round_trips(rows: list[dict]) -> list[dict]:
    by_pos: dict[int, list[dict]] = {}
    for r in rows:
        if int(r["magic"]) != MAGIC:
            continue
        by_pos.setdefault(int(r["position_id"]), []).append(r)
    trips = []
    for pos_id, ds in by_pos.items():
        ds = sorted(ds, key=lambda r: r["time_utc"])
        ins = [r for r in ds if int(r["entry"]) in (0, 2)]
        outs = [r for r in ds if int(r["entry"]) in (1, 2)]
        if not ins or not outs:
            continue
        ent = ins[0]
        ex = outs[-1]
        # MT5 type: 0 buy, 1 sell. Opening type gives position side.
        side = "BUY" if int(ent["type"]) == 0 else "SELL"
        trips.append({
            "position_id": pos_id,
            "entry_utc": ent["time_utc"],
            "exit_utc": ex["time_utc"],
            "side": side,
            "entry": ent["price"],
            "exit": ex["price"],
            "volume": ent["volume"],
            "profit": sum(float(r["profit"]) + float(r["commission"]) + float(r["swap"]) for r in ds),
            "deals": len(ds),
        })
    return sorted(trips, key=lambda r: r["entry_utc"])


def write_trips(path: Path, trips: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        fields = ["position_id", "entry_utc", "exit_utc", "side", "entry", "exit", "volume", "profit", "deals"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(trips)


def write_compare(path: Path, expected: list[ExpectedTrade], trips: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        fields = ["i", "bt_entry_utc", "bt_exit_utc", "bt_side", "bt_entry", "bt_exit", "bt_points", "bt_reason",
                  "mt5_entry_utc", "mt5_exit_utc", "mt5_side", "mt5_entry", "mt5_exit", "mt5_profit", "entry_diff_sec", "exit_diff_sec"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        n = max(len(expected), len(trips))
        for i in range(n):
            row = {"i": i + 1}
            if i < len(expected):
                e = expected[i]
                row.update({
                    "bt_entry_utc": e.entry_time.isoformat(),
                    "bt_exit_utc": e.exit_time.isoformat(),
                    "bt_side": "BUY" if e.side == 1 else "SELL",
                    "bt_entry": f"{e.entry:.5f}",
                    "bt_exit": f"{e.exit:.5f}",
                    "bt_points": f"{e.points:.2f}",
                    "bt_reason": e.reason,
                })
            if i < len(trips):
                t = trips[i]
                row.update({
                    "mt5_entry_utc": t["entry_utc"],
                    "mt5_exit_utc": t["exit_utc"],
                    "mt5_side": t["side"],
                    "mt5_entry": f"{float(t['entry']):.5f}",
                    "mt5_exit": f"{float(t['exit']):.5f}",
                    "mt5_profit": f"{float(t['profit']):.2f}",
                })
            if i < len(expected) and i < len(trips):
                e = expected[i]
                te = datetime.fromisoformat(trips[i]["entry_utc"])
                tx = datetime.fromisoformat(trips[i]["exit_utc"])
                row["entry_diff_sec"] = f"{(te - e.entry_time).total_seconds():.1f}"
                row["exit_diff_sec"] = f"{(tx - e.exit_time).total_seconds():.1f}"
            w.writerow(row)


def main() -> None:
    import MetaTrader5 as mt5

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start_utc = riyadh_to_utc_today(13, 0)
    end_utc = datetime.now(timezone.utc)
    tick_start = start_utc - timedelta(hours=4)
    print(f"window UTC {start_utc.isoformat()} -> {end_utc.isoformat()} (tick preload from {tick_start.isoformat()})")

    if not mt5.initialize():
        raise SystemExit(f"mt5.initialize failed: {mt5.last_error()}")
    try:
        if not mt5.symbol_select(SYMBOL, True):
            raise SystemExit(f"symbol_select failed: {SYMBOL}")
        ticks = mt5.copy_ticks_range(SYMBOL, tick_start, end_utc, mt5.COPY_TICKS_INFO)
        if ticks is None or len(ticks) == 0:
            raise SystemExit(f"no ticks loaded: {mt5.last_error()}")
        bid = np.asarray([float(t["bid"]) for t in ticks], dtype=np.float64)
        ask = np.asarray([float(t["ask"]) for t in ticks], dtype=np.float64)
        ts_ns = np.asarray(
            [int(t["time_msc"]) * 1_000_000 if int(t["time_msc"]) > 0 else int(t["time"]) * 1_000_000_000 for t in ticks],
            dtype=np.int64,
        )
        candles = build_bid_candles(bid, ask, ts_ns, "1m")
        print(f"ticks={len(ticks):,} candles={len(candles.ohlc):,}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, ns, point = load_torch_model(MODEL)
        preds = predict_model(model, ns, candles, float(point or default_point_size(SYMBOL)), device)
        p_up = preds[:, 0].astype(np.float64)
        if SIGNAL_MODE == "invert":
            p_up = 1.0 - p_up

        expected = simulate_expected(bid, ask, ts_ns, candles, p_up, start_utc)
        deals = export_mt5_history(mt5, start_utc, end_utc, OUT_DIR / "audusd_mt5_raw_deals.csv")
        trips = build_mt5_round_trips(deals)
    finally:
        mt5.shutdown()

    write_expected(OUT_DIR / "audusd_expected_backtest_trades.csv", expected)
    write_trips(OUT_DIR / "audusd_mt5_round_trips_magic936601.csv", trips)
    write_compare(OUT_DIR / "audusd_expected_vs_mt5_compare.csv", expected, trips)

    bt_pts = sum(t.points for t in expected)
    mt5_profit = sum(float(t["profit"]) for t in trips)
    print(f"expected_trades={len(expected)} expected_points={bt_pts:+.1f}")
    print(f"mt5_magic_trades={len(trips)} mt5_profit=${mt5_profit:+.2f}")
    print(f"wrote {OUT_DIR}")
    print("first expected:")
    for t in expected[:5]:
        print(f"  {t.entry_time.isoformat()} {('BUY' if t.side==1 else 'SELL')} -> {t.exit_time.isoformat()} {t.points:+.1f}pt {t.reason}")
    print("first mt5:")
    for t in trips[:5]:
        print(f"  {t['entry_utc']} {t['side']} -> {t['exit_utc']} profit={float(t['profit']):+.2f}")


if __name__ == "__main__":
    main()
