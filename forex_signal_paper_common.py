"""Shared MT5 paper helpers for candle-state signal strategies."""

from __future__ import annotations

import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from forex_synthetic_candles import Candle, SyntheticBidOHLC

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def round_volume(info, volume: float) -> float:
    step = float(info.volume_step or 0.01)
    vmin = float(info.volume_min or step)
    vmax = float(info.volume_max or volume)
    volume = max(vmin, min(vmax, volume))
    steps = math.floor((volume - vmin) / step + 1e-9)
    return round(vmin + steps * step, 8)


def filling_name(mt5, mode) -> str:
    names = {
        getattr(mt5, "ORDER_FILLING_IOC", None): "IOC",
        getattr(mt5, "ORDER_FILLING_FOK", None): "FOK",
        getattr(mt5, "ORDER_FILLING_RETURN", None): "RETURN",
    }
    return names.get(mode, f"BROKER({mode})")


def filling_candidates(mt5, symbol: str, selected: str = "auto"):
    info = mt5.symbol_info(symbol)
    first = getattr(info, "filling_mode", None) if info else None
    selected = (selected or "auto").lower().strip()
    if selected == "broker":
        return [first] if first is not None else []
    if selected == "ioc":
        return [mt5.ORDER_FILLING_IOC]
    if selected == "fok":
        return [mt5.ORDER_FILLING_FOK]
    if selected == "return":
        return [mt5.ORDER_FILLING_RETURN]
    modes = [first, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN]
    out = []
    for mode in modes:
        if mode is not None and mode not in out:
            out.append(mode)
    return out


def find_position(mt5, symbol: str, magic: int):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return None
    for p in positions:
        if int(getattr(p, "magic", 0) or 0) == int(magic):
            return p
    return None


def send_order(mt5, args, side: str, reason: str, close_ticket: int | None = None, tag: str = "sig"):
    tick = mt5.symbol_info_tick(args.symbol)
    info = mt5.symbol_info(args.symbol)
    if tick is None or info is None:
        print(f"[{tag}-paper] ORDER SKIP no tick/info", flush=True)
        return None
    volume = round_volume(info, args.lot)
    is_buy = side == "buy"
    order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
    price = float(tick.ask if is_buy else tick.bid)
    req_base = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": args.symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "deviation": args.deviation,
        "magic": args.magic,
        "comment": f"{tag}_{reason}",
        "type_time": mt5.ORDER_TIME_GTC,
    }
    if close_ticket is not None:
        req_base["position"] = int(close_ticket)
    if args.dry_run:
        print(
            f"[{tag}-paper] DRY {side.upper()} market vol={volume:g} "
            f"req_px={price:.5f} deviation={args.deviation} reason={reason}",
            flush=True,
        )
        return {"dry": True}
    last = None
    ps = float(getattr(info, "point", 0.01) or 0.01)
    for filling in filling_candidates(mt5, args.symbol, getattr(args, "filling_mode", "auto")):
        req = dict(req_base)
        req["type_filling"] = filling
        print(
            f"[{tag}-paper] ORDER SEND {side.upper()} market vol={volume:g} "
            f"fill={filling_name(mt5, filling)} req_px={price:.5f} "
            f"deviation={args.deviation} reason={reason}",
            flush=True,
        )
        start = time.perf_counter()
        res = mt5.order_send(req)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        last = res
        ret = getattr(res, "retcode", None)
        comment = getattr(res, "comment", "") if res is not None else "no result"
        if res is not None and res.retcode == mt5.TRADE_RETCODE_DONE:
            fill_px = float(getattr(res, "price", 0.0) or price)
            slip_pts = (fill_px - price) / ps
            if not is_buy:
                slip_pts = -slip_pts
            print(
                f"[{tag}-paper] ORDER OK {side.upper()} market vol={volume:g} "
                f"fill={filling_name(mt5, filling)} req_px={price:.5f} fill_px={fill_px:.5f} "
                f"slip={slip_pts:+.1f}pt ms={elapsed_ms:.1f} ret={ret} "
                f"deviation={args.deviation} reason={reason}",
                flush=True,
            )
            return res
        print(
            f"[{tag}-paper] ORDER TRY_FAIL {side.upper()} market vol={volume:g} "
            f"fill={filling_name(mt5, filling)} req_px={price:.5f} "
            f"ms={elapsed_ms:.1f} ret={ret} comment={comment!r} reason={reason}",
            flush=True,
        )
    print(
        f"[{tag}-paper] ORDER FAIL side={side} mode={getattr(args, 'filling_mode', 'auto')} "
        f"last={getattr(last, 'retcode', None)} reason={reason}",
        flush=True,
    )
    return None


def point_size(mt5, symbol: str) -> float:
    info = mt5.symbol_info(symbol)
    return float(getattr(info, "point", 0.01) or 0.01) if info else 0.01


def mt5_timeframe(mt5, timeframe: str):
    table = {
        "1m": mt5.TIMEFRAME_M1,
        "2m": mt5.TIMEFRAME_M2,
        "3m": mt5.TIMEFRAME_M3,
        "5m": mt5.TIMEFRAME_M5,
        "10m": mt5.TIMEFRAME_M10,
        "15m": mt5.TIMEFRAME_M15,
        "30m": mt5.TIMEFRAME_M30,
        "1h": mt5.TIMEFRAME_H1,
    }
    return table.get(timeframe.lower().strip())


def preload_bid_candles(mt5, args, candles: SyntheticBidOHLC, need: int, tag: str) -> int:
    """Seed candles from native MT5 bars when possible, otherwise recent ticks."""
    tf = args.timeframe.lower().strip()
    native = mt5_timeframe(mt5, tf)
    if native is not None:
        rates = mt5.copy_rates_from_pos(args.symbol, native, 1, need)
        if rates is not None and len(rates) >= max(need // 2, 2):
            candles.closed.clear()
            for r in rates:
                bucket = int(r["time"]) // candles.tf_sec
                candles.closed.append(Candle(
                    bucket=bucket,
                    open=float(r["open"]),
                    high=float(r["high"]),
                    low=float(r["low"]),
                    close=float(r["close"]),
                ))
            candles.last_closed_bucket = int(candles.closed[-1].bucket)
            print(f"[{tag}-paper] preload native candles={len(candles.closed)}", flush=True)
            return len(candles.closed)

    seconds = max((need + 5) * candles.tf_sec, 60)
    start = datetime.now(timezone.utc) - timedelta(seconds=seconds)
    ticks = mt5.copy_ticks_from(args.symbol, start, 250000, mt5.COPY_TICKS_INFO)
    if ticks is None or len(ticks) == 0:
        return 0
    candles.closed.clear()
    candles.current = None
    for t in ticks:
        class TickObj:
            pass
        obj = TickObj()
        obj.bid = float(t["bid"])
        obj.time = int(t["time"])
        candles.update(obj)
    if len(candles.closed):
        print(f"[{tag}-paper] preload tick candles={len(candles.closed)}", flush=True)
    return len(candles.closed)


def ema(values: np.ndarray, length: int) -> np.ndarray:
    out = np.empty(len(values), dtype=np.float64)
    alpha = 2.0 / (length + 1.0)
    val = float(values[0])
    for i, v in enumerate(values):
        val = alpha * float(v) + (1.0 - alpha) * val
        out[i] = val
    return out


def rma(values: np.ndarray, length: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float64)
    if len(values) < length:
        return out
    val = float(np.nanmean(values[:length]))
    out[length - 1] = val
    for i in range(length, len(values)):
        val = (val * (length - 1) + float(values[i])) / length
        out[i] = val
    first = np.where(np.isfinite(out))[0]
    if len(first):
        out[:first[0]] = out[first[0]]
    return out


@dataclass
class SignalInfo:
    state: int
    text: str
    exit_state: int | None = None


class StateMachinePaper:
    def __init__(self, args, tag: str):
        self.args = args
        self.tag = tag
        self.prev_state: int | None = None
        self.boot_ready = False
        self.entry_ts = 0.0
        self.best_px = 0.0

    def entry_blocked_now(self) -> bool:
        return not self.entry_allowed_now()

    def entry_allowed_now(self) -> bool:
        session_mode = int(getattr(self.args, "session", 0) or 0)
        if session_mode != 0:
            from forex_strategy_common import active_session_allowed
            now_ns = int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)
            if not bool(active_session_allowed(np.array([now_ns], dtype=np.int64), session_mode)[0]):
                return False
        raw = getattr(self.args, "block_entry_hours", "") or ""
        if not raw.strip():
            return True
        blocked = {int(x.strip()) for x in raw.split(",") if x.strip()}
        return datetime.now(timezone.utc).hour not in blocked

    def entry_session_text(self) -> str:
        now = datetime.now(timezone.utc)
        return (
            f"utc={now.hour:02d}:{now.minute:02d} "
            f"session={int(getattr(self.args, 'session', 0) or 0)} "
            f"entry_allowed={int(self.entry_allowed_now())}"
        )

    def on_state(self, mt5, state: int, text: str, exit_state: int | None = None) -> None:
        if exit_state is None:
            exit_state = state
        pos = find_position(mt5, self.args.symbol, self.args.magic)
        side = 0 if pos is None else (1 if int(pos.type) == mt5.POSITION_TYPE_BUY else -1)
        if self.prev_state is None:
            self.prev_state = state
            print(
                f"[{self.tag}-paper] initial state={state} exit_state={exit_state} "
                f"{text}; waiting for transition",
                flush=True,
            )
            return
        changed = state != self.prev_state
        prev = self.prev_state
        self.prev_state = state

        if side == 1 and exit_state == -1:
            pnl = float(getattr(pos, "profit", 0.0) or 0.0)
            if getattr(self.args, "ignore_signal_exit_when_bracket", False):
                return
            if self.args.tp_points <= 0 or pnl < 0:
                send_order(mt5, self.args, "sell", "signal_close_long", int(pos.ticket), tag=self.tag)
                return
        elif side == -1 and exit_state == 1:
            pnl = float(getattr(pos, "profit", 0.0) or 0.0)
            if getattr(self.args, "ignore_signal_exit_when_bracket", False):
                return
            if self.args.tp_points <= 0 or pnl < 0:
                send_order(mt5, self.args, "buy", "signal_close_short", int(pos.ticket), tag=self.tag)
                return

        if not changed:
            return

        if side == 0:
            if self.entry_blocked_now():
                print(f"[{self.tag}-paper] ENTRY BLOCK state={state} {self.entry_session_text()}", flush=True)
                return
            if state == 1:
                if send_order(mt5, self.args, "buy", "entry_long", tag=self.tag):
                    self.entry_ts = time.time()
                    tick = mt5.symbol_info_tick(self.args.symbol)
                    self.best_px = float(tick.ask) if tick else 0.0
            elif state == -1:
                if send_order(mt5, self.args, "sell", "entry_short", tag=self.tag):
                    self.entry_ts = time.time()
                    tick = mt5.symbol_info_tick(self.args.symbol)
                    self.best_px = float(tick.bid) if tick else 0.0
            return

    def on_tick_exits(self, mt5, candle_open: float | None = None) -> None:
        tick = mt5.symbol_info_tick(self.args.symbol)
        if tick is None:
            return
        ps = point_size(mt5, self.args.symbol)
        pos = find_position(mt5, self.args.symbol, self.args.magic)
        if pos is None:
            return
        side = 1 if int(pos.type) == mt5.POSITION_TYPE_BUY else -1
        entry = float(pos.price_open)
        pnl = float(getattr(pos, "profit", 0.0) or 0.0)
        if self.best_px <= 0.0:
            self.best_px = entry
        if side == 1:
            trail_source = getattr(self.args, "trail_source", "peak")
            if trail_source == "candle_open" and candle_open is not None and self.args.tp_points > 0 and self.args.trail_points > 0:
                open_ref = float(candle_open)
                if open_ref >= entry + self.args.tp_points * ps:
                    self.best_px = max(self.best_px, open_ref)
            else:
                self.best_px = max(self.best_px, float(tick.bid))
            move_pts = (float(tick.bid) - entry) / ps
            if self.args.tp_points > 0 and self.args.trail_points <= 0 and move_pts >= self.args.tp_points:
                send_order(mt5, self.args, "sell", "tp_long", int(pos.ticket), tag=self.tag)
            elif (
                self.args.tp_points > 0 and self.args.trail_points > 0
                and self.best_px >= entry + self.args.tp_points * ps
                and float(tick.bid) <= self.best_px - self.args.trail_points * ps
            ):
                reason = "trail_long_candle_open" if trail_source == "candle_open" else "trail_long"
                send_order(mt5, self.args, "sell", reason, int(pos.ticket), tag=self.tag)
            elif self.args.loss_cut_points > 0 and move_pts <= -self.args.loss_cut_points:
                send_order(mt5, self.args, "sell", "cut_long", int(pos.ticket), tag=self.tag)
            elif self.args.max_hold_minutes > 0 and pnl < 0 and time.time() - self.entry_ts >= self.args.max_hold_minutes * 60:
                send_order(mt5, self.args, "sell", "hold_cut_long", int(pos.ticket), tag=self.tag)
        else:
            trail_source = getattr(self.args, "trail_source", "peak")
            if trail_source == "candle_open" and candle_open is not None and self.args.tp_points > 0 and self.args.trail_points > 0:
                open_ref = float(candle_open)
                if open_ref <= entry - self.args.tp_points * ps:
                    self.best_px = min(self.best_px, open_ref) if self.best_px > 0.0 else open_ref
            else:
                self.best_px = min(self.best_px, float(tick.ask)) if self.best_px > 0.0 else float(tick.ask)
            move_pts = (entry - float(tick.ask)) / ps
            if self.args.tp_points > 0 and self.args.trail_points <= 0 and move_pts >= self.args.tp_points:
                send_order(mt5, self.args, "buy", "tp_short", int(pos.ticket), tag=self.tag)
            elif (
                self.args.tp_points > 0 and self.args.trail_points > 0
                and self.best_px <= entry - self.args.tp_points * ps
                and float(tick.ask) >= self.best_px + self.args.trail_points * ps
            ):
                reason = "trail_short_candle_open" if trail_source == "candle_open" else "trail_short"
                send_order(mt5, self.args, "buy", reason, int(pos.ticket), tag=self.tag)
            elif self.args.loss_cut_points > 0 and move_pts <= -self.args.loss_cut_points:
                send_order(mt5, self.args, "buy", "cut_short", int(pos.ticket), tag=self.tag)
            elif self.args.max_hold_minutes > 0 and pnl < 0 and time.time() - self.entry_ts >= self.args.max_hold_minutes * 60:
                send_order(mt5, self.args, "buy", "hold_cut_short", int(pos.ticket), tag=self.tag)


def add_common_args(ap, default_magic: int, default_tf: str, default_tp: float,
                    default_cut: float, default_hold: float) -> None:
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--timeframe", default=default_tf)
    ap.add_argument("--tp-points", type=float, default=default_tp)
    ap.add_argument("--trail-points", type=float, default=0.0,
                    help="TP-armed trailing giveback in points; 0 uses fixed TP")
    ap.add_argument("--trail-source", choices=["peak", "candle_open"], default="peak",
                    help="peak trails from best live price; candle_open arms/updates only from favorable candle opens")
    ap.add_argument("--loss-cut-points", type=float, default=default_cut)
    ap.add_argument("--max-hold-minutes", type=float, default=default_hold)
    ap.add_argument("--block-entry-hours", default="",
                    help="comma-separated UTC hours where new entries are blocked")
    ap.add_argument("--session", type=int, choices=[-1, 0, 1, 2], default=0,
                    help="-1 outside major sessions, 0 all hours, 1 inside major sessions, 2 block UTC 20:30-01:00")
    ap.add_argument("--lot", type=float, default=0.01)
    ap.add_argument("--magic", type=int, default=default_magic)
    ap.add_argument("--deviation", type=int, default=50)
    ap.add_argument("--filling-mode", choices=["auto", "broker", "ioc", "fok", "return"], default="auto",
                    help="MT5 market order filling mode. auto tries broker/default first, then IOC/FOK/RETURN.")
    ap.add_argument("--poll", type=float, default=0.25)
    ap.add_argument("--log-every", type=float, default=3.0)
    ap.add_argument("--dry-run", action="store_true")
