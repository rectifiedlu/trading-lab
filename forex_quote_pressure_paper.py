#!/usr/bin/env python3
"""Monitor quote pressure and optionally execute its sign strategy on MT5."""

from __future__ import annotations

import argparse
import time
from collections import deque
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5

from forex_signal_paper_common import find_position, send_order


def quote_update_pressure(
    bid_change: float,
    ask_change: float,
    retreat_weight: float,
) -> tuple[float, float]:
    """Return signed quote pressure and unweighted quote activity."""
    original_bid_change = bid_change
    original_ask_change = ask_change
    contribution = 0.0

    # A shared bid/ask shift is a full-strength market repricing. Classify only
    # the unmatched remainder as inward aggression or outward retreat.
    if bid_change * ask_change > 0.0:
        direction = 1.0 if bid_change > 0.0 else -1.0
        common_move = direction * min(abs(bid_change), abs(ask_change))
        contribution += 2.0 * common_move
        bid_change -= common_move
        ask_change -= common_move

    contribution += max(bid_change, 0.0)
    contribution -= max(-ask_change, 0.0)
    contribution += retreat_weight * max(ask_change, 0.0)
    contribution -= retreat_weight * max(-bid_change, 0.0)
    activity = abs(original_bid_change) + abs(original_ask_change)
    return contribution, activity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live MT5 quote-pressure paper trader")
    parser.add_argument("--symbol", default="USDJPY")
    parser.add_argument("--window", type=int, default=100, help="rolling quote-update window")
    parser.add_argument("--poll-ms", type=float, default=10.0, help="MT5 polling interval")
    parser.add_argument(
        "--retreat-weight",
        type=float,
        default=0.25,
        help="weight for outward quote retreats relative to inward aggression",
    )
    parser.add_argument("--trade", type=int, choices=[0, 1], default=0)
    parser.add_argument("--lot", type=float, default=0.01)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="absolute quoteP required to open or maintain a position",
    )
    parser.add_argument(
        "--deadband",
        type=int,
        choices=[0, 1],
        default=1,
        help="1 closes in the neutral band; 0 holds until the opposite threshold",
    )
    parser.add_argument(
        "--tp-points",
        type=float,
        default=0.0,
        help="take-profit distance in symbol points; 0 keeps normal signal exits",
    )
    parser.add_argument(
        "--sl-points",
        type=float,
        default=0.0,
        help="stop-loss distance in symbol points; 0 keeps normal signal exits",
    )
    parser.add_argument("--deviation", type=int, default=20)
    parser.add_argument("--magic", type=int, default=26081901)
    parser.add_argument(
        "--order-cooldown",
        type=float,
        default=1.0,
        help="minimum seconds between MT5 order attempts",
    )
    return parser.parse_args()


def order_succeeded(result) -> bool:
    return result is not None and getattr(result, "retcode", None) == mt5.TRADE_RETCODE_DONE


def pressure_side(quote_pressure: float, threshold: float) -> str | None:
    if quote_pressure > 0.0 and quote_pressure >= threshold:
        return "buy"
    if quote_pressure < 0.0 and quote_pressure <= -threshold:
        return "sell"
    return None


def pressure_supports(side: str, quote_pressure: float, threshold: float) -> bool:
    if side == "buy":
        return quote_pressure >= threshold
    return quote_pressure <= -threshold


def position_should_close(
    side: str,
    quote_pressure: float,
    threshold: float,
    deadband: int,
) -> bool:
    if deadband == 1:
        return not pressure_supports(side, quote_pressure, threshold)
    desired_side = pressure_side(quote_pressure, threshold)
    return desired_side is not None and desired_side != side


def position_exit_reason(
    args: argparse.Namespace,
    side: str,
    quote_pressure: float,
    pnl_points: float,
) -> str | None:
    if args.tp_points > 0.0 and pnl_points >= args.tp_points:
        return "tp"
    if args.sl_points > 0.0 and pnl_points <= -args.sl_points:
        return "sl"
    signal_close = position_should_close(
        side,
        quote_pressure,
        args.threshold,
        args.deadband,
    )
    if not signal_close:
        return None
    if pnl_points > 0.0 and args.tp_points > 0.0:
        return None
    if pnl_points < 0.0 and args.sl_points > 0.0:
        return None
    return "signal"


def position_pnl_points(position, bid: float, ask: float, point: float) -> float:
    if int(position.type) == mt5.POSITION_TYPE_BUY:
        return (bid - float(position.price_open)) / point
    return (float(position.price_open) - ask) / point


def trade_quote_pressure(
    args: argparse.Namespace,
    quote_pressure: float,
    bid: float,
    ask: float,
    point: float,
    blocked_sides: set[str],
) -> bool:
    desired_side = pressure_side(quote_pressure, args.threshold)
    positions = list(mt5.positions_get(symbol=args.symbol) or [])
    foreign = [position for position in positions if int(getattr(position, "magic", 0) or 0) != args.magic]
    if foreign:
        print(
            f"[quote-paper] TRADE BLOCKED {args.symbol} has {len(foreign)} foreign position(s)",
            flush=True,
        )
        return False

    position = find_position(mt5, args.symbol, args.magic)
    if position is None:
        if desired_side is None or desired_side in blocked_sides:
            return False
        return order_succeeded(
            send_order(mt5, args, desired_side, f"quoteP_{quote_pressure:+.3f}", tag="quote")
        )

    current_side = "buy" if int(position.type) == mt5.POSITION_TYPE_BUY else "sell"
    pnl_points = position_pnl_points(position, bid, ask, point)
    exit_reason = position_exit_reason(args, current_side, quote_pressure, pnl_points)
    if exit_reason is None:
        return False

    close_side = "sell" if current_side == "buy" else "buy"
    closed = send_order(
        mt5,
        args,
        close_side,
        f"{exit_reason}_{quote_pressure:+.3f}",
        close_ticket=int(position.ticket),
        tag="quote",
    )
    if not order_succeeded(closed):
        return False
    if exit_reason in ("tp", "sl") and desired_side is not None:
        blocked_sides.add(current_side)
        print(
            f"[quote-paper] {exit_reason.upper()} {current_side} "
            f"pnl={pnl_points:+.1f}pt; reset required",
            flush=True,
        )
    elif exit_reason in ("tp", "sl"):
        print(
            f"[quote-paper] {exit_reason.upper()} {current_side} "
            f"pnl={pnl_points:+.1f}pt; pressure already neutral",
            flush=True,
        )

    remaining = find_position(mt5, args.symbol, args.magic)
    if remaining is not None:
        print(
            f"[quote-paper] REVERSE BLOCKED position {remaining.ticket} remains open",
            flush=True,
        )
        return True

    if desired_side is not None and desired_side not in blocked_sides:
        send_order(
            mt5,
            args,
            desired_side,
            f"{exit_reason}_{quote_pressure:+.3f}",
            tag="quote",
        )
    return True


def preload_pressure_history(
    symbol: str,
    window: int,
    retreat_weight: float,
    movements: deque[float],
    quote_contributions: deque[float],
    quote_activity: deque[float],
) -> tuple[float | None, float | None, tuple[int, float, float] | None]:
    end = datetime.now(timezone.utc)
    ticks = None
    for lookback_days in (1, 7):
        ticks = mt5.copy_ticks_range(
            symbol,
            end - timedelta(days=lookback_days),
            end,
            mt5.COPY_TICKS_INFO,
        )
        if ticks is not None and len(ticks) >= window + 1:
            break
    if ticks is None or len(ticks) < 2:
        print(f"[quote-paper] history unavailable: {mt5.last_error()}", flush=True)
        return None, None, None

    recent = []
    newer_quote: tuple[float, float] | None = None
    for tick in reversed(ticks):
        bid = float(tick["bid"])
        ask = float(tick["ask"])
        quote = (bid, ask)
        if bid > 0.0 and ask >= bid and quote != newer_quote:
            recent.append(tick)
            newer_quote = quote
            if len(recent) == window + 1:
                break
    recent.reverse()
    if len(recent) < 2:
        print("[quote-paper] history contains fewer than two valid quotes", flush=True)
        return None, None, None
    for previous, current in zip(recent, recent[1:]):
        bid_change = float(current["bid"]) - float(previous["bid"])
        ask_change = float(current["ask"]) - float(previous["ask"])
        movements.append((bid_change + ask_change) / 2.0)
        contribution, activity = quote_update_pressure(
            bid_change,
            ask_change,
            retreat_weight,
        )
        quote_contributions.append(contribution)
        quote_activity.append(activity)

    latest = recent[-1]
    latest_bid = float(latest["bid"])
    latest_ask = float(latest["ask"])
    signature = (int(latest["time_msc"]), latest_bid, latest_ask)
    first_time = datetime.fromtimestamp(
        int(recent[0]["time_msc"]) / 1000.0,
        tz=timezone.utc,
    ).isoformat(timespec="milliseconds")
    last_time = datetime.fromtimestamp(
        int(latest["time_msc"]) / 1000.0,
        tz=timezone.utc,
    ).isoformat(timespec="milliseconds")
    print(
        f"[quote-paper] preloaded {len(movements)}/{window} historical updates "
        f"from {first_time} to {last_time}",
        flush=True,
    )
    return latest_bid, latest_ask, signature


def main() -> None:
    args = parse_args()
    if args.window < 2:
        raise SystemExit("--window must be at least 2")
    if args.poll_ms <= 0:
        raise SystemExit("--poll-ms must be positive")
    if not 0.0 <= args.retreat_weight <= 1.0:
        raise SystemExit("--retreat-weight must be between 0 and 1")
    if args.lot <= 0.0:
        raise SystemExit("--lot must be positive")
    if not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("--threshold must be between 0 and 1")
    if args.tp_points < 0.0:
        raise SystemExit("--tp-points cannot be negative")
    if args.sl_points < 0.0:
        raise SystemExit("--sl-points cannot be negative")
    if args.order_cooldown < 0.0:
        raise SystemExit("--order-cooldown cannot be negative")

    args.dry_run = False
    args.filling_mode = "ioc"

    if not mt5.initialize():
        raise SystemExit(f"MT5 initialize failed: {mt5.last_error()}")

    symbol = args.symbol.upper()
    args.symbol = symbol
    try:
        if not mt5.symbol_select(symbol, True):
            raise SystemExit(f"Could not select {symbol}: {mt5.last_error()}")

        info = mt5.symbol_info(symbol)
        if info is None:
            raise SystemExit(f"No MT5 symbol information for {symbol}")

        point = float(info.point)
        digits = int(info.digits)
        movements: deque[float] = deque(maxlen=args.window)
        quote_contributions: deque[float] = deque(maxlen=args.window)
        quote_activity: deque[float] = deque(maxlen=args.window)
        previous_bid, previous_ask, previous_signature = preload_pressure_history(
            symbol,
            args.window,
            float(args.retreat_weight),
            movements,
            quote_contributions,
            quote_activity,
        )
        blocked_sides: set[str] = set()
        last_order_attempt = 0.0

        print(
            f"Watching {symbol}: pressure=-1 down, 0 balanced, +1 up "
            f"(window={args.window} updates, retreat_weight={args.retreat_weight:g}, "
            "Ctrl+C to stop)",
            flush=True,
        )

        while True:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                time.sleep(args.poll_ms / 1000.0)
                continue

            bid = float(tick.bid)
            ask = float(tick.ask)
            signature = (int(tick.time_msc), bid, ask)
            if signature == previous_signature:
                time.sleep(args.poll_ms / 1000.0)
                continue
            previous_signature = signature

            if previous_bid is None or previous_ask is None:
                previous_bid = bid
                previous_ask = ask
                continue

            bid_change = bid - previous_bid
            ask_change = ask - previous_ask
            previous_bid = bid
            previous_ask = ask
            if bid_change == 0.0 and ask_change == 0.0:
                time.sleep(args.poll_ms / 1000.0)
                continue
            movement = (bid_change + ask_change) / 2.0
            movements.append(movement)

            quote_contribution, activity = quote_update_pressure(
                bid_change,
                ask_change,
                float(args.retreat_weight),
            )
            quote_contributions.append(quote_contribution)
            quote_activity.append(activity)

            total_movement = sum(movements)
            total_distance = sum(abs(value) for value in movements)
            pressure = total_movement / total_distance if total_distance else 0.0
            total_quote_activity = sum(quote_activity)
            quote_pressure = (
                sum(quote_contributions) / total_quote_activity
                if total_quote_activity else 0.0
            )
            desired_side = pressure_side(quote_pressure, args.threshold)
            if desired_side is None and blocked_sides:
                blocked_sides.clear()
                print("[quote-paper] pressure neutral; point-exit reset cleared", flush=True)
            spread_points = (ask - bid) / point
            movement_points = movement / point
            timestamp = datetime.fromtimestamp(
                tick.time_msc / 1000.0,
                tz=timezone.utc,
            ).strftime("%H:%M:%S.%f")[:-3]

            print(
                f"[{timestamp} UTC] {symbol} "
                f"bid={bid:.{digits}f} ask={ask:.{digits}f} "
                f"spread={spread_points:6.1f}pt move={movement_points:+6.1f}pt "
                f"midP={pressure:+.3f} quoteP={quote_pressure:+.3f} "
                f"samples={len(movements):3d}/{args.window}",
                flush=True,
            )

            if (
                args.trade == 1
                and len(movements) == args.window
                and time.monotonic() - last_order_attempt >= args.order_cooldown
            ):
                position_before = find_position(mt5, symbol, args.magic)
                current_side = None
                exit_reason = None
                if position_before is not None:
                    current_side = (
                        "buy" if int(position_before.type) == mt5.POSITION_TYPE_BUY else "sell"
                    )
                    pnl_points = position_pnl_points(position_before, bid, ask, point)
                    exit_reason = position_exit_reason(
                        args,
                        current_side,
                        quote_pressure,
                        pnl_points,
                    )
                needs_action = (
                    desired_side is not None and desired_side not in blocked_sides
                    if current_side is None
                    else exit_reason is not None
                )
                if needs_action:
                    last_order_attempt = time.monotonic()
                    trade_quote_pressure(
                        args,
                        quote_pressure,
                        bid,
                        ask,
                        point,
                        blocked_sides,
                    )
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
