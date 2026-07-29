"""Constant-memory final rankings for large forex parameter sweeps."""
from __future__ import annotations

import heapq

from forex_strategy_common import TradeResult


class TopResultTracker:
    """Keep exact top-N PnL and PnL/drawdown results from a streamed sweep."""

    def __init__(self, top: int):
        self.limit = max(1, int(top))
        self.serial = 0
        self.pnl_heap: list[tuple[tuple[float, ...], int, TradeResult]] = []
        self.pnl_dd_heap: list[tuple[tuple[float, ...], int, TradeResult]] = []

    def _push(self, heap, key: tuple[float, ...], result: TradeResult) -> None:
        item = (key, self.serial, result)
        if len(heap) < self.limit:
            heapq.heappush(heap, item)
        elif item[:2] > heap[0][:2]:
            heapq.heapreplace(heap, item)

    def add(self, result: TradeResult) -> None:
        self.serial += 1
        self._push(
            self.pnl_heap,
            (float(result.total), float(result.realised), float(result.profit_factor)),
            result,
        )
        if result.trades > 0 and result.total > 0.0 and result.max_drawdown > 0.0:
            self._push(
                self.pnl_dd_heap,
                (
                    float(result.total / result.max_drawdown),
                    float(result.total),
                    float(getattr(result, "median_day", result.total)),
                    float(result.profit_factor),
                ),
                result,
            )

    @staticmethod
    def _ranked(heap) -> list[TradeResult]:
        ordered = sorted(heap, key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in ordered]

    def by_pnl(self) -> list[TradeResult]:
        return self._ranked(self.pnl_heap)

    def by_pnl_dd(self) -> list[TradeResult]:
        return self._ranked(self.pnl_dd_heap)


def print_top_result_sections(tracker: TopResultTracker) -> None:
    sections = [
        ("top by total PnL", tracker.by_pnl()),
        ("top by total/account DD (PnL/DD)", tracker.by_pnl_dd()),
    ]
    headers = [
        "#", "pair", "strat", "tf", "tp", "sl", "total", "realised", "open",
        "tr", "wr%", "pf", "avg/day", "med/day", "acct_dd", "tr_max",
        "cum_dd", "worst_loss", "pnl/dd", "med_loss", "stops", "sig", "params",
    ]
    for title, ranked in sections:
        rows = []
        for index, result in enumerate(ranked, 1):
            ratio = result.total / result.max_drawdown if result.max_drawdown > 0.0 else 0.0
            rows.append([
                str(index), result.pair, result.strategy, result.timeframe,
                f"{result.tp_points:g}", f"{result.sl_points:g}",
                f"${result.total:+.4f}", f"${result.realised:+.4f}",
                f"${result.open_unrealized:+.4f}", str(result.trades),
                f"{result.win_rate:.1f}", f"{result.profit_factor:.4g}",
                f"${getattr(result, 'avg_day', result.total):+.4f}",
                f"${getattr(result, 'median_day', result.total):+.4f}",
                f"${result.max_drawdown:.2f}",
                f"${getattr(result, 'trade_max_drawdown', 0.0):.2f}",
                f"${getattr(result, 'cum_max_drawdown', 0.0):.2f}",
                f"${getattr(result, 'worst_trade_pnl', 0.0):+.4f}",
                f"{ratio:.2f}", f"${getattr(result, 'median_loss', 0.0):+.4f}",
                str(getattr(result, "stop_losses", 0)),
                str(getattr(result, "signal_exits", 0)),
                str(getattr(result, "params", "")),
            ])
        print("", flush=True)
        print(f"  {title}", flush=True)
        if not rows:
            print("  no qualifying results", flush=True)
            continue
        widths = [len(header) for header in headers]
        for row in rows:
            for column, cell in enumerate(row):
                widths[column] = max(widths[column], len(cell))
        print("  " + " ".join(header.rjust(widths[i]) for i, header in enumerate(headers)), flush=True)
        print("  " + "-" * (sum(widths) + len(widths) - 1), flush=True)
        for row in rows:
            print("  " + " ".join(cell.rjust(widths[i]) for i, cell in enumerate(row)), flush=True)