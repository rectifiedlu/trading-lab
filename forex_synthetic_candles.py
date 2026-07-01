from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time


def timeframe_seconds(tf: str) -> int:
    key = tf.lower().strip()
    if key.endswith("s"):
        return int(float(key[:-1]))
    if key.endswith("m"):
        return int(float(key[:-1]) * 60)
    if key.endswith("h"):
        return int(float(key[:-1]) * 3600)
    raise ValueError(f"unsupported timeframe: {tf}")


@dataclass
class Candle:
    bucket: int
    open: float
    high: float
    low: float
    close: float


class SyntheticBidOHLC:
    """Live bid-based OHLC candles built from MT5 ticks."""

    def __init__(self, timeframe: str, maxlen: int = 1000):
        self.tf_sec = timeframe_seconds(timeframe)
        self.maxlen = maxlen
        self.current: Candle | None = None
        self.closed: deque[Candle] = deque(maxlen=maxlen)
        self.last_closed_bucket = 0

    def update(self, tick) -> bool:
        bid = float(tick.bid)
        ts = int(getattr(tick, "time", 0) or time.time())
        bucket = ts // self.tf_sec
        if self.current is None:
            self.current = Candle(bucket, bid, bid, bid, bid)
            return False
        if bucket != self.current.bucket:
            self.closed.append(self.current)
            self.last_closed_bucket = int(self.current.bucket)
            self.current = Candle(bucket, bid, bid, bid, bid)
            return True
        self.current.high = max(self.current.high, bid)
        self.current.low = min(self.current.low, bid)
        self.current.close = bid
        return False
