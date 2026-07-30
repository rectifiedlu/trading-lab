"""Directional score-reset state shared by score-based paper traders."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DirectionalScoreReset:
    long_blocked: bool = False
    short_blocked: bool = False

    def mark_fixed_exit(self, side: int) -> None:
        if side == 1:
            self.long_blocked = True
        elif side == -1:
            self.short_blocked = True
        else:
            raise ValueError(f"side must be +1 or -1, got {side}")

    def observe(self, signal: int) -> None:
        if self.long_blocked and signal != 1:
            self.long_blocked = False
        if self.short_blocked and signal != -1:
            self.short_blocked = False

    def blocks(self, side: int) -> bool:
        return (side == 1 and self.long_blocked) or (side == -1 and self.short_blocked)

    def text(self) -> str:
        return f"reset_block=L{int(self.long_blocked)}/S{int(self.short_blocked)}"