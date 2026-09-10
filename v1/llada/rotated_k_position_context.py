from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass(frozen=True)
class BlockBounds:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class PositionState:
    # None 表示 full forward 或普通 append-cache。
    bounds: BlockBounds | None
    # 由第 0 层构建，后续 31 层复用。
    position_context: dict[str, Any] | None = None


_ACTIVE_STATE: ContextVar[PositionState | None] = ContextVar(
    "llada_rotated_k_position_state",
    default=None,
)


@contextmanager
def position_forward_scope(
    block_start: int | None = None,
    block_end: int | None = None,
) -> Iterator[PositionState]:
    if (block_start is None) != (block_end is None):
        raise ValueError(
            "block_start and block_end must either both be set "
            "or both be None"
        )

    bounds = None

    if block_start is not None and block_end is not None:
        if block_start < 0:
            raise ValueError("block_start must be non-negative")

        if block_end <= block_start:
            raise ValueError(
                f"Invalid block bounds: {block_start}:{block_end}"
            )

        bounds = BlockBounds(
            start=int(block_start),
            end=int(block_end),
        )

    state = PositionState(bounds=bounds)
    token = _ACTIVE_STATE.set(state)

    try:
        yield state
    finally:
        _ACTIVE_STATE.reset(token)


def get_position_state() -> PositionState:
    state = _ACTIVE_STATE.get()

    if state is None:
        raise RuntimeError(
            "No active position_forward_scope. "
            "Every model forward in v2b must run inside the scope."
        )

    return state
