"""Deterministic simulation-step to solver-evaluation scheduling."""

from __future__ import annotations

import threading
import time


def solver_grants(simulation_steps: int, solver_steps: int) -> tuple[int, ...]:
    """Distribute solver evaluations over simulation steps with integer arithmetic."""
    if simulation_steps <= 0 or solver_steps <= 0:
        raise ValueError("step counts must be positive")
    return tuple(
        ((index + 1) * solver_steps) // simulation_steps
        - (index * solver_steps) // simulation_steps
        for index in range(simulation_steps)
    )


class SolverClock:
    """Gate an inference thread at deterministic solver boundaries."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self.start(enabled=False)

    def start(self, *, enabled: bool) -> None:
        with self._condition:
            self.enabled = bool(enabled)
            self.closed = False
            self.available = 0
            self.issued = 0
            self.consumed = 0
            self.completed = 0
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self.closed = True
            self._condition.notify_all()

    def consume(self) -> bool:
        with self._condition:
            if not self.enabled:
                return not self.closed
            while self.available == 0 and not self.closed:
                self._condition.wait(timeout=0.1)
            if self.closed:
                return False
            self.available -= 1
            self.consumed += 1
            self._condition.notify_all()
            return True

    def complete(self) -> None:
        with self._condition:
            if not self.enabled:
                return
            if self.completed >= self.consumed:
                raise RuntimeError("solver completion has no matching consumed grant")
            self.completed += 1
            self._condition.notify_all()

    def grant_and_wait(self, count: int, timeout: float = 180.0) -> dict[str, int | bool]:
        if count <= 0:
            raise ValueError("grant count must be positive")
        with self._condition:
            if not self.enabled or self.closed:
                return self.snapshot(accepted=False)
            self.available += count
            self.issued += count
            target = self.issued
            self._condition.notify_all()
            deadline = time.monotonic() + timeout
            while self.completed < target and not self.closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return self.snapshot(accepted=False)
                self._condition.wait(timeout=remaining)
            return self.snapshot(accepted=self.completed >= target)

    def snapshot(self, *, accepted: bool = True) -> dict[str, int | bool]:
        return {
            "accepted": bool(accepted),
            "enabled": self.enabled,
            "closed": self.closed,
            "available": self.available,
            "issued": self.issued,
            "consumed": self.consumed,
            "completed": self.completed,
        }
