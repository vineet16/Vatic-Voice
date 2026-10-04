"""Event-loop lag monitor: sleeps ``interval`` and records the overshoot."""

from __future__ import annotations

import asyncio
import time


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = min(len(ordered) - 1, max(0, round(q / 100 * (len(ordered) - 1))))
    return ordered[k]


class LoopLagMonitor:
    def __init__(self, interval: float = 0.010, max_samples: int = 100_000) -> None:
        self.interval = interval
        self.max_samples = max_samples
        self.samples_ms: list[float] = []
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._run())

    async def _run(self) -> None:
        while True:
            t0 = time.perf_counter()
            await asyncio.sleep(self.interval)
            lag = (time.perf_counter() - t0 - self.interval) * 1000
            if len(self.samples_ms) < self.max_samples:
                self.samples_ms.append(max(0.0, lag))

    def p(self, q: float) -> float:
        return percentile(self.samples_ms, q)

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
