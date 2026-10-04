"""Test configuration.

Every async test runs on an event loop in debug mode with
``slow_callback_duration = 5 ms``; any slow-callback warning fails the test
(spec Section 14). Mark a test ``allow_slow_callbacks`` to opt out.
"""

from __future__ import annotations

import asyncio
import linecache
import logging
from collections.abc import Callable, Iterator

import pytest

# asyncio debug mode records a full traceback for every Future/Handle, and
# traceback extraction calls linecache.checkcache, which stat()s every file in the
# (deep, pytest) stack. That bookkeeping alone can exceed the 5 ms threshold, so it
# is disabled here; it only refreshes cached source lines for display.
linecache.checkcache = lambda filename=None: None


def _debug_loop() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()
    loop.set_debug(True)
    loop.slow_callback_duration = 0.005
    return loop


def pytest_asyncio_loop_factories(
    config: pytest.Config, item: pytest.Item
) -> dict[str, Callable[[], asyncio.AbstractEventLoop]]:
    return {"debug": _debug_loop}


class _SlowCallbackCatcher(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if msg.startswith("Executing") and " took " in msg:
            self.messages.append(msg)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "allow_slow_callbacks: do not fail on slow callbacks")


@pytest.fixture(autouse=True)
def _no_slow_callbacks(request: pytest.FixtureRequest) -> Iterator[None]:
    catcher = _SlowCallbackCatcher()
    logger = logging.getLogger("asyncio")
    logger.addHandler(catcher)
    try:
        yield
    finally:
        logger.removeHandler(catcher)
    if catcher.messages and request.node.get_closest_marker("allow_slow_callbacks") is None:
        pytest.fail("slow event-loop callbacks:\n" + "\n".join(catcher.messages[:5]))


# -- opt-in diagnostics: VATIC_LOOP_PROFILE=1 samples the loop thread's stack during
# -- steps longer than 3 ms and prints the hottest stacks per test.
if __import__("os").environ.get("VATIC_LOOP_PROFILE"):
    import collections
    import sys
    import threading
    import time
    import traceback
    from asyncio import events

    _state: dict[str, float | int | None] = {"start": None, "tid": None}
    _samples: collections.Counter[str] = collections.Counter()
    _orig_run = events.Handle._run
    _IDLE = {
        "wait",
        "_wait_for_tstate_lock",
        "get",
        "select",
        "poll",
        "_recv",
        "recv_bytes",
        "_recv_bytes",
        "_worker",
        "wait_result_broken_or_wakeup",
        "_watch",
        "accept",
    }

    def _timed_run(self: events.Handle) -> None:
        _state["start"] = time.perf_counter()
        _state["tid"] = threading.get_ident()
        try:
            _orig_run(self)
        finally:
            _state["start"] = None

    def _watch() -> None:
        while True:
            time.sleep(0.0005)
            start, tid = _state["start"], _state["tid"]
            if start is not None and time.perf_counter() - start > 0.003:
                me = threading.get_ident()
                for t_id, frame in sys._current_frames().items():
                    if t_id == me:
                        continue
                    st = traceback.extract_stack(frame)
                    if t_id != tid and st[-1].name in _IDLE:
                        continue
                    tag = "LOOP " if t_id == tid else "other"
                    _samples[
                        tag
                        + " "
                        + " <- ".join(
                            f"{f.filename.split('/')[-1]}:{f.lineno}:{f.name}" for f in st[-1:-8:-1]
                        )
                    ] += 1

    events.Handle._run = _timed_run  # type: ignore[method-assign]
    threading.Thread(target=_watch, daemon=True).start()

    @pytest.fixture(autouse=True)
    def _loop_profile() -> Iterator[None]:
        _samples.clear()
        yield
        for key, n in _samples.most_common(6):
            print(f"\n[loop-profile] {n:4d}  {key}")
