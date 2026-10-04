"""Trace storage: append-only JSONL files plus a SQLite index.

``TraceStore`` is synchronous and thread-safe; it must never be called from the
audio event loop directly. ``AsyncTraceWriter`` is the event-loop-facing side:
records go into a bounded in-memory queue (dropped with a counter when full) and
a background task writes them in batches on an executor thread.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Executor
from pathlib import Path

from vatic.core.tools import ToolInfo
from vatic.trace.schema import LifecycleEvent, SessionTrace, TurnTrace

Record = TurnTrace | SessionTrace | LifecycleEvent

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    trace_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    route TEXT NOT NULL,
    flow_id TEXT,
    flow_version INTEGER,
    flow_step TEXT,
    outcome TEXT,
    body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS turns_session ON turns(session_id, turn_index);
CREATE INDEX IF NOT EXISTS turns_route ON turns(route, flow_id);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    outcome TEXT,
    body TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    flow_id TEXT NOT NULL,
    body TEXT NOT NULL
);
"""


class TraceStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.root / "index.sqlite", check_same_thread=False)
        self._db.executescript(_SCHEMA)
        self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # -- writes ---------------------------------------------------------------

    def write_batch(self, records: Sequence[Record]) -> None:
        turns = [r for r in records if isinstance(r, TurnTrace)]
        sessions = [r for r in records if isinstance(r, SessionTrace)]
        events = [r for r in records if isinstance(r, LifecycleEvent)]
        with self._lock:
            if turns:
                with open(self.root / "turns.jsonl", "a", encoding="utf-8") as fh:
                    for t in turns:
                        fh.write(t.model_dump_json() + "\n")
                self._db.executemany(
                    "INSERT OR REPLACE INTO turns VALUES (?,?,?,?,?,?,?,?,?)",
                    [
                        (
                            t.trace_id,
                            t.session_id,
                            t.turn_index,
                            t.route,
                            t.flow_id,
                            t.flow_version,
                            t.flow_step,
                            t.outcome,
                            t.model_dump_json(),
                        )
                        for t in turns
                    ],
                )
            if sessions:
                with open(self.root / "sessions.jsonl", "a", encoding="utf-8") as fh:
                    for s in sessions:
                        fh.write(s.model_dump_json() + "\n")
                self._db.executemany(
                    "INSERT OR REPLACE INTO sessions VALUES (?,?,?)",
                    [(s.session_id, s.outcome, s.model_dump_json()) for s in sessions],
                )
            if events:
                with open(self.root / "events.jsonl", "a", encoding="utf-8") as fh:
                    for e in events:
                        fh.write(e.model_dump_json() + "\n")
                self._db.executemany(
                    "INSERT INTO events (flow_id, body) VALUES (?,?)",
                    [(e.flow_id, e.model_dump_json()) for e in events],
                )
            self._db.commit()

    def write(self, record: Record) -> None:
        self.write_batch([record])

    def write_tool_manifest(self, catalog: Mapping[str, ToolInfo]) -> None:
        data = {name: info.to_json() for name, info in sorted(catalog.items())}
        with self._lock:
            (self.root / "tools.json").write_text(json.dumps(data, indent=2, sort_keys=True))

    def tool_manifest(self) -> dict[str, ToolInfo]:
        path = self.root / "tools.json"
        if not path.exists():
            return {}
        data = json.loads(path.read_text())
        return {name: ToolInfo.from_json(v) for name, v in data.items()}

    # -- reads ----------------------------------------------------------------

    def iter_turns(
        self,
        *,
        session_id: str | None = None,
        route: str | None = None,
        flow_id: str | None = None,
    ) -> Iterator[TurnTrace]:
        sql = "SELECT body FROM turns WHERE 1=1"
        params: list[str] = []
        if session_id is not None:
            sql += " AND session_id = ?"
            params.append(session_id)
        if route is not None:
            sql += " AND route = ?"
            params.append(route)
        if flow_id is not None:
            sql += " AND flow_id = ?"
            params.append(flow_id)
        sql += " ORDER BY session_id, turn_index, trace_id"
        with self._lock:
            rows = self._db.execute(sql, params).fetchall()
        for (body,) in rows:
            yield TurnTrace.model_validate_json(body)

    def iter_sessions(self, *, outcome: str | None = None) -> Iterator[SessionTrace]:
        sql = "SELECT body FROM sessions"
        params: list[str] = []
        if outcome is not None:
            sql += " WHERE outcome = ?"
            params.append(outcome)
        sql += " ORDER BY session_id"
        with self._lock:
            rows = self._db.execute(sql, params).fetchall()
        for (body,) in rows:
            yield SessionTrace.model_validate_json(body)

    def iter_events(self, flow_id: str | None = None) -> Iterator[LifecycleEvent]:
        sql = "SELECT body FROM events"
        params: list[str] = []
        if flow_id is not None:
            sql += " WHERE flow_id = ?"
            params.append(flow_id)
        sql += " ORDER BY id"
        with self._lock:
            rows = self._db.execute(sql, params).fetchall()
        for (body,) in rows:
            yield LifecycleEvent.model_validate_json(body)

    def load_corpus(
        self, *, outcome: str | None = "success"
    ) -> list[tuple[SessionTrace, list[TurnTrace]]]:
        """Sessions with their non-shadow turns, ordered by turn index."""
        sessions = list(self.iter_sessions(outcome=outcome))
        by_session: dict[str, list[TurnTrace]] = {s.session_id: [] for s in sessions}
        for t in self.iter_turns():
            if t.route != "shadow" and t.session_id in by_session:
                by_session[t.session_id].append(t)
        return [(s, by_session[s.session_id]) for s in sessions]


class AsyncTraceWriter:
    """Non-blocking trace sink for the event loop."""

    def __init__(
        self,
        store: TraceStore,
        executor: Executor,
        *,
        maxsize: int = 10_000,
        batch_size: int = 64,
    ) -> None:
        self._store = store
        self._executor = executor
        self._queue: asyncio.Queue[Record] = asyncio.Queue(maxsize=maxsize)
        self._batch_size = batch_size
        self._task: asyncio.Task[None] | None = None
        self.dropped = 0
        self.written = 0

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._run())

    def submit(self, record: Record) -> bool:
        """Enqueue without blocking. Returns False (and counts a drop) if full."""
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            self.dropped += 1
            return False
        return True

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            first = await self._queue.get()
            batch = [first]
            while len(batch) < self._batch_size and not self._queue.empty():
                batch.append(self._queue.get_nowait())
            try:
                await loop.run_in_executor(self._executor, self._store.write_batch, batch)
                self.written += len(batch)
            finally:
                for _ in batch:
                    self._queue.task_done()

    async def flush(self) -> None:
        await self._queue.join()

    async def aclose(self) -> None:
        if self._task is not None:
            await self.flush()
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
