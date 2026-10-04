"""Mock clinic backend: SQLite in memory, deterministically seeded per session."""

from __future__ import annotations

import datetime as dt
import random
import sqlite3
import threading
from typing import Any

BASE_DATE = dt.date(2026, 10, 5)  # a Monday
CLINIC_TIMES = ["09:00", "10:00", "11:00", "13:00", "14:00", "15:00", "16:00"]
PROVIDERS = ["Dr. Patel", "Dr. Kim", "Dr. Lopez"]
MAX_OFFERED = 4

FIRST_NAMES = [
    "Jane",
    "John",
    "Maria",
    "David",
    "Sarah",
    "Michael",
    "Emily",
    "James",
    "Laura",
    "Robert",
    "Anna",
    "Daniel",
    "Olivia",
    "Thomas",
    "Grace",
    "Kevin",
    "Rachel",
    "Brian",
    "Megan",
    "Steven",
    "Hannah",
    "Peter",
    "Chloe",
    "Victor",
    "Nina",
    "Samuel",
    "Julia",
    "Oscar",
    "Lucy",
    "Henry",
]
LAST_NAMES = [
    "Doe",
    "Smith",
    "Garcia",
    "Johnson",
    "Brown",
    "Miller",
    "Davis",
    "Wilson",
    "Moore",
    "Taylor",
    "Anderson",
    "Thomas",
    "Jackson",
    "White",
    "Harris",
    "Martin",
    "Thompson",
    "Clark",
    "Lewis",
    "Walker",
    "Hall",
    "Young",
    "King",
    "Wright",
    "Scott",
    "Green",
    "Baker",
    "Adams",
    "Nelson",
    "Carter",
]


class SlotTaken(Exception):
    pass


class ClinicBackend:
    def __init__(self, seed: int, n_patients: int = 60) -> None:
        rng = random.Random(seed)
        self.seed = seed
        today = BASE_DATE + dt.timedelta(days=seed % 28)
        self.today = today + dt.timedelta(
            days=(7 - today.weekday()) % 7 if today.weekday() >= 5 else 0
        )
        self._lock = threading.Lock()
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self.db.executescript(
            """
            CREATE TABLE patients (id TEXT PRIMARY KEY, first_name TEXT, last_name TEXT);
            CREATE TABLE appointments (
                id TEXT PRIMARY KEY, patient_id TEXT, date TEXT, time TEXT,
                provider TEXT, status TEXT
            );
            """
        )
        names = rng.sample([(f, la) for f in FIRST_NAMES for la in LAST_NAMES], n_patients)
        self.patients = [(f"P{1001 + i}", f, la) for i, (f, la) in enumerate(names)]
        self.db.executemany("INSERT INTO patients VALUES (?,?,?)", self.patients)
        self._next_appt = 5001
        days = self.clinic_days(21)
        # Busy slots held by other patients.
        for d in days:
            for t in CLINIC_TIMES:
                if rng.random() < 0.45:
                    self._insert("P0000", d, t, rng.choice(PROVIDERS))
        # 40% of our patients have one upcoming appointment.
        for pid, _, _ in self.patients:
            if rng.random() < 0.4:
                free = [(d, t) for d in days[1:15] for t in self.open_times(d)]
                if free:
                    d, t = rng.choice(free)
                    self._insert(pid, d, t, rng.choice(PROVIDERS))
        self.db.commit()
        self.initial_state = self.snapshot()

    def _insert(self, pid: str, date: str, time: str, provider: str) -> str:
        aid = f"A{self._next_appt}"
        self._next_appt += 1
        self.db.execute(
            "INSERT INTO appointments VALUES (?,?,?,?,?, 'booked')",
            (aid, pid, date, time, provider),
        )
        return aid

    def clinic_days(self, horizon: int) -> list[str]:
        out = []
        for i in range(1, horizon + 1):
            d = self.today + dt.timedelta(days=i)
            if d.weekday() < 5:
                out.append(d.isoformat())
        return out

    def open_times(self, date: str) -> list[str]:
        try:
            d = dt.date.fromisoformat(date)
        except ValueError:
            return []
        if d.weekday() >= 5 or d <= self.today:
            return []
        taken = {
            r[0]
            for r in self.db.execute(
                "SELECT time FROM appointments WHERE date=? AND status='booked'", (date,)
            )
        }
        return [t for t in CLINIC_TIMES if t not in taken]

    # -- tool operations --------------------------------------------------------------

    def lookup_patient(self, name: str) -> dict[str, Any]:
        parts = name.strip().lower().split()
        with self._lock:
            row = None
            if len(parts) >= 2:
                row = self.db.execute(
                    "SELECT id, first_name, last_name FROM patients "
                    "WHERE lower(first_name)=? AND lower(last_name)=?",
                    (parts[0], " ".join(parts[1:])),
                ).fetchone()
            if row is None:
                return {"found": False, "patient": None}
            appt = self.db.execute(
                "SELECT id, date, time, provider FROM appointments "
                "WHERE patient_id=? AND status='booked' AND date>? ORDER BY date, time LIMIT 1",
                (row[0], self.today.isoformat()),
            ).fetchone()
        nxt = None
        if appt:
            nxt = {"id": appt[0], "date": appt[1], "time": appt[2], "provider": appt[3]}
        return {
            "found": True,
            "patient": {
                "id": row[0],
                "first_name": row[1],
                "last_name": row[2],
                "next_appointment": nxt,
            },
        }

    def check_availability(self, date: str) -> dict[str, Any]:
        with self._lock:
            times = self.open_times(date)[:MAX_OFFERED]
        return {"date": date, "available": bool(times), "times": times}

    def book_appointment(self, patient_id: str, date: str, time: str) -> dict[str, Any]:
        with self._lock:
            if time not in self.open_times(date):
                raise SlotTaken(f"{date} {time} is not available")
            provider = PROVIDERS[(int(time[:2]) + len(date)) % len(PROVIDERS)]
            aid = self._insert(patient_id, date, time, provider)
            self.db.commit()
        return {"appointment_id": aid, "date": date, "time": time, "provider": provider}

    def reschedule_appointment(self, appointment_id: str, date: str, time: str) -> dict[str, Any]:
        with self._lock:
            row = self.db.execute(
                "SELECT provider FROM appointments WHERE id=? AND status='booked'",
                (appointment_id,),
            ).fetchone()
            if row is None:
                raise SlotTaken(f"no booked appointment {appointment_id}")
            if time not in self.open_times(date):
                raise SlotTaken(f"{date} {time} is not available")
            self.db.execute(
                "UPDATE appointments SET date=?, time=? WHERE id=?", (date, time, appointment_id)
            )
            self.db.commit()
        return {"appointment_id": appointment_id, "date": date, "time": time, "provider": row[0]}

    def cancel_appointment(self, appointment_id: str) -> dict[str, Any]:
        with self._lock:
            cur = self.db.execute(
                "UPDATE appointments SET status='cancelled' WHERE id=? AND status='booked'",
                (appointment_id,),
            )
            self.db.commit()
            if cur.rowcount == 0:
                raise SlotTaken(f"no booked appointment {appointment_id}")
        return {"cancelled": True, "appointment_id": appointment_id}

    # -- state checks (task success is judged from backend state, never by an LLM) ------

    def snapshot(self) -> list[tuple[Any, ...]]:
        with self._lock:
            return list(self.db.execute("SELECT * FROM appointments ORDER BY id"))

    def appointments_for(self, patient_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.db.execute(
                "SELECT id, date, time, status FROM appointments WHERE patient_id=? ORDER BY id",
                (patient_id,),
            ).fetchall()
        return [{"id": r[0], "date": r[1], "time": r[2], "status": r[3]} for r in rows]


class BackendPool:
    """Maps session ids to their backend so tools can be registered once."""

    def __init__(self) -> None:
        self._backends: dict[str, ClinicBackend] = {}

    def add(self, session_id: str, backend: ClinicBackend) -> None:
        self._backends[session_id] = backend

    def get(self, session_id: str) -> ClinicBackend:
        return self._backends[session_id]

    def remove(self, session_id: str) -> None:
        self._backends.pop(session_id, None)
