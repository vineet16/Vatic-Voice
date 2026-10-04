"""Simulated callers.

``ScriptedCaller`` is deterministic (seeded) and needs no network; ``LLMCaller``
lets a real LLM play the caller. Each utterance carries a ground-truth label
(on_path / side_question / multi_intent / correction / ...) used by the
benchmark to measure wrong-route rate independently of shadow mode.
"""

from __future__ import annotations

import datetime as dt
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from examples.scheduling.backend import ClinicBackend
from vatic.core.extract import extract_date_phrase, extract_time_phrase
from vatic.core.formatters import human_date, human_time
from vatic.core.transforms import NUMBER_WORDS, TransformContext, normalize_date, normalize_time
from vatic.llm.client import LLMClient

PERSONAS: dict[str, Any] = yaml.safe_load(
    (Path(__file__).parent / "personas.yaml").read_text(encoding="utf-8")
)
_ORD_SUFFIX = {1: "st", 2: "nd", 3: "rd"}


def _weighted(rng: random.Random, weights: dict[str, int]) -> str:
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else _ORD_SUFFIX.get(n % 10, "th")
    return f"{n}{suffix}"


def date_phrases(target: str, today: dt.date) -> list[str]:
    d = dt.date.fromisoformat(target)
    wd, month = d.strftime("%A"), d.strftime("%B")
    cands = [
        wd,
        f"next {wd}",
        f"this {wd}",
        "tomorrow",
        f"{month} {d.day}",
        f"{month} {_ordinal(d.day)}",
        f"the {_ordinal(d.day)}",
        f"{wd} the {_ordinal(d.day)}",
    ]
    ctx = TransformContext(today)
    return [c for c in cands if normalize_date(c, ctx) == target]


def time_phrases(target: str) -> list[str]:
    hour = int(target[:2])
    h12 = hour % 12 or 12
    suffix = "am" if hour < 12 else "pm"
    word = next(w for w, n in NUMBER_WORDS.items() if n == h12)
    part = "in the morning" if suffix == "am" else "in the afternoon"
    cands = [
        f"{h12} {suffix}",
        f"{h12}{suffix}",
        f"{h12} o'clock",
        f"{word} {suffix}",
        f"{h12}:00",
        f"{h12} {part}",
    ]
    ctx = TransformContext(dt.date(2026, 1, 1))
    return [c for c in cands if normalize_time(c, ctx) == target]


@dataclass
class Persona:
    goal: str
    style: str
    name: str
    patient_id: str
    today: str
    opener: str = "name"
    target_date: str | None = None
    date_phrase: str | None = None
    target_time: str | None = None
    time_phrase: str | None = None
    alt_date: str | None = None
    alt_date_phrase: str | None = None
    alt_time: str | None = None
    alt_time_phrase: str | None = None
    appointment_id: str | None = None
    first_choice_phrase: str | None = None  # a day with no openings, tried first
    event: tuple[str, str] | None = None  # (at question, kind)

    def describe(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def _pick_slot(
    backend: ClinicBackend, rng: random.Random, avoid: str | None = None
) -> tuple[str, str] | None:
    days = [d for d in backend.clinic_days(14)[:10] if d != avoid]
    rng.shuffle(days)
    for d in days:
        offered = backend.check_availability(d)["times"]
        if offered:
            return d, rng.choice(offered)
    return None


def make_persona(backend: ClinicBackend, rng: random.Random) -> Persona:
    goal = _weighted(rng, PERSONAS["goals"])
    style = _weighted(rng, PERSONAS["styles"])
    with_appt, without = [], []
    for pid, first, last in backend.patients:
        booked = [a for a in backend.appointments_for(pid) if a["status"] == "booked"]
        (with_appt if booked else without).append((pid, f"{first} {last}", booked))
    if goal in ("cancel", "reschedule") and not with_appt:
        goal = "book"
    pid, name, booked = rng.choice(with_appt if goal in ("cancel", "reschedule") else without)
    p = Persona(goal=goal, style=style, name=name, patient_id=pid, today=backend.today.isoformat())
    today = backend.today
    if goal in ("book", "reschedule"):
        slot = _pick_slot(backend, rng)
        alt = _pick_slot(backend, rng, avoid=slot[0] if slot else None)
        if slot is None or alt is None:
            goal = p.goal = "hours"
        else:
            p.target_date, p.target_time = slot
            p.date_phrase = rng.choice(date_phrases(slot[0], today))
            p.time_phrase = rng.choice(time_phrases(slot[1]))
            p.alt_date, p.alt_time = alt
            p.alt_date_phrase = rng.choice(date_phrases(alt[0], today))
            p.alt_time_phrase = rng.choice(time_phrases(alt[1]))
    if goal in ("cancel", "reschedule"):
        p.appointment_id = booked[0]["id"]
    if goal == "book":
        p.opener = _weighted(rng, PERSONAS["book_openers"])
        if p.opener == "name" and rng.randrange(100) < PERSONAS["unavailable_first_choice"]:
            weekend = today + dt.timedelta(days=(5 - today.weekday()) % 7 or 7)
            p.first_choice_phrase = rng.choice(date_phrases(weekend.isoformat(), today))
    if style == "interrupts":
        if goal in ("book", "reschedule"):
            p.event = rng.choice(
                [
                    ("date", "side_question"),
                    ("date", "multi_intent"),
                    ("time", "multi_intent"),
                    ("time", "correction"),
                    ("time", "side_question"),
                ]
            )
        elif goal == "cancel":
            p.event = ("confirm", "side_question")
    return p


@dataclass
class Utterance:
    text: str
    label: str


@dataclass
class ScriptedCaller:
    persona: Persona
    rng: random.Random
    said_bye: bool = False
    unsure_done: bool = False
    event_done: bool = False
    repeats: int = 0
    log: list[str] = field(default_factory=list)

    def _style_bank_style(self) -> str:
        return "chatty" if self.persona.style == "interrupts" else self.persona.style

    def _say(self, key: str, label: str = "on_path", **values: str) -> Utterance:
        bank = PERSONAS["phrases"][key]
        if isinstance(bank, dict):
            bank = bank.get(self._style_bank_style()) or bank["chatty"]
        text = str(self.rng.choice(bank)).format(**values)
        return Utterance(_cap(text), label)

    def _opener_key(self) -> str:
        p = self.persona
        if p.goal == "book":
            return {"name": "book", "name_date": "book_date", "none": "book_noname"}[p.opener]
        return p.goal

    def opening(self) -> Utterance:
        return self._open(self._opener_key())

    def _open(self, key: str) -> Utterance:
        bank = PERSONAS["phrases"]["open"][key]
        bank = bank.get(self._style_bank_style()) or bank["chatty"]
        p = self.persona
        return Utterance(
            _cap(str(self.rng.choice(bank)).format(name=p.name, date=p.date_phrase or "")), "opener"
        )

    def _event(self, at: str, slot: str | None = None) -> Utterance | None:
        p = self.persona
        if p.event is None or self.event_done or p.event[0] != at:
            return None
        self.event_done = True
        kind = p.event[1]
        if kind == "side_question":
            return Utterance(
                self.rng.choice(PERSONAS["phrases"]["side_questions"]), "side_question"
            )
        if kind == "multi_intent" and slot:
            text = self.rng.choice(PERSONAS["phrases"]["multi_intent"]).format(slot=slot)
            return Utterance(_cap(text), "multi_intent")
        if kind == "correction" and p.alt_date_phrase:
            text = self.rng.choice(PERSONAS["phrases"]["corrections"]).format(
                date2=p.alt_date_phrase
            )
            p.target_date, p.date_phrase = p.alt_date, p.alt_date_phrase
            p.target_time, p.time_phrase = p.alt_time, p.alt_time_phrase
            return Utterance(text, "correction")
        return None

    def _mentioned(self, text: str) -> tuple[set[str], set[str]]:
        """Dates (ISO) and times (HH:MM) the agent mentioned, however it phrased them."""
        ctx = TransformContext(dt.date.fromisoformat(self.persona.today))
        dates = {normalize_date(s.text, ctx) for s in extract_date_phrase(text)}
        times = {normalize_time(s.text, ctx) for s in extract_time_phrase(text)}
        return {d for d in dates if d}, {t for t in times if t}

    def _details_ok(self, text: str, low: str) -> bool:
        p = self.persona
        hd, ht = human_date(p.target_date), human_time(p.target_time)
        if hd and ht and hd.lower() in low and ht.lower() in low:
            return True
        dates, times = self._mentioned(text)
        return p.target_date in dates and p.target_time in times

    def respond(self, agent_text: str) -> Utterance | None:
        p = self.persona
        text = re.sub("[\u2010-\u2015]", "-", agent_text).replace("\u202f", " ")
        low = text.lower()
        if self.said_bye or "goodbye" in low:
            return None
        if any(k in low for k in _CLOSING) and not any(k in low for k in _CONFIRM):
            self.said_bye = True
            return self._say("bye", "closing")
        if p.goal == "hours" and any(k in low for k in _HOURS_ANSWER):
            # Got what they called for; a real agent may not add "anything else?".
            self.said_bye = True
            return self._say("bye", "closing")
        if "full name" in low or "couldn't find" in low:
            return self._say("name", name=p.name)
        if "how can i help" in low:
            return self._open(self._opener_key())
        if (
            "shall i" in low
            or "go ahead?" in low
            or ("?" in low and any(k in low for k in _CONFIRM))
        ):
            ev = self._event("confirm")
            if ev is not None:
                return ev
            if p.goal in ("book", "reschedule"):
                if not self._details_ok(text, low):
                    text = self.rng.choice(PERSONAS["phrases"]["wrong_details"]).format(
                        date=p.date_phrase, time=p.time_phrase
                    )
                    return Utterance(text, "correction")
            return self._say("affirm")
        if any(
            k in low for k in ("what day", "another day", "which day", "move it to", "day works")
        ) or ("?" in low and any(k in low for k in _DATE_Q)):
            if self.persona.style == "confused" and not self.unsure_done:
                self.unsure_done = True
                return Utterance(PERSONAS["phrases"]["unsure_date"][0], "other")
            if p.first_choice_phrase and "no openings" not in low:
                first, p.first_choice_phrase = p.first_choice_phrase, None
                return self._say("date", date=first)
            ev = self._event("date", slot=p.date_phrase)
            if ev is not None:
                return ev
            return self._say("date", date=p.date_phrase or "")
        if (
            "which time" in low
            or "which would you like" in low
            or ("?" in low and any(k in low for k in _TIME_Q))
        ):
            hd = human_date(p.target_date)
            mentioned, _ = self._mentioned(text)
            # Correct only if the agent named a different day (not if it named none).
            if mentioned and (hd and hd.lower() not in low) and p.target_date not in mentioned:
                # The agent is offering the wrong day (e.g. ASR noise): correct it.
                return Utterance(f"No, I meant {p.date_phrase}.", "correction")
            ev = self._event("time", slot=p.time_phrase)
            if ev is not None:
                return ev
            return self._say("time", time=p.time_phrase or "")
        self.repeats += 1
        if self.repeats > 2:
            return None
        return Utterance(PERSONAS["phrases"]["repeat"][0], "other")


# Broader cues so a real LLM agent's phrasing is understood (the scripted agent's
# exact phrasings are matched first, above).
_CLOSING = (
    "anything else",
    "else i can help",
    "help you with anything",
    "anything more",
    "let me know",
    "see you then",
    "have a great day",
    "you're all set",
    "all set!",
)
_HOURS_ANSWER = ("monday", "8 am", "8 a.m", "8:00", "5 pm", "5 p.m", "17:00")
_CONFIRM = (
    "should i",
    "want me to",
    "would you like me to",
    "do you want me",
    "lock that",
    "confirm",
    "book that",
    "book it",
    "cancel it",
    "cancel that",
    "reschedule it",
)
_DATE_Q = (
    "day in mind",
    "what date",
    "which date",
    "preferred day",
    "when would you",
    "what day",
    "day would",
    "date would",
    "particular day",
    "new date",
    "which day",
)
_TIME_Q = (
    "what time",
    "which time",
    "time works",
    "time would",
    "slot",
    "times available",
    "prefer",
)


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


class LLMCaller:
    """A real LLM plays the caller (text mode). Labels are unknown ('llm')."""

    def __init__(self, persona: Persona, llm: LLMClient) -> None:
        self.persona = persona
        self.llm = llm
        goal = persona.describe()
        self.messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a patient phoning a clinic. Speak in one short sentence per turn, "
                    f"style: {persona.style}. Your situation: {goal}. Use the date/time phrases "
                    "given. Confirm only if the details are right. When done, say goodbye."
                ),
            }
        ]

    async def opening(self) -> Utterance:
        return await self._next("(The receptionist answers the phone.)")

    async def respond(self, agent_text: str) -> Utterance | None:
        if "goodbye" in agent_text.lower():
            return None
        return await self._next(agent_text)

    async def _next(self, heard: str) -> Utterance:
        self.messages.append({"role": "user", "content": heard})
        resp = await self.llm.complete(self.messages, temperature=0.7)
        self.messages.append({"role": "assistant", "content": resp.text})
        return Utterance(re.sub(r"\s+", " ", resp.text).strip(), "llm")
