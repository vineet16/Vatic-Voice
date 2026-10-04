"""Offline stand-in for the clinic agent's LLM.

A deterministic policy that reads the OpenAI-style message history (user turns,
assistant tool calls, tool results, the ``[vatic-flow]`` context note) and
returns the next tool call or reply, exactly like a tool-calling model would.
It exists so the simulator, tests and benchmark run without network access or
API keys; with ``OPENAI_API_KEY`` set, the real model can be used instead.

It is intentionally *not* built on Vatic's membership checks: it handles
corrections, digressions and multi-intent turns more leniently than the
compiled path, like a capable LLM would.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import multiprocessing
import re
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from vatic.core.extract import (
    extract_date_phrase,
    extract_person_name,
    extract_time_phrase,
    tokenize,
)
from vatic.core.formatters import human_date, human_time, human_time_list
from vatic.core.transforms import TransformContext, normalize_date, normalize_time
from vatic.llm.client import LLMResponse, ToolCallRequest

HOURS = "We're open Monday to Friday, 8 AM to 5 PM."
DIGRESSIONS: list[tuple[str, str]] = [
    (r"\bhours\b|\bopen\b|\bclose\b", HOURS),
    (r"\binsurance\b", "Yes, we accept most major insurance plans."),
    (r"\baddress\b|\bwhere are you\b|\blocated\b", "We're at 120 River Road."),
    (r"\bbring\b", "Please bring your insurance card and a photo ID."),
    (r"\bpark", "There's free parking behind the building."),
]
YES = {"yes", "yeah", "yep", "yup", "sure", "correct", "right", "perfect", "ok", "okay"}
NO = {"no", "nope", "not", "wrong", "wait", "actually"}
BYE = re.compile(r"\b(bye|that's all|that's it|nothing else|no thanks|no, thank|all set)\b")
TARGET_TOOL = {
    "book": "book_appointment",
    "cancel": "cancel_appointment",
    "reschedule": "reschedule_appointment",
}


def classify_ask(text: str) -> str | None:
    t = text.lower()
    if "goodbye" in t:
        return "bye"
    if "shall i" in t or "go ahead?" in t:
        return "confirm"
    if "anything else" in t:
        return "anything_else"
    if "full name" in t:
        return "name"
    if "which time" in t or "which would you like" in t:
        return "time"
    if any(k in t for k in ("what day", "another day", "which day", "move it to", "day works")):
        return "date"
    if "how can i help" in t:
        return "intent"
    return None


def detect_intent(low: str) -> str | None:
    if "cancel" in low:
        return "cancel"
    if "reschedul" in low or "move my" in low or "change my appointment" in low:
        return "reschedule"
    if any(
        k in low for k in ("book", "schedule", "make an appointment", "appointment", "see a doctor")
    ):
        return "book"
    if re.search(r"\bhours\b|\bopen\b", low):
        return "hours"
    return None


@dataclass
class Conv:
    today: dt.date = dt.date(2026, 10, 5)
    first_user: str = ""
    turn_no: int = 0
    intent: str | None = None
    name: str | None = None
    patient: dict[str, Any] | None = None
    failed_name: str | None = None
    date_raw: str | None = None
    date: str | None = None
    avail: dict[str, Any] | None = None
    time_raw: str | None = None
    time: str | None = None
    confirmed: bool = False
    declined: bool = False
    awaiting: str | None = None
    task_done: bool = False
    result: dict[str, Any] | None = None
    closing: bool = False
    wants_suggestion: bool = False
    prefix: list[str] = field(default_factory=list)
    turn_calls: list[str] = field(default_factory=list)
    entered: bool = False
    rejected_flows: set[str] = field(default_factory=set)
    resumed: bool = False
    resume_failed: bool = False
    flow_ctx: dict[str, Any] | None = None
    pending_flow_ctx: dict[str, Any] | None = None

    @property
    def ctx(self) -> TransformContext:
        return TransformContext(self.today)

    def reset_task(self) -> None:
        self.date_raw = self.date = self.time_raw = self.time = None
        self.avail = None
        self.confirmed = self.declined = self.task_done = self.wants_suggestion = False

    # -- user turns -------------------------------------------------------------------

    def on_user(self, text: str) -> None:
        self.turn_no += 1
        if not self.first_user:
            self.first_user = text
        self.prefix, self.turn_calls, self.rejected_flows = [], [], set()
        self.entered = self.resumed = self.resume_failed = False
        self.flow_ctx, self.pending_flow_ctx = self.pending_flow_ctx, None
        words = [t.text for t in tokenize(text) if t.text not in ("um", "uh", "er", "hmm")]
        low = " ".join(words)

        if (self.task_done or self.intent == "hours" or self.awaiting == "anything_else") and (
            BYE.search(low) or words[:1] == ["no"]
        ):
            self.closing = True
            return
        new_intent = detect_intent(low)
        if self.intent is None:
            self.intent = new_intent
        elif self.task_done and new_intent not in (None, "hours"):
            self.intent = new_intent
            self.reset_task()
        if self.intent not in (None, "hours"):
            for pattern, answer in DIGRESSIONS:
                if re.search(pattern, low) and answer not in self.prefix:
                    self.prefix.append(answer)
        if self.patient is None:
            names = extract_person_name(text)
            if names:
                self.name = names[-1].text
        changed = False
        if self.intent in ("book", "reschedule"):
            dates = extract_date_phrase(text)
            if dates:
                iso = normalize_date(dates[-1].text, self.ctx)
                if iso is not None and iso != self.date:
                    self.date_raw, self.date = dates[-1].text, iso
                    self.avail, self.time, self.time_raw = None, None, None
                    self.confirmed = False
                    changed = True
            times = extract_time_phrase(text)
            if times:
                hhmm = normalize_time(times[-1].text, self.ctx)
                if hhmm is not None and hhmm != self.time:
                    self.time_raw, self.time = times[-1].text, hhmm
                    self.confirmed = False
                    changed = True
            if (
                self.awaiting == "date"
                and not dates
                and re.search(r"not sure|what do you have|whatever|available", low)
            ):
                self.wants_suggestion = True
        if self.awaiting == "confirm":
            if any(w in NO for w in words):
                self.confirmed = False
                if self.intent == "cancel":
                    self.declined = True
                elif not changed:
                    self.date = self.date_raw = self.time = self.time_raw = None
                    self.avail = None
            elif any(w in YES for w in words) or "go ahead" in low or "sounds good" in low:
                self.confirmed = True

    # -- tool results ----------------------------------------------------------------------

    def on_result(self, name: str, args: dict[str, Any], out: dict[str, Any]) -> None:
        self.turn_calls.append(name)
        if name == "enter_flow":
            for ex in out.get("executed") or []:
                if ex.get("error") is None and ex.get("output") is not None:
                    self.on_result(ex["tool"], ex["args"], ex["output"])
            if out.get("status") == "entered":
                self.entered = True
            else:
                self.rejected_flows.add(str(args.get("flow_id")))
            return
        if name == "resume_flow":
            if out.get("status") == "resumed":
                self.resumed = True
            else:
                self.resume_failed = True
            return
        if "error" in out:
            if name in TARGET_TOOL.values():
                self.avail, self.time, self.time_raw, self.confirmed = None, None, None, False
            return
        if name == "lookup_patient":
            if out.get("found"):
                self.patient = out["patient"]
            else:
                self.failed_name = args.get("name")
        elif name == "check_availability":
            self.avail = out
        elif name in TARGET_TOOL.values():
            self.task_done = True
            self.result = out


_POOL: ProcessPoolExecutor | None = None


def _process_pool() -> ProcessPoolExecutor:
    """Create (and spawn) the shared worker pool. Blocking: call off the event loop."""
    global _POOL
    if _POOL is None:
        pool = ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn"))
        list(pool.map(_noop, range(2)))
        _POOL = pool
    return _POOL


def _noop(_: int) -> None:
    return None


def respond(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
    """Pure function: next model output for a message history."""
    return Policy(replay(messages), tools).decide()


class ScriptedClinicLLM:
    """Implements ``LLMClient`` with the deterministic clinic policy.

    The policy runs in a worker process, like a remote model: it never holds the
    GIL of the process running the voice event loop.
    """

    def __init__(
        self,
        latency_s: float = 0.0,
        jitter_s: float = 0.0,
        executor: Executor | None = None,
    ) -> None:
        self.latency_s = latency_s
        self.jitter_s = jitter_s
        self.calls = 0
        self._executor = executor

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        self.calls += 1
        loop = asyncio.get_running_loop()
        pool = self._executor or _POOL or await asyncio.to_thread(_process_pool)
        resp = await loop.run_in_executor(pool, respond, messages, tools or [])
        if self.latency_s or self.jitter_s:
            h = int(hashlib.sha256(json.dumps(messages[-1], default=str).encode()).hexdigest(), 16)
            await asyncio.sleep(self.latency_s + self.jitter_s * ((h % 1000) / 1000))
        return resp


def replay(messages: list[dict[str, Any]]) -> Conv:
    """Rebuild conversation state from an OpenAI-style message history."""
    conv = Conv()
    pending: dict[str, tuple[str, dict[str, Any]]] = {}
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if role == "system" and isinstance(content, str):
            if content.startswith("[vatic-flow] "):
                conv.pending_flow_ctx = json.loads(content[len("[vatic-flow] ") :])
            elif mt := re.search(r"Today's date is (\d{4}-\d{2}-\d{2})", content):
                conv.today = dt.date.fromisoformat(mt.group(1))
        elif role == "user":
            conv.on_user(str(content or ""))
        elif role == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = tc["function"]
                pending[tc["id"]] = (fn["name"], json.loads(fn.get("arguments") or "{}"))
            if content:
                conv.awaiting = classify_ask(str(content))
        elif role == "tool":
            name, args = pending.get(str(m.get("tool_call_id")), ("", {}))
            conv.on_result(name, args, json.loads(str(content or "{}")))
    return conv


class Policy:
    def __init__(self, conv: Conv, tools: list[dict[str, Any]]) -> None:
        self.c = conv
        self.tools = {t["function"]["name"]: t for t in tools}

    # -- helpers -----------------------------------------------------------------

    def _pick(self, key: str, main: str, alt: str, alt_pct: int = 8) -> str:
        h = hashlib.sha256(f"{self.c.first_user}|{key}|{self.c.turn_no}".encode()).hexdigest()
        return alt if int(h, 16) % 100 < alt_pct else main

    def say(self, text: str) -> LLMResponse:
        return LLMResponse(text=" ".join([*self.c.prefix, text]).strip())

    def call(self, name: str, args: dict[str, Any]) -> LLMResponse:
        cid = f"call_{self.c.turn_no}_{len(self.c.turn_calls)}"
        return LLMResponse(tool_calls=[ToolCallRequest(id=cid, name=name, arguments=args)])

    def _known_slots(self) -> dict[str, str]:
        out = {}
        if self.c.name:
            out["name"] = self.c.name
        if self.c.date_raw:
            out["date"] = self.c.date_raw
        if self.c.time_raw:
            out["time"] = self.c.time_raw
        return out

    def _maybe_enter_flow(self) -> LLMResponse | None:
        spec = self.tools.get("enter_flow")
        target = TARGET_TOOL.get(self.c.intent or "")
        if spec is None or target is None:
            return None
        known = self._known_slots()
        for line in spec["function"]["description"].splitlines():
            m = re.match(r"- (\S+): (.*) Entry slots: (.*)\.$", line)
            if not m or target not in m.group(2) or m.group(1) in self.c.rejected_flows:
                continue
            slots = [s.split(" ", 1)[0] for s in re.split(r", (?=\w+ \()", m.group(3))]
            if set(slots) == set(known):
                return self.call("enter_flow", {"flow_id": m.group(1), "slots": known})
        return None

    def ask(self, kind: str, text: str) -> LLMResponse:
        """Ask something; first hand control back to a paused flow if it asks the same."""
        fc = self.c.flow_ctx
        if fc and "resume_flow" in self.tools and not (self.c.resumed or self.c.resume_failed):
            for st in fc.get("resumable_steps", []):
                if (kind == "confirm" and st["kind"] == "confirm") or (
                    st["kind"] == "ask" and st.get("expects") == [kind]
                ):
                    return self.call(
                        "resume_flow",
                        {
                            "flow_id": fc["flow_id"],
                            "step_id": st["step_id"],
                            "slots": self._known_slots(),
                        },
                    )
        return self.say(text)

    # -- the policy ---------------------------------------------------------------

    def decide(self) -> LLMResponse:
        c = self.c
        if c.entered:
            return LLMResponse(text="")
        if c.closing:
            return LLMResponse(text="Thank you for calling. Goodbye!")
        if c.intent is None:
            return self.say("How can I help you today?")
        if c.intent == "hours":
            return self.say(f"{HOURS} Is there anything else I can help you with?")
        target = TARGET_TOOL.get(c.intent)
        if c.task_done and target in c.turn_calls and c.result is not None:
            return self._done(target, c.result)
        if c.task_done or c.declined:
            if c.declined:
                return self.say(
                    "Okay, I won't cancel it. Is there anything else I can help you with?"
                )
            return self.say("Is there anything else I can help you with?")
        if c.patient is None:
            if c.name and c.name != c.failed_name and "lookup_patient" not in c.turn_calls:
                return self._maybe_enter_flow() or self.call("lookup_patient", {"name": c.name})
            if c.name and c.name == c.failed_name:
                return self.say(
                    f"I couldn't find a patient named {c.name}. "
                    "Could you please repeat your full name?"
                )
            return self.say("Sure, I can help with that. May I have your full name, please?")
        first = c.patient["first_name"]
        greet = f"Thanks, {first}. " if "lookup_patient" in c.turn_calls else ""
        if c.intent == "book":
            return self._book(first, greet)
        appt = c.patient.get("next_appointment")
        if appt is None:
            c.task_done = True
            return self.say(
                f"{greet}I don't see any upcoming appointments for you. "
                "Is there anything else I can help you with?"
            )
        if c.intent == "cancel":
            return self._cancel(appt, greet)
        return self._reschedule(appt, greet)

    def _done(self, target: str, result: dict[str, Any]) -> LLMResponse:
        c = self.c
        assert c.patient is not None
        if target == "cancel_appointment":
            appt = c.patient["next_appointment"]
            return self.say(
                f"Your appointment on {human_date(appt['date'])} at {human_time(appt['time'])} "
                "has been cancelled. Is there anything else I can help you with?"
            )
        hd, ht = human_date(result["date"]), human_time(result["time"])
        if target == "book_appointment":
            return self.say(
                f"You're all set, {c.patient['first_name']}. Your appointment is on {hd} at {ht}. "
                "Is there anything else I can help you with?"
            )
        return self.say(
            f"Done. Your appointment is now on {hd} at {ht}. "
            "Is there anything else I can help you with?"
        )

    def _offer_and_confirm(self, greet: str, confirm_text: str) -> LLMResponse | None:
        c = self.c
        if c.avail is None or c.avail.get("date") != c.date:
            if "check_availability" in c.turn_calls:
                return self.ask(
                    "date", "Sorry, I couldn't check that day. What day would you like?"
                )
            return self.call("check_availability", {"date": c.date})
        hd = human_date(c.date) or c.date
        if not c.avail.get("available"):
            c.date = c.date_raw = None
            return self.ask(
                "date",
                f"{greet}Sorry, we have no openings on {hd}. "
                "Is there another day that works for you?",
            )
        times = human_time_list(c.avail["times"])
        if c.time is None:
            return self.ask(
                "time",
                greet
                + self._pick(
                    "offer",
                    f"On {hd} I have {times}. Which time works best for you?",
                    f"I have {times} available on {hd}. Which time works best for you?",
                ),
            )
        if c.time not in c.avail["times"]:
            ht = human_time(c.time) or c.time
            c.time = c.time_raw = None
            return self.ask(
                "time",
                f"Sorry, {ht} isn't available on {hd}. I have {times}. Which would you like?",
            )
        if not c.confirmed:
            return self.ask("confirm", confirm_text.format(hd=hd, ht=human_time(c.time)))
        return None

    def _book(self, first: str, greet: str) -> LLMResponse:
        c = self.c
        if c.date is None:
            if c.wants_suggestion:
                return self.ask(
                    "date", "We have openings most weekdays. What day works best for you?"
                )
            return self.ask(
                "date",
                self._pick(
                    "ask_date",
                    f"Thanks, {first}. What day would you like to come in?",
                    f"Thank you, {first}. Which day works best for you?",
                ),
            )
        confirm = self._pick(
            "confirm",
            "Just to confirm: {hd} at {ht}. Shall I book it?",
            "So that's {hd} at {ht}. Shall I book it?",
        )
        pending = self._offer_and_confirm(greet, confirm)
        if pending is not None:
            return pending
        assert c.patient is not None and c.date is not None and c.time is not None
        if "book_appointment" not in c.turn_calls:
            return self.call(
                "book_appointment",
                {"patient_id": c.patient["id"], "date": c.date, "time": c.time},
            )
        return self.say("Sorry, something went wrong. Could you tell me the day again?")

    def _cancel(self, appt: dict[str, Any], greet: str) -> LLMResponse:
        c = self.c
        hd, ht = human_date(appt["date"]), human_time(appt["time"])
        if not c.confirmed:
            return self.ask(
                "confirm",
                f"{greet}I see your appointment on {hd} at {ht} with {appt['provider']}. "
                "Shall I cancel it?",
            )
        if "cancel_appointment" not in c.turn_calls:
            return self.call("cancel_appointment", {"appointment_id": appt["id"]})
        return self.say("Sorry, I couldn't cancel that. Is there anything else I can help with?")

    def _reschedule(self, appt: dict[str, Any], greet: str) -> LLMResponse:
        c = self.c
        if c.date is None:
            return self.ask(
                "date",
                f"{greet}Your current appointment is on {human_date(appt['date'])} at "
                f"{human_time(appt['time'])}. What day would you like to move it to?",
            )
        pending = self._offer_and_confirm(
            "", "Just to confirm: move your appointment to {hd} at {ht}. Shall I go ahead?"
        )
        if pending is not None:
            return pending
        assert c.date is not None and c.time is not None
        if "reschedule_appointment" not in c.turn_calls:
            return self.call(
                "reschedule_appointment",
                {"appointment_id": appt["id"], "date": c.date, "time": c.time},
            )
        return self.say("Sorry, that didn't work. Could you tell me the day again?")
