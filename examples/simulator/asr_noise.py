"""Deterministic, seeded text-level ASR noise."""

from __future__ import annotations

import random
import re

HOMOPHONES = {"two": "to", "four": "for", "eight": "ate", "right": "write", "great": "grate"}
WEEKDAY_CONFUSION = {
    "Tuesday": "Thursday",
    "Thursday": "Tuesday",
    "tuesday": "thursday",
    "thursday": "tuesday",
}
FILLERS = ["um", "uh", "like"]


class AsrNoise:
    """Applies at most one perturbation per utterance with probability ``rate``."""

    def __init__(self, rate: float = 0.1) -> None:
        self.rate = rate

    def apply(self, text: str, rng: random.Random) -> tuple[str, str | None]:
        if self.rate <= 0 or rng.random() >= self.rate:
            return text, None
        ops = [
            self._insert_filler,
            self._drop_word,
            self._homophone,
            self._lowercase,
            self._digit_slip,
            self._weekday,
        ]
        rng.shuffle(ops)
        for op in ops:
            out = op(text, rng)
            if out is not None and out != text:
                return out, op.__name__.lstrip("_")
        return text, None

    @staticmethod
    def _insert_filler(text: str, rng: random.Random) -> str | None:
        words = text.split()
        if len(words) < 2:
            return None
        i = rng.randrange(1, len(words))
        return " ".join([*words[:i], rng.choice(FILLERS) + ",", *words[i:]])

    @staticmethod
    def _drop_word(text: str, rng: random.Random) -> str | None:
        words = text.split()
        if len(words) < 3:
            return None
        i = rng.randrange(len(words))
        return " ".join(words[:i] + words[i + 1 :])

    @staticmethod
    def _homophone(text: str, rng: random.Random) -> str | None:
        hits = [w for w in HOMOPHONES if re.search(rf"\b{w}\b", text, re.IGNORECASE)]
        if not hits:
            return None
        w = rng.choice(hits)
        return re.sub(rf"\b{w}\b", HOMOPHONES[w], text, count=1, flags=re.IGNORECASE)

    @staticmethod
    def _lowercase(text: str, rng: random.Random) -> str | None:
        return text.lower()

    @staticmethod
    def _digit_slip(text: str, rng: random.Random) -> str | None:
        digits = list(re.finditer(r"\b([1-9])\b(?!:)", text))
        if not digits:
            return None
        m = rng.choice(digits)
        new = str(int(m.group(1)) % 9 + 1)
        return text[: m.start()] + new + text[m.end() :]

    @staticmethod
    def _weekday(text: str, rng: random.Random) -> str | None:
        for w, v in WEEKDAY_CONFUSION.items():
            if re.search(rf"\b{w}\b", text):
                return re.sub(rf"\b{w}\b", v, text, count=1)
        return None
