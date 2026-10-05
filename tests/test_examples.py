"""The README's offline demo must actually complete a call (it once used a patient name
that does not exist in the clinic data, so every turn asked for the name again)."""

from __future__ import annotations

from pathlib import Path

import pytest

from examples.scheduling.pipeline_handbuilt import run


@pytest.mark.allow_slow_callbacks  # a whole demo call, not a latency test
async def test_handbuilt_demo_books_an_appointment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # always the offline scripted model
    await run(None, tmp_path / "store", True, 1)
    out = capsys.readouterr().out
    assert "couldn't find a patient" not in out
    assert "You're all set" in out and "Goodbye" in out
