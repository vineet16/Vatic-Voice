"""Per-step membership classifier: a small cross-encoder exported to ONNX, CPU only.

Input is (step prompt, caller utterance); output is the probability that the
utterance belongs to the step. One ``InferenceSession`` per model, created at
startup with capped intra-op threads so inference cannot starve the audio
pipeline; ONNX Runtime releases the GIL while it runs. Scoring is synchronous
and is called by the runtime on its bounded executor with a timeout.

Model directory layout (written by ``vatic train``)::

    model.onnx  tokenizer.json  meta.json
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from vatic.ir.schema import AskStep

_PLACEHOLDER = re.compile(r"\{([^{}|]+)(?:\|[^{}]*)?\}")


def step_prompt(step: AskStep) -> str:
    """The step's question with placeholders reduced to their field names."""
    template = step.say.template or (step.say.llm.example if step.say.llm else "") or ""

    def name(m: re.Match[str]) -> str:
        return "[" + m.group(1).split(".")[-1] + "]"

    return _PLACEHOLDER.sub(name, template).replace("{{", "{").replace("}}", "}")


class ClassifierUnavailable(RuntimeError):
    pass


class StepClassifier:
    def __init__(self, model_dir: str | Path, *, threads: int = 1) -> None:
        try:
            import numpy as np
            import onnxruntime as ort  # type: ignore[import-untyped]
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - depends on installed extras
            raise ClassifierUnavailable("pip install vatic[classifier]") from exc
        model_dir = Path(model_dir)
        self.meta: dict[str, Any] = json.loads((model_dir / "meta.json").read_text())
        self._np = np
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=int(self.meta.get("max_length", 64)))
        self.tokenizer.no_padding()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self._inputs = {i.name for i in self.session.get_inputs()}

    def score(self, prompt: str, utterance: str) -> float:
        enc = self.tokenizer.encode(prompt, utterance)
        np = self._np
        feeds = {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
            "token_type_ids": np.array([enc.type_ids], dtype=np.int64),
        }
        logits = self.session.run(None, {k: v for k, v in feeds.items() if k in self._inputs})[0]
        return 1.0 / (1.0 + math.exp(-float(logits[0][0])))

    def warmup(self) -> None:
        self.score("What day would you like?", "next tuesday")
