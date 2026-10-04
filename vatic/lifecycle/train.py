"""Train per-step membership classifiers (``vatic train``). Needs torch + transformers.

One small cross-encoder per flow, conditioned on the step prompt, so it serves
every eligible step of that flow; steps with fewer than ``min_on`` on-path or
``min_off`` off-path examples get no classifier (rules only). Off-path data is
the compiler's deviation + seeded examples plus other steps' on-path answers
(a time is not an answer to "what day?"). Thresholds are set later by
calibration on shadow outcomes, never here.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from vatic.core.classifier import step_prompt
from vatic.ir.schema import AskStep, FlowGraph, save_flow

BASE_MODEL = "cross-encoder/ms-marco-TinyBERT-L2-v2"


@dataclass
class TrainReport:
    flow_id: str
    steps: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    examples: int = 0
    holdout_accuracy: float | None = None


def _load_examples(path: Path) -> list[tuple[str, int]]:
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out.append((row["text"], 1 if row["label"] == "on_path" else 0))
    return out


def _holdout(prompt: str, text: str) -> bool:
    return int(hashlib.sha256(f"{prompt}|{text}".encode()).hexdigest(), 16) % 5 == 0


def train_flow(
    flow: FlowGraph,
    flows_dir: Path,
    *,
    base_model: str = BASE_MODEL,
    epochs: int = 3,
    min_on: int = 50,
    min_off: int = 20,
    max_length: int = 64,
    seed: int = 0,
) -> TrainReport:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    report = TrainReport(flow.flow_id)
    ex_dir = flows_dir / "examples" / flow.flow_id
    per_step: dict[str, list[tuple[str, int]]] = {}
    for step in flow.steps:
        if not isinstance(step, AskStep):
            continue
        path = ex_dir / f"{step.id}.jsonl"
        data = _load_examples(path) if path.exists() else []
        on = sum(1 for _, y in data if y)
        off = len(data) - on
        if on < min_on or off < min_off:
            report.skipped[step.id] = f"{on} on-path / {off} off-path examples"
            continue
        per_step[step.id] = data
    if not per_step:
        return report

    prompts = {sid: step_prompt(flow.step(sid)) for sid in per_step}  # type: ignore[arg-type]
    rows: list[tuple[str, str, int]] = []
    for sid, data in sorted(per_step.items()):
        rows += [(prompts[sid], text, y) for text, y in data]
        for other, odata in sorted(per_step.items()):  # cross-step negatives
            if other != sid:
                rows += [(prompts[sid], text, 0) for text, y in odata[:100] if y]
    rows = sorted(set(rows))
    train = [r for r in rows if not _holdout(r[0], r[1])]
    held = [r for r in rows if _holdout(r[0], r[1])]
    rng = random.Random(seed)
    torch.manual_seed(seed)

    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=1)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        rng.shuffle(train)
        for i in range(0, len(train), 16):
            batch = train[i : i + 16]
            enc = tok(
                [p for p, _, _ in batch],
                [t for _, t, _ in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            logits = model(**enc).logits.squeeze(-1)
            loss = loss_fn(logits, torch.tensor([float(y) for _, _, y in batch]))
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    if held:
        with torch.no_grad():
            enc = tok(
                [p for p, _, _ in held],
                [t for _, t, _ in held],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            pred = (model(**enc).logits.squeeze(-1) > 0).long().tolist()
        report.holdout_accuracy = round(
            sum(int(p == y) for p, (_, _, y) in zip(pred, held, strict=True)) / len(held), 4
        )

    out = flows_dir / "classifiers" / flow.flow_id
    out.mkdir(parents=True, exist_ok=True)
    sample = tok("What day?", "Tuesday", return_tensors="pt")
    names = ["input_ids", "attention_mask", "token_type_ids"]
    axes = {n: {0: "batch", 1: "seq"} for n in names} | {"logits": {0: "batch"}}
    torch.onnx.export(
        model,
        tuple(sample[n] for n in names),
        str(out / "model.onnx"),
        input_names=names,
        output_names=["logits"],
        dynamic_axes=axes,
        opset_version=17,
        dynamo=False,
    )
    tok.backend_tokenizer.save(str(out / "tokenizer.json"))
    meta = {
        "base_model": base_model,
        "max_length": max_length,
        "epochs": epochs,
        "seed": seed,
        "steps": {sid: prompts[sid] for sid in sorted(per_step)},
        "examples": len(rows),
        "holdout_accuracy": report.holdout_accuracy,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

    for step in flow.steps:
        if isinstance(step, AskStep) and step.id in per_step:
            step.membership.classifier = f"classifiers/{flow.flow_id}"
            step.membership.accept_threshold = None  # set by calibration only
            step.membership.reject_threshold = None
    flow.content_hash = flow.compute_hash()
    save_flow(flow, flows_dir / f"{flow.flow_id}.yaml")
    report.steps = sorted(per_step)
    report.examples = len(rows)
    return report
