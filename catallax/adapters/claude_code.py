"""Read a Claude Code session transcript and turn it into one Run.

Claude Code writes one JSONL per session under
~/.claude/projects/<project>/<session>.jsonl. Every assistant content
block is its own line and repeats the parent message's `usage`, so we
dedupe by message id (keeping the last line seen) before summing.

Usage: catallax ingest-claude <transcript.jsonl> --variant my-skill@v2
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..ledger import Run
from ..pricing import PriceTable, normalize_usage


def usage_by_model(transcript: str | Path) -> dict[str, dict[str, int]]:
    seen: dict[str, tuple[str, dict[str, Any]]] = {}
    with Path(transcript).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") != "assistant":
                continue
            msg = d.get("message") or {}
            usage = msg.get("usage")
            model = msg.get("model")
            # Claude Code inserts bookkeeping messages with model "<synthetic>";
            # they are not billed.
            if not usage or not model or model.startswith("<"):
                continue
            key = msg.get("id") or d.get("uuid")
            seen[key] = (model, usage)

    totals: dict[str, dict[str, int]] = {}
    for model, usage in seen.values():
        u = normalize_usage(usage)
        t = totals.setdefault(model, {k: 0 for k in u})
        for k, v in u.items():
            t[k] += v
    return totals


def run_from_transcript(transcript: str | Path, variant: str, prices: PriceTable,
                        task: str | None = None) -> Run:
    per_model = usage_by_model(transcript)
    if not per_model:
        raise ValueError(f"no assistant usage found in {transcript}")
    cost = sum(prices.cost(m, u) for m, u in per_model.items())
    models = sorted(per_model)
    model = models[0] if len(models) == 1 else "multi"
    keys = next(iter(per_model.values())).keys()
    usage = {k: sum(u[k] for u in per_model.values()) for k in keys}
    return Run(
        variant=variant, model=model, cost_usd=cost, usage=usage, task=task,
        meta={"source": "claude-code", "transcript": Path(transcript).name,
              "models": {m: round(prices.cost(m, u), 6) for m, u in per_model.items()}},
    )
