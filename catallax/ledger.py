"""Append-only ledger of runs and outcomes (two JSONL files).

A *run* is one execution of a harness variant: what it cost.
An *outcome* is a signal that arrived later and points back at a run:
what the world (or the user) said about it.

Nothing is ever rewritten. If a record is wrong, append a correction.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class Run:
    """One execution of a harness variant."""

    variant: str                      # e.g. "cover-letter@v3"
    model: str                        # model id, or "multi" for mixed sessions
    cost_usd: float                   # real money, computed from a PriceTable
    usage: dict[str, int] = field(default_factory=dict)
    task: str | None = None           # free label for the thing being done
    meta: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: _new_id("run"))
    ts: str = field(default_factory=_now)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Run":
        return cls(**d)


@dataclass
class Outcome:
    """A signal about a run that arrived from outside the agent."""

    run_id: str
    kind: str                         # e.g. "reply", "accepted", "rejected"
    value: float | None = None        # optional USD-equivalent revenue
    meta: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: _new_id("out"))
    ts: str = field(default_factory=_now)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Outcome":
        return cls(**d)


class Ledger:
    """Two append-only JSONL files in a directory."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.runs_path = self.root / "runs.jsonl"
        self.outcomes_path = self.root / "outcomes.jsonl"

    # -- writing ---------------------------------------------------------
    def append_run(self, run: Run) -> Run:
        self._append(self.runs_path, run.to_json())
        return run

    def append_outcome(self, outcome: Outcome) -> Outcome:
        if not self.has_run(outcome.run_id):
            raise KeyError(f"unknown run_id {outcome.run_id!r}")
        self._append(self.outcomes_path, outcome.to_json())
        return outcome

    @staticmethod
    def _append(path: Path, line: str) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    # -- reading ---------------------------------------------------------
    def runs(self) -> Iterator[Run]:
        for d in self._read(self.runs_path):
            yield Run.from_dict(d)

    def outcomes(self) -> Iterator[Outcome]:
        for d in self._read(self.outcomes_path):
            yield Outcome.from_dict(d)

    def has_run(self, run_id: str) -> bool:
        return any(r.id == run_id for r in self.runs())

    def outcomes_by_run(self) -> dict[str, list[Outcome]]:
        by: dict[str, list[Outcome]] = {}
        for o in self.outcomes():
            by.setdefault(o.run_id, []).append(o)
        return by

    @staticmethod
    def _read(path: Path) -> Iterator[dict[str, Any]]:
        if not path.exists():
            return
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
