"""Turn token usage into dollars using a price table you control.

Prices are per million tokens. Lookup is by longest matching prefix of
the model id, so "claude-opus-5-5" matches an entry for "claude-opus-5-5"
before one for "claude-opus-5". Unknown models raise: a cost you cannot
compute is not a cost of zero.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

# Anthropic first-party list prices, USD per 1M tokens, as published
# 2026-10. cache_write is 1.25x input and cache_read 0.1x input where the
# vendor does not publish a figure. VERIFY before relying on these for
# billing; prices drift.
DEFAULT_PRICES: dict[str, dict[str, float]] = {
    "claude-fable-5-1": {"input": 10.0, "output": 50.0, "cache_read": 0.25, "cache_write": 12.5},
    "claude-fable-5":   {"input": 10.0, "output": 50.0, "cache_read": 1.0,  "cache_write": 12.5},
    "claude-opus-5-5":  {"input": 4.0,  "output": 20.0, "cache_read": 0.20, "cache_write": 5.0},
    "claude-opus-5":    {"input": 5.0,  "output": 25.0, "cache_read": 0.50, "cache_write": 6.25},
    "claude-opus-4-8":  {"input": 5.0,  "output": 25.0, "cache_read": 0.50, "cache_write": 6.25},
    "claude-opus-4-7":  {"input": 5.0,  "output": 25.0, "cache_read": 0.50, "cache_write": 6.25},
    "claude-opus-4-6":  {"input": 5.0,  "output": 25.0, "cache_read": 0.50, "cache_write": 6.25},
    "claude-sonnet-5-5": {"input": 2.0, "output": 10.0, "cache_read": 0.20, "cache_write": 2.5},
    "claude-sonnet-5":  {"input": 2.0,  "output": 10.0, "cache_read": 0.20, "cache_write": 2.5},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cache_read": 0.30, "cache_write": 3.75},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.01, "cache_write": 0.125},
    "claude-haiku-4-5": {"input": 1.0,  "output": 5.0,  "cache_read": 0.10, "cache_write": 1.25},
}

USAGE_KEYS = ("input", "output", "cache_read", "cache_write")

# Field names different SDKs use for the same thing.
_ALIASES = {
    # snake_case: Anthropic/OpenAI API bodies; camelCase: Claude Code's `--output-format json` modelUsage
    "input": ("input", "input_tokens", "prompt_tokens", "inputTokens"),
    "output": ("output", "output_tokens", "completion_tokens", "outputTokens"),
    "cache_read": ("cache_read", "cache_read_input_tokens", "cached_tokens", "cacheReadInputTokens"),
    "cache_write": ("cache_write", "cache_creation_input_tokens", "cacheCreationInputTokens"),
}


def normalize_usage(raw: Mapping[str, Any]) -> dict[str, int]:
    """Map vendor usage dicts onto the four keys we price."""
    out: dict[str, int] = {}
    for key, names in _ALIASES.items():
        for name in names:
            if name in raw and raw[name] is not None:
                out[key] = int(raw[name])
                break
        else:
            out[key] = 0
    return out


class PriceTable:
    def __init__(self, prices: Mapping[str, Mapping[str, float]] | None = None):
        self.prices: dict[str, dict[str, float]] = {
            k: dict(v) for k, v in (prices or DEFAULT_PRICES).items()
        }

    @classmethod
    def load(cls, path: str | Path) -> "PriceTable":
        with Path(path).open(encoding="utf-8") as fh:
            return cls(json.load(fh))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.prices, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def lookup(self, model: str) -> dict[str, float]:
        best = None
        for prefix in self.prices:
            if model.startswith(prefix) and (best is None or len(prefix) > len(best)):
                best = prefix
        if best is None:
            raise KeyError(f"no price for model {model!r}; add it to the price table")
        return self.prices[best]

    def cost(self, model: str, usage: Mapping[str, Any]) -> float:
        p = self.lookup(model)
        u = normalize_usage(usage)
        return sum(u[k] * p[k] for k in USAGE_KEYS) / 1_000_000
