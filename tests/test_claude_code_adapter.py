import json

import pytest

from catallax import PriceTable
from catallax.adapters.claude_code import run_from_transcript, usage_by_model


def _line(msg_id, model, usage, type_="assistant"):
    return json.dumps({"type": type_, "message": {"id": msg_id, "model": model, "usage": usage}})


@pytest.fixture
def transcript(tmp_path):
    u1 = {"input_tokens": 2, "output_tokens": 100, "cache_read_input_tokens": 1000,
          "cache_creation_input_tokens": 500}
    u2 = {"input_tokens": 10, "output_tokens": 50, "cache_read_input_tokens": 0,
          "cache_creation_input_tokens": 0}
    lines = [
        _line("msg_1", "claude-opus-5-5", u1),
        _line("msg_1", "claude-opus-5-5", u1),          # same message, second content block
        _line("msg_2", "claude-haiku-5-5", u2),
        json.dumps({"type": "user", "message": {"role": "user", "content": "hi"}}),
        "not json at all",
    ]
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_dedupes_repeated_message_ids(transcript):
    per = usage_by_model(transcript)
    assert per["claude-opus-5-5"] == {"input": 2, "output": 100, "cache_read": 1000, "cache_write": 500}
    assert per["claude-haiku-5-5"]["output"] == 50


def test_run_from_mixed_models(transcript):
    pt = PriceTable()
    run = run_from_transcript(transcript, "skill@v1", pt, task="t")
    assert run.model == "multi"
    assert set(run.meta["models"]) == {"claude-opus-5-5", "claude-haiku-5-5"}
    expected = pt.cost("claude-opus-5-5", {"input": 2, "output": 100, "cache_read": 1000, "cache_write": 500}) \
        + pt.cost("claude-haiku-5-5", {"input": 10, "output": 50})
    assert run.cost_usd == pytest.approx(expected)
    assert run.usage["output"] == 150


def test_empty_transcript_raises(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        run_from_transcript(p, "v", PriceTable())
