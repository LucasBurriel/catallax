import pytest

from catallax import Ledger, Outcome, PriceTable, Run
from catallax.pricing import normalize_usage


def test_ledger_roundtrip(tmp_path):
    led = Ledger(tmp_path)
    run = led.append_run(Run(variant="a@v1", model="m", cost_usd=0.5, usage={"input": 10}))
    led.append_outcome(Outcome(run_id=run.id, kind="accepted", value=2.0))
    runs = list(led.runs())
    outs = list(led.outcomes())
    assert len(runs) == 1 and runs[0].id == run.id and runs[0].cost_usd == 0.5
    assert outs[0].run_id == run.id and outs[0].value == 2.0


def test_outcome_requires_known_run(tmp_path):
    led = Ledger(tmp_path)
    with pytest.raises(KeyError):
        led.append_outcome(Outcome(run_id="run_nope", kind="accepted"))


def test_ledger_is_append_only(tmp_path):
    led = Ledger(tmp_path)
    led.append_run(Run(variant="a", model="m", cost_usd=1))
    led.append_run(Run(variant="a", model="m", cost_usd=2))
    assert [r.cost_usd for r in led.runs()] == [1, 2]


def test_price_prefix_match_prefers_longest():
    pt = PriceTable({"claude-opus-5": {"input": 5, "output": 25, "cache_read": 0.5, "cache_write": 6.25},
                     "claude-opus-5-5": {"input": 4, "output": 20, "cache_read": 0.2, "cache_write": 5}})
    assert pt.lookup("claude-opus-5-5")["input"] == 4
    assert pt.lookup("claude-opus-5")["input"] == 5
    assert pt.lookup("claude-opus-5-5-20270101")["input"] == 4


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        PriceTable().cost("gpt-unknown", {"input": 1})


def test_cost_math():
    pt = PriceTable({"m": {"input": 1.0, "output": 10.0, "cache_read": 0.1, "cache_write": 1.25}})
    usage = {"input_tokens": 1_000_000, "output_tokens": 100_000,
             "cache_read_input_tokens": 1_000_000, "cache_creation_input_tokens": 0}
    assert pt.cost("m", usage) == pytest.approx(1.0 + 1.0 + 0.1)


def test_normalize_usage_aliases():
    u = normalize_usage({"prompt_tokens": 3, "completion_tokens": 4})
    assert u == {"input": 3, "output": 4, "cache_read": 0, "cache_write": 0}
