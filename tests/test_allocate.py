import pytest

from catallax import Config, Ledger, Outcome, Run, VariantStats, allocate, summarize
from catallax.simulate import simulate


def _stats(name, n, ok, cost_each):
    s = VariantStats(variant=name)
    s.n_runs, s.successes, s.cost_total = n, ok, n * cost_each
    return s


def test_budget_is_conserved():
    stats = {"a": _stats("a", 20, 5, 0.1), "b": _stats("b", 20, 2, 0.1), "c": VariantStats("c")}
    alloc = allocate(stats, 37.5, Config(seed=1))
    assert sum(v.budget_usd for v in alloc.variants.values()) == pytest.approx(37.5)
    assert sum(alloc.routing().values()) == pytest.approx(1.0)


def test_better_yield_gets_more_budget():
    stats = {"cheap-good": _stats("cheap-good", 40, 10, 0.05),
             "dear-same": _stats("dear-same", 40, 10, 0.50)}
    alloc = allocate(stats, 10, Config(seed=1))
    assert alloc.variants["cheap-good"].share > alloc.variants["dear-same"].share


def test_unseen_variant_keeps_exploration_floor():
    stats = {"vet": _stats("vet", 100, 60, 0.05), "new": VariantStats("new")}
    cfg = Config(seed=1, explore_min=0.1, prior_cost_usd=0.05)
    alloc = allocate(stats, 10, cfg)
    assert alloc.variants["new"].share >= 0.1 - 1e-9
    assert alloc.variants["new"].status == "live"


def test_clear_loser_is_retired_after_enough_runs():
    stats = {"good": _stats("good", 50, 30, 0.05), "bad": _stats("bad", 50, 1, 0.50)}
    alloc = allocate(stats, 10, Config(seed=1, min_runs_to_retire=8))
    assert alloc.variants["bad"].status == "retired"
    assert alloc.variants["bad"].budget_usd == 0
    assert alloc.variants["good"].budget_usd == pytest.approx(10)


def test_loser_is_not_retired_before_min_runs():
    stats = {"good": _stats("good", 50, 30, 0.05), "bad": _stats("bad", 3, 0, 0.50)}
    alloc = allocate(stats, 10, Config(seed=1, min_runs_to_retire=8))
    assert alloc.variants["bad"].status == "live"


def test_never_retire_everything():
    stats = {"only": _stats("only", 50, 0, 0.5)}
    alloc = allocate(stats, 5, Config(seed=1))
    assert alloc.variants["only"].status == "live"
    assert alloc.variants["only"].budget_usd == pytest.approx(5)


def test_summarize_counts_success_once_per_run(tmp_path):
    led = Ledger(tmp_path)
    r = led.append_run(Run(variant="v", model="m", cost_usd=0.2))
    led.append_outcome(Outcome(run_id=r.id, kind="reply"))
    led.append_outcome(Outcome(run_id=r.id, kind="reply"))
    led.append_outcome(Outcome(run_id=r.id, kind="rejected"))
    led.append_run(Run(variant="v", model="m", cost_usd=0.2))
    s = summarize(led, ["reply"])["v"]
    assert s.n_runs == 2 and s.successes == 1
    assert s.outcome_counts == {"reply": 2, "rejected": 1}
    assert s.p_mean == pytest.approx(2 / 4)   # Beta(1+1, 1+1)


def test_simulation_beats_even_split_and_conserves_budget():
    res = simulate(periods=15, budget_per_period=10.0, seed=3, verbose=False)
    assert res["thompson"]["per_usd"] > res["even"]["per_usd"]
    # spend can only be below budget by the change that no run was cheap enough to use
    assert res["thompson"]["spent"] <= 15 * 10.0 + 1e-6
