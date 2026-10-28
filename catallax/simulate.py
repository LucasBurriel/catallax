"""A toy market to show the allocator does what it claims.

Each variant has a hidden true success probability and cost per run.
Every period the allocator hands out a fixed budget, each variant runs as
many times as its budget pays for, outcomes are drawn, and the ledger
grows. We compare against spreading the budget evenly.

This is a demo of the mechanism, not evidence about any real harness:
the "market" here is a coin flip the simulator controls, which is exactly
the thing the real system refuses to do.
"""

from __future__ import annotations

import random
import tempfile
from dataclasses import dataclass

from .allocate import Config, allocate
from .ledger import Ledger, Outcome, Run
from .stats import VariantStats, summarize


@dataclass
class TrueVariant:
    name: str
    p_success: float
    cost_usd: float


DEFAULT_WORLD = [
    TrueVariant("big-model@v1",   p_success=0.30, cost_usd=0.40),
    TrueVariant("small-model@v1", p_success=0.12, cost_usd=0.04),   # best yield
    TrueVariant("small-model@v2", p_success=0.08, cost_usd=0.04),
    TrueVariant("verbose@v1",     p_success=0.28, cost_usd=0.90),   # worst yield
]


def _run_period(ledger: Ledger, world: dict[str, TrueVariant], budgets: dict[str, float],
                rng: random.Random) -> tuple[float, int]:
    spent, successes = 0.0, 0
    for name, budget in budgets.items():
        tv = world[name]
        n = int(budget // tv.cost_usd)
        for _ in range(n):
            run = ledger.append_run(Run(variant=name, model="sim", cost_usd=tv.cost_usd))
            spent += tv.cost_usd
            if rng.random() < tv.p_success:
                ledger.append_outcome(Outcome(run_id=run.id, kind="accepted"))
                successes += 1
    return spent, successes


def simulate(periods: int = 12, budget_per_period: float = 10.0, seed: int = 7,
             world: list[TrueVariant] | None = None, verbose: bool = True) -> dict:
    world_list = world or DEFAULT_WORLD
    world_map = {v.name: v for v in world_list}
    cfg = Config(success_kinds=["accepted"], seed=seed, prior_cost_usd=0.2,
                 min_runs_to_retire=10)
    rng_a, rng_u = random.Random(seed), random.Random(seed)

    with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_u:
        led_a, led_u = Ledger(tmp_a), Ledger(tmp_u)
        tot_a = tot_u = 0
        spent_a = spent_u = 0.0
        even = {n: budget_per_period / len(world_map) for n in world_map}
        history = []
        for t in range(periods):
            stats = summarize(led_a, cfg.success_kinds)
            for n in world_map:
                stats.setdefault(n, VariantStats(variant=n))
            alloc = allocate(stats, budget_per_period, cfg)
            budgets = {n: v.budget_usd for n, v in alloc.variants.items()}
            s, k = _run_period(led_a, world_map, budgets, rng_a)
            spent_a += s
            tot_a += k
            s, k = _run_period(led_u, world_map, even, rng_u)
            spent_u += s
            tot_u += k
            history.append({n: round(v.share, 3) for n, v in alloc.variants.items()})
            if verbose:
                shares = "  ".join(
                    f"{n}={v.share:.2f}{'*' if v.status == 'retired' else ''}"
                    for n, v in alloc.variants.items()
                )
                print(f"period {t + 1:2d}  thompson={tot_a:4d} ok  even={tot_u:4d} ok   {shares}")
        result = {
            "periods": periods,
            "budget_per_period": budget_per_period,
            "thompson": {"successes": tot_a, "spent": round(spent_a, 2),
                         "per_usd": round(tot_a / spent_a, 3) if spent_a else 0.0},
            "even": {"successes": tot_u, "spent": round(spent_u, 2),
                     "per_usd": round(tot_u / spent_u, 3) if spent_u else 0.0},
            "final_shares": history[-1],
            "true_yield": {n: round(v.p_success / v.cost_usd, 2) for n, v in world_map.items()},
        }
        if verbose:
            print()
            print(f"thompson: {tot_a} successes for ${spent_a:.2f}  ({result['thompson']['per_usd']}/USD)")
            print(f"even:     {tot_u} successes for ${spent_u:.2f}  ({result['even']['per_usd']}/USD)")
            print(f"true yield per USD: {result['true_yield']}   (* = retired)")
        return result
