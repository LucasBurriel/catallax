"""Budget-conserving allocation across harness variants.

Thompson sampling on yield (successes per dollar): each draw samples a
success rate from every live variant's posterior, divides by its mean
cost, and the highest yield wins the draw. A variant's share of the
budget is its share of wins, floored at `explore_min` so nothing is
starved before it has had a chance.

Conservation: the shares sum to one and the budgets sum to the period
budget. There is no way for a variant to earn allocation except by
having been cheaper or more successful in the ledger.

Retirement: a variant with at least `min_runs_to_retire` runs whose
sampled yield falls below `retire_fraction` of the best sampled yield in
at least `retire_confidence` of the draws is retired and gets nothing.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Mapping

from .stats import VariantStats


@dataclass
class Config:
    success_kinds: list[str] = field(default_factory=lambda: ["accepted"])
    prior_cost_usd: float = 0.05        # assumed cost for a variant with no runs
    explore_min: float = 0.05           # floor share for every live variant
    retire_fraction: float = 0.3        # retire if yield < this x best
    retire_confidence: float = 0.95
    min_runs_to_retire: int = 8
    draws: int = 4000
    seed: int | None = None

    @classmethod
    def from_dict(cls, d: Mapping) -> "Config":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VariantAllocation:
    variant: str
    status: str                 # "live" | "retired"
    share: float
    budget_usd: float
    n_runs: int
    p_mean: float
    mean_cost: float
    yield_mean: float
    win_rate: float
    p_below_floor: float


@dataclass
class Allocation:
    budget_usd: float
    variants: dict[str, VariantAllocation]
    seed: int | None

    def to_dict(self) -> dict:
        return {
            "budget_usd": self.budget_usd,
            "seed": self.seed,
            "variants": {k: asdict(v) for k, v in self.variants.items()},
        }

    def routing(self) -> dict[str, float]:
        """variant -> share, live variants only. Feed this to your router."""
        return {k: v.share for k, v in self.variants.items() if v.status == "live"}


def allocate(stats: Mapping[str, VariantStats], budget_usd: float,
             config: Config | None = None) -> Allocation:
    cfg = config or Config()
    if budget_usd < 0:
        raise ValueError("budget must be non-negative")
    names = sorted(stats)
    if not names:
        return Allocation(budget_usd, {}, cfg.seed)

    rng = random.Random(cfg.seed)
    costs = {n: stats[n].mean_cost(cfg.prior_cost_usd) for n in names}
    wins = {n: 0 for n in names}
    below = {n: 0 for n in names}

    for _ in range(cfg.draws):
        ys = {n: stats[n].sample_p(rng) / costs[n] for n in names}
        best_name = max(ys, key=ys.get)
        wins[best_name] += 1
        floor = cfg.retire_fraction * ys[best_name]
        for n in names:
            if ys[n] < floor:
                below[n] += 1

    retired = {
        n for n in names
        if stats[n].n_runs >= cfg.min_runs_to_retire
        and below[n] / cfg.draws >= cfg.retire_confidence
    }
    live = [n for n in names if n not in retired]
    if not live:                       # never retire everything
        live = [max(names, key=lambda n: wins[n])]
        retired.discard(live[0])

    # shares among live variants: win share, floored, renormalized
    total_live_wins = sum(wins[n] for n in live) or 1
    raw = {n: wins[n] / total_live_wins for n in live}
    floor = min(cfg.explore_min, 1.0 / len(live))
    floored = {n: max(raw[n], floor) for n in live}
    z = sum(floored.values())
    shares = {n: floored[n] / z for n in live}

    out: dict[str, VariantAllocation] = {}
    for n in names:
        s = stats[n]
        share = shares.get(n, 0.0)
        out[n] = VariantAllocation(
            variant=n,
            status="retired" if n in retired else "live",
            share=share,
            budget_usd=share * budget_usd,
            n_runs=s.n_runs,
            p_mean=s.p_mean,
            mean_cost=costs[n],
            yield_mean=s.yield_mean(cfg.prior_cost_usd),
            win_rate=wins[n] / cfg.draws,
            p_below_floor=below[n] / cfg.draws,
        )

    # conservation: budgets must sum to the period budget
    total = sum(v.budget_usd for v in out.values())
    assert abs(total - budget_usd) < 1e-6 * max(1.0, budget_usd), (total, budget_usd)
    return Allocation(budget_usd, out, cfg.seed)
