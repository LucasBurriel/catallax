"""Per-variant accounting with honest uncertainty for small samples.

Success rate is a Beta posterior (uniform prior). Yield is successes per
dollar. Credible intervals come from Monte Carlo on the posterior because
the standard library has no Beta quantile function and this package has
no dependencies.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable

from .ledger import Ledger, Outcome, Run


@dataclass
class VariantStats:
    variant: str
    n_runs: int = 0
    cost_total: float = 0.0
    successes: int = 0
    revenue: float = 0.0
    outcome_counts: dict[str, int] = field(default_factory=dict)
    prior_a: float = 1.0
    prior_b: float = 1.0

    # -- posterior -------------------------------------------------------
    @property
    def alpha(self) -> float:
        return self.prior_a + self.successes

    @property
    def beta(self) -> float:
        return self.prior_b + (self.n_runs - self.successes)

    @property
    def p_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def mean_cost(self, fallback: float) -> float:
        return self.cost_total / self.n_runs if self.n_runs else fallback

    def yield_mean(self, fallback_cost: float) -> float:
        """Expected successes per USD."""
        return self.p_mean / self.mean_cost(fallback_cost)

    def cost_per_success(self, fallback_cost: float) -> float:
        return self.mean_cost(fallback_cost) / self.p_mean

    @property
    def margin(self) -> float:
        """Only meaningful when outcomes carry a USD value."""
        return self.revenue - self.cost_total

    def sample_p(self, rng: random.Random) -> float:
        return rng.betavariate(self.alpha, self.beta)

    def p_interval(self, rng: random.Random, lo: float = 0.05, hi: float = 0.95,
                   draws: int = 4000) -> tuple[float, float]:
        xs = sorted(self.sample_p(rng) for _ in range(draws))
        return xs[int(lo * draws)], xs[int(hi * draws) - 1]


def summarize(ledger: Ledger, success_kinds: Iterable[str],
              runs: Iterable[Run] | None = None,
              outcomes: Iterable[Outcome] | None = None) -> dict[str, VariantStats]:
    """Aggregate the ledger per variant.

    A run counts as one success if it has at least one outcome whose kind is
    in `success_kinds`. Revenue is the sum of outcome values on that run.
    """
    success_kinds = set(success_kinds)
    runs = list(ledger.runs()) if runs is None else list(runs)
    by_run: dict[str, list[Outcome]] = {}
    for o in (ledger.outcomes() if outcomes is None else outcomes):
        by_run.setdefault(o.run_id, []).append(o)

    stats: dict[str, VariantStats] = {}
    for r in runs:
        s = stats.setdefault(r.variant, VariantStats(variant=r.variant))
        s.n_runs += 1
        s.cost_total += r.cost_usd
        outs = by_run.get(r.id, [])
        if any(o.kind in success_kinds for o in outs):
            s.successes += 1
        for o in outs:
            s.outcome_counts[o.kind] = s.outcome_counts.get(o.kind, 0) + 1
            if o.value:
                s.revenue += o.value
    return stats
