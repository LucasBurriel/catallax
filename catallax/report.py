"""Plain-text report of the ledger, per variant."""

from __future__ import annotations

import random
from typing import Mapping

from .allocate import Config
from .stats import VariantStats


def render(stats: Mapping[str, VariantStats], cfg: Config) -> str:
    if not stats:
        return "ledger is empty"
    rng = random.Random(cfg.seed)
    rows = []
    for name in sorted(stats, key=lambda n: -stats[n].yield_mean(cfg.prior_cost_usd)):
        s = stats[name]
        lo, hi = s.p_interval(rng)
        rows.append((
            name, s.n_runs, f"{s.cost_total:.4f}", f"{s.mean_cost(cfg.prior_cost_usd):.4f}",
            s.successes, f"{s.p_mean:.2f} [{lo:.2f},{hi:.2f}]",
            f"{s.yield_mean(cfg.prior_cost_usd):.2f}",
            f"{s.cost_per_success(cfg.prior_cost_usd):.3f}",
            f"{s.margin:+.4f}" if s.revenue else "-",
        ))
    head = ("variant", "runs", "cost$", "cost/run", "ok", "p(ok) 90%ci", "ok/$", "$/ok", "margin$")
    widths = [max(len(str(r[i])) for r in [head, *rows]) for i in range(len(head))]
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    out = [fmt.format(*head), fmt.format(*("-" * w for w in widths))]
    out += [fmt.format(*map(str, r)) for r in rows]
    out.append("")
    out.append(f"success kinds: {', '.join(cfg.success_kinds)}   "
               f"prior cost for unseen variants: ${cfg.prior_cost_usd}")
    return "\n".join(out)
