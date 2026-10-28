# catallax

**Let real prices, not self-grading, decide which agent harness survives.**

`catallax` is a small, dependency-free Python library and CLI that keeps a ledger of
what each run of an LLM agent *cost* (tokens × market price) and what it *earned*
(a signal from outside the agent), and allocates a fixed budget across harness
variants according to which ones the world actually rewards.

It is an experiment in applying an Austrian-economics intuition to agent
engineering: an entrepreneur does not learn by grading their own work, they learn
from prices and from profit and loss. Most "self-improving agent" systems today
have the model judge its own traces. This project asks what happens if you refuse
to do that and let the market be the judge instead.

> Status: v0.1, early. The mechanism works and is tested; the real-world data is
> still being collected. See *Limitations* before believing anything.

## The idea in one paragraph

A harness (prompts, tools, model choice, orchestration) is a hypothesis about how
to do a job. Running it costs real money, priced by a real market (token prices).
Whether it worked is sometimes also decided by a real market: a job application
gets a reply or it does not, a sales email converts or it does not, a user accepts
the output or rewrites it. When both sides of the ledger are external, the margin
is a signal no designer wrote and no model can game by reasoning harder. Give each
variant a budget, conserve it strictly, shift it toward what pays, retire what
never does. That is all this does.

## Install

```bash
pip install -e ".[dev]"
python -m pytest
```

Python 3.11+, no runtime dependencies.

## Quick start

```bash
catallax init                                   # creates .catallax/ (ledger, prices.json, config.json)

# record a run of a harness variant (cost is computed from the price table)
catallax run --variant cover-letter@v3 --model claude-sonnet-5-5 --input 18000 --output 900
# -> run_3f9a1c2b7d10

# later, when the world answers
catallax outcome --run run_3f9a1c2b7d10 --kind reply

# or record a whole Claude Code session as one run
catallax ingest-claude ~/.claude/projects/<project>/<session>.jsonl --variant cover-letter@v3

catallax report                                 # per-variant cost, success posterior, successes per dollar
catallax allocate --budget 20 --write routing.json   # how to split next week's $20
catallax simulate                               # toy market demo of the allocator
```

Edit `.catallax/config.json` so `success_kinds` lists the outcome kinds that count
as a win for your use case (`["reply", "interview"]`, `["accepted"]`, ...).

## What the simulation shows

Four variants with hidden true yields (successes per dollar) of 3.0, 2.0, 0.75 and
0.31. Same seed, same budget per period, allocator vs. even split:

```
period  1  thompson=  20 ok  even=  22 ok   big=0.26  small-v1=0.24  small-v2=0.25  verbose=0.25
period  4  thompson=  96 ok  even=  69 ok   big=0.05  small-v1=0.49  small-v2=0.42  verbose=0.05
period 12  thompson= 299 ok  even= 189 ok   big=0.05  small-v1=0.83  small-v2=0.08  verbose=0.05

thompson: 299 successes for $113.00  (2.646/USD)
even:     189 successes for $109.92  (1.719/USD)
```

The allocator finds the best variant by period 2 and keeps a 5% exploration floor
on the losers. This proves the mechanism, not the thesis: in the simulation the
"market" is a coin the simulator controls, which is exactly what the real system
refuses to do.

## How it works

**Ledger** (`ledger.py`). Two append-only JSONL files. A `Run` is one execution of a
variant with its token usage and USD cost. An `Outcome` points at a run and says
what happened (`kind`) and optionally what it was worth (`value`, USD). Nothing is
ever rewritten.

**Pricing** (`pricing.py`). Token usage → dollars from a price table you own
(`.catallax/prices.json`, seeded with Anthropic list prices). Longest-prefix match
on model id. Unknown model raises; a cost you cannot compute is not zero.

**Stats** (`stats.py`). Per variant: runs, total cost, successes, a Beta posterior
on success rate (uniform prior), yield = successes per dollar, and margin when
outcomes carry a value. Intervals are Monte Carlo on the posterior, which is why
small samples show wide bars instead of false confidence.

**Allocator** (`allocate.py`). Thompson sampling on yield. Each draw samples a
success rate from every variant's posterior, divides by its mean cost, and the
best yield wins. Budget share = win share, floored at `explore_min`. Hard
conservation: shares sum to one, budgets sum to the period budget, asserted. A
variant with enough runs whose yield is below `retire_fraction` of the best in
`retire_confidence` of the draws is retired to zero. Never retires everything.

**Adapters** (`adapters/`). `claude_code.py` reads a Claude Code transcript,
dedupes the per-block repeated usage by message id, prices each model, and emits
one `Run`. Other vendors are a small function away.

## Where this is being used

The author runs personal agents (job applications, scheduled research routines,
daily tasks) on Claude Code. Each routine is a variant. Costs come from the
transcripts. The external signal for the job-search routines is literal: a
recruiter replies or does not. For routines with no external market, the proxy is
whether the user accepted the output untouched, edited it, or discarded it, which
is a weaker signal and is treated as one.

Real data lives in a private repository. This repository holds the mechanism and
synthetic data only.

## Limitations, honestly

- **A market inside one owner's system is not a market.** If both the cost and the
  "revenue" are set by the designer, this is central planning with transfer prices
  and extra steps, and transfer prices get gamed. The method only earns its claim
  when at least the revenue side comes from outside. Where it does not, say so and
  treat the result as a heuristic.
- **Slow, noisy signals.** A job application answers in weeks, if ever. Expect
  months before the posterior separates variants that differ by less than 2×.
  Log from day one; decide later.
- **Attribution.** An outcome is attributed to one run. Multi-step pipelines need
  you to decide which step gets the credit. The ledger does not solve that.
- **Thompson on a point estimate of cost.** Cost variance is ignored; only success
  rate is sampled. Fine while cost per variant is stable, wrong if it is not.
- **Subscriptions.** If you pay a flat fee, list token prices are shadow prices.
  They are still the market's valuation of the compute, so they still carry
  information, but they are not your marginal cost.

## Roadmap

- **Proposer.** Today the variants are hand-made. The natural next step is to plug
  in a generator of harness variants (prompt edits, tool-set changes, model swaps)
  along the lines of Google's Harness² (Propose–Probe–Compose over execution
  traces) and keep this allocator as the *selector*. The proposer plays the
  entrepreneur; the ledger plays the market. The point of the split is that the
  proposer never grades itself.
- Cost sampling in the allocator (log-normal per variant).
- Adapters for the OpenAI and Gemini usage shapes.
- A `hook` subcommand for Claude Code's `Stop` hook so runs log themselves.

## Background

Miller & Drexler's *agoric computing* (1988) and Baum's *Hayek Machine* (2000) built
internal economies of agents with conserved money and property rights. They worked
on toy problems and stalled because the terminal payoff was always set by the
experimenter. This project's bet is that LLM agents doing real work for real
people finally have access to the thing those systems lacked: an external price.

## License

MIT
