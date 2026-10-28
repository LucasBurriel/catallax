"""Command line interface.

    catallax init                       create .catallax/ with prices + config
    catallax run --variant V --model M --input N --output N [...]
    catallax outcome --run ID --kind K [--value USD]
    catallax ingest-claude transcript.jsonl --variant V
    catallax report
    catallax allocate --budget 20 [--write routing.json]
    catallax simulate [--periods 12 --budget 10 --seed 7]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .adapters.claude_code import run_from_transcript
from .allocate import Config, allocate
from .ledger import Ledger, Outcome, Run
from .pricing import PriceTable
from .report import render
from .simulate import simulate
from .stats import summarize

DEFAULT_ROOT = ".catallax"


def _root(args) -> Path:
    return Path(args.root)


def _load(args) -> tuple[Ledger, PriceTable, Config]:
    root = _root(args)
    if not root.exists():
        sys.exit(f"{root} does not exist; run `catallax init` first")
    ledger = Ledger(root / "ledger")
    prices_path = root / "prices.json"
    prices = PriceTable.load(prices_path) if prices_path.exists() else PriceTable()
    cfg_path = root / "config.json"
    cfg = Config.from_dict(json.loads(cfg_path.read_text(encoding="utf-8"))) if cfg_path.exists() else Config()
    return ledger, prices, cfg


def cmd_init(args) -> None:
    root = _root(args)
    (root / "ledger").mkdir(parents=True, exist_ok=True)
    prices_path = root / "prices.json"
    if not prices_path.exists():
        PriceTable().save(prices_path)
    cfg_path = root / "config.json"
    if not cfg_path.exists():
        cfg_path.write_text(json.dumps(Config().to_dict(), indent=2) + "\n", encoding="utf-8")
    print(f"initialized {root}/ (ledger/, prices.json, config.json)")
    print("edit config.json: success_kinds is the list of outcome kinds that count as a win")


def cmd_run(args) -> None:
    ledger, prices, _ = _load(args)
    usage = {"input": args.input, "output": args.output,
             "cache_read": args.cache_read, "cache_write": args.cache_write}
    cost = args.cost if args.cost is not None else prices.cost(args.model, usage)
    run = ledger.append_run(Run(variant=args.variant, model=args.model, cost_usd=cost,
                                usage=usage, task=args.task))
    print(run.id)


def cmd_outcome(args) -> None:
    ledger, _, _ = _load(args)
    out = ledger.append_outcome(Outcome(run_id=args.run, kind=args.kind, value=args.value))
    print(out.id)


def cmd_ingest_claude(args) -> None:
    ledger, prices, _ = _load(args)
    run = run_from_transcript(args.transcript, args.variant, prices, task=args.task)
    ledger.append_run(run)
    print(f"{run.id}  {run.model}  ${run.cost_usd:.4f}")


def cmd_report(args) -> None:
    ledger, _, cfg = _load(args)
    print(render(summarize(ledger, cfg.success_kinds), cfg))


def cmd_allocate(args) -> None:
    ledger, _, cfg = _load(args)
    alloc = allocate(summarize(ledger, cfg.success_kinds), args.budget, cfg)
    for name, v in sorted(alloc.variants.items(), key=lambda kv: -kv[1].share):
        flag = "" if v.status == "live" else "  (retired)"
        print(f"{name:<28} {v.share:6.1%}  ${v.budget_usd:8.2f}  "
              f"runs={v.n_runs:<4} p={v.p_mean:.2f} ok/$={v.yield_mean:.2f}{flag}")
    if args.write:
        Path(args.write).write_text(json.dumps(alloc.to_dict(), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.write}")


def cmd_simulate(args) -> None:
    simulate(periods=args.periods, budget_per_period=args.budget, seed=args.seed)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="catallax", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=DEFAULT_ROOT, help=f"data directory (default {DEFAULT_ROOT})")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init").set_defaults(fn=cmd_init)

    r = sub.add_parser("run", help="record one run of a variant")
    r.add_argument("--variant", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--input", type=int, default=0)
    r.add_argument("--output", type=int, default=0)
    r.add_argument("--cache-read", type=int, default=0)
    r.add_argument("--cache-write", type=int, default=0)
    r.add_argument("--cost", type=float, help="override computed cost (USD)")
    r.add_argument("--task")
    r.set_defaults(fn=cmd_run)

    o = sub.add_parser("outcome", help="attach an external signal to a run")
    o.add_argument("--run", required=True)
    o.add_argument("--kind", required=True)
    o.add_argument("--value", type=float, help="USD-equivalent, if you have one")
    o.set_defaults(fn=cmd_outcome)

    i = sub.add_parser("ingest-claude", help="record a Claude Code session transcript as a run")
    i.add_argument("transcript")
    i.add_argument("--variant", required=True)
    i.add_argument("--task")
    i.set_defaults(fn=cmd_ingest_claude)

    sub.add_parser("report").set_defaults(fn=cmd_report)

    a = sub.add_parser("allocate", help="split a budget across variants")
    a.add_argument("--budget", type=float, required=True)
    a.add_argument("--write", help="write allocation JSON here")
    a.set_defaults(fn=cmd_allocate)

    s = sub.add_parser("simulate", help="toy market demo")
    s.add_argument("--periods", type=int, default=12)
    s.add_argument("--budget", type=float, default=10.0)
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(fn=cmd_simulate)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
