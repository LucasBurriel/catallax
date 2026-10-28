"""catallax: let real prices, not self-grading, decide which agent harness survives.

A ledger of what each agent run cost (tokens x market price) and what it
earned (an external signal), plus a budget-conserving allocator that shifts
spend toward the variants the market rewards.
"""

from .allocate import Allocation, Config, allocate
from .ledger import Ledger, Outcome, Run
from .pricing import PriceTable
from .stats import VariantStats, summarize

__version__ = "0.1.0"
__all__ = [
    "Ledger", "Run", "Outcome", "PriceTable", "Config",
    "VariantStats", "summarize", "Allocation", "allocate",
]
