"""Shared count-matched selection envelope; training batch size is separate."""

import math

from self_curate.contracts.types import integer, require
from self_curate.data import MetricView
from self_curate.operators import BudgetPolicy, QualityPolicy, ExecutionContext


def count_context(records, *, model, count, seed, quality=None):
    records = tuple(records)
    integer(count, "selected sample count", 1)
    require(count <= len(records), "Insufficient candidate pool")
    lengths = sorted(r.stats.supervised_tokens for r in records)
    lower, upper = sum(lengths[:count]), sum(lengths[-count:])
    target = math.ceil((lower + upper) / 2)
    tolerance = max(abs(lower / target - 1), abs(upper / target - 1)) + 1e-12
    # P1 effective_batch=1 preserves the exact count. SFT flushes its explicit
    # smaller tail windows, using the separately frozen training effective batch.
    return ExecutionContext(
        MetricView(records),
        quality or QualityPolicy(),
        BudgetPolicy(target, tolerance, 1, max_samples=count),
        model,
        (),
        training_seed=seed,
        training_order="seeded",
    )
