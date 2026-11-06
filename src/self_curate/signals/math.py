"""Stable full-vocabulary teacher-forced token statistics, measured in nats."""

import math

from self_curate.contracts.types import require, integer


def token_statistics(logits, target_ids, *, rows=None, chunk_size=128):
    """Return NLL(target) and categorical entropy per row, without gradients.

    rows maps each target to an explicit logit row. Dense causal backends use
    p-1; external row providers supply the requested rows directly. No softmax
    temperature/top-k/top-p or smoothing/clipping is applied.
    """
    import numpy as np

    integer(chunk_size, "statistics chunk size", 1)
    values = logits if hasattr(logits, "shape") else np.asarray(logits)
    require(
        len(values.shape) == 2 and values.shape[1] > 0,
        "Need a [rows, full_vocab] logit matrix",
    )
    targets = tuple(target_ids)
    require(
        rows is not None or values.shape[0] == len(targets),
        "Default logit rows must match targets exactly; specify an explicit row mapping",
    )
    indices = tuple(range(len(targets))) if rows is None else tuple(rows)
    require(bool(targets) and len(indices) == len(targets), "Targets/rows must align")
    for target, row in zip(targets, indices):
        integer(target, "target token")
        integer(row, "logit row")
        require(
            target < values.shape[1] and row < values.shape[0],
            "Token/logit row outside supplied vocabulary/context",
        )
    nll, entropy = [], []
    for offset in range(0, len(targets), chunk_size):
        wanted = list(indices[offset : offset + chunk_size])
        block = values[wanted]
        if hasattr(block, "detach"):
            require(
                str(block.dtype) != "torch.bool" and not block.is_complex(),
                "Logits must be real numeric values",
            )
            block = block.detach().double().cpu().numpy()
        else:
            require(
                block.dtype.kind in ("f", "i", "u"),
                "Logits must be real numeric values",
            )
            block = np.asarray(block, dtype=np.float64)
        require(np.isfinite(block).all(), "Raw logits contain NaN or infinity")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            shifted = block - np.max(block, axis=1, keepdims=True)
            log_z = np.log(
                np.sum(np.exp(shifted), axis=1, keepdims=True, dtype=np.float64)
            )
            negative_log_p = log_z - shifted
            p = np.exp(-negative_log_p)
            selected = negative_log_p[
                np.arange(len(wanted)), list(targets[offset : offset + len(wanted)])
            ]
            h = np.sum(p * negative_log_p, axis=1, dtype=np.float64)
        require(
            np.isfinite(selected).all() and np.isfinite(h).all(),
            "Token statistics exceed finite range",
        )
        require((selected >= 0).all() and (h >= 0).all(), "Negative token statistic")
        nll.extend(float(v) for v in selected)
        entropy.extend(float(v) for v in h)
    return tuple(nll), tuple(entropy)


def aggregate(values, spec):
    require(bool(values), "Cannot aggregate empty token statistics")
    try:
        total = (
            math.fsum(v / len(values) for v in values)
            if spec.reduction == "token_mean"
            else math.fsum(values)
        )
    except OverflowError:
        require(False, "Signal aggregate exceeds finite range")
    if spec.log_base == "2":
        total /= math.log(2)
    require(
        math.isfinite(total) and total >= 0, "Signal aggregate exceeds finite range"
    )
    return total
