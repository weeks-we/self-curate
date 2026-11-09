"""Exact assistant-only causal loss with bounded vocabulary-logit activations."""


def activation_context(enabled, device):
    from contextlib import nullcontext

    if not enabled:
        return nullcontext()
    import torch

    return torch.autograd.graph.save_on_cpu(
        pin_memory=torch.device(device).type == "cuda"
    )


def causal_loss_sum(hidden, labels, head, *, chunk_size=32):
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    from self_curate.contracts.types import require

    require(
        type(chunk_size) is int and chunk_size > 0, "Positive loss chunk size required"
    )
    features = hidden[:, :-1].reshape(-1, hidden.shape[-1])
    targets = labels[:, 1:].reshape(-1)
    positions = torch.nonzero(targets != -100, as_tuple=False).flatten()
    require(positions.numel() > 0, "No causal supervised positions")

    def loss_block(x, y):
        return F.cross_entropy(head(x).float(), y, reduction="sum")

    total = hidden.new_zeros((), dtype=torch.float32)
    for offset in range(0, positions.numel(), chunk_size):
        selected = positions[offset : offset + chunk_size]
        x, y = features[selected], targets[selected]
        # Backward recomputes only this small output-head block; the language
        # backbone is traversed once, not once per vocabulary chunk.
        total = total + checkpoint(loss_block, x, y, use_reentrant=False)
    return total
