"""Cross-process task slots and exclusive GPU reservations."""

from contextlib import contextmanager, ExitStack
from pathlib import Path
import time
from self_curate.training.recovery import lease
from self_curate.contracts.types import require


@contextmanager
def slots(root, names, *, timeout):
    root = Path(root)
    deadline = time.monotonic() + timeout
    while True:
        for name in names:
            context = lease(root / (name + ".lease"))
            try:
                context.__enter__()
            except RuntimeError:
                continue
            try:
                yield name
            finally:
                context.__exit__(None, None, None)
            return
        require(
            time.monotonic() < deadline,
            "Timed out waiting for an authorized shared resource slot",
        )
        time.sleep(0.1)


@contextmanager
def exclusive_gpus(root, uuids, count, *, timeout):
    require(
        type(count) is int and 0 <= count <= len(uuids),
        "Native task GPU demand exceeds reserved resources",
    )
    with ExitStack() as stack:
        chosen = []
        # Serialize multi-GPU acquisition, avoiding partial-allocation deadlocks.
        if count:
            with slots(root, ["allocation"], timeout=timeout):
                for _ in range(count):
                    name = stack.enter_context(
                        slots(
                            root, [u for u in uuids if u not in chosen], timeout=timeout
                        )
                    )
                    chosen.append(name)
        yield tuple(chosen)
