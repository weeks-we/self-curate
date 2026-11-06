"""Version identities for injected implementations and torch numeric execution.

No torch import until an actual torch backend is configured. Adapter authors may
provide implementation_hash to additionally bind remote code/config identities.
"""

import inspect
from pathlib import Path
import platform

from self_curate.contracts import digest
from self_curate.contracts.types import require, sha256
from self_curate.data.pool import file_sha256


def code_identity(instance):
    cls = (
        instance
        if inspect.isclass(instance) or inspect.isfunction(instance)
        else type(instance)
    )
    declared = getattr(instance, "implementation_hash", None)
    if declared is not None:
        sha256(declared, "adapter implementation_hash")
    try:
        source = inspect.getsourcefile(cls)
    except TypeError:
        source = None
    source_hash = (
        file_sha256(Path(source)) if source and Path(source).is_file() else None
    )
    require(
        source_hash is not None or declared is not None,
        "Opaque model/task adapter must declare implementation_hash",
    )
    return {
        "class": cls.__module__ + "." + cls.__qualname__,
        "source_hash": source_hash,
        "declared_hash": declared,
    }


def torch_execution(device):
    import torch

    device = torch.device(device)
    require(device.type in ("cpu", "cuda"), "Supported devices are CPU and CUDA")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return {
        "policy": "native-parameter-dtype/no-ambient-autocast/ce-float32-v1",
        "device": str(device),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "default_dtype": str(torch.get_default_dtype()),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "threads": torch.get_num_threads(),
    }


def execution_source_hash():
    return digest({"execution.py": file_sha256(Path(__file__))})
