"""Small immutable selection receipts; no raw trajectories or model weights."""

from pathlib import Path

from self_curate.contracts import ContractError, ErrorCode, parse_batch_manifest
from self_curate.contracts.parsing import load_json
from self_curate.contracts.program import parse_program
from self_curate.contracts.types import require
from self_curate.data import read_pool
from self_curate.data.pool import _publish, _staging_directory, file_sha256, write_json
from .context import load_context
from .runtime import execute_program


def load_program(path):
    return parse_program(load_json(Path(path).read_text(encoding="utf-8")))


def load_batch(path):
    return parse_batch_manifest(load_json(Path(path).read_text(encoding="utf-8")))


def select_to_directory(pool_path, program_path, context_path, output):
    output = Path(output).resolve()
    require(
        not output.exists(),
        "Selection output already exists",
        ErrorCode.OUTPUT_EXISTS,
        path=str(output),
    )
    program = load_program(program_path)
    pool = read_pool(pool_path)
    context = load_context(context_path, pool.metric_view())
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = _staging_directory(output)
    write_json(staging / "program.json", program.to_dict())
    write_json(staging / "context.json", context.to_dict())
    try:
        result = execute_program(program, context)
    except ContractError as exc:
        write_json(
            staging / "result.json", {"status": "failed", "error": exc.to_dict()}
        )
        write_json(
            staging / "checksums.json",
            {p.name: file_sha256(p) for p in staging.iterdir() if p.is_file()},
        )
        _publish(staging, output)
        raise
    write_json(staging / "result.json", result.to_dict())
    write_json(
        staging / "trace.json",
        [{**t.to_dict(), "details": load_json(t.details_json)} for t in result.traces],
    )
    if result.batch is not None:
        write_json(staging / "batch.json", result.batch.to_dict())
    write_json(
        staging / "checksums.json",
        {p.name: file_sha256(p) for p in staging.iterdir() if p.is_file()},
    )
    _publish(staging, output)
    return result
