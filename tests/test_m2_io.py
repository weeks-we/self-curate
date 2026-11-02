"""Public M1 -> M2 -> immutable downstream receipt acceptance."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from helpers_m2 import context, fixture_program, records
from self_curate.cli import main
from self_curate.contracts import (
    ContractError,
    ErrorCode,
    canonical_json,
    digest,
    parse_batch_manifest,
    verify_batch,
)
from self_curate.data import MetricView, read_pool
from self_curate.data.pool import build_pool, load_config, write_json
from self_curate.operators import (
    execute_program,
    load_batch,
    parse_context,
    select_to_directory,
)

ROOT = Path(__file__).resolve().parents[1]


def hashes(path):
    return {
        p.relative_to(path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in path.rglob("*")
        if p.is_file()
    }


def test_full_cli_and_frozen_context_replay_without_mutating_m1(tmp_path, capsys):
    pool = tmp_path / "pool"
    assert (
        main(
            [
                "normalize",
                "--config",
                str(ROOT / "configs/m2.data.fixture.json"),
                "--output",
                str(pool),
            ]
        )
        == 0
    )
    capsys.readouterr()
    original = hashes(pool)
    program = ROOT / "configs/programs/m2.fixture.json"
    context_file = ROOT / "configs/m2.fixture.json"
    assert main(["validate-program", "--program", str(program)]) == 0
    assert json.loads(capsys.readouterr().out)["nodes"] == 10
    output = tmp_path / "selected"
    args = [
        "select",
        "--pool",
        str(pool),
        "--program",
        str(program),
        "--context",
        str(context_file),
        "--output",
        str(output),
    ]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "feasible"
    manifest = load_batch(output / "batch.json")
    assert len(manifest.members) == 8 and manifest.supervised_tokens == 64
    assert (
        manifest.actual_branch_tokens == {"a": 32, "b": 32}
        and manifest.synthetic is True
    )
    assert (
        verify_batch(manifest, read_pool(pool).metric_view()) == manifest.training_order
    )
    replay = tmp_path / "replay"
    second = select_to_directory(
        pool, output / "program.json", output / "context.json", replay
    )
    assert canonical_json(second.to_dict()) == canonical_json(result)
    assert hashes(output) == hashes(replay)
    receipt = hashes(output)
    assert main(args) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "OUTPUT_EXISTS"
    assert hashes(output) == receipt and hashes(pool) == original
    checksum = json.loads((output / "checksums.json").read_text())
    assert checksum == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.iterdir()
        if p.name != "checksums.json"
    }


def test_infeasible_and_missing_signal_publish_no_fake_batch(tmp_path, capsys):
    pool = tmp_path / "pool"
    build_pool(load_config(ROOT / "configs/m2.data.fixture.json"), pool)
    context_file = tmp_path / "context.json"
    raw = json.loads((ROOT / "configs/m2.fixture.json").read_text())
    raw["budget_pack"]["token_target"] = 65
    write_json(context_file, raw)
    args = [
        "select",
        "--pool",
        str(pool),
        "--program",
        str(ROOT / "configs/programs/m2.fixture.json"),
        "--context",
        str(context_file),
        "--output",
        str(tmp_path / "infeasible"),
    ]
    assert main(args) == 3
    assert json.loads(capsys.readouterr().out)["status"] == "infeasible"
    assert not (tmp_path / "infeasible/batch.json").exists()
    raw["budget_pack"]["token_target"] = 64
    raw["model_ref"]["weights_hash"] = digest("another-model-checkpoint")
    raw["signal_contexts"][0]["references"] = []
    write_json(context_file, raw)
    recipe = fixture_program().to_dict()
    for node in recipe["nodes"]:
        if node["op"] == "NLLSelect":
            node["params"]["missing_policy"] = "request"
    program_file = tmp_path / "program.json"
    write_json(program_file, recipe)
    args[args.index("--program") + 1] = str(program_file)
    args[-1] = str(tmp_path / "missing")
    assert main(args) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "MISSING_SIGNAL"
    assert error["details"]["requests"][0]["metric"] == "nll"
    assert len(error["details"]["requests"][0]["data_ids"]) == 6
    assert not (tmp_path / "missing/batch.json").exists()
    receipt = json.loads((tmp_path / "missing/result.json").read_text())
    assert receipt["status"] == "failed" and receipt["error"] == error
    assert receipt["error"]["details"]["completed_traces"]


def test_context_snapshot_policy_and_signal_identity_are_explicit():
    rows = records()
    ctx = context(rows)
    value = json.loads(canonical_json(ctx.to_dict()))
    restored = parse_context(value, ctx.view)
    assert restored.context_hash == ctx.context_hash
    with pytest.raises(ContractError, match="another pool"):
        parse_context(value, MetricView(rows[:-1]))
    with pytest.raises(ContractError) as exc:
        parse_context({**value, "operator_version": "future"}, ctx.view)
    assert exc.value.code == ErrorCode.STALE_CACHE
    for field, val in [
        ("training_seed", 9),
        ("training_order", "selection"),
        ("max_nodes", 127),
    ]:
        assert (
            parse_context({**value, field: val}, ctx.view).context_hash
            != ctx.context_hash
        )
    changed = replace(ctx, budget=replace(ctx.budget, token_target=100))
    assert changed.policy_hash != ctx.policy_hash
    assert (
        execute_program(fixture_program(), ctx).execution_hash
        != execute_program(fixture_program(), changed).execution_hash
    )
    with pytest.raises(ContractError):
        parse_context({**value, "unrecognized": True}, ctx.view)


@pytest.mark.parametrize(
    "field,value",
    [("training_order", [{}]), ("selection_order", [[]]), ("branch_shares", [[{}, 1]])],
)
def test_malformed_downstream_receipt_is_a_structured_error(field, value):
    manifest = execute_program(
        fixture_program(),
        context(records(), target=64, tolerance=0, batch=8, share_tolerance=0),
    ).batch
    raw = json.loads(canonical_json(manifest.to_dict()))
    raw[field] = value
    with pytest.raises(ContractError):
        parse_batch_manifest(raw)
