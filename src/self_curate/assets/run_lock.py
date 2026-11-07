"""Freeze resolved inputs/config before a run; does not launch or resume experiments."""

from pathlib import Path

from self_curate.contracts import ArtifactRef, digest, canonical_json
from self_curate.contracts.parsing import artifact, fields, load_json
from self_curate.contracts.types import require, text
from .contracts import Producer, parse_producer
from .store import structured


def _config(value):
    if type(value) is dict:
        for name, item in value.items():
            require(type(name) is str, "Configuration keys must be strings")
            if name.lower() in (
                "api_key",
                "access_token",
                "auth_token",
                "password",
                "secret",
            ):
                require(
                    item in (None, ""),
                    "Store credential environment-variable names, not credential values",
                )
            _config(item)
    elif type(value) is list:
        for item in value:
            _config(item)


def _resolve(store, inputs):
    require(type(inputs) is dict, "Named shared inputs required")
    for name, ref in inputs.items():
        text(name, "input name")
        require(
            isinstance(ref, ArtifactRef)
            and ref.uri.startswith("asset://")
            and ref.uri.endswith("/manifest.json"),
            "Run inputs must pin asset manifests",
        )
        key = ref.uri.split("/")[2]
        asset = store.inspect(key)
        require(asset.reference == ref, "Frozen run input changed")


@structured
def freeze_run_inputs(store, output, *, producer, config, inputs):
    require(
        isinstance(producer, Producer) and type(config) is dict,
        "Resolved configuration/producer required",
    )
    _config(config)
    require(
        digest(config) == producer.config_hash,
        "Producer stamp differs from effective configuration",
    )
    _resolve(store, inputs)
    path = Path(output).resolve()
    require(
        not path.is_relative_to(store.root),
        "Run lock belongs in the experiment output directory",
    )
    row = {
        "schema_version": "1",
        "producer": producer.to_dict(),
        "config": config,
        "inputs": {name: ref.to_dict() for name, ref in sorted(inputs.items())},
    }
    result = {**row, "lock_hash": digest(row)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical_json(result) + "\n")
    return result


@structured
def load_run_inputs(
    store,
    path,
    *,
    expected_lock_hash,
    expected_source_tree_hash,
    expected_environment_hash,
):
    row = fields(
        load_json(Path(path).read_text(encoding="utf-8")),
        ("schema_version", "producer", "config", "inputs", "lock_hash"),
    )
    producer = parse_producer(row["producer"])
    body = {k: v for k, v in row.items() if k != "lock_hash"}
    require(
        row["schema_version"] == "1"
        and row["lock_hash"] == expected_lock_hash == digest(body),
        "Frozen configuration/input identity changed",
    )
    require(
        producer.source_tree_hash == expected_source_tree_hash
        and producer.environment_hash == expected_environment_hash,
        "Current code/environment does not match this run; use its pinned checkout/environment",
    )
    require(
        type(row["config"]) is dict
        and digest(row["config"]) == producer.config_hash
        and type(row["inputs"]) is dict,
        "Run configuration does not match producer stamp",
    )
    _config(row["config"])
    _resolve(store, {name: artifact(ref) for name, ref in row["inputs"].items()})
    return row
