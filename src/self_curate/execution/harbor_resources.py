"""Map native per-service GPU counts onto exclusively reserved UUIDs."""

from pathlib import Path
import tomllib, json
from self_curate.contracts.types import require


def gpu_layout(task_path):
    path = Path(task_path)
    native = tomllib.loads((path / "task.toml").read_text(encoding="utf-8"))
    count = native.get("environment", {}).get("gpus", 0)
    require(type(count) is int and count >= 0, "Native task GPU count is not resolved")
    result = {"main": count} if count else {}
    compose = next(
        (
            p
            for p in (
                path / "environment/docker-compose.yaml",
                path / "environment/docker-compose.yml",
            )
            if p.exists()
        ),
        None,
    )
    if compose:
        import yaml

        data = yaml.safe_load(compose.read_text(encoding="utf-8"))
        for name, service in data.get("services", {}).items():
            values = (
                service.get("deploy", {})
                .get("resources", {})
                .get("reservations", {})
                .get("devices", [])
            )
            explicit = []
            for row in values:
                if any(
                    x == "gpu" or x.startswith("nvidia-")
                    for x in row.get("capabilities", [])
                ):
                    requested = (
                        len(row["device_ids"])
                        if row.get("device_ids")
                        else row.get("count")
                    )
                    require(
                        type(requested) is int and requested > 0,
                        "Native Compose GPU demand must be explicit; do not reinterpret all/unresolved devices",
                    )
                    explicit.append(requested)
            if service.get("gpus"):
                require(
                    type(service["gpus"]) is int and service["gpus"] > 0,
                    "Native Compose gpus=all/list needs an explicit resource profile",
                )
                explicit.append(service["gpus"])
            if explicit:
                result[name] = sum(explicit)
    return result


def write_override(layout, uuids, output):
    require(
        sum(layout.values()) == len(uuids),
        "GPU assignment changed the native task counts",
    )
    lines = ["services:"]
    offset = 0
    for name, count in layout.items():
        assigned = uuids[offset : offset + count]
        offset += count
        lines.extend(
            [
                "  " + json.dumps(name) + ":",
                "    gpus: !reset null",
                "    deploy:",
                "      resources:",
                "        reservations:",
                "          devices: !override",
                "            - driver: nvidia",
                "              device_ids: " + json.dumps(list(assigned)),
                "              capabilities: [gpu]",
            ]
        )
    path = Path(output)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
