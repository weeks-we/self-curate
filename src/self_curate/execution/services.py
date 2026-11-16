"""Owned vLLM services, phase-aware GPU allocation, and actual utilization logs."""

from pathlib import Path
import os, subprocess, sys, time, threading, signal, json
from self_curate.contracts.types import require
from self_curate.training.recovery import atomic_json
from .model import http_json


def port_is_free(entry):
    import socket

    with socket.socket() as connection:
        connection.settimeout(0.2)
        return (
            connection.connect_ex((entry.get("host", "127.0.0.1"), entry["port"])) != 0
        )


def phase_uuids(uuids, phase, *, reserved=(), training_count=4):
    require(
        len(uuids) == 8 and len(set(uuids)) == 8,
        "Eight unique authorized GPU UUIDs required",
    )
    require(set(reserved) <= set(uuids), "Unapproved task GPU reservation")
    if phase == "scoring":
        return ()
    if phase == "training":
        require(
            type(training_count) is int and 1 <= training_count <= 8,
            "Invalid independent training branch count",
        )
        return tuple(uuids[training_count:])
    if phase == "terminal":
        return tuple(u for u in uuids if u not in reserved)
    require(
        phase in ("development", "policy", "benchmark", "smoke"),
        "Unknown execution phase",
    )
    return tuple(uuids)


class ServiceManager:
    def __init__(
        self,
        settings,
        entries,
        uuids,
        *,
        execution,
        popen=subprocess.Popen,
        probe=http_json,
        terminate=None,
        port_free=port_is_free,
    ):
        require(
            execution == "server",
            "Real service launch requires explicit server execution",
        )
        self.settings, self.entries, self.uuids = settings, entries, tuple(uuids)
        self.popen, self.probe = popen, probe
        self.children = {}
        self.relays = {}
        self.stop_event = threading.Event()
        self.monitor = None
        self.terminate = terminate
        self.port_free = port_free
        self.root = Path(settings["runtime_root"])
        self.root.mkdir(parents=True, exist_ok=True)
        require(
            {e["gpu_uuid"] for e in entries} == set(uuids),
            "Each approved GPU needs one declared inference service",
        )

    def command(self, entry):
        value = self.settings
        require(type(value.get("max_num_seqs",max(e["slots"] for e in self.entries))) is int
                and value.get("max_num_seqs",max(e["slots"] for e in self.entries))>=max(e["slots"] for e in self.entries),
                "Serving sequence capacity is below declared task concurrency")
        require(
            entry["base_url"]
            in (
                "http://127.0.0.1:" + str(entry["port"]),
                "http://localhost:" + str(entry["port"]),
            ),
            "Managed inference services bind loopback only",
        )
        return [
            value.get("python", sys.executable),
            "-m",
            "self_curate.execution.vllm_capture",
            "--middleware",
            "self_curate.execution.vllm_capture.SeedMiddleware",
            "--model",
            value["base_path"],
            "--host",
            "127.0.0.1",
            "--port",
            str(entry["port"]),
            "--dtype",
            "bfloat16",
            "--max-model-len",
            str(value["max_context_tokens"]),
            "--max-num-seqs",
            str(value.get("max_num_seqs",max(e["slots"] for e in self.entries))),
            "--gpu-memory-utilization",
            str(value["gpu_memory_utilization"]),
            "--enable-lora",
            "--max-lora-rank",
            str(value["lora_rank"]),
            "--max-loras",
            "2",
            "--max-cpu-loras",
            "16",
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            "--max-num-batched-tokens",
            str(value.get("max_num_batched_tokens",8192)),
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            value["tool_call_parser"],
            "--reasoning-parser",
            value["reasoning_parser"],
        ]

    def transition(self, phase, *, reserved=()):
        desired = set(
            phase_uuids(
                self.uuids,
                phase,
                reserved=reserved,
                training_count=self.settings.get("training_processes", 4),
            )
        )
        for uuid in tuple(self.children):
            if uuid not in desired:
                self._stop(uuid)
        started = []
        for entry in self.entries:
            uuid = entry["gpu_uuid"]
            if uuid not in desired or uuid in self.children:
                continue
            require(
                self.port_free(entry),
                "Managed service port is already occupied; reconcile it rather than attaching to an unverified server",
            )
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=uuid,
                VLLM_ALLOW_RUNTIME_LORA_UPDATING="True",
                OMP_NUM_THREADS="1",
                SELF_CURATE_NATIVE_CAPTURE_ROOT=str(Path(entry.get("native_capture_root", str(self.root / "native-tokens" / uuid))).resolve()),
            )
            if entry.get("api_key_env"):
                require(
                    bool(os.environ.get(entry["api_key_env"])),
                    "Managed service API key environment variable is unset",
                )
                env["VLLM_API_KEY"] = os.environ[entry["api_key_env"]]
            index = len(list(self.root.glob("launch-*.json")))
            stream = (self.root / ("service-" + str(index) + ".log")).open(
                "w", encoding="utf-8"
            )
            argv = self.command(entry)
            atomic_json(
                self.root / ("launch-" + str(index) + ".json"),
                {
                    "gpu_uuid": uuid,
                    "argv": argv,
                    "phase": phase,
                    "started_at": time.time(),
                },
            )
            process = self.popen(
                argv,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=os.name != "nt",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            self.children[uuid] = (process, stream, entry)
            if entry.get("container_base_url"):
                require(
                    self.port_free(
                        {"host": entry["relay_host"], "port": entry["relay_port"]}
                    ),
                    "Container relay port is already occupied",
                )
                relay_stream = (self.root / ("relay-" + str(index) + ".log")).open(
                    "w", encoding="utf-8"
                )
                relay_argv = [
                    self.settings.get("python", sys.executable),
                    "-m",
                    "self_curate.execution.relay",
                    "--bind",
                    entry["relay_host"],
                    "--port",
                    str(entry["relay_port"]),
                    "--target-port",
                    str(entry["port"]),
                    "--grants",
                    entry["relay_grant_root"],
                    "--key-env",
                    entry["api_key_env"],
                    "--execution",
                    "server",
                ]
                relay = self.popen(
                    relay_argv,
                    env=env,
                    stdout=relay_stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=os.name != "nt",
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                self.relays[uuid] = (relay, relay_stream)
            atomic_json(
                self.root / ("owner-" + str(index) + ".json"),
                {
                    "pid": getattr(process, "pid", None),
                    "gpu_uuid": uuid,
                    "argv": argv,
                    "phase": phase,
                },
            )
            started.append(uuid)
        # Launch all servers first, then await readiness, so model loads overlap.
        deadline = time.monotonic() + self.settings["startup_timeout_seconds"]
        pending = set(started)
        while pending:
            for uuid in tuple(pending):
                process, _, entry = self.children[uuid]
                if uuid in self.relays:
                    require(
                        self.relays[uuid][0].poll() is None,
                        "Container inference relay exited during startup",
                    )
                require(
                    process.poll() is None,
                    "Inference service exited during startup: " + uuid,
                )
                try:
                    self.probe(
                        entry["base_url"],
                        "/v1/models",
                        None,
                        key_env=entry.get("api_key_env"),
                        timeout=3,
                    )
                    pending.remove(uuid)
                except Exception:
                    pass
            require(
                not pending or time.monotonic() < deadline,
                "Inference startup timeout; inspect per-GPU logs",
            )
            if pending:
                time.sleep(0.2)
        atomic_json(
            self.root / "phase.json",
            {
                "phase": phase,
                "active_uuids": sorted(desired),
                "reserved": list(reserved),
                "timestamp": time.time(),
            },
        )
        if self.monitor is None and self.settings.get("telemetry", True):
            self.monitor = threading.Thread(target=self._telemetry, daemon=True)
            self.monitor.start()
        return tuple(u for u in self.uuids if u in desired)

    def _stop(self, uuid):
        process, stream, _ = self.children.pop(uuid)
        relay = self.relays.pop(uuid, None)
        if relay is not None:
            child, log = relay
            if self.terminate:
                self.terminate(child)
            elif child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            log.close()
        if self.terminate is not None:
            self.terminate(process)
            stream.close()
            return
        if process.poll() is None:
            if os.name == "nt":
                process.terminate()
            else:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        stream.close()

    def _telemetry(self):
        with (self.root / "gpu-utilization.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            while not self.stop_event.is_set():
                try:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=uuid,utilization.gpu,memory.used,memory.total,power.draw",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=True,
                    )
                    rows = [
                        line.strip().split(", ") for line in result.stdout.splitlines()
                    ]
                    value = {
                        "timestamp": time.time(),
                        "measurements": [r for r in rows if r[0] in self.uuids],
                        "columns": [
                            "uuid",
                            "utilization_percent",
                            "memory_used_mib",
                            "memory_total_mib",
                            "power_watts",
                        ],
                    }
                except Exception as exc:
                    value = {
                        "timestamp": time.time(),
                        "measurement_error": type(exc).__name__,
                    }
                stream.write(json.dumps(value) + "\n")
                stream.flush()
                self.stop_event.wait(5)

    def close(self):
        self.stop_event.set()
        if self.monitor:
            self.monitor.join(timeout=15)
        for uuid in tuple(self.children):
            self._stop(uuid)
