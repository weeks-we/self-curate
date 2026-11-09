"""Exact-checkpoint inference routing and per-call usage; no automatic model retry."""

from contextlib import contextmanager
from pathlib import Path
from threading import BoundedSemaphore, RLock
import json
import os
import time
import urllib.request

from self_curate.contracts import digest
from self_curate.contracts.types import require
from self_curate.training.recovery import atomic_json
from .artifacts import local_path


def http_json(base, path, payload, *, key_env=None, timeout=300):
    headers = {"Content-Type": "application/json"}
    if key_env:
        key = os.environ.get(key_env)
        require(bool(key), "Inference API key environment variable is unset")
        headers["Authorization"] = "Bearer " + key
    request = urllib.request.Request(
        base.rstrip("/") + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


class AdapterRouter:
    """Pool entries bind the base/tokenizer/initial adapter; trained adapters load by hash.

    Inference services are started separately on authorized UUIDs. vLLM must have
    dynamic LoRA loading enabled. A model alias is never silently reassigned.
    """

    def __init__(
        self,
        entries,
        initial_model,
        initial_adapter,
        *,
        transport=http_json,
        slot_root=None,
    ):
        self.entries = entries
        self.model = initial_model
        self.initial_adapter = Path(initial_adapter).resolve()
        self.transport = transport
        self.lock = RLock()
        self.loaded = set()
        self.next = 0
        self.slot_root = slot_root
        self.active_uuids = {e["gpu_uuid"] for e in entries}
        require(bool(entries), "At least one inference service required")
        self.slots = [BoundedSemaphore(e["slots"]) for e in entries]
        for e in entries:
            require(
                e["base_weights_hash"] == initial_model.weights_hash
                and e["tokenizer_hash"] == initial_model.tokenizer.tokenizer_hash,
                "Serving pool base/tokenizer mismatch",
            )
            require(
                type(e["slots"]) is int
                and 1 <= e["slots"] <= 2
                and e["gpu_uuid"].startswith("GPU-"),
                "Invalid authorized serving slot",
            )

    def set_active(self, uuids):
        require(
            bool(uuids) and set(uuids) <= {e["gpu_uuid"] for e in self.entries},
            "Invalid active inference GPUs",
        )
        with self.lock:
            self.active_uuids = set(uuids)
            self.loaded.clear()

    @contextmanager
    def _reservation(self, timeout):
        with self.lock:
            indices = [
                i
                for i, e in enumerate(self.entries)
                if e["gpu_uuid"] in self.active_uuids
            ]
            require(bool(indices), "No inference GPUs allocated in this phase")
            offset = self.next % len(indices)
            self.next += 1
            indices = indices[offset:] + indices[:offset]
        if self.slot_root:
            from .resources import slots

            mapping = {
                self.entries[i]["gpu_uuid"] + "-" + str(j): i
                for i in indices
                for j in range(self.entries[i]["slots"])
            }
            with slots(self.slot_root, list(mapping), timeout=timeout) as name:
                yield mapping[name]
        else:
            index = indices[0]
            with self.slots[index]:
                yield index

    @contextmanager
    def bind(self, request, output):
        model = request.model_ref
        require(
            model.base_model_id == self.model.base_model_id
            and model.revision == self.model.revision
            and model.weights_hash == self.model.weights_hash
            and model.tokenizer == self.model.tokenizer
            and model.parameterization == self.model.parameterization,
            "Inference substituted base/tokenizer/parameterization",
        )
        require(model.adapter_hash is not None, "This experiment serves LoRA adapters")
        if request.checkpoint is None:
            require(model == self.model, "Only M0 can omit a checkpoint")
            from self_curate.training.artifacts import checkpoint_fingerprint

            require(
                checkpoint_fingerprint(self.initial_adapter)[0] == model.adapter_hash,
                "Initial adapter changed",
            )
            adapter = self.initial_adapter
        else:
            folder = local_path(request.checkpoint, directory=True)
            from self_curate.contracts.parsing import model_ref
            from self_curate.training.recovery import read_json
            from self_curate.training.artifacts import checkpoint_fingerprint

            require(
                model_ref(read_json(folder / "model.json")) == model
                and checkpoint_fingerprint(folder / "weights")[0] == model.adapter_hash,
                "Wrong adapter checkpoint",
            )
            adapter = folder / "weights"
        timeout = getattr(getattr(request, "task", None), "budget", None)
        with self._reservation(
            3600 if timeout is None else timeout.max_wall_seconds
        ) as index:
            entry = self.entries[index]
            alias = "sc-" + model.model_hash
            with self.lock:
                if (index, alias) not in self.loaded:
                    listing = self.transport(
                        entry["base_url"],
                        "/v1/models",
                        None,
                        key_env=entry.get("api_key_env"),
                    )
                    ids = {r["id"] for r in listing["data"]}
                    # The alias contains the complete model hash; publishers must
                    # not preload this reserved prefix with other adapter bytes.
                    if alias not in ids:
                        self.transport(
                            entry["base_url"],
                            "/v1/load_lora_adapter",
                            {"lora_name": alias, "lora_path": str(adapter)},
                            key_env=entry.get("api_key_env"),
                        )
                    self.loaded.add((index, alias))
            binding = {
                **{k: v for k, v in entry.items() if k != "api_key"},
                "model_name": alias,
                "model_ref": model.to_dict(),
                "checkpoint": (
                    None if request.checkpoint is None else request.checkpoint.to_dict()
                ),
                "adapter_path": str(adapter),
            }
            atomic_json(Path(output) / "serving-binding.json", binding)
            yield binding


class CompletionClient:
    def __init__(self, binding, output, *, transport=http_json):
        self.binding, self.output, self.transport = binding, Path(output), transport
        self.calls = []

    def complete(
        self,
        messages,
        *,
        max_tokens,
        seed,
        temperature=0,
        tools=None,
        timeout=300,
        response_format=None,
    ):
        index = len(self.calls)
        start = time.perf_counter()
        payload = {
            "model": self.binding["model_name"],
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "stream": False,
        }
        if tools is not None:
            payload["tools"] = tools
        if response_format is not None:
            payload["response_format"] = response_format
        folder = self.output / ("call-" + str(index).zfill(4))
        folder.mkdir(parents=True)
        atomic_json(folder / "request.json", payload)
        atomic_json(folder / "dispatched.json", {"payload_hash": digest(payload)})
        try:
            response = self.transport(
                self.binding["base_url"],
                "/v1/chat/completions",
                payload,
                key_env=self.binding.get("api_key_env"),
                timeout=timeout,
            )
            atomic_json(folder / "response.json", response)
            usage = response.get("usage", {})
            require(
                all(
                    type(usage.get(n)) is int and usage[n] >= 0
                    for n in ("prompt_tokens", "completion_tokens")
                ),
                "Inference omitted actual token usage; do not infer zero",
            )
            value = {
                "input_tokens": usage["prompt_tokens"],
                "output_tokens": usage["completion_tokens"],
                "cached_input_tokens": usage.get("prompt_tokens_details", {}).get(
                    "cached_tokens"
                ),
                "wall_seconds": time.perf_counter() - start,
                "status": "completed",
            }
            self.calls.append(value)
            atomic_json(folder / "usage.json", value)
            return response
        except Exception as exc:
            value = {
                "input_tokens": None,
                "output_tokens": None,
                "wall_seconds": time.perf_counter() - start,
                "status": "unknown",
                "error_type": type(exc).__name__,
            }
            self.calls.append(value)
            atomic_json(folder / "usage.json", value)
            raise
