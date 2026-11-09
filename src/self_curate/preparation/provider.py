"""Bounded OpenAI-compatible annotation calls; disk cache and unknown-call guards."""

from pathlib import Path
import json, os, time, urllib.request, urllib.error
from self_curate.contracts import digest
from self_curate.contracts.types import require
from self_curate.training.recovery import atomic_json, read_json, lease


class AnnotationUnavailable(RuntimeError):
    """Infrastructure/configuration failure, not a reason to remove training data."""


class AnnotationClient:
    def __init__(self, settings, output, *, transport=None):
        self.settings, self.output, self.transport = (
            settings,
            Path(output),
            transport or self._http,
        )
        require(
            settings["backend"] in ("real", "mock"),
            "Explicit annotation domain required",
        )
        require(
            type(settings["max_calls"]) is int and settings["max_calls"] > 0,
            "Annotation call budget required",
        )
        require(
            type(settings["max_output_tokens"]) is int
            and settings["max_output_tokens"] > 0,
            "Annotation output budget required",
        )
        if settings["backend"] == "real":
            require(
                bool(settings["model"]) and settings["execution"] == "server",
                "Real Teacher needs explicit server execution and model",
            )

    def _http(self, payload):
        key = os.environ.get(self.settings["api_key_env"])
        if not key:
            raise AnnotationUnavailable("Teacher API key environment variable is unset")
        request = urllib.request.Request(
            self.settings["base_url"].rstrip("/") + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + key,
            },
        )
        with urllib.request.urlopen(
            request, timeout=self.settings["timeout_seconds"]
        ) as response:
            return json.load(response)

    def ask(self, system, payload):
        request = {
            "model": self.settings["model"],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "temperature": 0,
            "max_tokens": self.settings["max_output_tokens"],
            "response_format": {"type": "json_object"},
        }
        key = digest(
            {
                "request": request,
                "endpoint": self.settings["base_url"],
                "backend": self.settings["backend"],
            }
        )
        folder = self.output / key
        with lease(self.output / ".provider.lease"):
            if (folder / "response.json").exists():
                require(
                    read_json(folder / "request.json") == request,
                    "Annotation cache request changed",
                )
                response = read_json(folder / "response.json")
            else:
                if (folder / "dispatched.json").exists():
                    raise AnnotationUnavailable(
                        "Unknown in-flight Teacher call; reconcile before retrying"
                    )
                used = len(list(self.output.glob("*/dispatched.json")))
                if used >= self.settings["max_calls"]:
                    raise AnnotationUnavailable("Teacher call budget exhausted")
                folder.mkdir(parents=True, exist_ok=True)
                atomic_json(folder / "request.json", request)
                atomic_json(
                    folder / "dispatched.json",
                    {"request_hash": key, "started_at": time.time()},
                )
                started = time.perf_counter()
                try:
                    response = self.transport(request)
                    atomic_json(folder / "response.json", response)
                    atomic_json(
                        folder / "timing.json",
                        {
                            "wall_seconds": time.perf_counter() - started,
                            "finished_at": time.time(),
                        },
                    )
                except Exception as exc:
                    atomic_json(
                        folder / "failure.json",
                        {
                            "error_type": type(exc).__name__,
                            "usage": "unknown",
                            "wall_seconds": time.perf_counter() - started,
                        },
                    )
                    raise
        choice = response["choices"][0]
        require(
            choice.get("finish_reason") == "stop",
            "Teacher response truncated or unfinished; not a valid annotation",
        )
        usage = response.get("usage", {})
        require(
            all(
                type(usage.get(n)) is int and usage[n] >= 0
                for n in ("prompt_tokens", "completion_tokens")
            ),
            "Teacher omitted actual token usage",
        )
        require(
            type(response.get("model")) is str and bool(response["model"]),
            "Teacher omitted returned model identity",
        )
        value = json.loads(choice["message"]["content"])
        require(type(value) is dict, "Teacher must return a JSON object")
        return value, folder / "response.json"
