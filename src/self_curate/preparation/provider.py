"""Bounded OpenAI-compatible annotation calls; disk cache and unknown-call guards."""

from pathlib import Path
import json, math, os, time, urllib.request, urllib.error
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
        self._settled_ceiling_costs={}
        require(type(settings.get("max_input_bytes",900000)) is int and settings.get("max_input_bytes",900000)>0,
                "Positive Teacher input byte guard required")
        if settings.get("max_cost_cny") is not None:
            cap=settings["max_cost_cny"];rates=settings.get("price_ceiling_cny_per_million",{})
            require(type(cap) in (int,float) and math.isfinite(cap) and cap>0,"Finite positive currency cap required")
            require(all(type(rates.get(k)) in (int,float) and math.isfinite(rates[k]) and rates[k]>0
                        for k in ("input","output")),"Finite positive price ceilings required")
        if settings["backend"] == "real":
            require(
                bool(settings["model"]) and settings["execution"] in ("server","workstation"),
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
        if self.settings.get("thinking") is not None:
            request["thinking"]=self.settings["thinking"]
        if self.settings.get("reasoning_effort"):
            request["reasoning_effort"]=self.settings["reasoning_effort"]
        request_bytes=len(json.dumps(request,ensure_ascii=False).encode("utf-8"))
        if request_bytes>self.settings.get("max_input_bytes",900000):
            raise AnnotationUnavailable("Teacher request exceeds explicit byte guard; never truncate")
        key = digest(
            {
                "request": request,
                "endpoint": self.settings["base_url"],
                "backend": self.settings["backend"],
            }
        )
        usage_path = getattr(self, "usage_path", None)
        if usage_path is not None:
            usage_path = Path(usage_path)
            with lease(usage_path.with_suffix(".lease")):
                used_keys = read_json(usage_path) if usage_path.exists() else []
                require(type(used_keys) is list and all(type(v) is str for v in used_keys), "Invalid annotation consumption receipt")
                atomic_json(usage_path, sorted(set(used_keys) | {key}))
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
                scope = None if usage_path is None else set(read_json(usage_path))
                used = sum(scope is None or p.parent.name in scope for p in self.output.glob("*/dispatched.json"))
                if used >= self.settings["max_calls"]:
                    raise AnnotationUnavailable("Teacher call budget exhausted")
                reservation=None
                if self.settings.get("max_cost_cny") is not None:
                    rates=self.settings.get("price_ceiling_cny_per_million")
                    require(isinstance(rates,dict) and rates.get("input",0)>0 and rates.get("output",0)>0,
                            "Currency guard requires frozen conservative prices")
                    # UTF-8 bytes plus framing allowance is a conservative planning ceiling,
                    # never reported as measured tokens or actual provider billing.
                    reservation=((request_bytes+8192)*rates["input"]+
                                 request["max_tokens"]*rates["output"])/1e6
                    spent=0.0
                    for old in self.output.glob("*/dispatched.json"):
                        cost_key=(old.parent.name,rates["input"],rates["output"])
                        if cost_key in self._settled_ceiling_costs:
                            spent+=self._settled_ceiling_costs[cost_key];continue
                        old_response=old.parent/"response.json"
                        raw=read_json(old_response) if old_response.exists() else {}
                        usage=raw.get("usage",{})
                        if all(type(usage.get(n)) is int and usage[n]>=0 for n in ("prompt_tokens","completion_tokens")):
                            settled=(usage["prompt_tokens"]*rates["input"]+usage["completion_tokens"]*rates["output"])/1e6
                            self._settled_ceiling_costs[cost_key]=settled;spent+=settled
                        else:spent+=(read_json(old).get("reserved_cost_cny") or self.settings["max_cost_cny"])
                    if spent+reservation>self.settings["max_cost_cny"]:
                        raise AnnotationUnavailable("Teacher currency ceiling reached; unknown calls keep reservations")
                folder.mkdir(parents=True, exist_ok=True)
                atomic_json(folder / "request.json", request)
                atomic_json(
                    folder / "dispatched.json",
                    {"request_hash": key, "started_at": time.time(),"reserved_cost_cny":reservation,
                     "cost_scope":"provider_cache_root","price_ceiling":self.settings.get("price_ceiling_cny_per_million")},
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
        from self_curate.contracts.parsing import load_json
        value = load_json(choice["message"]["content"])
        require(type(value) is dict, "Teacher must return a JSON object")
        return value, folder / "response.json"
