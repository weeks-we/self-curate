"""Local Transformers/PEFT LoRA store; no implicit model download or base update."""

from dataclasses import dataclass, asdict
from pathlib import Path
import re
from self_curate.contracts import digest
from self_curate.contracts.parsing import model_ref
from self_curate.contracts.types import require
from self_curate.data.pool import file_sha256
from self_curate.training.recovery import read_json, atomic_json
from .artifacts import checkpoint_fingerprint
from .torch_backend import LoadedModel


@dataclass(frozen=True)
class LoRASettings:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    language_path: str = "model.language_model"
    bias: str = "none"
    seed: int = 42
    gradient_checkpointing: bool = True
    use_rslora: bool = False
    cpu_activation_offload: bool = False
    offload_min_tokens: int = 65536
    loss_chunk_tokens: int = 32

    def __post_init__(self):
        require(type(self.loss_chunk_tokens) is int and self.loss_chunk_tokens>0,"Positive loss chunk tokens required")
        require(
            type(self.rank) is int
            and self.rank > 0
            and type(self.alpha) is int
            and self.alpha > 0,
            "Invalid LoRA rank/alpha",
        )
        require(
            type(self.dropout) in (int, float) and 0 <= self.dropout < 1,
            "Invalid LoRA dropout",
        )
        require(
            self.bias == "none"
            and type(self.target_modules) is tuple
            and bool(self.target_modules),
            "Use explicit text LoRA modules with bias=none",
        )
        require(
            all(type(x) is str and x for x in self.target_modules),
            "Invalid target modules",
        )
        require(
            type(self.gradient_checkpointing) is bool
            and type(self.use_rslora) is bool
            and type(self.seed) is int
            and self.seed >= 0,
            "Invalid LoRA controls",
        )
        require(
            type(self.language_path) is str and self.language_path,
            "Explicit text model path required",
        )
        require(
            type(self.cpu_activation_offload) is bool
            and type(self.offload_min_tokens) is int
            and self.offload_min_tokens > 0,
            "Invalid activation offload policy",
        )

    @classmethod
    def from_dict(cls, value):
        row = dict(value)
        if "target_modules" in row:
            row["target_modules"] = tuple(row["target_modules"])
        return cls(**row)

    @property
    def config_hash(self):
        return digest(asdict(self))

    def resolve(self, model):
        import torch

        found = [
            name
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
            and (
                name == self.language_path or name.startswith(self.language_path + ".")
            )
            and name.rsplit(".", 1)[-1] in self.target_modules
        ]
        require(
            bool(found),
            "No configured text linear modules matched; inspect the pinned model architecture",
        )
        require(
            all(
                not any(p in n.split(".") for p in ("visual", "vision", "lm_head"))
                for n in found
            ),
            "LoRA target leaked outside the text backbone",
        )
        return tuple(sorted(found))


class PeftModelStore:
    checkpoint_kind = "adapter"

    def __init__(
        self,
        *,
        base_path,
        initial_model,
        initial_adapter,
        settings,
        dtype="bfloat16",
        loader="AutoModelForImageTextToText",
        base_fingerprint=None,
    ):
        self.base_path, self.initial_adapter = (
            Path(base_path).resolve(),
            Path(initial_adapter).resolve(),
        )
        self.model_ref, self.settings, self.dtype, self.loader = (
            initial_model,
            settings,
            dtype,
            loader,
        )
        require(
            isinstance(settings, LoRASettings)
            and initial_model.adapter_hash is not None,
            "LoRA identity required",
        )
        require(
            base_fingerprint is not None
            and checkpoint_fingerprint(self.base_path)[0]
            == base_fingerprint
            == initial_model.weights_hash,
            "Verify downloaded base bytes before creating the model store",
        )
        require(
            checkpoint_fingerprint(self.initial_adapter)[0]
            == initial_model.adapter_hash,
            "Initial adapter bytes changed",
        )
        require(
            read_json(self.initial_adapter / "self-curate-lora.json")["config_hash"]
            == settings.config_hash,
            "Initial LoRA hyperparameters differ from configuration",
        )
        self.resolved_targets = tuple(
            read_json(self.initial_adapter / "self-curate-lora.json")[
                "resolved_targets"
            ]
        )
        require(
            bool(self.resolved_targets)
            and all(
                n.startswith(settings.language_path + ".")
                and n.rsplit(".", 1)[-1] in settings.target_modules
                for n in self.resolved_targets
            ),
            "Initial adapter targets differ from the configured text scope",
        )
        self._verify_adapter_config(self.initial_adapter)

    def _verify_adapter_config(self, path):
        actual = read_json(Path(path) / "adapter_config.json")
        expected = "^(?:" + "|".join(re.escape(n) for n in self.resolved_targets) + ")$"
        require(
            actual.get("r") == self.settings.rank
            and actual.get("lora_alpha") == self.settings.alpha
            and actual.get("lora_dropout") == self.settings.dropout
            and actual.get("bias") == self.settings.bias
            and actual.get("use_rslora", False) == self.settings.use_rslora
            and actual.get("task_type") == "CAUSAL_LM"
            and not actual.get("modules_to_save")
            and not actual.get("use_dora", False)
            and actual.get("target_modules") == expected,
            "PEFT adapter config disagrees with frozen LoRA settings",
        )

    @property
    def implementation_hash(self):
        return digest(
            {
                "code": file_sha256(Path(__file__)),
                "loss_code": file_sha256(Path(__file__).with_name("chunked_loss.py")),
                "model": self.model_ref.to_dict(),
                "lora": asdict(self.settings),
                "dtype": self.dtype,
                "loader": self.loader,
            }
        )

    def _base(self):
        import torch, transformers

        require(self.dtype in ("bfloat16", "float32"), "Unsupported model precision")
        require(
            self.loader in ("AutoModelForImageTextToText", "AutoModelForMultimodalLM"),
            "Use a compatible multimodal auto loader for the pinned model",
        )
        model = getattr(transformers, self.loader).from_pretrained(
            str(self.base_path),
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=getattr(torch, self.dtype),
        )
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    def _load(self, reference, path):
        from peft import PeftModel

        require(
            checkpoint_fingerprint(path)[0] == reference.adapter_hash,
            "Adapter bytes changed",
        )
        metadata = read_json(path / "self-curate-lora.json")
        self._verify_adapter_config(path)
        require(
            metadata["config_hash"] == self.settings.config_hash,
            "Saved adapter hyperparameters changed",
        )
        model = PeftModel.from_pretrained(
            self._base(), str(path), is_trainable=True, local_files_only=True
        )
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        require(
            names
            and all(
                "lora_" in n and "visual" not in n and "vision" not in n for n in names
            ),
            "Only configured LoRA parameters may train",
        )
        if self.settings.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.enable_input_require_grads()
        model.config.use_cache = False
        from types import MethodType
        from .chunked_loss import causal_loss_sum

        language = model.get_base_model().get_submodule(self.settings.language_path)
        head = model.get_base_model().get_output_embeddings()

        def loss_sum(this, input_ids, attention_mask, labels):
            from .chunked_loss import activation_context

            enabled = (
                self.settings.cpu_activation_offload
                and input_ids.shape[-1] >= self.settings.offload_min_tokens
            )
            with activation_context(enabled, input_ids.device):
                hidden = language(
                    input_ids=input_ids, attention_mask=attention_mask, use_cache=False
                ).last_hidden_state
                return causal_loss_sum(hidden, labels, head, chunk_size=self.settings.loss_chunk_tokens)

        model.self_curate_loss_sum = MethodType(loss_sum, model)

        def signal_logits(this, input_ids, attention_mask, positions, chunk_size=32):
            import torch

            hidden = language(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).last_hidden_state
            for offset in range(0, len(positions), chunk_size):
                selected = positions[offset : offset + chunk_size]
                indices = torch.tensor([p - 1 for p in selected], device=hidden.device)
                yield selected, head(hidden[0, indices])

        model.self_curate_signal_logits = MethodType(signal_logits, model)
        return LoadedModel(model, reference)

    def load_branch(self, reference):
        require(
            reference == self.model_ref,
            "Training branches must share the exact initial adapter",
        )
        return self._load(reference, self.initial_adapter)

    def load_checkpoint(self, reference, checkpoint):
        folder = Path(checkpoint)
        require(
            model_ref(read_json(folder / "model.json")) == reference
            and reference.weights_hash == self.model_ref.weights_hash,
            "Wrong model/base in checkpoint",
        )
        return self._load(reference, folder / "weights")

    def save_branch(self, model, output):
        path = Path(output)
        path.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(path, safe_serialization=True)
        atomic_json(
            path / "self-curate-lora.json",
            {
                "config_hash": self.settings.config_hash,
                "settings": asdict(self.settings),
                "base_model": self.model_ref.base_model_id,
                "revision": self.model_ref.revision,
                "resolved_targets": list(self.resolved_targets),
            },
        )


def initialize_adapter(base_model, output, settings, *, base_model_id, revision):
    """Call once on the server; publish one common deterministic initial adapter."""
    import torch
    from peft import LoraConfig, get_peft_model

    path = Path(output)
    require(not path.exists(), "Do not replace an initial adapter")
    targets = settings.resolve(base_model)
    for p in base_model.parameters():
        p.requires_grad_(False)
    with torch.random.fork_rng():
        torch.manual_seed(settings.seed)
        model = get_peft_model(
            base_model,
            LoraConfig(
                r=settings.rank,
                lora_alpha=settings.alpha,
                lora_dropout=settings.dropout,
                bias=settings.bias,
                use_rslora=settings.use_rslora,
                target_modules="^(?:" + "|".join(re.escape(n) for n in targets) + ")$",
                task_type="CAUSAL_LM",
            ),
        )
    path.mkdir(parents=True)
    model.save_pretrained(path, safe_serialization=True)
    atomic_json(
        path / "self-curate-lora.json",
        {
            "config_hash": settings.config_hash,
            "settings": asdict(settings),
            "base_model": base_model_id,
            "revision": revision,
            "resolved_targets": list(targets),
        },
    )
    return checkpoint_fingerprint(path)[0]
