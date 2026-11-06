"""Single-checkpoint NICE-inspired cosine, matching the referenced repository.

Reference: weeks-we/NICE-Subset-Validation/nice_validation/scoring.py.
The score is dimensionless, not an observed reward delta or a full NICE replication.
"""

import math
from threading import RLock
from typing import Protocol

from self_curate.contracts import ArtifactRef, ResourceUsage, digest
from self_curate.contracts.types import require
from self_curate.data.pool import write_json
from .common import (
    candidate_data,
    require_isolation,
    make_result,
    new_run,
    ValuationFailure,
    implementation_identity,
)
from .contracts import NICERecipe, NICEReference, NICEScores


def normalize(vector, epsilon=1e-12):
    import numpy as np

    value = np.asarray(vector, dtype=np.float64)
    require(value.ndim == 1 and np.isfinite(value).all(), "Invalid gradient vector")
    norm = float(np.linalg.norm(value))
    require(math.isfinite(norm), "Gradient norm overflow")
    return value / norm if norm > epsilon else np.zeros_like(value)


def fresh_adam(gradient, epsilon, clip):
    import numpy as np

    value = np.asarray(gradient, dtype=np.float64)
    require(value.ndim == 1 and np.isfinite(value).all(), "Invalid raw gradient")
    length = float(np.linalg.norm(value))
    require(math.isfinite(length), "Gradient norm overflow")
    clipped = value * min(1.0, clip / (length + 1e-6))
    return clipped / (np.abs(clipped) + epsilon)


def count_sketch(vector, dimension, seed):
    """CountSketch-v1, original global coordinates; Adam precedes projection."""
    import numpy as np

    projected = np.zeros(dimension, dtype=np.float64)
    for start in range(0, len(vector), 262144):
        values = vector[start : start + 262144]
        idx = np.arange(start, start + len(values), dtype=np.uint64)
        with np.errstate(over="ignore"):
            bits = idx + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
            bits = (bits ^ (bits >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            bits = (bits ^ (bits >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            bits ^= bits >> np.uint64(31)
        buckets = (bits % np.uint64(dimension)).astype(np.int64)
        signs = np.where((bits >> np.uint64(63)) == 0, 1.0, -1.0)
        projected += np.bincount(buckets, weights=values * signs, minlength=dimension)
    require(np.isfinite(projected).all(), "Nonfinite projected gradient")
    return projected


def score_matrix(samples, targets, epsilon=1e-12):
    """One column per task, including zero tasks. No normalization of their mean."""
    import numpy as np

    require(len(samples) > 0 and len(targets) > 0, "Empty NICE axes")
    dimension = len(samples[0])
    require(
        dimension > 0 and all(len(v) == dimension for v in (*samples, *targets)),
        "Gradient coordinates differ",
    )
    return tuple(
        tuple(
            float(np.dot(normalize(s, epsilon), normalize(t, epsilon))) for t in targets
        )
        for s in samples
    )


class NICEBackend(Protocol):
    backend: str
    implementation_hash: str

    def compute(self, request, data, reference, recipe, output) -> NICEScores: ...


class NICEValidator:
    def __init__(self, *, model, view, reference, recipe, engine, output):
        require(
            isinstance(reference, NICEReference) and isinstance(recipe, NICERecipe),
            "Typed NICE inputs required",
        )
        require(
            reference.model_ref == model
            and engine.backend == reference.evaluation.suite.backend,
            "NICE model/domain mismatch",
        )
        self.model, self.view, self.reference, self.recipe = (
            model,
            view,
            reference,
            recipe,
        )
        self.engine, self.output = engine, output
        self.backend = reference.evaluation.suite.backend
        self.protocol = ArtifactRef(
            "record://leval/nice/" + self.identity, self.identity
        )
        self.results = []
        self._lock = RLock()

    @property
    def identity(self):
        return digest(
            {
                "method": "nice",
                "model": self.model.to_dict(),
                "reference": self.reference.record_hash,
                "recipe": self.recipe.record_hash,
                "engine": self.engine.implementation_hash,
                "implementation": implementation_identity(self.engine),
                "aggregation": "member_mean/task_mean-v1",
            }
        )

    def evaluate(self, request):
        with self._lock:
            return self._evaluate(request)

    def _evaluate(self, request):
        require(
            request.model_ref == self.model
            and request.backend == self.backend
            and request.measure == "predicted_gain"
            and request.protocol == self.protocol
            and self.protocol.sha256 == self.identity,
            "NICE request changed model/method/protocol",
        )
        data = candidate_data(request, self.view)
        suite = self.reference.evaluation.suite
        require_isolation(self.view, data, suite)
        skills = {w.skill for w in request.objective.weights}
        require(
            skills.issubset({s for t in suite.tasks for s in t.skills}),
            "NICE targets miss objective skills",
        )
        run = new_run(self.output, request)
        scores = None
        try:
            scores = self.engine.compute(
                request, data, self.reference, self.recipe, run
            )
            return self._consume(request, data, suite, skills, scores, run)
        except Exception as exc:
            usage = getattr(scores, "usage", None)
            if not isinstance(usage, ResourceUsage):
                usage = getattr(exc, "usage", ResourceUsage())
            if not isinstance(usage, ResourceUsage):
                usage = ResourceUsage()
            try:
                write_json(
                    run / "failure.json",
                    {
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "usage": usage.to_dict(),
                    },
                )
            except Exception:
                pass
            raise ValuationFailure(str(exc), usage=usage) from exc

    def _consume(self, request, data, suite, skills, scores, run):
        require(isinstance(scores, NICEScores), "NICE backend must return typed scores")
        require(
            scores.request_hash == request.request_hash
            and scores.data_hash == data.record_hash
            and scores.reference_hash == self.reference.record_hash
            and scores.recipe_hash == self.recipe.record_hash
            and scores.sample_ids == request.batch.training_order
            and scores.task_ids == tuple(t.task_id for t in suite.tasks)
            and scores.backend == self.backend,
            "NICE output identity/order mismatch",
        )
        require(
            self.backend != "real"
            or not scores.evidence.uri.strip()
            .lower()
            .startswith(("fixture://", "mock://")),
            "Mock NICE vectors cannot be real",
        )
        nonzero = any(n > self.recipe.zero_epsilon for n in scores.target_norms)
        values = {}
        if nonzero:
            for skill in skills:
                columns = [j for j, t in enumerate(suite.tasks) if skill in t.skills]
                values[skill] = math.fsum(
                    row[j] / len(columns) / len(scores.sample_ids)
                    for row in scores.scores
                    for j in columns
                )
        result = make_result(
            request,
            run,
            values=values,
            usage=scores.usage,
            reason=None if nonzero else "ALL_VALIDATION_DIRECTIONS_ZERO",
            details={
                "method": "nice",
                "estimator": "single-checkpoint-fresh-adam-cosine",
                "unit": "dimensionless_proxy",
                "score_receipt": scores.to_dict(),
                "reference_evaluation": self.reference.evaluation.record_hash,
                "reference_cost_imported_not_recharged": True,
                "weights_updated": False,
                "real_reward_gain_verified": False,
            },
        )
        self.results.append(result)
        return result
