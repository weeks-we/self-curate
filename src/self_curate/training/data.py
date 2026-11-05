"""Materialize the exact historical best batch, preserving IDs, order and masks."""

from self_curate.contracts import ArtifactRef, validate_workflow, verify_batch
from self_curate.contracts.types import require
from .contracts import TrainingData, TrainingExample


def prepare_subset(best, view, *, expected_model, expected_batch_id):
    request = best.request
    require(
        best.feasible
        and request.model_ref == expected_model
        and request.batch.batch_id == expected_batch_id,
        "Training must consume the system-pinned historical best batch",
    )
    validate_workflow(
        best.prepared.workflow,
        view,
        expected_model=expected_model,
        expected_objective=request.objective,
        expected_selection_context_hash=request.selection_context_hash,
        expected_batch_id=expected_batch_id,
    )
    order = verify_batch(request.batch, view, model_ref=expected_model)
    examples = tuple(
        TrainingExample(
            i,
            view.record(i).content_hash,
            view.record(i).tokenization.input_ids,
            view.record(i).labels,
        )
        for i in order
    )
    return TrainingData(
        "subset",
        request.batch.batch_id,
        expected_model.tokenizer,
        examples,
        (
            ArtifactRef(
                "record://selected-batch/" + expected_batch_id, expected_batch_id
            ),
        ),
        request.backend,
    )
