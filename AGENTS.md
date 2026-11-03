# Self-Curate project instructions

Read `docs/self_curate_codex_blueprint.md`, `docs/decisions.md`, and
`docs/progress.md` before changing behavior. This is an independent project;
do not import neighboring projects or inherit their experiment configuration.

## Scope and architecture

- Implement the milestone requested by the user; distinguish M1 data work from P0/P1 milestones.
- Freeze weights in inner search. Shared roles use the same accepted ModelRef and separate memories.
- Use deterministic code for filtering, numerical evaluation, and acceptance.
- Train the exact historically best evaluated manifest; never resample it at training time.
- Compute missing sample gradients only after selecting data, and bind caches to complete identities.
- Keep final test data outside optimization and agent context.
- Preserve rejected attempts, exposure history, evidence, and costs.
- Read `docs/p0_interfaces.md` before implementing workflow consumers. Reuse the versioned task/report/round contracts and validate handoffs with `validate_workflow(bundle, view)`.
- Declared mock contract records are not executed search/training events. Keep P0 validation distinct from the P2 state machine and later real backends.
- For P1-to-workflow handoffs, reuse `prepare_selection`/`prepare_to_directory`; read `docs/p1_handoff_interfaces.md`. Keep downstream outputs empty until they execute.
- Bind workflow consumption to system-owned expected model/objective/context values. Distinguish original-snapshot selection replay from consuming a fixed batch after feature enrichment.
- When consuming a historical best selection, pin `expected_batch_id` from the system's saved BatchManifest. A self-consistent alternative batch or training order is not the original selection.

## M1 contracts

- Preserve user/system/tool context; supervise only assistant payload token positions.
- Token positions are global half-open offsets bound to tokenizer and template hashes.
- Never fill missing annotations/signals with zero. Union overlapping spans before counting.
- A Teacher-only judgment cannot establish quality pass.
- Reject unsupported multimodal input or schema fields explicitly; never silently drop them.
- Keep scoring, gradient, candidate-SFT, and accepted-SFT histories distinct.
- Do not truncate complete trajectories to fit budgets. Record explicit exclusion reasons.
- A precomputed tokenizer declaration is not proof of source authenticity.
- Use ID-keyed, version-scoped in-memory metric views for operators; do not introduce persistent operator indexes without a new request.
- Batch columns must preserve requested ID order, label missing cells, and carry matching content/feature versions.
- External label envelopes bind data ID, content, source, and tokenization; never join labels by row position.
- Store sample gradient references with full model/data/coordinate/loss/projection/precision/optimizer identities; do not calculate gradients in M1.
- Keep usage transitions lossless; terminal training decisions cannot be reversed.

## Implementation and delivery

- Keep data logic separate from model/GPU backends. No model or machine path is hard-coded.
- No GPU/model downloads/real containers unless the user authorizes that later milestone.
- Mark fixture artifacts as synthetic; never silently fall back from real to fixture.
- Use content hashes, strict schemas, immutable records, and complete publication markers.
- Run the entire project test suite after code changes, including independent mask/metric references.
- Update decisions, interfaces, and progress when public behavior changes.
- Do not commit datasets, weights, caches, credentials, or generated run outputs.
- Do not commit, push, or merge on the user's behalf.
- Report only checks actually run; separate local acceptance from real-model/research validation.
