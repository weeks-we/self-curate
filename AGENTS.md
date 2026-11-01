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

## M1 contracts

- Preserve user/system/tool context; supervise only assistant payload token positions.
- Token positions are global half-open offsets bound to tokenizer and template hashes.
- Never fill missing annotations/signals with zero. Union overlapping spans before counting.
- A Teacher-only judgment cannot establish quality pass.
- Reject unsupported multimodal input or schema fields explicitly; never silently drop them.
- Keep scoring, gradient, candidate-SFT, and accepted-SFT histories distinct.
- Do not truncate complete trajectories to fit budgets. Record explicit exclusion reasons.
- A precomputed tokenizer declaration is not proof of source authenticity.

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
