> 2026-11-21 length/budget update: EXP001 evaluates every epoch. Per-benchmark budget authority is exp001.evaluation.benchmark_budgets; see docs/TRAINING_LENGTH_AND_BENCHMARK_BUDGETS_ZH.md. Keep training8192 pending final50k pre-filter length audit.

> 2026-11-21 active recipe: EXP001 only LR5e-5/1e-4, two disjoint four-GPU FSDP2 lanes, main global32 and NICE global8, async CPU grading. See docs/EXP001_QWEN35_9B_RECIPE_AND_ENV_ZH.md. Old one-LR counts below are superseded.

> 2026-11-20最新确认：Random50取消；Initial/Full/Random5/10/20/30，统一3 epoch，18分支/15训练/45 checkpoint/72测评。当前执行请读 docs/HANDOFF.md 与 docs/CHECKPOINT_ACCOUNTING_ZH.md；此前相反数字为历史说明。

# Self-Curate project instructions

Read `docs/self_curate_codex_blueprint.md`, `docs/decisions.md`, and
`docs/progress.md` before changing behavior. This is an independent project;
do not import neighboring projects or inherit their experiment configuration.

## Scope and architecture

- Implement the milestone requested by the user; distinguish M1 data work from P0/P1 milestones.
- Freeze the shared roles and reference ModelRef in inner search. Per the user's 2026-11-06 extension, optional candidate-SFT validation may train isolated branches from that fixed reference; never promote them or chain candidates. NICE only differentiates, without optimizer steps. Shared roles keep separate memories.
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
- The user requested a runtime-only tree on 2026-11-09. Development tests/demos are archived outside source; do not recreate them without a request. Validate runtime changes with temporary CPU checks, the retained EXP001 dry-run, and server preflight/smoke. Keep check scripts and outputs under ignored runs/. Never report removed tests as executed.
- Update decisions, interfaces, and progress when public behavior changes.
- Do not commit datasets, weights, caches, credentials, or generated run outputs.
- Do not commit, push, or merge on the user's behalf.
- Report only checks actually run; separate local acceptance from real-model/research validation.

## Current ordinary-task experiment (2026-11-21)

This is the independent self-curate project. Do not import the neighboring NICE
repository. Active entrypoint: self_curate.experiments.ordinary; authority:
configs/ordinary/exp001.json and docs/HANDOFF.md. The user added NICE validation
inside EXP001; its config is configs/ordinary/exp001_nice.json and documentation
docs/EXP001_NICE_ZH.md. Old meetings/NICE redo proposals are retired.

Ordinary branches: Qwen3.5-9B, thinking off, coding/math/IF; Initial once, Full and
Random5/10/20/30 at both shared LR values. Values 1e-5/2e-5 are proposals pending
user check, approved=false. Target SFT 3 epochs, save each epoch, evaluate only3.
With two rates: 33 ordinary branches, 30 SFT, 90 checkpoints, 132 evaluations.

NICE: 20 TACO TRAIN tasks excluded from 3000 candidates, four rollouts/task,
fixed reward-weighted token-SUM direction /4; candidate assistant CE token-MEAN.
LoRA only, no optimizer step during gradients. Top/bottom1000, two shared LR,
four independent one-epoch SFT branches, original effective batch8 and r16/alpha32,
three native progress checkpoints42/84/125. Evaluate each on the same20 diagnostic
tasks plus four coding benchmarks: 12 checkpoints/60 result groups. Keep zero
directions in the fixed20 denominator; never select checkpoints by final benchmark.

A executes full task. B executes ordinary environment/short-training acceptance
plus NICE backward-only probe (2 target/8 candidates, no NICE grading/SFT/evaluation).
Formal execution requires B's identity-bound receipt. Preserve source/model/data/
tokenizer/template/adapter/coordinates/RNG/exposure/cost provenance. Native checkpoints
must have optimizer/scheduler/matching-rank RNG (four ranks per EXP001 lane), not just adapter weights.

Math uses native NuminaMath1.5 stratified50k by source then problem_type, seed42.
Proofs may train but not auto-grade. Development50/skill is published after EXP001,
from TRAIN only in all domains; TACO test remains final-only. Record EXP001 exposure
and start future experiments from the original base for independent dev evaluation.

Workstation prepares fixed model/data/checker/image assets; cluster runs offline.
No local target weights/GPU/container runs. CPU synthetic acceptance is distinct
from real server backward, capacity, FSDP2 resume and official container grading.
Do not commit datasets/credentials/weights/results or private meetings. User commits.
