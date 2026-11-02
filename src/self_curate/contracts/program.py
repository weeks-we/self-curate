"""Strict data-only selection programs and operator parameter schemas."""

from dataclasses import dataclass
import math
import re

from .errors import ContractError, ErrorCode
from .parsing import fields, load_json, sequence
from .types import Record, canonical_json, digest, integer, require, text

OPERATOR_VERSION = "m2-v2"
OPERATOR_NAMES = (
    "SkillSelect",
    "EvidenceSelect",
    "NLLSelect",
    "EntropySelect",
    "LengthSelect",
    "DedupSelect",
    "SampleSelect",
    "MixAllocate",
)


def _choice(value, choices, name):
    require(isinstance(value, str) and value in choices, f"Invalid {name}")
    return value


def _number(value, name, minimum=0, maximum=None):
    require(type(value) in (int, float), f"{name} must be numeric")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ContractError(ErrorCode.INVALID_PROGRAM, f"{name} overflows") from exc
    require(
        math.isfinite(value)
        and value >= minimum
        and (maximum is None or value <= maximum),
        f"Invalid {name}",
    )
    return value


def _names(value, name, *, nonempty=True):
    sequence(value, name)
    require(bool(value) or not nonempty, f"{name} must not be empty")
    for v in value:
        text(v, name)
    require(len(set(value)) == len(value), f"Duplicate {name}")
    return sorted(value)


def _range(value, name, *, integral=False, maximum=None):
    require(
        type(value) is list and len(value) == 2, f"{name} must be [minimum, maximum]"
    )
    lo, hi = value
    for bound in value:
        if bound is not None:
            if integral:
                integer(bound, name)
            else:
                _number(bound, name, maximum=maximum)
    require(lo is not None or hi is not None, f"{name} has no bounds")
    require(lo is None or hi is None or lo <= hi, f"Reversed {name}")
    return value


def normalize_params(operator, params):
    try:
        _choice(operator, OPERATOR_NAMES, "operator")
        if operator == "SkillSelect":
            fields(params, ("skills",), ("match", "missing_policy"))
            return {
                "skills": _names(params["skills"], "skills"),
                "match": _choice(params.get("match", "any"), ("any", "all"), "match"),
                "missing_policy": _choice(
                    params.get("missing_policy", "exclude"),
                    ("error", "exclude", "include"),
                    "missing_policy",
                ),
            }
        if operator == "EvidenceSelect":
            fields(
                params,
                ("target_skill",),
                (
                    "min_share",
                    "min_density",
                    "min_tokens",
                    "min_events",
                    "required_behaviors",
                    "missing_policy",
                ),
            )
            text(params["target_skill"], "target_skill")
            out = {
                "target_skill": params["target_skill"],
                "missing_policy": _choice(
                    params.get("missing_policy", "exclude"),
                    ("error", "exclude", "include"),
                    "missing_policy",
                ),
            }
            for name in ("min_share", "min_density", "min_tokens", "min_events"):
                value = params.get(name)
                if value is not None:
                    (
                        integer(value, name)
                        if name in ("min_tokens", "min_events")
                        else _number(value, name, maximum=1)
                    )
                out[name] = value
            out["required_behaviors"] = _names(
                params.get("required_behaviors", []),
                "required_behaviors",
                nonempty=False,
            )
            return out
        if operator in ("NLLSelect", "EntropySelect"):
            fields(
                params,
                (),
                (
                    "scope",
                    "skill",
                    "scale",
                    "interval",
                    "keep_fraction",
                    "rank_direction",
                    "reference_group",
                    "missing_policy",
                ),
            )
            scope = _choice(
                params.get("scope", "assistant"),
                ("assistant", "skill_evidence"),
                "scope",
            )
            skill = params.get("skill")
            require(
                (skill is not None) == (scope == "skill_evidence"),
                "skill_evidence scope requires skill",
            )
            if skill is not None:
                text(skill, "skill")
            scale = _choice(params.get("scale", "raw"), ("raw", "percentile"), "scale")
            ref = params.get("reference_group")
            require(
                (ref is not None) == (scale == "percentile"),
                "Only percentile scale requires reference_group",
            )
            if ref is not None:
                text(ref, "reference_group")
            interval = params.get("interval")
            if interval is not None:
                _range(
                    interval, "interval", maximum=1 if scale == "percentile" else None
                )
            fraction = params.get("keep_fraction")
            if fraction is not None:
                fraction = _number(fraction, "keep_fraction", maximum=1)
                require(fraction > 0, "keep_fraction must be positive")
            rank = params.get(
                "rank_direction", "ascending" if fraction is not None else None
            )
            require(
                fraction is None or rank is not None,
                "Fraction selection requires ranking",
            )
            if rank is not None:
                _choice(rank, ("ascending", "descending"), "rank_direction")
            require(
                interval is not None or fraction is not None or rank is not None,
                "Score selector requires interval, fraction or ranking",
            )
            return {
                "scope": scope,
                "skill": skill,
                "scale": scale,
                "interval": interval,
                "keep_fraction": fraction,
                "rank_direction": rank,
                "reference_group": ref,
                "missing_policy": _choice(
                    params.get("missing_policy", "error"),
                    ("error", "exclude", "request"),
                    "missing_policy",
                ),
            }
        if operator == "LengthSelect":
            allowed = (
                "assistant_turn_range",
                "tool_call_range",
                "supervised_token_range",
                "context_token_range",
            )
            fields(params, (), allowed)
            require(
                any(params.get(n) is not None for n in allowed),
                "LengthSelect needs a range",
            )
            return {
                n: (
                    None
                    if params.get(n) is None
                    else _range(params[n], n, integral=True)
                )
                for n in allowed
            }
        if operator == "DedupSelect":
            fields(
                params,
                ("mode",),
                ("threshold", "representative_rule", "missing_policy"),
            )
            mode = _choice(
                params["mode"], ("exact", "annotated_group", "semantic"), "dedup mode"
            )
            threshold = params.get("threshold")
            require(
                (threshold is not None) == (mode == "semantic"),
                "Only semantic dedup requires threshold",
            )
            if threshold is not None:
                threshold = _number(threshold, "threshold", minimum=-1, maximum=1)
            return {
                "mode": mode,
                "threshold": threshold,
                "representative_rule": _choice(
                    params.get("representative_rule", "first"),
                    ("first", "lowest_id", "shortest"),
                    "representative_rule",
                ),
                "missing_policy": _choice(
                    params.get("missing_policy", "error"),
                    ("error", "exclude", "include"),
                    "missing_policy",
                ),
            }
        if operator == "SampleSelect":
            fields(
                params,
                ("method",),
                ("size", "token_target", "seed", "strata", "missing_policy"),
            )
            method = _choice(
                params["method"], ("uniform", "ranked", "stratified"), "sampling method"
            )
            size, tokens = params.get("size"), params.get("token_target")
            require(
                (size is None) != (tokens is None),
                "Choose exactly one size or token_target",
            )
            integer(size if size is not None else tokens, "sampling target", 1)
            seed = params.get("seed", 0)
            integer(seed, "sampling seed")
            strata = _names(
                params.get("strata", []), "strata", nonempty=method == "stratified"
            )
            require(
                method == "stratified" or not strata,
                "Only stratified sampling uses strata",
            )
            require(
                all(
                    n in ("source_id", "repository", "task_type", "skills")
                    for n in strata
                ),
                "Unsupported stratum field",
            )
            return {
                "method": method,
                "size": size,
                "token_target": tokens,
                "seed": seed,
                "strata": strata,
                "missing_policy": _choice(
                    params.get("missing_policy", "error"),
                    ("error", "exclude", "include"),
                    "missing_policy",
                ),
            }
        fields(
            params,
            ("branch_token_shares",),
            ("overlap_rule", "shortfall_policy", "seed"),
        )
        shares = params["branch_token_shares"]
        require(
            type(shares) is dict and len(shares) >= 2,
            "MixAllocate needs at least two shares",
        )
        for name, value in shares.items():
            text(name, "branch")
            _number(value, "branch share", maximum=1)
            require(value > 0, "Branch shares must be positive")
        require(abs(sum(shares.values()) - 1) <= 1e-9, "Branch shares must sum to 1")
        integer(params.get("seed", 0), "mix seed")
        return {
            "branch_token_shares": shares,
            "overlap_rule": _choice(
                params.get("overlap_rule", "fixed_priority"),
                ("fixed_priority", "seeded_priority"),
                "overlap_rule",
            ),
            "shortfall_policy": _choice(
                params.get("shortfall_policy", "report_infeasible"),
                ("report_infeasible",),
                "shortfall_policy",
            ),
            "seed": params.get("seed", 0),
        }
    except ContractError as exc:
        raise ContractError(ErrorCode.INVALID_PROGRAM, str(exc), **exc.details) from exc


@dataclass(frozen=True)
class NodeSpec(Record):
    id: str
    op: str
    inputs: tuple[str, ...]
    params_json: str

    def __post_init__(self):
        require(
            isinstance(self.id, str)
            and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", self.id) is not None
            and self.id not in ("eligible_pool", "final_batch"),
            "Invalid/reserved node ID",
            ErrorCode.INVALID_PROGRAM,
        )
        require(
            type(self.inputs) is tuple and bool(self.inputs),
            "Node inputs must be nonempty",
            ErrorCode.INVALID_PROGRAM,
        )
        for item in self.inputs:
            text(item, "input dependency")
        require(
            len(set(self.inputs)) == len(self.inputs),
            "Node inputs must be unique",
            ErrorCode.INVALID_PROGRAM,
        )
        normalized = normalize_params(self.op, load_json(self.params_json))
        object.__setattr__(self, "params_json", canonical_json(normalized))

    @property
    def params(self):
        return load_json(self.params_json)

    def to_dict(self):
        return {
            "id": self.id,
            "op": self.op,
            "inputs": list(self.inputs),
            "params": self.params,
        }


@dataclass(frozen=True)
class ProgramSpec(Record):
    program_id: str
    nodes: tuple[NodeSpec, ...]
    output: str
    seed: int
    parent_id: str | None = None
    schema_version: str = "1"

    def __post_init__(self):
        text(self.program_id, "program_id")
        text(self.output, "output")
        integer(self.seed, "program.seed")
        require(
            self.schema_version == "1",
            "Unsupported program schema",
            ErrorCode.INVALID_PROGRAM,
        )
        require(
            type(self.nodes) is tuple
            and bool(self.nodes)
            and all(isinstance(n, NodeSpec) for n in self.nodes),
            "Program needs nodes",
            ErrorCode.INVALID_PROGRAM,
        )
        require(
            len(self.nodes) <= 256,
            "Program exceeds hard node limit",
            ErrorCode.INVALID_PROGRAM,
        )
        if self.parent_id is not None:
            text(self.parent_id, "parent_id")
        self.ordered_nodes()

    def ordered_nodes(self):
        by_id = {n.id: n for n in self.nodes}
        require(
            len(by_id) == len(self.nodes),
            "Duplicate node ID",
            ErrorCode.INVALID_PROGRAM,
        )
        require(self.output in by_id, "Unknown output node", ErrorCode.INVALID_PROGRAM)
        for node in self.nodes:
            require(
                all(i == "eligible_pool" or i in by_id for i in node.inputs),
                "Unknown dependency",
                ErrorCode.INVALID_PROGRAM,
                node_id=node.id,
            )
            if node.op == "MixAllocate":
                require(
                    node.id == self.output
                    and len(node.inputs) >= 2
                    and set(node.params["branch_token_shares"]) == set(node.inputs),
                    "Mix must be terminal with one share per input",
                    ErrorCode.INVALID_PROGRAM,
                )
                require(
                    all(
                        i != "eligible_pool" and by_id[i].op != "MixAllocate"
                        for i in node.inputs
                    ),
                    "Mix inputs must be branch chains",
                    ErrorCode.INVALID_PROGRAM,
                )
            else:
                require(
                    len(node.inputs) == 1,
                    "Unary operator needs exactly one input",
                    ErrorCode.INVALID_PROGRAM,
                )
                require(
                    node.inputs[0] == "eligible_pool"
                    or by_id[node.inputs[0]].op != "MixAllocate",
                    "Operators cannot run after MixAllocate",
                    ErrorCode.INVALID_PROGRAM,
                )
        visited, visiting, ordered = set(), set(), []

        def visit(node_id):
            if node_id == "eligible_pool" or node_id in visited:
                return
            require(
                node_id not in visiting,
                "Program contains a cycle",
                ErrorCode.INVALID_PROGRAM,
            )
            visiting.add(node_id)
            for dep in by_id[node_id].inputs:
                visit(dep)
            visiting.remove(node_id)
            visited.add(node_id)
            ordered.append(by_id[node_id])

        visit(self.output)
        require(
            len(visited) == len(self.nodes),
            "Program contains unreachable nodes",
            ErrorCode.INVALID_PROGRAM,
        )
        return tuple(ordered)

    @property
    def recipe_hash(self):
        return digest(
            {
                "operator_version": OPERATOR_VERSION,
                "schema_version": self.schema_version,
                "seed": self.seed,
                "nodes": [n.to_dict() for n in sorted(self.nodes, key=lambda n: n.id)],
                "output": self.output,
            }
        )

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "program_id": self.program_id,
            "parent_id": self.parent_id,
            "seed": self.seed,
            "nodes": [n.to_dict() for n in self.nodes],
            "output": self.output,
        }


def parse_program(value):
    try:
        fields(
            value,
            ("schema_version", "program_id", "nodes", "output", "seed"),
            ("parent_id",),
        )
        nodes = []
        for node in sequence(value["nodes"], "nodes"):
            fields(node, ("id", "op", "inputs", "params"))
            nodes.append(
                NodeSpec(
                    node["id"],
                    node["op"],
                    tuple(sequence(node["inputs"], "inputs")),
                    canonical_json(node["params"]),
                )
            )
        return ProgramSpec(**{**value, "nodes": tuple(nodes)})
    except ContractError as exc:
        raise ContractError(ErrorCode.INVALID_PROGRAM, str(exc), **exc.details) from exc
