"""Source-family/identity/near-text isolation without a persistent operator index."""

import random, re, math
from collections import defaultdict
from self_curate.contracts import digest
from self_curate.contracts.types import require


def family_ids(task):
    value = task.get("family_ids") or (
        [task["source_family"]]
        if task.get("source_family")
        else ["task:" + task["instance_id"]]
    )
    require(
        type(value) in (list, tuple)
        and bool(value)
        and all(type(x) is str and x for x in value),
        "Explicit source family identities required",
    )
    return tuple(sorted(set(value)))


def shingles(text):
    words = re.findall(r"\w+", text.lower())
    return frozenset(
        " ".join(words[i : i + 5]) for i in range(max(0, len(words) - 4))
    ) or frozenset([" ".join(words)])


class IsolationIndex:
    def __init__(self, threshold=0.85):
        require(
            type(threshold) in (float, int) and 0 < threshold <= 1,
            "Explicit near-duplicate threshold required",
        )
        self.threshold = threshold
        self.rows = []
        self.postings = defaultdict(set)
        self.ids = {}
        self.families = defaultdict(set)

    def conflicts(self, task):
        found = set(self.ids.get(task["instance_id"], ()))
        for name in family_ids(task):
            found.update(self.families.get(name, ()))
        tokens = shingles(task["problem_statement"])
        candidates = set()
        # Any match needs at least ceil(t*|query|) shared shingles. A prefix
        # longer than its allowed missing count must therefore hit every match.
        prefix = sorted(tokens, key=lambda x: (len(self.postings.get(x, ())), x))[
            : len(tokens) - math.ceil(self.threshold * len(tokens)) + 1
        ]
        for token in prefix:
            candidates.update(self.postings.get(token, ()))
        for i in candidates:
            other = self.rows[i][1]
            if (
                not self.threshold * len(tokens)
                <= len(other)
                <= len(tokens) / self.threshold
            ):
                continue
            if len(tokens & other) / len(tokens | other) >= self.threshold:
                found.add(i)
        return tuple(self.rows[i][0] for i in sorted(found))

    def add(self, task):
        i = len(self.rows)
        tokens = shingles(task["problem_statement"])
        self.rows.append((task["instance_id"], tokens))
        self.ids.setdefault(task["instance_id"], set()).add(i)
        for token in tokens:
            self.postings[token].add(i)
        for name in family_ids(task):
            self.families[name].add(i)


def select_development(candidates, labels, *, seed, per_skill, blocked):
    from .skills import SKILLS

    require(
        type(per_skill) is int and per_skill > 0, "Development quota must be positive"
    )
    ids = [t["instance_id"] for t in candidates]
    require(len(set(ids)) == len(ids), "Duplicate development task IDs")
    rng = random.Random(seed)
    ordered = list(candidates)
    rng.shuffle(ordered)
    counts = {s: 0 for s in SKILLS}
    selected = []
    excluded = []
    seen = IsolationIndex(blocked.threshold)
    for task in ordered:
        groups = labels[task["instance_id"]]
        require(set(groups) <= set(SKILLS), "Unknown task skill")
        conflicts = blocked.conflicts(task) or seen.conflicts(task)
        if conflicts:
            excluded.append(
                {"instance_id": task["instance_id"], "conflicts": list(conflicts)}
            )
            continue
        if any(counts[s] < per_skill for s in groups):
            selected.append(task)
            seen.add(task)
            for s in groups:
                counts[s] += 1
    require(bool(selected), "No isolated development tasks remain")
    return tuple(selected), {
        "seed": seed,
        "target_per_skill": per_skill,
        "counts": counts,
        "shortfalls": {s: max(0, per_skill - n) for s, n in counts.items()},
        "excluded": excluded,
        "selected_ids": [t["instance_id"] for t in selected],
        "labels_hash": digest(labels),
    }
