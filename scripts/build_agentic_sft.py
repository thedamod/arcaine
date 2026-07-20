"""Convert blind oracle records into verified tool-using SFT trajectories.

Only rows that the blind candidate search can independently recover and
round-trip verify are retained. Unsupported or underdetermined rows are
reported, never silently relabelled with oracle answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crypto_agent.trajectory import build_tool_trajectory, validate_trajectory  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as src:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as dst:
        for row in rows:
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).upper()


def group_key(row: dict) -> str:
    metadata = row["metadata"]
    # The trajectory stores only a hash of normalized plaintext for grouping;
    # no oracle plaintext is exposed outside assistant/tool messages.
    payload = metadata["solution_group"]
    return hashlib.sha256(payload.encode()).hexdigest()


def stratified_group_split(rows: list[dict], test_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    by_category = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_category[row["metadata"]["category"]][group_key(row)].append(row)

    rng = random.Random(seed)
    test_ids = set()
    for category in sorted(by_category):
        groups = list(by_category[category].values())
        rng.shuffle(groups)
        target = round(sum(len(group) for group in groups) * test_fraction)
        selected = 0
        for group in groups:
            if selected >= target:
                break
            test_ids.update(id(row) for row in group)
            selected += len(group)
    return (
        [row for row in rows if id(row) not in test_ids],
        [row for row in rows if id(row) in test_ids],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build verified agentic cryptanalysis SFT data")
    parser.add_argument("--input", required=True, type=Path, help="Blind raw JSONL with artifact and oracle fields")
    parser.add_argument("--output", required=True, type=Path, help="Output prefix ending in .jsonl")
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-ratio", type=float, default=0.35)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--max-per-category", type=int, default=4000)
    args = parser.parse_args()

    if not 0 <= args.test_split < 1:
        raise ValueError("--test-split must be in [0, 1)")
    if not 0 <= args.search_ratio <= 1:
        raise ValueError("--search-ratio must be in [0, 1]")

    source = load_jsonl(args.input)
    rng = random.Random(args.seed)
    rng.shuffle(source)
    stats = Counter()
    category_counts = Counter()
    seen_artifacts = set()
    rows = []

    for record in source:
        category = str(record.get("category") or "unknown")
        artifact = str(record.get("artifact") or "")
        artifact_key = normalized(artifact)
        if not artifact or artifact_key in seen_artifacts:
            stats["empty_or_duplicate_artifact"] += 1
            continue
        if args.max_per_category > 0 and category_counts[category] >= args.max_per_category:
            stats["category_cap"] += 1
            continue
        try:
            row = build_tool_trajectory(
                record,
                search_ratio=args.search_ratio,
                max_candidates=args.max_candidates,
            )
            validate_trajectory(row)
        except (KeyError, TypeError, ValueError) as exc:
            stats[f"drop:{category}"] += 1
            stats[f"reason:{str(exc)[:100]}"] += 1
            continue
        seen_artifacts.add(artifact_key)
        category_counts[category] += 1
        rows.append(row)

    rows.sort(key=lambda row: row["metadata"]["id"] or "")
    rng.shuffle(rows)
    train, test = stratified_group_split(rows, args.test_split, args.seed)

    write_jsonl(args.output, rows)
    write_jsonl(args.output.with_suffix(".train.jsonl"), train)
    write_jsonl(args.output.with_suffix(".test.jsonl"), test)

    train_groups = {group_key(row) for row in train}
    test_groups = {group_key(row) for row in test}
    if train_groups & test_groups:
        raise RuntimeError("solution groups leaked across train/test")

    print(json.dumps({
        "input_rows": len(source),
        "kept_rows": len(rows),
        "train_rows": len(train),
        "test_rows": len(test),
        "categories": dict(category_counts.most_common()),
        "search_rows": sum(row["metadata"]["tool_calls"] == 3 for row in rows),
        "python_only_rows": sum(row["metadata"]["tool_calls"] == 2 for row in rows),
        "drops": dict(stats.most_common()),
        "solution_group_overlap": 0,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
