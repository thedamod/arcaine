"""Shared schema for blind cryptanalysis records.

Ground-truth family and key data live under ``oracle`` for validation and
trajectory construction. They must never be copied into the user prompt.
"""

from __future__ import annotations

import hashlib
from typing import Any

BLIND_INSTRUCTION = (
    "Recover the plaintext from the artifact below. The transformation family, key, "
    "and number of layers are intentionally omitted. Infer them from observable "
    "structure, use tools when useful, keep competing hypotheses until evidence "
    "separates them, and verify the result by reproducing the artifact."
)


def blind_problem(artifact: str, key_material: str | None = None) -> str:
    prompt = f"{BLIND_INSTRUCTION}\n\nArtifact:\n{artifact}"
    if key_material is not None:
        prompt += (
            "\n\nAuxiliary key material (unlabelled):\n"
            f"{key_material}\n\nThe key material may fit more than one construction; test plausible "
            "interpretations instead of treating its shape as proof."
        )
    return prompt


def make_record(
    *,
    artifact: str,
    family: str,
    key: dict[str, Any],
    difficulty: str,
    reasoning: str,
    solution: str,
    source: str = "synthetic",
    steps: list[dict[str, Any]] | None = None,
    category: str | None = None,
    key_material: str | None = None,
) -> dict[str, Any]:
    artifact = str(artifact)
    solution = str(solution).strip()
    record_id = hashlib.sha256(f"{family}\0{artifact}\0{solution}".encode()).hexdigest()[:20]
    return {
        "id": f"crypto-{record_id}",
        "problem": blind_problem(artifact, key_material),
        "artifact": artifact,
        "key_material": key_material,
        "category": category or f"{family}_cipher",
        "difficulty": difficulty,
        "chain_of_thought": reasoning.strip(),
        "solution": solution,
        "source": source,
        "oracle": {
            "family": family,
            "key": key,
            "steps": steps or [{"family": family, "key": key}],
        },
    }


def assert_blind_prompt(record: dict[str, Any]) -> None:
    problem = str(record.get("problem", "")).lower()
    family = str(record.get("oracle", {}).get("family", "")).lower()
    forbidden = {family, family.replace("_", " ")}
    aliases = {
        "caesar": {"caesar"},
        "rot13": {"rot13"},
        "single_byte_xor": {"single-byte xor", "single byte xor"},
        "columnar_transposition": {"columnar transposition"},
        "vigenere": {"vigenere", "vigenère"},
        "substitution": {"substitution cipher"},
        "word_reverse": {"word reversal", "word-reversal"},
    }
    forbidden.update(aliases.get(family, set()))
    leaked = sorted(token for token in forbidden if token and token in problem)
    if leaked:
        raise ValueError(f"prompt leaks oracle family via {leaked}: {record.get('id')}")
