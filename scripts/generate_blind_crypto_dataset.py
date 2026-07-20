"""Generate blind, oracle-backed cryptanalysis records for agentic SFT.

The user prompt never names the transformation. Oracle family/key fields are
retained only for validation and trajectory construction.
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.generators.caesar_generator import generate_caesar_sample  # noqa: E402
from dataset.generators.rot13_generator import generate_rot13_sample  # noqa: E402
from dataset.generators.schema import assert_blind_prompt, blind_problem, make_record  # noqa: E402
from dataset.generators.substitution_generator import generate_substitution_sample  # noqa: E402
from dataset.generators.transposition_generator import generate_transposition_sample  # noqa: E402
from dataset.generators.vigenere_generator import generate_vigenere_sample  # noqa: E402
from dataset.generators.xor_generator import generate_xor_sample  # noqa: E402


def clean_plaintext(line: str) -> str | None:
    text = re.sub(r"[^A-Za-z0-9\s.,;:!?'\-]", "", line).strip()
    text = re.sub(r"\s+", " ", text).upper()
    alpha_ratio = sum(ch.isalpha() for ch in text) / max(1, len(text))
    if not 24 <= len(text) <= 140 or alpha_ratio < 0.72:
        return None
    return text


def reservoir_plaintexts(path: Path, count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    sample: list[str] = []
    seen = set()
    eligible = 0
    with path.open(encoding="utf-8", errors="ignore") as src:
        for line in src:
            text = clean_plaintext(line)
            if not text or text in seen:
                continue
            seen.add(text)
            eligible += 1
            if len(sample) < count:
                sample.append(text)
            else:
                index = rng.randrange(eligible)
                if index < count:
                    sample[index] = text
    if len(sample) < count:
        raise ValueError(f"only found {len(sample)} suitable unique plaintexts in {path}")
    rng.shuffle(sample)
    return sample


def word_reverse_record(plaintext: str) -> dict:
    artifact = " ".join(word[::-1] for word in plaintext.split())
    return make_record(
        artifact=artifact,
        family="word_reverse",
        key={},
        category="word_reversed",
        difficulty="easy" if len(plaintext) <= 60 else "medium",
        reasoning=(
            "Token boundaries survived while character order inside each token was reversed. "
            f"Applying that self-inverse operation produced {plaintext}, and repeating it reproduced the artifact."
        ),
        solution=plaintext,
    )


def base64_record(plaintext: str) -> dict:
    artifact = base64.b64encode(plaintext.encode()).decode()
    return make_record(
        artifact=artifact,
        family="base64",
        key={},
        category="encoded_artifact",
        difficulty="easy",
        reasoning="The restricted alphabet and padding fit Base64. Strict decoding yielded readable text, and re-encoding matched exactly.",
        solution=plaintext,
    )


def hex_record(plaintext: str) -> dict:
    artifact = plaintext.encode().hex()
    return make_record(
        artifact=artifact,
        family="hex",
        key={},
        category="encoded_artifact",
        difficulty="easy",
        reasoning="The even-length hexadecimal artifact decoded to UTF-8 text, and encoding those bytes back to hex matched exactly.",
        solution=plaintext,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate blind cryptanalysis oracle records")
    parser.add_argument("--plaintext-file", type=Path, default=Path("data/Gigaword/clean.train.article.txt"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    random.seed(args.seed)
    plaintexts = reservoir_plaintexts(args.plaintext_file, args.rows, args.seed)
    generators = [
        (0.22, generate_caesar_sample),
        (0.18, generate_xor_sample),
        (0.15, generate_transposition_sample),
        (0.08, generate_rot13_sample),
        (0.09, word_reverse_record),
        (0.06, base64_record),
        (0.04, hex_record),
        (0.11, generate_vigenere_sample),
        (0.07, generate_substitution_sample),
    ]
    cutoffs = []
    total = 0.0
    for weight, generator in generators:
        total += weight
        cutoffs.append((total, generator))

    rows = []
    for plaintext in plaintexts:
        draw = rng.random()
        generator = next(generator for cutoff, generator in cutoffs if draw <= cutoff)
        record = generator(plaintext)
        family = record["oracle"]["family"]
        if family in {"caesar", "single_byte_xor"} and rng.random() < 0.30:
            if family == "caesar":
                material = str(record["oracle"]["key"]["shift"])
            else:
                material = str(record["oracle"]["key"]["key_hex"])
            record["key_material"] = material
            record["problem"] = blind_problem(record["artifact"], material)
        assert_blind_prompt(record)
        rows.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as dst:
        for record in rows:
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts = {}
    for record in rows:
        family = record["oracle"]["family"]
        counts[family] = counts.get(family, 0) + 1
    print(json.dumps({"rows": len(rows), "families": counts, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
