"""Deterministic cryptanalysis probes used by training traces and runtime tools.

This is deliberately a *candidate generator*, not an oracle. It inspects an
unlabelled artifact, tries transformations whose signatures fit, scores the
results, and returns alternatives. Dataset rows are accepted only when this
blind search actually rediscovers the generated plaintext.
"""

from __future__ import annotations

import base64
import itertools
import json
import math
import re
import string
from typing import Any

ALPHABET = string.ascii_uppercase
COMMON_WORDS = {
    "A", "I", "THE", "OF", "TO", "AND", "IN", "IS", "IT", "YOU", "THAT",
    "HE", "WAS", "FOR", "ON", "ARE", "AS", "WITH", "HIS", "THEY", "AT",
    "BE", "THIS", "HAVE", "FROM", "OR", "ONE", "HAD", "BY", "WORD", "BUT",
    "NOT", "WHAT", "ALL", "WERE", "WE", "WHEN", "YOUR", "CAN", "SAID",
    "THERE", "USE", "AN", "EACH", "WHICH", "SHE", "DO", "HOW", "THEIR",
    "IF", "WILL", "UP", "OTHER", "ABOUT", "OUT", "MANY", "THEN", "THEM",
    "THESE", "SO", "SOME", "HER", "WOULD", "MAKE", "LIKE", "HIM", "INTO",
    "TIME", "HAS", "LOOK", "TWO", "MORE", "WRITE", "GO", "SEE", "NUMBER",
    "NO", "WAY", "COULD", "PEOPLE", "MY", "THAN", "FIRST", "WATER", "BEEN",
    "CALL", "WHO", "OIL", "ITS", "NOW", "FIND", "LONG", "DOWN", "DAY",
    "DID", "GET", "COME", "MADE", "MAY", "PART", "SECRET", "MESSAGE",
    "KEY", "CIPHER", "ATTACK", "MEET", "DATA", "HIDDEN", "CODE", "TEXT",
}
COMMON_NGRAMS = (
    " THE ", " AND ", "ING", "ION", "TH", "HE", "IN", "ER", "AN", "RE",
    "ON", "AT", "EN", "ND", "TI", "ES", "OR", "TE", "OF", "ED", "IS",
    "IT", "AL", "AR", "ST", "TO", "NT", "NG", "SE", "HA", "AS", "OU",
)


def normalize_plaintext(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).upper()


def english_score(text: str) -> float:
    """Cheap deterministic English-likeness score; higher is better."""
    if not text:
        return -1e9
    printable = sum(ch in string.printable for ch in text) / len(text)
    if printable < 0.85:
        return -500 + printable

    upper = text.upper()
    letters = [ch for ch in upper if ch in ALPHABET]
    if not letters:
        return -100
    letter_ratio = len(letters) / len(text)
    spaces = upper.count(" ")
    words = re.findall(r"[A-Z]+", upper)
    known = sum(1 + min(len(word), 8) / 4 for word in words if word in COMMON_WORDS)
    ngrams = sum(upper.count(ngram) for ngram in COMMON_NGRAMS)
    vowels = sum(ch in "AEIOU" for ch in letters) / len(letters)
    vowel_penalty = abs(vowels - 0.38) * 12
    weird = sum(ch not in string.printable or (ord(ch) < 32 and ch not in "\n\t") for ch in text)
    long_token_penalty = sum(max(0, len(word) - 16) for word in words) * 0.2
    return (
        printable * 6
        + letter_ratio * 5
        + min(spaces, 20) * 0.08
        + known * 2.5
        + ngrams * 0.45
        - vowel_penalty
        - weird * 5
        - long_token_penalty
    )


def _caesar(text: str, shift: int) -> str:
    out = []
    for char in text.upper():
        out.append(ALPHABET[(ALPHABET.index(char) + shift) % 26] if char in ALPHABET else char)
    return "".join(out)


def _word_reverse(text: str) -> str:
    return " ".join(word[::-1] for word in text.split())


def _vigenere(text: str, keyword: str, decrypt: bool) -> str:
    keyword = "".join(ch for ch in keyword.upper() if ch in ALPHABET)
    if not keyword:
        raise ValueError("Vigenere keyword contains no letters")
    output = []
    index = 0
    direction = -1 if decrypt else 1
    for char in text.upper():
        if char in ALPHABET:
            shift = ALPHABET.index(keyword[index % len(keyword)])
            output.append(ALPHABET[(ALPHABET.index(char) + direction * shift) % 26])
            index += 1
        else:
            output.append(char)
    return "".join(output)


def _substitution_decrypt(text: str, cipher_alphabet: str) -> str:
    cipher_alphabet = cipher_alphabet.upper()
    if len(cipher_alphabet) != 26 or set(cipher_alphabet) != set(ALPHABET):
        raise ValueError("substitution key must be a 26-letter permutation")
    inverse = {cipher: plain for plain, cipher in zip(ALPHABET, cipher_alphabet)}
    return "".join(inverse.get(char, char) for char in text.upper())


def _transposition_decrypt(ciphertext: str, cols: int, order: list[int]) -> str:
    if cols < 2 or len(ciphertext) % cols:
        raise ValueError("ciphertext length is not divisible by column count")
    rows = len(ciphertext) // cols
    matrix = [[""] * cols for _ in range(rows)]
    pos = 0
    for column in order:
        for row in range(rows):
            matrix[row][column] = ciphertext[pos]
            pos += 1
    return "".join("".join(row) for row in matrix).rstrip()


def _candidate(family: str, key: dict[str, Any], plaintext: str, evidence: str) -> dict[str, Any]:
    return {
        "family": family,
        "key": key,
        "plaintext": plaintext,
        "score": round(english_score(plaintext), 4),
        "evidence": evidence,
    }


def inspect_artifact(artifact: str) -> dict[str, Any]:
    stripped = artifact.strip()
    letters = sum(ch.isalpha() for ch in stripped)
    return {
        "length": len(stripped),
        "even_length": len(stripped) % 2 == 0,
        "hex_like": bool(stripped) and len(stripped) % 2 == 0 and all(ch in string.hexdigits for ch in stripped),
        "base64_like": bool(re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", stripped)) and len(stripped) % 4 == 0,
        "alphabetic_ratio": round(letters / max(1, len(stripped)), 4),
        "preserves_spaces": " " in stripped,
        "unique_symbols": len(set(stripped)),
    }


def analyze_artifact(
    artifact: str,
    *,
    key_material: str | None = None,
    max_candidates: int = 12,
    max_transposition_columns: int = 6,
) -> dict[str, Any]:
    """Try signature-compatible decoders and return ranked candidates."""
    artifact = artifact.strip("\n")
    inspection = inspect_artifact(artifact)
    candidates: list[dict[str, Any]] = []

    # Unlabelled key material is evidence, not an algorithm label. Try every
    # interpretation that fits its shape before the broad keyless probes.
    material = key_material.strip() if isinstance(key_material, str) else ""
    if material:
        try:
            numeric_key = int(material, 0)
        except ValueError:
            numeric_key = None
        if numeric_key is not None:
            if inspection["alphabetic_ratio"] >= 0.55:
                shift = numeric_key % 26
                if shift:
                    family = "rot13" if shift == 13 else "caesar"
                    candidates.append(_candidate(
                        family, {"shift": shift}, _caesar(artifact, -shift),
                        "unlabelled numeric key tested as an alphabet shift",
                    ))
            if inspection["hex_like"] and 0 <= numeric_key <= 255:
                raw = bytes.fromhex(artifact)
                decoded = bytes(byte ^ numeric_key for byte in raw)
                try:
                    plaintext = decoded.decode("utf-8")
                    candidates.append(_candidate(
                        "single_byte_xor",
                        {"key": numeric_key, "key_hex": f"0x{numeric_key:02x}"},
                        plaintext,
                        "unlabelled numeric key tested as a repeated byte key",
                    ))
                except UnicodeDecodeError:
                    pass

        letters_only = "".join(ch for ch in material.upper() if ch in ALPHABET)
        if inspection["alphabetic_ratio"] >= 0.55 and letters_only:
            if 2 <= len(letters_only) <= 32:
                candidates.append(_candidate(
                    "vigenere", {"keyword": letters_only},
                    _vigenere(artifact, letters_only, decrypt=True),
                    "unlabelled alphabetic key tested as a repeating shift sequence",
                ))
            if len(letters_only) == 26 and set(letters_only) == set(ALPHABET):
                candidates.append(_candidate(
                    "substitution", {"cipher_alphabet": letters_only},
                    _substitution_decrypt(artifact, letters_only),
                    "unlabelled 26-letter permutation tested as a substitution alphabet",
                ))

    # Identity is useful for distinguishing encoding from encryption.
    candidates.append(_candidate("identity", {}, artifact, "baseline for score comparison"))

    if inspection["preserves_spaces"]:
        reversed_words = _word_reverse(artifact)
        candidates.append(_candidate(
            "word_reverse", {}, reversed_words,
            "reversing each token preserves token boundaries",
        ))

    if inspection["alphabetic_ratio"] >= 0.55:
        for shift in range(1, 26):
            plaintext = _caesar(artifact, -shift)
            family = "rot13" if shift == 13 else "caesar"
            candidates.append(_candidate(
                family, {"shift": shift}, plaintext,
                "fixed alphabet shift tested without assuming a family label",
            ))

        # Exhaustive column order search is bounded at six columns (873 total
        # permutations across widths), suitable for the generated exercises.
        for cols in range(2, min(max_transposition_columns, len(artifact)) + 1):
            if len(artifact) % cols:
                continue
            for order in itertools.permutations(range(cols)):
                plaintext = _transposition_decrypt(artifact, cols, list(order))
                candidates.append(_candidate(
                    "columnar_transposition",
                    {"columns": cols, "order": list(order)},
                    plaintext,
                    "bounded grid-width and column-order search",
                ))

    if inspection["hex_like"]:
        raw = bytes.fromhex(artifact)
        try:
            decoded = raw.decode("utf-8")
            candidates.append(_candidate("hex", {}, decoded, "valid UTF-8 after hexadecimal decoding"))
        except UnicodeDecodeError:
            pass
        for key in range(256):
            decoded_bytes = bytes(byte ^ key for byte in raw)
            try:
                plaintext = decoded_bytes.decode("utf-8")
            except UnicodeDecodeError:
                continue
            candidates.append(_candidate(
                "single_byte_xor", {"key": key, "key_hex": f"0x{key:02x}"}, plaintext,
                "exhaustive single-byte key search over hex-decoded bytes",
            ))

    if inspection["base64_like"]:
        try:
            raw = base64.b64decode(artifact, validate=True)
            plaintext = raw.decode("utf-8")
            candidates.append(_candidate("base64", {}, plaintext, "strict Base64 decode succeeded"))
        except (ValueError, UnicodeDecodeError):
            pass

    # Collapse exact duplicate plaintexts while preserving the strongest score.
    best_by_text: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        key = normalize_plaintext(candidate["plaintext"])
        previous = best_by_text.get(key)
        if previous is None or candidate["score"] > previous["score"]:
            best_by_text[key] = candidate
    ranked = sorted(best_by_text.values(), key=lambda row: (-row["score"], row["family"]))
    return {
        "inspection": inspection,
        "candidates": ranked[:max_candidates],
        "attempted_candidates": len(candidates),
    }


def encrypt_candidate(plaintext: str, family: str, key: dict[str, Any] | None = None) -> str:
    key = key or {}
    if family == "identity":
        return plaintext
    if family == "caesar":
        return _caesar(plaintext, int(key["shift"]))
    if family == "rot13":
        return _caesar(plaintext, 13)
    if family == "single_byte_xor":
        value = int(key["key"])
        return "".join(f"{ord(ch) ^ value:02x}" for ch in plaintext)
    if family == "word_reverse":
        return _word_reverse(plaintext)
    if family == "columnar_transposition":
        cols = int(key["columns"])
        order = [int(value) for value in key["order"]]
        rows = math.ceil(len(plaintext) / cols)
        padded = plaintext + " " * (rows * cols - len(plaintext))
        matrix = [padded[row * cols:(row + 1) * cols] for row in range(rows)]
        return "".join(matrix[row][column] for column in order for row in range(rows))
    if family == "vigenere":
        return _vigenere(plaintext, str(key["keyword"]), decrypt=False)
    if family == "substitution":
        cipher_alphabet = str(key["cipher_alphabet"]).upper()
        if len(cipher_alphabet) != 26 or set(cipher_alphabet) != set(ALPHABET):
            raise ValueError("substitution key must be a 26-letter permutation")
        mapping = dict(zip(ALPHABET, cipher_alphabet))
        return "".join(mapping.get(char, char) for char in plaintext.upper())
    if family == "base64":
        return base64.b64encode(plaintext.encode()).decode()
    if family == "hex":
        return plaintext.encode().hex()
    raise ValueError(f"unsupported family: {family}")


def verify_candidate(artifact: str, plaintext: str, family: str, key: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        reproduced = encrypt_candidate(plaintext, family, key)
    except (KeyError, TypeError, ValueError) as exc:
        return {"verified": False, "error": str(exc)}
    return {
        "verified": reproduced == artifact,
        "reproduced_artifact": reproduced,
        "expected_artifact": artifact,
    }


def dumps_analysis(artifact: str, max_candidates: int = 12) -> str:
    return json.dumps(analyze_artifact(artifact, max_candidates=max_candidates), ensure_ascii=False, indent=2)
