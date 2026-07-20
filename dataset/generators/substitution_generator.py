import random
import string

from .schema import make_record


ALPHABET = string.ascii_uppercase


def substitution_encrypt(plaintext: str):
    shuffled = list(ALPHABET)
    random.shuffle(shuffled)
    key_map = dict(zip(ALPHABET, shuffled))
    ciphertext = ''.join(key_map.get(c, c) for c in plaintext.upper())
    return ciphertext, key_map


def _freq_str(cipher: str) -> str:
    """Build a frequency table summary string for the CoT."""
    from collections import Counter
    freq = Counter(c for c in cipher if c in ALPHABET).most_common(6)
    return ', '.join(f"'{l}'={n}" for l, n in freq)


def _pick_substitution_cot(cipher: str, key_map: dict, plaintext: str) -> str:
    top6 = _freq_str(cipher)
    templates = [
        # Variant A: full frequency analysis with top mappings
        (
            f"This is a monoalphabetic substitution cipher — each plaintext letter is replaced by a "
            f"fixed ciphertext letter. I begin by counting letter frequencies: the top ciphertext letters "
            f"are {top6}. In English, E > T > A > O > I > N is the typical order. Mapping the most frequent "
            f"ciphertext letter to 'E', the second to 'T', and so on gives a partial key. "
            f"Testing this key on short words and common patterns (e.g., 'THE', 'AND', 'FOR') confirms the "
            f"mapping. Decrypting fully yields: {plaintext}."
        ),
        # Variant B: short-word cribs first
        (
            f"Short one-, two-, and three-letter words in the ciphertext are powerful cribs. "
            f"A single-letter word is almost always 'A' or 'I'. Two-letter words often map to 'OF', 'TO', 'IN', 'IS'. "
            f"Using these constraints, I build a partial substitution table and propagate it through longer words. "
            f"The complete key recovers plaintext: {plaintext}."
        ),
        # Variant C: concise
        (
            f"Substitution cipher. Partial key from frequency + crib words, then full decryption: {plaintext}."
        ),
        # Variant D: pattern word attack
        (
            f"I search for words with unique letter-pattern signatures (e.g., 'THAT' = ABAC pattern). "
            f"Matching these patterns against a dictionary of common words reveals several ciphertext-to-plaintext "
            f"letter pairs. Building the key from these pairs and filling gaps with frequency estimates produces: {plaintext}."
        ),
        # Variant E: combined approach
        (
            f"Step 1 — Frequency count: {top6} are the most common ciphertext letters. "
            f"Step 2 — Hypothesize E, T, A mappings. Step 3 — Check short-word consistency. "
            f"Step 4 — Use pattern words (e.g., double letters like 'LL' or 'SS') to confirm. "
            f"Step 5 — Decrypt: {plaintext}."
        ),
    ]
    from .cot_style import weighted_choice
    return weighted_choice(templates)


def generate_substitution_sample(plaintext: str):
    cipher, key_map = substitution_encrypt(plaintext)
    ln = len(plaintext)
    difficulty = "medium" if ln <= 40 else "hard"

    return make_record(
        artifact=cipher,
        family="substitution",
        key={"plaintext_to_cipher": key_map},
        category="substitution_cipher",
        difficulty=difficulty,
        reasoning=_pick_substitution_cot(cipher, key_map, plaintext),
        solution=plaintext,
        key_material="".join(key_map[letter] for letter in ALPHABET),
    )
