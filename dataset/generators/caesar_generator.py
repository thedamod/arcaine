import random
import string

from .schema import make_record


ALPHABET = string.ascii_uppercase


def caesar_encrypt(text: str, shift: int) -> str:
    result = ""
    for char in text.upper():
        if char in ALPHABET:
            idx = (ALPHABET.index(char) + shift) % 26
            result += ALPHABET[idx]
        else:
            result += char
    return result


def difficulty_from_text(text):
    ln = len(text)
    if ln <= 30:
        return "easy"
    elif ln <= 80:
        return "medium"
    return "hard"


def _pick_caesar_cot(shift: int, plaintext: str) -> str:
    """Return a randomly selected CoT variant for Caesar cipher."""
    templates = [
        # Variant A: brute-force emphasis
        (
            f"This is a Caesar cipher — every letter is shifted by a fixed amount. "
            f"Brute-forcing all 25 possible shifts and scoring each candidate for English word frequency "
            f"reveals that a backward shift of {shift} produces coherent text: {plaintext}."
        ),
        # Variant B: frequency analysis lead
        (
            f"The ciphertext preserves word lengths and spaces, suggesting a monoalphabetic substitution. "
            f"Frequency analysis points to a Caesar cipher: the most common letter likely maps to 'E'. "
            f"Testing this hypothesis, a backward shift of {shift} yields readable plaintext: {plaintext}."
        ),
        # Variant C: concise
        (
            f"Caesar cipher, shift = {shift}. Decrypting by shifting backward {shift} positions gives: {plaintext}."
        ),
        # Variant D: verbose pedagogical
        (
            f"A Caesar cipher replaces each plaintext letter with another letter a fixed number of positions "
            f"down the alphabet. Since the shift is unknown, I systematically test each value from 1 to 25. "
            f"Scoring candidates for English n-gram likelihood, shift {shift} stands out. "
            f"Applying a backward shift of {shift} to every character recovers: {plaintext}."
        ),
        # Variant E: pattern / short-word heuristic
        (
            f"Short two- and three-letter words in the ciphertext provide strong cribs. "
            f"Testing which shift turns the most common short ciphertext words into English words "
            f"(e.g., 'THE', 'AND', 'FOR') points to a backward shift of {shift}. "
            f"Full decryption: {plaintext}."
        ),
    ]
    from .cot_style import weighted_choice
    return weighted_choice(templates)


def generate_caesar_sample(plaintext: str):
    shift = random.randint(1, 25)
    cipher = caesar_encrypt(plaintext, shift)
    difficulty = difficulty_from_text(plaintext)

    return make_record(
        artifact=cipher,
        family="caesar",
        key={"shift": shift},
        category="caesar_cipher",
        difficulty=difficulty,
        reasoning=_pick_caesar_cot(shift, plaintext),
        solution=plaintext,
    )
