import random
import string

from .schema import make_record


ALPHABET = string.ascii_uppercase


def rot13_encrypt(text: str) -> str:
    result = ""
    for char in text.upper():
        if char in ALPHABET:
            idx = (ALPHABET.index(char) + 13) % 26
            result += ALPHABET[idx]
        else:
            result += char
    return result


def _pick_rot13_cot(plaintext: str) -> str:
    templates = [
        # Variant A: standard
        (
            f"This is a ROT13 cipher, which shifts each letter by 13 positions. "
            f"Applying ROT13 again (or shifting by 13) decodes it: {plaintext}."
        ),
        # Variant B: Caesar-13 framing
        (
            f"The ciphertext preserves spaces and punctuation, suggesting a simple letter shift. "
            f"Testing shift values reveals that a shift of 13 (ROT13, its own inverse) produces coherent text: {plaintext}."
        ),
        # Variant C: concise
        (
            f"ROT13 is self-inverse: applying it twice returns the original. Decryption: {plaintext}."
        ),
        # Variant D: recognizing the tell
        (
            f"A quick scan shows the ciphertext contains common ROT13 giveaways — short words like 'GUR' or 'NA' "
            f"that map to 'THE' and 'AN' under a 13-shift. Confirming with a full ROT13 pass gives: {plaintext}."
        ),
        # Variant E: as a special case
        (
            f"ROT13 is a Caesar cipher with shift exactly 13, which means encryption and decryption are identical. "
            f"Shifting every letter 13 positions backward (or forward) recovers: {plaintext}."
        ),
    ]
    from .cot_style import weighted_choice
    return weighted_choice(templates)


def generate_rot13_sample(plaintext: str):
    cipher = rot13_encrypt(plaintext)

    return make_record(
        artifact=cipher,
        family="rot13",
        key={"shift": 13},
        category="rot13_cipher",
        difficulty="easy",
        reasoning=_pick_rot13_cot(plaintext),
        solution=plaintext,
    )
