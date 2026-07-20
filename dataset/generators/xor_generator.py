import random

from .schema import make_record


def xor_encrypt(text: str, key: int):
    return ''.join(format(ord(c) ^ key, '02x') for c in text)


def _pick_xor_cot(key: int, plaintext: str) -> str:
    templates = [
        # Variant A: brute-force scoring
        (
            f"A single-byte XOR cipher encrypts each ASCII character with the same byte. "
            f"I brute-force all 256 possible keys, decoding the hex string for each candidate and "
            f"scoring the output for printable-ASCII density and English letter frequency. "
            f"Key 0x{key:02x} ({key}) produces the only coherent English text: {plaintext}."
        ),
        # Variant B: frequency on hex-decoded bytes
        (
            f"Converting the hex string to raw bytes, the most frequent byte in the ciphertext "
            f"likely corresponds to space (0x20) or 'E' (0x45) in the plaintext. "
            f"XORing the top byte with 0x20 gives candidate key 0x{key:02x} ({key}). "
            f"Applying this to the full ciphertext yields: {plaintext}."
        ),
        # Variant C: concise
        (
            f"Single-byte XOR key = 0x{key:02x} ({key}). XOR every byte to decode: {plaintext}."
        ),
        # Variant D: entropy / structure detection
        (
            f"The hex string has even length, consistent with byte-wise encoding. "
            f"Testing single-byte XOR, I compute the Shannon entropy of each decoded candidate. "
            f"The minimum-entropy decryption uses key 0x{key:02x} ({key}), giving: {plaintext}."
        ),
        # Variant E: common-prefix heuristic
        (
            f"Many English sentences start with common trigrams like 'THE' or common words like 'I'. "
            f"Assuming the first three plaintext bytes are 'THE' gives candidate key bytes; only one "
            f"is consistent across all ciphertext bytes. That key is 0x{key:02x} ({key}), producing: {plaintext}."
        ),
    ]
    from .cot_style import weighted_choice
    return weighted_choice(templates)


def generate_xor_sample(plaintext: str):
    key = random.randint(1, 255)
    cipher = xor_encrypt(plaintext, key)
    ln = len(plaintext)
    difficulty = "easy" if ln <= 30 else ("medium" if ln <= 80 else "hard")

    return make_record(
        artifact=cipher,
        family="single_byte_xor",
        key={"key": key, "key_hex": f"0x{key:02x}"},
        category="xor_cipher",
        difficulty=difficulty,
        reasoning=_pick_xor_cot(key, plaintext),
        solution=plaintext,
    )
