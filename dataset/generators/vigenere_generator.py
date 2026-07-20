import random
import string

from .schema import make_record


ALPHABET = string.ascii_uppercase


def vigenere_encrypt(plaintext: str, key: str):
    ciphertext = ""
    key = key.upper()
    key_index = 0

    for char in plaintext.upper():
        if char in ALPHABET:
            shift = ALPHABET.index(key[key_index % len(key)])
            new_char = ALPHABET[(ALPHABET.index(char) + shift) % 26]
            ciphertext += new_char
            key_index += 1
        else:
            ciphertext += char

    return ciphertext


def _difficulty(key_len: int, text_len: int) -> str:
    if key_len <= 3 and text_len <= 40:
        return "easy"
    if key_len <= 5:
        return "medium"
    return "hard"


def _pick_vigenere_cot(key: str, plaintext: str) -> str:
    templates = [
        # Variant A: Kasiski / repeated pattern lead
        (
            f"Looking for repeated trigrams and bigrams in the ciphertext suggests a polyalphabetic "
            f"cipher. Measuring the distances between repeated sequences and factoring those distances "
            f"reveals a probable key length. Once the length is known, each key letter can be solved as a "
            f"separate Caesar cipher using frequency analysis. The recovered key is '{key}', giving plaintext: {plaintext}."
        ),
        # Variant B: index of coincidence
        (
            f"The ciphertext's index of coincidence is close to English (~0.065) only when tested against "
            f"specific key-length assumptions. Testing key lengths 1 through 8, the IC peaks at len={len(key)}. "
            f"Splitting the text into {len(key)} columns and solving each as a monoalphabetic shift yields "
            f"key '{key}' and plaintext: {plaintext}."
        ),
        # Variant C: concise
        (
            f"Vigenere cipher with key '{key}'. Decrypting column-by-column recovers: {plaintext}."
        ),
        # Variant D: brute-force short key
        (
            f"For a short keyword, brute-forcing all possible keys of length 3-5 against common English "
            f"trigrams is feasible. Scoring each candidate key for n-gram likelihood, '{key}' scores highest. "
            f"Full decryption: {plaintext}."
        ),
        # Variant E: educational / verbose
        (
            f"A Vigenere cipher uses a repeating keyword to shift each plaintext letter. Unlike a Caesar cipher, "
            f"the same plaintext letter can map to different ciphertext letters depending on its position. "
            f"To break it, I look for repeated ciphertext segments and measure their spacing. The greatest common "
            f"divisor of these spacings suggests key length {len(key)}. Treating every {len(key)}th character as a "
            f"Caesar cipher reveals the keyword '{key}', which decrypts to: {plaintext}."
        ),
    ]
    from .cot_style import weighted_choice
    return weighted_choice(templates)


def generate_vigenere_sample(plaintext: str):
    key_length = random.randint(3, 8)
    key = ''.join(random.choice(ALPHABET) for _ in range(key_length))
    cipher = vigenere_encrypt(plaintext, key)
    difficulty = _difficulty(key_length, len(plaintext))

    return make_record(
        artifact=cipher,
        family="vigenere",
        key={"keyword": key},
        category="vigenere_cipher",
        difficulty=difficulty,
        reasoning=_pick_vigenere_cot(key, plaintext),
        solution=plaintext,
        key_material=key,
    )
