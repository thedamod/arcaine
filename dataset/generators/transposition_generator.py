import random
import math

from .schema import make_record


def transposition_encrypt(text: str):
    if len(text) < 2:
        return text, 1, [0]
    cols = random.randint(2, min(6, len(text)))
    rows = (len(text) + cols - 1) // cols
    padded = text + ' ' * (rows * cols - len(text))

    matrix = [list(padded[i*cols:(i+1)*cols]) for i in range(rows)]
    col_order = list(range(cols))
    random.shuffle(col_order)

    cipher = ''
    for c in col_order:
        for r in range(rows):
            cipher += matrix[r][c]

    return cipher, cols, col_order


def _pick_transposition_cot(cols: int, col_order: list, plaintext: str, cipher: str) -> str:
    ln = len(plaintext)
    # Compute a likely divisor hint
    divisors = [d for d in range(2, min(8, ln+1)) if ln % d == 0 or (ln % d) > (d//2)]
    divisor_hint = random.choice(divisors) if divisors else cols

    templates = [
        # Variant A: systematic column testing
        (
            f"This looks like a columnar transposition — the plaintext was written row-wise into a grid "
            f"and read out column-by-column in a permuted order. The ciphertext length is {ln}, so I test "
            f"divisors near {divisor_hint} as possible column counts. Trying {cols} columns and every permutation "
            f"of column reading order, the arrangement {col_order} produces readable English: {plaintext}."
        ),
        # Variant B: anagram / word fragment heuristic
        (
            f"I look for recognizable word fragments or common letter pairs (e.g., 'TH', 'HE', 'IN') "
            f"at regular intervals in the ciphertext. Their spacing suggests a grid width of {cols} columns. "
            f"Reassembling the text into a {cols}-column grid and permuting read order yields {col_order}, "
            f"which gives: {plaintext}."
        ),
        # Variant C: concise
        (
            f"Columnar transposition, {cols} columns, read order {col_order}. Grid reassembly gives: {plaintext}."
        ),
        # Variant D: brute-force permutation search
        (
            f"For a ciphertext of length {ln}, possible column counts are divisors of {ln} or {ln+1}. "
            f"Testing each column count and all permutations of reading order, I score each reconstruction for "
            f"English word and n-gram frequency. The best-scoring reconstruction uses {cols} columns in order "
            f"{col_order}, producing: {plaintext}."
        ),
        # Variant E: padding-aware
        (
            f"Columnar transposition often pads the final row to make the rectangle complete. The ciphertext length "
            f"{ln} suggests {cols} columns with {math.ceil(ln/cols)} rows. Writing the ciphertext down columns "
            f"in order {col_order} and reading across rows recovers the plaintext: {plaintext}."
        ),
    ]
    from .cot_style import weighted_choice
    return weighted_choice(templates)


def generate_transposition_sample(plaintext: str):
    cipher, cols, col_order = transposition_encrypt(plaintext)
    ln = len(plaintext)
    difficulty = "easy" if cols <= 3 and ln <= 35 else ("medium" if cols <= 4 else "hard")

    return make_record(
        artifact=cipher,
        family="columnar_transposition",
        key={"columns": cols, "order": col_order},
        category="transposition_cipher",
        difficulty=difficulty,
        reasoning=_pick_transposition_cot(cols, col_order, plaintext, cipher),
        solution=plaintext,
    )
