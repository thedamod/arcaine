"""Global CoT style configuration for all generators.

Set via set_cot_style() before dataset generation to bias template selection.
"""

import random

_STYLE = "balanced"
_WEIGHTS = None  # populated on demand


def set_cot_style(style: str = "balanced"):
    """Set the global CoT template selection style.

    Styles:
      balanced   — uniform random selection across all variants (default)
      verbose    — bias toward longer, pedagogical templates (weights ~3x for verbose variants)
      concise    — bias toward short, direct templates (weights ~3x for concise variants)
    """
    global _STYLE, _WEIGHTS
    _STYLE = style
    _WEIGHTS = None  # force recomputation


def get_cot_style() -> str:
    return _STYLE


def weighted_choice(templates: list) -> str:
    """Select a template according to the current style.

    Template order differs by generator, so verbosity is inferred from rendered
    template length instead of assuming fixed indices.
    """
    if not templates:
        raise ValueError("weighted_choice requires at least one template")

    lengths = [len(t) for t in templates]
    min_len = min(lengths)
    max_len = max(lengths)
    span = max(1, max_len - min_len)

    weights = []
    for length in lengths:
        normalized = (length - min_len) / span
        if _STYLE == "verbose":
            weights.append(1.0 + 4.0 * normalized)
        elif _STYLE == "concise":
            weights.append(1.0 + 4.0 * (1.0 - normalized))
        else:
            weights.append(1.0)

    total = sum(weights)
    r = random.uniform(0, total)
    cum = 0.0
    for weight, template in zip(weights, templates):
        cum += weight
        if r <= cum:
            return template
    return templates[-1]
