"""Build verified Qwen/OpenAI-style tool-use trajectories from blind records."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from dataset.generators.schema import assert_blind_prompt

from .solver import analyze_artifact, inspect_artifact, normalize_plaintext, verify_candidate
from .tools import TOOL_SCHEMAS, run_tool

SYSTEM_PROMPT = ""

_SUPPORTED = {
    "caesar": {"caesar", "rot13"},
    "rot13": {"rot13"},
    "single_byte_xor": {"single_byte_xor"},
    "word_reverse": {"word_reverse"},
    "columnar_transposition": {"columnar_transposition"},
    "base64": {"base64"},
    "hex": {"hex"},
    "vigenere": {"vigenere"},
    "substitution": {"substitution"},
}


def _tool_call(call_id: str, name: str, arguments: dict[str, Any], thinking: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": f"<think>\n{thinking.strip()}\n</think>",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                # Qwen's Transformers chat template iterates arguments as a
                # mapping. OpenAI servers serialize this mapping on output.
                "arguments": arguments,
            },
        }],
    }


def _tool_result(
    call_id: str,
    name: str,
    arguments: dict[str, Any],
    *,
    content: str | None = None,
) -> dict[str, Any]:
    return {
        "role": "tool",
        "name": name,
        "tool_call_id": call_id,
        "content": content if content is not None else run_tool(name, arguments),
    }


def _python_success(payload: dict[str, Any]) -> str:
    """Represent deterministic in-process execution in the Python tool envelope."""
    return json.dumps({
        "exit_code": 0,
        "stdout": json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        "stderr": "",
        "timed_out": False,
    }, ensure_ascii=False)


def _search_query(inspection: dict[str, Any]) -> str:
    features = []
    if inspection["hex_like"]:
        features.extend(["even-length hexadecimal", "byte-level repeated-key identification"])
    if inspection["base64_like"]:
        features.extend(["restricted alphabet with padding", "outer encoding layer"])
    if inspection["preserves_spaces"]:
        features.extend(["spaces and token boundaries preserved", "alphabetic cryptogram"])
    if inspection["alphabetic_ratio"] > 0.7:
        features.extend(["letter-frequency analysis", "fixed shift versus transposition"])
    return "cryptanalysis methods for artifact with " + ", ".join(features or ["unknown printable structure"])


def _pick_verified_candidate(record: dict[str, Any], max_candidates: int) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = str(record["artifact"])
    expected = normalize_plaintext(str(record["solution"]))
    oracle_family = str(record["oracle"]["family"])
    allowed = _SUPPORTED.get(oracle_family)
    if not allowed:
        raise ValueError(f"unsupported oracle family for verified tool trace: {oracle_family}")

    analysis = analyze_artifact(
        artifact,
        key_material=record.get("key_material"),
        max_candidates=max_candidates,
    )
    for candidate in analysis["candidates"]:
        if candidate["family"] not in allowed:
            continue
        if normalize_plaintext(candidate["plaintext"]) != expected:
            continue
        verification = verify_candidate(
            artifact,
            candidate["plaintext"],
            candidate["family"],
            candidate["key"],
        )
        if verification.get("verified"):
            return analysis, candidate
    raise ValueError("blind candidate search did not recover and verify the labelled plaintext")


def build_tool_trajectory(
    record: dict[str, Any],
    *,
    search_ratio: float = 0.35,
    max_candidates: int = 8,
) -> dict[str, Any]:
    """Convert one oracle record into an executable, verified tool trajectory."""
    assert_blind_prompt(record)
    analysis, candidate = _pick_verified_candidate(record, max_candidates=max_candidates)
    artifact = str(record["artifact"])
    key_material = record.get("key_material")
    inspection = analysis["inspection"]
    digest = hashlib.sha256(str(record.get("id", artifact)).encode()).digest()
    use_search = digest[0] / 255 < search_ratio

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": str(record["problem"])},
    ]

    if use_search:
        search_args = {"query": _search_query(inspection), "top_k": 4}
        messages.append(_tool_call(
            "search-1",
            "search",
            search_args,
            (
                f"The artifact is {inspection['length']} characters long. Its observable flags are "
                f"hex_like={inspection['hex_like']}, base64_like={inspection['base64_like']}, "
                f"alphabetic_ratio={inspection['alphabetic_ratio']}, and "
                f"preserves_spaces={inspection['preserves_spaces']}. These features support several "
                "hypotheses, so I should search by signature rather than naming a cipher prematurely."
            ),
        ))
        messages.append(_tool_result("search-1", "search", search_args))

    analysis_code = (
        "import json\n"
        "from crypto_agent.solver import analyze_artifact\n"
        f"artifact = {artifact!r}\n"
        f"key_material = {key_material!r}\n"
        f"print(json.dumps(analyze_artifact(artifact, key_material=key_material, max_candidates={max_candidates}), "
        "ensure_ascii=False, indent=2))\n"
    )
    python_args = {"code": analysis_code, "timeout_seconds": 20}
    messages.append(_tool_call(
        "python-1",
        "python",
        python_args,
        (
            "I need an assumption-light comparison. I will run format checks, fixed-shift trials, "
            "byte-key search when syntax permits it, word-boundary transforms, and a bounded grid "
            "search, then rank all readable candidates."
        ),
    ))
    messages.append(_tool_result(
        "python-1", "python", python_args, content=_python_success(analysis)
    ))

    original_trace = str(record.get("chain_of_thought") or "").strip()
    trace_summary = original_trace if len(original_trace) <= 1200 else original_trace[:1197] + "..."
    verification_code = (
        "import json\n"
        "from crypto_agent.solver import verify_candidate\n"
        f"artifact = {artifact!r}\n"
        f"plaintext = {candidate['plaintext']!r}\n"
        f"family = {candidate['family']!r}\n"
        f"key = {candidate['key']!r}\n"
        "print(json.dumps(verify_candidate(artifact, plaintext, family, key), "
        "ensure_ascii=False, indent=2))\n"
    )
    verify_args = {"code": verification_code, "timeout_seconds": 10}
    messages.append(_tool_call(
        "python-2",
        "python",
        verify_args,
        (
            f"The blind search recovered a strong {candidate['family']} candidate with score "
            f"{candidate['score']} and parameters {candidate['key']}. The detailed reasoning trace is: "
            f"{trace_summary} I should not trust readability alone; I will apply the proposed forward "
            "transformation and require exact equality with the supplied artifact."
        ),
    ))
    verification = verify_candidate(
        artifact, candidate["plaintext"], candidate["family"], candidate["key"]
    )
    messages.append(_tool_result(
        "python-2", "python", verify_args, content=_python_success(verification)
    ))

    key_text = json.dumps(candidate["key"], ensure_ascii=False, sort_keys=True)
    messages.append({
        "role": "assistant",
        "content": (
            "<think>\nThe round-trip check returned verified=true, so the candidate is supported by "
            "both language evidence and an exact mechanical reproduction. I can now report it without "
            "claiming certainty from appearance alone.\n</think>\n"
            f"**Identification:** `{candidate['family']}`\n\n"
            f"**Recovered key/parameters:** `{key_text}`\n\n"
            f"**Plaintext:**\n{candidate['plaintext']}\n\n"
            "**Verification:** Reapplying the recovered transformation reproduced the artifact exactly.\n\n"
            "**Confidence:** High."
        ),
    })

    return {
        "messages": messages,
        "tools": TOOL_SCHEMAS,
        "metadata": {
            "id": record.get("id"),
            "category": record.get("category"),
            "difficulty": record.get("difficulty"),
            "source": record.get("source"),
            "oracle_family": record["oracle"]["family"],
            "reasoning_trace": original_trace,
            "solution_group": hashlib.sha256(normalize_plaintext(str(record["solution"])).encode()).hexdigest(),
            "tool_calls": 3 if use_search else 2,
            "verified": True,
        },
    }


def validate_trajectory(row: dict[str, Any]) -> None:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 6:
        raise ValueError("trajectory is too short")
    if messages[0].get("role") != "user":
        raise ValueError("trajectory must start with user message")
    calls = {}
    results = set()
    for message in messages:
        for call in message.get("tool_calls") or []:
            call_id = call.get("id")
            calls[call_id] = call["function"]["name"]
            arguments = call["function"]["arguments"]
            if not isinstance(arguments, dict):
                raise ValueError("tool-call arguments must be a mapping for Qwen's chat template")
        if message.get("role") == "tool":
            results.add(message.get("tool_call_id"))
    if set(calls) != results:
        raise ValueError(f"tool call/result mismatch: calls={set(calls)}, results={results}")
    if not row.get("metadata", {}).get("verified"):
        raise ValueError("trajectory is not verified")
