"""Tool contracts and local reference implementations for the cryptanalysis agent.

The Python runner is intended for a disposable container. It is a timeout and
resource boundary, not a hardened security sandbox. Never expose it directly to
untrusted users on a host containing secrets.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "python",
            "description": (
                "Run a Python 3 script in an isolated working directory. Use it for decoding, "
                "frequency analysis, bounded key search, and round-trip verification. Print "
                "compact JSON or text. Network access is not guaranteed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete Python source code to execute.",
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 30,
                        "default": 10,
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search cryptography references for transformation signatures, attack methods, "
                "formats, or implementation guidance. Search from observed properties; do not "
                "assume the cipher family in the query. A deployment may back this with a local "
                "knowledge base or a web-search provider."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                },
                "required": ["query"],
            },
        },
    },
]

_REFERENCE = [
    {
        "title": "Fixed alphabet shifts",
        "keywords": "alphabetic spaces punctuation fixed shift frequency brute force 26 caesar rot13",
        "content": "If spaces survive and every letter appears shifted by one constant amount, test all 26 shifts and rank candidates with language statistics. Shift 13 is self-inverse but should be inferred, not assumed.",
    },
    {
        "title": "Single-byte XOR signatures",
        "keywords": "hex hexadecimal even length xor byte printable brute force 256",
        "content": "Even-length hexadecimal often represents bytes. For a repeated one-byte XOR key, test 0..255, reject invalid text, rank printable candidates, then re-XOR the plaintext to verify exact equality.",
    },
    {
        "title": "Transposition signatures",
        "keywords": "spaces preserved letter frequencies scrambled grid columns permutation transposition",
        "content": "Transposition preserves the symbol histogram but disrupts adjacency. For short exercises, enumerate plausible grid widths and column orders, score reconstructed text, and verify by applying the same write/read convention.",
    },
    {
        "title": "Word-level reversal",
        "keywords": "spaces tokens words reversed backwards boundaries",
        "content": "If token boundaries remain and reversing characters inside each token produces common words, test word-wise reversal. Verify by applying the operation again.",
    },
    {
        "title": "Base64 and hexadecimal are encodings",
        "keywords": "base64 padding equals alphabet hex encoding decode layer",
        "content": "Base64 and hex are encodings rather than encryption. Strictly validate syntax, decode one layer, inspect the result, and continue only when the decoded bytes have a plausible structure.",
    },
    {
        "title": "Repeating-key polyalphabetic analysis",
        "keywords": "repeated sequences index coincidence columns key length vigenere polyalphabetic",
        "content": "Estimate repeating-key length with repeated-sequence distances and index of coincidence. Solve each key position as a shift, then compare several lengths and verify by re-encryption. Short texts can be underdetermined.",
    },
    {
        "title": "Monoalphabetic substitution caution",
        "keywords": "substitution frequency word pattern mapping annealing underdetermined",
        "content": "Short substitution cryptograms are often underdetermined. Use word patterns, tetragram scoring, and stochastic key search; report uncertainty rather than inventing a unique plaintext. Longer text is substantially more reliable.",
    },
    {
        "title": "Layered artifact workflow",
        "keywords": "multiple layers chain recursive decode beam search verify",
        "content": "For layered artifacts, identify the outer representation first, retain several candidates, and recursively probe each result. Avoid committing to a chain until every inverse step can be round-trip verified.",
    },
]


def search(query: str, top_k: int = 5) -> dict[str, Any]:
    # Optional deployment hook. The endpoint should accept
    # {"query": str, "top_k": int} and return a JSON object. Keeping the local
    # fallback makes dataset generation reproducible and network-free.
    endpoint = os.environ.get("ARCAINE_SEARCH_URL")
    if endpoint:
        request = urllib.request.Request(
            endpoint,
            data=json.dumps({"query": query, "top_k": top_k}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return {"query": query, "results": payload.get("results", payload), "backend": endpoint}
        except Exception as exc:
            return {"query": query, "results": [], "backend": endpoint, "error": str(exc)}

    tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    ranked = []
    for item in _REFERENCE:
        haystack = f"{item['title']} {item['keywords']} {item['content']}".lower()
        score = sum(1 for token in tokens if token in haystack)
        ranked.append((score, item))
    ranked.sort(key=lambda pair: (-pair[0], pair[1]["title"]))
    results = [dict(item, score=score) for score, item in ranked[:max(1, min(top_k, 10))] if score > 0]
    return {"query": query, "results": results, "backend": "local_crypto_reference"}


def execute_python(code: str, timeout_seconds: int = 10) -> dict[str, Any]:
    timeout_seconds = max(1, min(int(timeout_seconds), 30))
    project_root = Path(__file__).resolve().parents[1]
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(project_root),
        "PYTHONIOENCODING": "utf-8",
    }
    with tempfile.TemporaryDirectory(prefix="arcaine-tool-") as tmp:
        script = Path(tmp) / "main.py"
        script.write_text(code, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, str(script)],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            return {
                "exit_code": completed.returncode,
                "stdout": completed.stdout[-20_000:],
                "stderr": completed.stderr[-8_000:],
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "exit_code": None,
                "stdout": (exc.stdout or "")[-20_000:] if isinstance(exc.stdout, str) else "",
                "stderr": (exc.stderr or "")[-8_000:] if isinstance(exc.stderr, str) else "",
                "timed_out": True,
            }


def run_tool(name: str, arguments: dict[str, Any]) -> str:
    if name == "python":
        result = execute_python(
            code=str(arguments.get("code", "")),
            timeout_seconds=int(arguments.get("timeout_seconds", 10)),
        )
    elif name == "search":
        result = search(
            query=str(arguments.get("query", "")),
            top_k=int(arguments.get("top_k", 5)),
        )
    else:
        result = {"error": f"unknown tool: {name}"}
    return json.dumps(result, ensure_ascii=False)
