#!/usr/bin/env python3
"""Bridge Pi's Arcaine tools to the bounded local crypto tool implementations."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from crypto_agent.tools import run_tool  # noqa: E402

# The model was trained to call analyze_artifact/verify_candidate through this
# tool. Reject obvious host-access primitives; this remains a bounded local
# runner, not a hardened sandbox for untrusted public input.
FORBIDDEN = re.compile(
    r"(?:import\s+(?:os|subprocess|socket|shutil|pathlib)|"
    r"from\s+(?:os|subprocess|socket|shutil|pathlib)|"
    r"__import__|\bopen\s*\(|/checkpoints|/home/|environ)",
    re.IGNORECASE,
)


def main() -> None:
    request = json.load(sys.stdin)
    name = str(request.get("name", ""))
    arguments = request.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise SystemExit("arguments must be an object")
    if name == "python" and FORBIDDEN.search(str(arguments.get("code", ""))):
        print(json.dumps({"error": "generated code rejected by Arcaine Pi safety policy"}))
        return
    print(run_tool(name, arguments))


if __name__ == "__main__":
    main()
