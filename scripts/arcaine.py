#!/usr/bin/env python3
"""Small CLI for the deployed Arcaine OpenAI-compatible Modal API.

Examples:
    ./arcaine "A RIAOE TETAAO.A LEAENB TB AEOCAASSF,IESENSF""KLD T  SN R T JFT M NTFMTHM E O S ROUOIET RN"
    echo "ciphertext" | ./arcaine

Configuration is read from flags first, then environment variables.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from crypto_agent.tools import TOOL_SCHEMAS, run_tool

DEFAULT_BASE_URL = "https://aether6430--arcaine-openai-api-serve.modal.run/v1"
DEFAULT_MODEL = "arcaine"
FORBIDDEN_PYTHON = re.compile(
    r"(?:import\s+(?:os|subprocess|socket|shutil|pathlib)|"
    r"from\s+(?:os|subprocess|socket|shutil|pathlib)|"
    r"__import__|\bopen\s*\(|/checkpoints|/home/|environ)",
    re.IGNORECASE,
)

try:
    from dataset.generators.schema import blind_problem
except Exception:  # pragma: no cover - fallback keeps the CLI usable standalone.
    def blind_problem(artifact: str, key_material: str | None = None) -> str:
        prompt = (
            "Recover the plaintext from the artifact below. The transformation "
            "family, key, and number of layers are intentionally omitted. Infer "
            "them from observable structure, keep competing hypotheses until "
            "evidence separates them, and verify the result by reproducing the "
            "artifact.\n\nArtifact:\n"
            f"{artifact}"
        )
        if key_material:
            prompt += f"\n\nAuxiliary key material (unlabelled):\n{key_material}"
        return prompt


def config_value(names: tuple[str, ...], fallback: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return fallback


def normalize_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def read_artifact(parts: list[str]) -> str:
    if parts:
        return " ".join(parts)
    if not sys.stdin.isatty():
        return sys.stdin.read().strip("\n")
    raise SystemExit("Pass a ciphertext argument, or pipe one on stdin. Try: ./arcaine --help")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="arcaine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Call the deployed Arcaine cryptanalysis model and print its reply.",
        epilog=textwrap.dedent(
            """
            Environment:
              ARCAINE_API_KEY or OPENAI_API_KEY   API key
              ARCAINE_BASE_URL or OPENAI_BASE_URL Modal/OpenAI-compatible base URL
              ARCAINE_MODEL or OPENAI_MODEL       Model name (default: arcaine)

            """
        ).strip(),
    )
    parser.add_argument("artifact", nargs="*", help="ciphertext/artifact to send to the model")
    parser.add_argument("--base-url", help=f"OpenAI-compatible base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--api-key", help="API key; prefer ARCAINE_API_KEY/OPENAI_API_KEY for shell history safety")
    parser.add_argument("--model", help=f"model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--system", default="You are Arcaine, a careful cryptanalysis assistant. Return the best recovered plaintext and concise verification.")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--raw", action="store_true", help="send the argument exactly as the user message, without the blind cryptanalysis wrapper")
    parser.add_argument("--thinking", action="store_true", help="enable Qwen thinking in chat template kwargs; default is off for concise CLI output")
    parser.add_argument("--no-tools", action="store_true", help="make one plain completion instead of running the agent tool loop")
    parser.add_argument("--allow-python", action="store_true", help="allow the model to run bounded local Python analysis")
    parser.add_argument("--max-rounds", type=int, default=6, help="maximum model/tool rounds (default: 6)")
    parser.add_argument("--json", action="store_true", help="print the raw JSON response instead of just message content")
    return parser.parse_args()


def chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    payload["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{normalize_base_url(base_url)}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"API request failed: HTTP {exc.code}\n{detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"API request failed: {exc.reason}") from exc


def main() -> None:
    args = parse_args()
    api_key = args.api_key or config_value(("ARCAINE_API_KEY", "OPENAI_API_KEY"))
    if not api_key:
        raise SystemExit(
            "Missing API key. Set ARCAINE_API_KEY/OPENAI_API_KEY, pass --api-key, "
            "or pass --api-key."
        )

    base_url = args.base_url or config_value(("ARCAINE_BASE_URL", "OPENAI_BASE_URL"), DEFAULT_BASE_URL)
    model = args.model or config_value(("ARCAINE_MODEL", "OPENAI_MODEL"), DEFAULT_MODEL)
    artifact = read_artifact(args.artifact)
    user_message = artifact if args.raw else blind_problem(artifact)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": user_message},
    ]
    tools = None if args.no_tools else TOOL_SCHEMAS

    for _ in range(max(1, args.max_rounds) if tools else 1):
        response = chat_completion(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            enable_thinking=args.thinking,
            tools=tools,
        )

        try:
            message = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise SystemExit(f"Unexpected API response:\n{json.dumps(response, indent=2)}") from exc

        if args.json and not message.get("tool_calls"):
            print(json.dumps(response, indent=2, ensure_ascii=False))
            return

        messages.append(message)
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            print(message.get("content") or "")
            return

        for call in tool_calls:
            function = call.get("function") or {}
            name = str(function.get("name", ""))
            try:
                arguments = json.loads(function.get("arguments", "{}"))
            except json.JSONDecodeError as exc:
                result = {"error": f"invalid tool arguments: {exc}"}
            else:
                if not isinstance(arguments, dict):
                    result = {"error": "tool arguments must be an object"}
                elif name == "python" and not args.allow_python:
                    result = {"error": "Python analysis is disabled; rerun with --allow-python"}
                elif name == "python" and FORBIDDEN_PYTHON.search(str(arguments.get("code", ""))):
                    result = {"error": "generated code rejected by Arcaine CLI safety policy"}
                else:
                    result = json.loads(run_tool(name, arguments))
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "name": name,
                "content": json.dumps(result, ensure_ascii=False),
            })

    raise SystemExit(f"agent exceeded --max-rounds={args.max_rounds} without a final response")


if __name__ == "__main__":
    main()
