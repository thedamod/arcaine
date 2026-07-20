"""Authenticated OpenAI-compatible serving for the Arcaine Qwen3.5 LoRA.

Deploy:
    modal deploy modal_app/serve.py

The deployment exposes vLLM's OpenAI-compatible API. Use model ``arcaine``.
"""

from __future__ import annotations

import os
import subprocess

import modal

APP_NAME = "arcaine-openai-api"
BASE_MODEL = "Qwen/Qwen3.5-4B"
ADAPTER_REPO = "aether5896/arcaine-agentic-lora-v8"
SERVED_MODEL = "arcaine"
PORT = 8000

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git")
    .pip_install("vllm>=0.15.0", "huggingface_hub[hf_transfer]")
)

app = modal.App(APP_NAME, image=image)
model_cache = modal.Volume.from_name("qwen35-model-cache", create_if_missing=True)


@app.function(
    gpu="L4",
    timeout=86_400,
    scaledown_window=600,
    volumes={"/root/.cache/huggingface": model_cache},
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("arcaine-api"),
    ],
)
@modal.concurrent(max_inputs=16)
@modal.web_server(PORT, startup_timeout=1_200)
def serve() -> None:
    api_key = os.environ.get("ARCAINE_API_KEY")
    if not api_key:
        raise RuntimeError("Modal secret arcaine-api must define ARCAINE_API_KEY")

    command = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--model",
        BASE_MODEL,
        "--served-model-name",
        "arcaine-base",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "16384",
        "--gpu-memory-utilization",
        "0.90",
        "--max-num-seqs",
        "4",
        # Interactive Pi sessions benefit more from fast cold starts than
        # maximum batch throughput. Avoid multi-minute torch.compile startup.
        "--enforce-eager",
        "--enable-lora",
        "--max-lora-rank",
        "16",
        "--lora-modules",
        f"{SERVED_MODEL}={ADAPTER_REPO}",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_xml",
        "--api-key",
        api_key,
        "--trust-remote-code",
    ]
    subprocess.Popen(command)
