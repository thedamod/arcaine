"""Modal SFT training for Qwen3.5 LoRA adapters.

Prereqs:
  modal setup
  modal volume create qwen35-sft-checkpoints
  modal secret create huggingface HF_TOKEN=hf_...

Run:
  modal run modal_app/train.py

Run a fresh 1,000-update V3 agent job with explicit repos:
  modal run modal_app/train.py \
    --dataset-repo your-user/your-private-agentic-dataset-v3 \
    --model-repo your-user/your-agentic-lora-v3 \
    --max-steps 1000 \
    --no-resume

The dataset should expose OpenAI/Qwen-style ``messages`` plus ``tools`` in train and test splits.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parents[1]

APP_NAME = "qwen35-cipher-sft"
VOLUME_NAME = "qwen35-sft-checkpoints"
# V3 intentionally uses a fresh checkpoint directory. It trains structured
# assistant/tool-call turns with assistant-only loss, so old optimizer and
# scheduler state is incompatible.
CHECKPOINT_DIR = Path("/checkpoints/qwen35-cryptanalysis-agent-v8")
METRICS_PATH = CHECKPOINT_DIR / "metrics.jsonl"

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_DATASET_REPO = "aether5896/arcaine_dataset"
DEFAULT_MODEL_REPO = "aether5896/arcaine-agentic-lora-v8"
DEFAULT_EXCLUDED_CATEGORIES = ""

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "build-essential")
    .pip_install(
        "torch",
        "datasets",
        "accelerate",
        "peft",
        "bitsandbytes",
        "huggingface_hub",
        "sentencepiece",
        "protobuf",
        "einops",
        "pillow",
        "torchvision",
    )
    # Qwen3.5 requires Transformers v5. Force-reinstall Unsloth/Unsloth Zoo so
    # Modal doesn't resolve an older combo that still expects Transformers v4/TRl internals.
    .run_commands(
        "pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo 'transformers>=5.2.0' trl"
    )
    .add_local_dir(str(PROJECT_ROOT / "crypto_agent"), remote_path="/root/crypto_agent", copy=True)
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
model_cache = modal.Volume.from_name("qwen35-model-cache", create_if_missing=True)


def _latest_checkpoint(output_dir: Path) -> str | None:
    if not output_dir.exists():
        return None
    checkpoints = [p for p in output_dir.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return None
    return str(max(checkpoints, key=lambda p: int(p.name.rsplit("-", 1)[-1])))


@app.function(
    gpu="L4",
    timeout=43_200,
    volumes={
        "/checkpoints": volume,
        "/root/.cache/huggingface": model_cache,
    },
    secrets=[modal.Secret.from_name("huggingface")],
)
def train(
    dataset_repo: str = DEFAULT_DATASET_REPO,
    model_repo: str = DEFAULT_MODEL_REPO,
    base_model: str = DEFAULT_BASE_MODEL,
    dataset_split: str = "train",
    eval_dataset_split: str = "test",
    excluded_categories: str = DEFAULT_EXCLUDED_CATEGORIES,
    max_seq_length: int = 4096,
    num_train_epochs: float = 1,
    max_steps: int = -1,
    max_eval_samples: int = 64,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 5e-5,
    warmup_steps: int = 50,
    resume: bool = True,
) -> None:
    # Disable Unsloth's external statistics/status probe. The probe can time
    # out even when model files are accessible and previously killed healthy
    # provisioning attempts before weights were downloaded.
    os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    # Unsloth should be imported before transformers/trl so its patches apply cleanly.
    from unsloth import FastLanguageModel
    import json
    import time

    import torch
    from datasets import load_dataset
    from huggingface_hub import login
    from transformers import DataCollatorForSeq2Seq, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "Missing HF token. Create it with: modal secret create huggingface HF_TOKEN=hf_..."
        )
    login(token=hf_token)

    if "YOUR_HF_USERNAME" in dataset_repo or "YOUR_HF_USERNAME" in model_repo:
        raise RuntimeError(
            "Set DATASET_REPO and MODEL_REPO env vars, or edit DEFAULT_DATASET_REPO/DEFAULT_MODEL_REPO."
        )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    volume.reload()

    print(f"Loading base model: {base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        token=hf_token,
    )

    model_cache.commit()

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    blocked = {category.strip() for category in excluded_categories.split(",") if category.strip()}

    def category_of(example: dict) -> str | None:
        category = example.get("category")
        if category:
            return str(category)
        metadata = example.get("metadata")
        if isinstance(metadata, dict) and metadata.get("category"):
            return str(metadata["category"])
        return None

    def to_agentic_messages(example: dict) -> dict:
        """Preserve all assistant/tool turns and structured tool calls."""
        if example.get("messages"):
            messages = example["messages"]
            if not any(message.get("role") == "assistant" for message in messages):
                raise ValueError("conversation has no assistant message")
            result = {"messages": messages}
            if example.get("tools"):
                result["tools"] = example["tools"]
            return result

        if example.get("prompt") and example.get("completion"):
            return {"messages": [*example["prompt"], *example["completion"]]}

        problem = str(example.get("problem") or "").strip()
        cot = str(example.get("chain_of_thought") or "").strip()
        solution = str(example.get("solution") or "").strip()
        if not problem or not cot or not solution:
            raise ValueError("raw row has an empty problem, chain_of_thought, or solution")
        return {"messages": [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": f"<think>\n{cot}\n</think>\n{solution}"},
        ]}

    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    template_owner = tokenizer if hasattr(tokenizer, "apply_chat_template") else text_tokenizer
    assistant_start_ids = text_tokenizer(
        "<|im_start|>assistant\n", add_special_tokens=False
    )["input_ids"]
    turn_end_ids = text_tokenizer("<|im_end|>\n", add_special_tokens=False)["input_ids"]

    def find_subsequence(sequence: list[int], pattern: list[int], start: int) -> int:
        if not pattern:
            return -1
        stop = len(sequence) - len(pattern) + 1
        for index in range(start, max(start, stop)):
            if sequence[index:index + len(pattern)] == pattern:
                return index
        return -1

    def to_masked_features(example: dict) -> dict:
        normalized = to_agentic_messages(example)
        rendered = template_owner.apply_chat_template(
            normalized["messages"],
            tools=normalized.get("tools"),
            tokenize=False,
            add_generation_prompt=False,
        )
        encoded = text_tokenizer(
            rendered,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_length,
        )
        input_ids = encoded["input_ids"]
        labels = [-100] * len(input_ids)
        cursor = 0
        assistant_turns = 0
        while True:
            marker = find_subsequence(input_ids, assistant_start_ids, cursor)
            if marker < 0:
                break
            content_start = marker + len(assistant_start_ids)
            turn_end = find_subsequence(input_ids, turn_end_ids, content_start)
            if turn_end < 0:
                turn_end = len(input_ids)
                next_cursor = len(input_ids)
            else:
                turn_end += len(turn_end_ids)
                next_cursor = turn_end
            labels[content_start:turn_end] = input_ids[content_start:turn_end]
            assistant_turns += 1
            cursor = next_cursor
        if assistant_turns == 0 or not any(label != -100 for label in labels):
            raise ValueError("failed to locate assistant spans in rendered Qwen conversation")
        return {
            "input_ids": input_ids,
            "attention_mask": encoded.get("attention_mask", [1] * len(input_ids)),
            "labels": labels,
        }

    def prepare_split(split, label: str):
        before = len(split)
        if blocked:
            split = split.filter(lambda row: category_of(row) not in blocked)
        removed = before - len(split)
        if removed:
            print(f"Dropped {removed} blocked-category rows from {label}: {before} -> {len(split)}")
        elif blocked and not ({"category", "metadata"} & set(split.column_names)):
            print(f"WARNING: {label} has no category metadata; category blocklist could not be applied")
        return split.map(
            to_masked_features,
            remove_columns=split.column_names,
            desc=f"Rendering and masking {label} agentic conversations",
        )

    print(f"Loading dataset: {dataset_repo} [{dataset_split}]")
    ds = prepare_split(
        load_dataset(dataset_repo, split=dataset_split, token=hf_token),
        "train",
    )

    eval_ds = None
    if eval_dataset_split:
        try:
            print(f"Loading evaluation split: {dataset_repo} [{eval_dataset_split}]")
            raw_eval_ds = load_dataset(dataset_repo, split=eval_dataset_split, token=hf_token)
            if max_eval_samples > 0 and len(raw_eval_ds) > max_eval_samples:
                raw_eval_ds = raw_eval_ds.select(range(max_eval_samples))
                print(f"Using {max_eval_samples} deterministic evaluation samples")
            eval_ds = prepare_split(raw_eval_ds, "eval")
        except Exception as exc:
            print(f"WARNING: evaluation split unavailable ({exc.__class__.__name__}: {exc})")

    class MetricsAndVolumeCallback(TrainerCallback):
        def _write_event(self, event: dict) -> None:
            event.setdefault("time", time.time())
            event.setdefault("time_iso", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            event.setdefault("metrics_file", str(METRICS_PATH))
            with METRICS_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

        def on_train_begin(self, args, state, control, **kwargs):
            self._write_event({
                "event": "train_begin",
                "global_step": state.global_step,
                "epoch": state.epoch,
                "max_steps": state.max_steps,
                "num_train_epochs": args.num_train_epochs,
            })
            volume.commit()

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            self._write_event({
                "event": "log",
                "global_step": state.global_step,
                "epoch": state.epoch,
                **logs,
            })
            volume.commit()

        def on_save(self, args, state, control, **kwargs):
            print("Committing Modal volume after checkpoint save...")
            self._write_event({
                "event": "save",
                "global_step": state.global_step,
                "epoch": state.epoch,
            })
            volume.commit()

        def on_train_end(self, args, state, control, **kwargs):
            self._write_event({
                "event": "train_end",
                "global_step": state.global_step,
                "epoch": state.epoch,
            })
            volume.commit()

    has_eval = eval_ds is not None and len(eval_ds) > 0
    training_args = SFTConfig(
        output_dir=str(CHECKPOINT_DIR),
        max_seq_length=max_seq_length,
        eos_token="<|im_end|>",
        packing=False,
        assistant_only_loss=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=4,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_steps=50,
        save_total_limit=3,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=250 if has_eval else None,
        prediction_loss_only=True,
        # Keep frequent interruption-safe checkpoints while evaluating less
        # often. Transformers otherwise requires save_steps to be a multiple
        # of eval_steps when loading the best checkpoint automatically.
        load_best_model_at_end=False,
        metric_for_best_model=None,
        greater_is_better=None,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        report_to="none",
        seed=3407,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        eval_dataset=eval_ds if has_eval else None,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=text_tokenizer,
            model=model,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        ),
        args=training_args,
        callbacks=[MetricsAndVolumeCallback()],
    )

    # Fail before the first optimizer update if TRL/Qwen template integration
    # did not create assistant-only labels. This prevents another expensive run
    # that accidentally learns random ciphertext and tool-result tokens.
    objective_batch = next(iter(trainer.get_train_dataloader()))
    labels = objective_batch.get("labels")
    if labels is None:
        raise RuntimeError("SFT preflight failed: training batch has no labels")
    masked_tokens = int((labels == -100).sum().item())
    trained_tokens = int((labels != -100).sum().item())
    if masked_tokens == 0 or trained_tokens == 0:
        raise RuntimeError(
            "SFT preflight failed: assistant-only mask is missing or masks every token "
            f"(masked={masked_tokens}, trained={trained_tokens})"
        )
    print(
        "Assistant-only objective preflight passed: "
        f"masked_tokens={masked_tokens}, trained_tokens={trained_tokens}"
    )
    del objective_batch, labels

    resume_from = _latest_checkpoint(CHECKPOINT_DIR) if resume else None
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
    else:
        print("No checkpoint found; starting fresh.")

    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(str(CHECKPOINT_DIR / "final_adapter"))
    tokenizer.save_pretrained(str(CHECKPOINT_DIR / "final_adapter"))
    volume.commit()

    print(f"Pushing LoRA adapter to Hugging Face: {model_repo}")
    model.push_to_hub(model_repo, token=hf_token, private=True)
    tokenizer.push_to_hub(model_repo, token=hf_token, private=True)
    print("Done.")


@app.function(
    volumes={"/checkpoints": volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def publish_adapter(model_repo: str = DEFAULT_MODEL_REPO) -> None:
    """Upload the completed adapter from the persistent Modal volume to the Hub."""
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("Missing Hugging Face token")

    volume.reload()
    adapter_dir = CHECKPOINT_DIR / "final_adapter"
    required = ("adapter_config.json", "adapter_model.safetensors", "tokenizer.json")
    missing = [name for name in required if not (adapter_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Final adapter is incomplete at {adapter_dir}: missing {missing}")

    api = HfApi(token=token)
    api.create_repo(repo_id=model_repo, repo_type="model", private=True, exist_ok=True)
    info = api.upload_folder(
        repo_id=model_repo,
        repo_type="model",
        folder_path=str(adapter_dir),
        commit_message="Upload completed Arcaine LoRA adapter",
    )
    print(f"Published {adapter_dir} to {info.repo_url}")


@app.function(
    gpu="L4",
    timeout=3_600,
    volumes={
        "/checkpoints": volume,
        "/root/.cache/huggingface": model_cache,
    },
    secrets=[modal.Secret.from_name("huggingface")],
)
def test_checkpoint(
    dataset_repo: str = DEFAULT_DATASET_REPO,
    base_model: str = DEFAULT_BASE_MODEL,
    test_split: str = "test",
    start_index: int = 64,
    num_samples: int = 3,
    max_seq_length: int = 4096,
    max_new_tokens: int = 768,
    max_assistant_turns: int = 3,
) -> None:
    """Run a small autonomous tool-use test against the latest committed adapter."""
    os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    import json
    import re
    import time

    import torch
    from datasets import load_dataset
    from unsloth import FastLanguageModel

    from crypto_agent.tools import run_tool

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("Missing Hugging Face token")

    volume.reload()
    checkpoint = _latest_checkpoint(CHECKPOINT_DIR)
    if not checkpoint:
        raise RuntimeError(f"No committed checkpoint found in {CHECKPOINT_DIR}")
    checkpoint_step = int(Path(checkpoint).name.rsplit("-", 1)[-1])
    print(f"Testing checkpoint: {checkpoint}")

    # Unsloth recognizes a PEFT checkpoint directory and loads its declared
    # base model plus adapter. Keeping base_model as a fallback makes failures
    # explicit rather than silently testing an unadapted model.
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint,
            max_seq_length=max_seq_length,
            dtype=torch.bfloat16,
            load_in_4bit=False,
            token=token,
        )
    except Exception as adapter_exc:
        print(f"Direct adapter load failed; loading base + PEFT adapter: {adapter_exc}")
        from peft import PeftModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_length,
            dtype=torch.bfloat16,
            load_in_4bit=False,
            token=token,
        )
        model = PeftModel.from_pretrained(model, checkpoint)

    FastLanguageModel.for_inference(model)
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    # Agent loops must preserve the newest tool result and generation prompt
    # when a long analysis output exceeds the context window.
    text_tokenizer.truncation_side = "left"
    template_owner = tokenizer if hasattr(tokenizer, "apply_chat_template") else text_tokenizer

    dataset = load_dataset(dataset_repo, split=test_split, token=token)
    stop_index = min(len(dataset), start_index + max(1, min(num_samples, 10)))
    if start_index < 0 or start_index >= stop_index:
        raise ValueError(f"Invalid test range {start_index}:{stop_index} for {len(dataset)} rows")

    json_tool_pattern = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    qwen_tool_pattern = re.compile(
        r"<tool_call>\s*<function=([^>]+)>\s*(.*?)</function>\s*</tool_call>",
        re.DOTALL,
    )
    qwen_parameter_pattern = re.compile(
        r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
        re.DOTALL,
    )
    plaintext_pattern = re.compile(
        r"\*\*Plaintext:\*\*\s*(.*?)(?:\n\s*\n\*\*Verification:|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    forbidden_code = re.compile(
        r"(?:import\s+(?:os|subprocess|socket|shutil|pathlib)|from\s+(?:os|subprocess|socket|shutil|pathlib)"
        r"|__import__|\bopen\s*\(|/checkpoints|environ)",
        re.IGNORECASE,
    )

    def normalize(value: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", value.lower()))

    def expected_plaintext(messages: list[dict]) -> str:
        for message in reversed(messages):
            if message.get("role") != "assistant":
                continue
            match = plaintext_pattern.search(str(message.get("content") or ""))
            if match:
                return match.group(1).strip()
        return ""

    def parse_tool_call(text: str) -> tuple[str, dict, str] | None:
        json_match = json_tool_pattern.search(text)
        if json_match:
            try:
                payload = json.loads(json_match.group(1))
                name = str(payload["name"])
                arguments = payload.get("arguments") or {}
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    return None
                return name, arguments, text[:json_match.start()].strip()
            except (KeyError, TypeError, json.JSONDecodeError):
                return None

        # Qwen3.5 emits its native XML function-call representation rather
        # than JSON inside <tool_call> tags.
        qwen_match = qwen_tool_pattern.search(text)
        if not qwen_match:
            return None
        name = qwen_match.group(1).strip()
        arguments = {}
        for parameter in qwen_parameter_pattern.finditer(qwen_match.group(2)):
            key = parameter.group(1).strip()
            raw_value = parameter.group(2).strip()
            if raw_value.lower() in {"none", "null"}:
                continue
            if raw_value.isdigit():
                arguments[key] = int(raw_value)
            else:
                arguments[key] = raw_value
        return name, arguments, text[:qwen_match.start()].strip()

    def execute_checked(name: str, arguments: dict) -> str:
        if name == "python":
            code = str(arguments.get("code", ""))
            if forbidden_code.search(code):
                return json.dumps({"error": "generated code rejected by checkpoint-test safety policy"})
        return run_tool(name, arguments)

    def generate(messages: list[dict], tools: list[dict]) -> str:
        rendered = template_owner.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = text_tokenizer(
            rendered,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        prompt_length = encoded["input_ids"].shape[-1]
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=text_tokenizer.eos_token_id,
                pad_token_id=text_tokenizer.pad_token_id or text_tokenizer.eos_token_id,
            )
        generated = output[0, prompt_length:]
        return text_tokenizer.decode(generated, skip_special_tokens=False).replace("<|im_end|>", "").strip()

    started = time.time()
    results = []
    for index in range(start_index, stop_index):
        row = dataset[index]
        source_messages = row.get("messages") or []
        tools = row.get("tools") or []
        first_assistant = next(
            (position for position, message in enumerate(source_messages) if message.get("role") == "assistant"),
            None,
        )
        if first_assistant is None:
            raise RuntimeError(f"Row {index} has no assistant turn")
        messages = [dict(message) for message in source_messages[:first_assistant]]
        expected = expected_plaintext(source_messages)
        generated_turns = []
        tool_names = []
        tool_successes = 0
        final_response = ""

        print(f"TEST sample={index} category={(row.get('metadata') or {}).get('category')}")
        for turn_index in range(max_assistant_turns):
            response = generate(messages, tools)
            generated_turns.append(response)
            parsed = parse_tool_call(response)
            if parsed is None:
                final_response = response
                break

            name, arguments, reasoning = parsed
            tool_names.append(name)
            call_id = f"checkpoint-test-{index}-{turn_index + 1}"
            # Qwen's generation prompt itself contributes the opening
            # <think> tag, so it is absent from newly decoded tokens. Restore
            # it before replaying this turn through the chat template.
            replay_reasoning = reasoning
            if "</think>" in replay_reasoning and "<think>" not in replay_reasoning:
                replay_reasoning = "<think>\n" + replay_reasoning
            messages.append({
                "role": "assistant",
                "content": replay_reasoning,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }],
            })
            tool_result = execute_checked(name, arguments)
            try:
                parsed_result = json.loads(tool_result)
                if not parsed_result.get("error") and not parsed_result.get("timed_out"):
                    tool_successes += 1
            except json.JSONDecodeError:
                pass
            messages.append({
                "role": "tool",
                "name": name,
                "tool_call_id": call_id,
                "content": tool_result,
            })

        expected_normalized = normalize(expected)
        generated_normalized = normalize(final_response)
        exact_plaintext = bool(expected_normalized) and expected_normalized in generated_normalized
        verified_signal = any(
            "verified" in turn.lower() and "true" in turn.lower()
            for turn in generated_turns
        )
        result = {
            "sample_index": index,
            "category": (row.get("metadata") or {}).get("category"),
            "tool_names": tool_names,
            "valid_tool_calls": len(tool_names),
            "successful_tool_calls": tool_successes,
            "reached_final_response": bool(final_response),
            "exact_plaintext_in_response": exact_plaintext,
            "verification_claim": verified_signal,
            "expected_plaintext": expected,
            "final_response": final_response,
            "generated_turns": generated_turns,
        }
        results.append(result)
        print(json.dumps({key: value for key, value in result.items() if key != "generated_turns"}, ensure_ascii=False))

    summary = {
        "checkpoint": checkpoint,
        "checkpoint_step": checkpoint_step,
        "dataset": dataset_repo,
        "split": test_split,
        "sample_range": [start_index, stop_index],
        "samples": len(results),
        "exact_plaintext_matches": sum(item["exact_plaintext_in_response"] for item in results),
        "final_responses": sum(item["reached_final_response"] for item in results),
        "valid_tool_calls": sum(item["valid_tool_calls"] for item in results),
        "successful_tool_calls": sum(item["successful_tool_calls"] for item in results),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    report = {"summary": summary, "results": results}
    report_path = CHECKPOINT_DIR / f"checkpoint_test_{checkpoint_step}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    volume.commit()
    print("CHECKPOINT_TEST_SUMMARY " + json.dumps(summary, ensure_ascii=False))
    print(f"Saved report: {report_path}")


@app.function(secrets=[modal.Secret.from_name("huggingface")])
def inspect_dataset(dataset_repo: str = DEFAULT_DATASET_REPO) -> None:
    """Validate Hub files, splits, and the agentic schema without provisioning a GPU."""
    import json
    from datasets import get_dataset_split_names, load_dataset
    from huggingface_hub import list_repo_files

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    files = list_repo_files(dataset_repo, repo_type="dataset", token=token)
    splits = get_dataset_split_names(dataset_repo, token=token)
    print(json.dumps({"repo": dataset_repo, "files": files, "splits": splits}, ensure_ascii=False))
    for split in splits:
        ds = load_dataset(dataset_repo, split=split, token=token)
        if not len(ds):
            raise RuntimeError(f"dataset split is empty: {split}")
        row = ds[0]
        messages = row.get("messages") or []
        tools = row.get("tools") or []
        if not messages or not any(message.get("role") == "assistant" for message in messages):
            raise RuntimeError(f"split {split} is not agentic messages data")
        if not tools:
            raise RuntimeError(f"split {split} has no tools column/data")
        print(json.dumps({
            "split": split,
            "rows": len(ds),
            "columns": ds.column_names,
            "roles": [message.get("role") for message in messages],
            "tool_names": [tool.get("function", {}).get("name") for tool in tools],
        }, ensure_ascii=False))


@app.function(volumes={"/checkpoints": volume})
def tail_metrics(lines: int = 20) -> None:
    volume.reload()
    if not METRICS_PATH.exists():
        print(f"No metrics file yet: {METRICS_PATH}")
        return
    rows = METRICS_PATH.read_text(encoding="utf-8").splitlines()
    for row in rows[-lines:]:
        print(row)


@app.local_entrypoint()
def main(
    dataset_repo: str = DEFAULT_DATASET_REPO,
    model_repo: str = DEFAULT_MODEL_REPO,
    base_model: str = DEFAULT_BASE_MODEL,
    dataset_split: str = "train",
    eval_dataset_split: str = "test",
    excluded_categories: str = DEFAULT_EXCLUDED_CATEGORIES,
    max_seq_length: int = 4096,
    num_train_epochs: float = 1,
    max_steps: int = -1,
    max_eval_samples: int = 64,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 5e-5,
    warmup_steps: int = 50,
    resume: bool = True,
    background: bool = False,
) -> None:
    kwargs = dict(
        dataset_repo=dataset_repo,
        model_repo=model_repo,
        base_model=base_model,
        dataset_split=dataset_split,
        eval_dataset_split=eval_dataset_split,
        excluded_categories=excluded_categories,
        max_seq_length=max_seq_length,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        max_eval_samples=max_eval_samples,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        resume=resume,
    )
    if background:
        call = train.spawn(**kwargs)
        print(f"Spawned Modal training job: {call.object_id}")
        print("If your Modal workspace supports spawned calls, it should continue in the background.")
        print("If it disappears, run without --background under nohup/tmux instead.")
        print("Monitor logs in the Modal dashboard, or run:")
        print("  modal run modal_app/train.py::tail_metrics --lines 30")
    else:
        train.remote(**kwargs)
