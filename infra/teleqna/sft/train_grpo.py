#!/usr/bin/env python3
"""GRPO over teleqna thinking traces.

The difference that matters versus the earlier GRPO script on the tools track:
thinking is ON. Rewarding `ANSWER: X` with no reasoning would be the same
1-bit letter channel that SFT and four DPO recipes already exhausted (net
+4..6 rows, fix-rate flat against training-pair similarity). With thinking on,
the unit of credit is a whole sampled reasoning path, so the policy is trained
on *how it reaches* the letter — the pass@k -> pass@1 gap that the any-branch
oracle (83.85% vs 74.71% voted) says is worth ~9 points.

Reward = the benchmark's own scorer:
  correctness  1.0  parsed letter == gold, using run_baseline.py's exact
                    STRICT -> LOOSE -> BARE cascade, last match wins
  format       0.05 a well-formed 'ANSWER: X' line exists at all
Nothing else is rewarded. Length is not penalised directly — truncated
rollouts already score zero because no ANSWER line survives, which is the
honest signal.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
from pathlib import Path

# venv-grpo carries trl/peft/transformers but no vllm; venv carries vllm 0.11
# against the same container torch. Append (never prepend) so this venv's own
# transformers/trl keep precedence and only vllm resolves from the other tree.
_EXTRA = os.environ.get("EXTRA_SITE")
if _EXTRA:
    # numpy is the one package that has to come from the OTHER tree. venv-grpo
    # ships numpy 2.3.2 and no numba; venv ships the matched pair numpy 2.2.6 +
    # numba 0.61.2, and numba refuses to load against anything above 2.2. With
    # EXTRA_SITE merely appended, venv-grpo's 2.3.2 wins and vllm dies on import
    # with "Numba needs NumPy 2.2 or less".
    #
    # So resolve numpy — and only numpy — from the other tree first, then put
    # the path back at the end. Once numpy 2.2.6 is in sys.modules every later
    # import gets it, while trl/peft/transformers still resolve venv-grpo-first
    # as intended. This has to happen before datasets/peft/trl are imported
    # below, because those pull numpy in themselves.
    sys.path.insert(0, _EXTRA)
    import numpy as _np
    sys.path.pop(0)
    sys.path.append(_EXTRA)
    print(f"pinned numpy {_np.__version__} from EXTRA_SITE for numba/vllm",
          flush=True)

from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

# ported verbatim from run_baseline.py / eval_dev.py
STRICT = re.compile(r"(?i)^ANSWER\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)", re.MULTILINE)
LOOSE = re.compile(r"(?i)ANSWER\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)")
BARE = re.compile(r"^\s*([A-Ea-e])(?:[).:,\s]|$)")


def completion_text(completion) -> str:
    if isinstance(completion, list):
        return " ".join(str(m.get("content", "")) for m in completion)
    return str(completion)


def parse(text: str, n: int) -> str:
    m = STRICT.findall(text or "") or LOOSE.findall(text or "") or BARE.findall(text or "")
    if not m:
        return ""
    got = m[-1].strip().rstrip(".").upper()
    return got if got in {chr(65 + i) for i in range(n)} else ""


def reward_correct(completions, answer, n_choices, **kwargs) -> list[float]:
    return [1.0 if parse(completion_text(c), n) == gold else 0.0
            for c, gold, n in zip(completions, answer, n_choices)]


def reward_format(completions, n_choices, **kwargs) -> list[float]:
    return [0.05 if parse(completion_text(c), n) else 0.0
            for c, n in zip(completions, n_choices)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model-path", required=True,
                    help="base weights to start from. To build on CPT, merge "
                         "that adapter first (merge_adapter.py) and point here "
                         "at the merged directory — GRPOTrainer attaches its "
                         "own fresh LoRA and will not stack onto a loaded one")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    # LoRA takes ~10x the full-finetune LR (Schulman et al., "LoRA Without
    # Regret"): the 1/r scaling makes the optimum roughly rank-independent,
    # and their RL reference runs use 1e-5 LoRA against 1e-6 full FT.
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    # Reference GRPO recipes now run beta=0. We keep a whisper of KL because
    # this project has a measured drift failure (SFT erased 114-155 dev rows);
    # with LoRA the reference pass is free (disable the adapter), so the
    # insurance costs nothing. Raise it if dev-1000 shows drift, drop to 0 if
    # the policy stops exploring.
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--generation-batch-size", type=int, default=None)
    ap.add_argument("--per-device-batch", type=int, default=8)
    # LoRA tolerates large batches poorly (same source); keep the effective
    # batch under ~32 completions.
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-prompt-length", type=int, default=1024)
    ap.add_argument("--max-completion-length", type=int, default=1536,
                    help="thinking averages ~968 tokens on this benchmark")
    ap.add_argument("--temperature", type=float, default=1.0)
    # RL extracts ~1 bit per episode, so it needs far less adapter capacity
    # than SFT: the same source recommends rank 1-32 for RL against 256 for
    # post-training-scale SFT.
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--vllm-gpu-mem", type=float, default=0.3)
    ap.add_argument("--save-steps", type=int, default=20)
    args = ap.parse_args()

    try:
        import vllm  # noqa: F401
    except ImportError:
        raise SystemExit(
            "vllm is not importable. venv-grpo does not ship it; point "
            "EXTRA_SITE at a site-packages tree that has vllm (venv has "
            "0.11.0 against the same container torch). Refusing to fall back "
            "to HF generate silently — rollouts would be several times slower "
            "and the run would look merely slow rather than misconfigured.")

    records = [json.loads(l) for l in Path(args.dataset).read_text(encoding="utf-8").splitlines()]

    # Thinking is rendered into the prompt here rather than requested through
    # GRPOConfig. trl 0.23.1 has no chat_template_kwargs field, and the earlier
    # version of this script refused to run without it — correctly, because
    # rewarding a bare `ANSWER: X` is the one-bit letter channel that SFT and
    # four DPO recipes already exhausted. The fix is to stop asking the config:
    # trl's maybe_apply_chat_template leaves a `prompt` that is already a plain
    # string untouched (it only templates conversational examples), so applying
    # the template ourselves with enable_thinking=True is exact and needs no
    # upgrade over a network that is currently unusable.
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)

    def render(msgs) -> str:
        return _tok.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True,
                                        enable_thinking=True)

    probe_on = render([{"role": "user", "content": "probe"}])
    probe_off = _tok.apply_chat_template([{"role": "user", "content": "probe"}],
                                          tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False)
    if probe_on == probe_off:
        raise SystemExit(
            "this tokenizer renders enable_thinking=True and False identically, "
            "so thinking cannot be guaranteed in rollouts — refusing to train a "
            "letter-only policy by accident")
    print(f"thinking prompts verified: enabled/disabled renderings differ by "
          f"{len(probe_off) - len(probe_on)} chars", flush=True)

    for r in records:
        if isinstance(r.get("prompt"), list):
            r["prompt"] = render(r["prompt"])

    dataset = Dataset.from_list(records)
    print(f"dataset: {len(dataset)} prompts (pre-rendered, thinking on)", flush=True)

    cfg = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        num_generations=args.num_generations,
        generation_batch_size=args.generation_batch_size or args.num_generations,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_steps=args.save_steps,
        save_total_limit=50,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_mem,
        log_completions=True,
        report_to=[],
        seed=42,
    )
    valid = {f.name for f in dataclasses.fields(GRPOConfig)}
    # Thinking is already baked into the prompt strings above. On a trl that
    # does have the field, set it too — it is a no-op for prompts that are
    # already rendered, and keeps the run correct if the prompts ever go back
    # to conversational form.
    if "chat_template_kwargs" in valid:
        cfg["chat_template_kwargs"] = {"enable_thinking": True}
    if "mask_truncated_completions" in valid:
        cfg["mask_truncated_completions"] = True
    for k in [k for k in cfg if k not in valid]:
        print(f"WARN: GRPOConfig lacks {k}; dropping", flush=True)
        cfg.pop(k)

    trainer = GRPOTrainer(
        model=args.model_path,
        reward_funcs=[reward_correct, reward_format],
        args=GRPOConfig(**cfg),
        train_dataset=dataset,
        peft_config=LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]),
    )
    trainer.train()
    trainer.save_model(args.output_dir + "/final")
    print(f"adapter saved -> {args.output_dir}/final", flush=True)


if __name__ == "__main__":
    main()
