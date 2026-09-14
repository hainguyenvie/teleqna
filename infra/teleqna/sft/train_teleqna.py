#!/usr/bin/env python3
"""Completion-only LoRA SFT for the teleqna specialist, on train_v3.jsonl.

Adapted from the proven v13 trainer (same base model, same loss masking, same
LearnGuard no-learning tripwire). Differences, each deliberate:

- Single task: rows are {"prompt", "completion"} with completion "ANSWER: X".
  The prompt is already byte-identical to run_baseline.py's harness prompt.
- Chat template is applied with enable_thinking=False — the exact flag the
  eval harness sends to vLLM. Training and eval must tokenize identically.
- LoRA rank comes from $LORA_R: the plan is twin probe runs (r=16, r=64);
  if r=64 shows no dev advantage the task is not capacity-bound.
- Loss is per-example-normalised (inherited from v13). With uniform
  ~5-token completions this equals plain token averaging, but it keeps the
  trainer correct if longer completions are ever mixed in.
- eval set = a deterministic 1,000-row slice of the training file, used for
  loss curves only. Model selection happens on the real dev-1000 afterwards,
  scored by eval_dev.py with the harness contract — loss on synthetic rows
  does not decide anything.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

ROOT = Path(os.environ.get("PROJ_ROOT", "/workspace/teleqna-sft"))
BASE = os.environ["BASE_MODEL"]
DATA = Path(os.environ.get("TRAIN_DATA", ROOT / "data" / "train_v3.jsonl"))
LORA_R = int(os.environ.get("LORA_R", "64"))
MAXLEN = int(os.environ.get("MAXLEN", "1024"))
BATCH = int(os.environ.get("BATCH", "16"))
ACCUM = int(os.environ.get("ACCUM", "2"))
EPOCHS = float(os.environ.get("EPOCHS", "2"))
LR = float(os.environ.get("LR", "1e-4"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "200"))
SAVE_LIMIT = int(os.environ.get("SAVE_LIMIT", "3"))
RUN = os.environ.get("RUN_NAME", f"r{LORA_R}")
set_seed(42)

ADAPTER = ROOT / "models" / f"teleqna-{RUN}-adapter"
CHECKPOINTS = ROOT / "models" / f"teleqna-{RUN}-checkpoints"
if ADAPTER.exists() and os.environ.get("ALLOW_OVERWRITE") != "1":
    raise SystemExit(f"{ADAPTER} exists; refusing overwrite")

tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def encode(rows: list[dict], name: str) -> list[dict]:
    out, lens = [], []
    for i, row in enumerate(rows, 1):
        messages = [{"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["completion"]}]
        prompt = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        full = tokenizer.apply_chat_template(
            messages, tokenize=False, enable_thinking=False)
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
        if f_ids[: len(p_ids)] != p_ids:
            raise SystemExit(f"{name}:{i} prompt not a prefix of full")
        if len(f_ids) > MAXLEN:
            raise SystemExit(f"{name}:{i} len={len(f_ids)} > {MAXLEN}")
        labels = list(f_ids)
        labels[: len(p_ids)] = [-100] * len(p_ids)
        out.append({"input_ids": f_ids, "labels": labels})
        lens.append(len(f_ids))
    lens.sort()
    print(f"{name}: n={len(out)} tokens min/med/p99/max="
          f"{lens[0]}/{lens[len(lens)//2]}/{lens[int(len(lens)*.99)]}/{lens[-1]}",
          flush=True)
    return out


rows = [json.loads(l) for l in DATA.open(encoding="utf-8")]
# Deterministic 1,000-row eval slice by sample_id hash — not position.
def h(r):
    return hashlib.sha256(("evalsplit|" + r["sample_id"]).encode()).digest()[0]
eval_rows = [r for r in rows if h(r) < 6][:1000]
eval_ids = {r["sample_id"] for r in eval_rows}
train_rows = [r for r in rows if r["sample_id"] not in eval_ids]
print(f"train={len(train_rows)} eval={len(eval_rows)}", flush=True)

train_enc = encode(train_rows, "train")
eval_enc = encode(eval_rows, "eval")


@dataclass
class Collator:
    pad_id: int

    def __call__(self, feats):
        m = max(len(f["input_ids"]) for f in feats)
        return {
            "input_ids": torch.tensor(
                [f["input_ids"] + [self.pad_id] * (m - len(f["input_ids"])) for f in feats]),
            "labels": torch.tensor(
                [f["labels"] + [-100] * (m - len(f["labels"])) for f in feats]),
            "attention_mask": torch.tensor(
                [[1] * len(f["input_ids"]) + [0] * (m - len(f["input_ids"])) for f in feats]),
        }


base = AutoModelForCausalLM.from_pretrained(
    BASE, dtype=torch.bfloat16, attn_implementation="sdpa",
    device_map=None, local_files_only=True).cuda()
base.config.use_cache = False
base.enable_input_require_grads()

model = get_peft_model(base, LoraConfig(
    r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()


class PerExampleTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        sl = outputs.logits[:, :-1, :].contiguous()
        st = labels[:, 1:].contiguous()
        tok = F.cross_entropy(sl.view(-1, sl.size(-1)), st.view(-1),
                              ignore_index=-100, reduction="none").view_as(st)
        mask = st.ne(-100)
        per_ex = (tok * mask).sum(1) / mask.sum(1).clamp_min(1)
        loss = per_ex.mean()
        return (loss, outputs) if return_outputs else loss


class LearnGuard(TrainerCallback):
    def __init__(self):
        self.name = self.ref = self.step = None

    def on_train_begin(self, args, state, control, model=None, **kw):
        for n, p in model.named_parameters():
            if p.requires_grad and "lora_B" in n:
                self.name, self.ref = n, p.detach().clone()
                self.step = state.global_step + 2
                print(f"LearnGuard tracking {n} at step {self.step}", flush=True)
                return
        raise SystemExit("LearnGuard: no trainable LoRA B found")

    def on_step_end(self, args, state, control, model=None, **kw):
        if state.global_step == self.step and self.ref is not None:
            p = dict(model.named_parameters())[self.name]
            d = (p.detach() - self.ref.to(p.device)).abs().max().item()
            print(f"LearnGuard step={state.global_step} max_abs_delta={d:.3e}",
                  flush=True)
            if d == 0:
                raise SystemExit("NO LEARNING: LoRA B did not change")
            self.ref = None


trainer = PerExampleTrainer(
    model=model,
    args=TrainingArguments(
        output_dir=str(CHECKPOINTS),
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH * 2,
        gradient_accumulation_steps=ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_LIMIT,
        report_to=[],
        seed=42,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    ),
    train_dataset=train_enc,
    eval_dataset=eval_enc,
    data_collator=Collator(tokenizer.pad_token_id),
    callbacks=[LearnGuard()],
)
trainer.train()
model.save_pretrained(str(ADAPTER))
tokenizer.save_pretrained(str(ADAPTER))
print(f"adapter saved -> {ADAPTER}", flush=True)
