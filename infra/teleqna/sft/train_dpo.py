#!/usr/bin/env python3
"""Single-token DPO for the teleqna specialist.

chosen and rejected differ in exactly ONE token (the answer letter), so the
sequence-level DPO log-ratio collapses to a difference of two logits at one
position — the log-softmax normaliser cancels. One forward per batch gives
the policy delta; the same forward with the LoRA adapter disabled gives the
reference delta (standard LoRA-DPO trick, no second model in memory).

    loss = -logsigmoid(beta * ((pol_gold - pol_rej) - (ref_gold - ref_rej)))

This objective moves the *relative* preference between the two letters the
model actually confused (rejected = the student's real wrong pick), and the
reference anchor penalises drift — the failure mode that sank both SFT runs
(immediate -3.6pp at 0.24 epoch, gains saturating at ~85 rows).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(os.environ.get("PROJ_ROOT", "/workspace/teleqna-sft"))
BASE = os.environ["BASE_MODEL"]
DATA = Path(os.environ.get("TRAIN_DATA", ROOT / "data" / "dpo_pairs.jsonl"))
LORA_R = int(os.environ.get("LORA_R", "16"))
BETA = float(os.environ.get("BETA", "0.2"))
# Asymmetric beta (run 3): fix pairs get a low beta (weak KL leash — allowed
# to learn), anchor pairs a high one (hard leash — forbidden to drift). Run 2
# showed a symmetric beta protects (8 lost vs SFT's 114+) but also throttles
# learning (14 gained vs SFT's ~85); the two roles need different tensions.
BETA_FIX = float(os.environ.get("BETA_FIX", str(BETA)))
BETA_ANCHOR = float(os.environ.get("BETA_ANCHOR", str(BETA)))
BATCH = int(os.environ.get("BATCH", "16"))
ACCUM = int(os.environ.get("ACCUM", "2"))
EPOCHS = float(os.environ.get("EPOCHS", "1"))
LR = float(os.environ.get("LR", "1e-5"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "200"))
RUN = os.environ.get("RUN_NAME", f"dpo-r{LORA_R}")
set_seed(42)

ADAPTER = ROOT / "models" / f"teleqna-{RUN}-adapter"
CHECKPOINTS = ROOT / "models" / f"teleqna-{RUN}-checkpoints"
if ADAPTER.exists() and os.environ.get("ALLOW_OVERWRITE") != "1":
    raise SystemExit(f"{ADAPTER} exists; refusing overwrite")

tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def encode(rows):
    out = []
    for i, row in enumerate(rows, 1):
        msgs_c = [{"role": "user", "content": row["prompt"]},
                  {"role": "assistant", "content": row["chosen"]}]
        msgs_r = [{"role": "user", "content": row["prompt"]},
                  {"role": "assistant", "content": row["rejected"]}]
        full_c = tokenizer.apply_chat_template(
            msgs_c, tokenize=False, enable_thinking=False)
        full_r = tokenizer.apply_chat_template(
            msgs_r, tokenize=False, enable_thinking=False)
        ids_c = tokenizer(full_c, add_special_tokens=False)["input_ids"]
        ids_r = tokenizer(full_r, add_special_tokens=False)["input_ids"]
        if len(ids_c) != len(ids_r):
            raise SystemExit(f"row {i}: chosen/rejected token length differs")
        diff = [j for j, (a, b) in enumerate(zip(ids_c, ids_r)) if a != b]
        if len(diff) != 1:
            raise SystemExit(f"row {i}: {len(diff)} differing tokens, want 1")
        out.append({"input_ids": ids_c, "pos": diff[0],
                    "tok_c": ids_c[diff[0]], "tok_r": ids_r[diff[0]],
                    "beta": BETA_ANCHOR if row.get("kind") == "anchor" else BETA_FIX})
    lens = sorted(len(o["input_ids"]) for o in out)
    print(f"pairs={len(out)} tokens med/p99/max="
          f"{lens[len(lens)//2]}/{lens[int(len(lens)*.99)]}/{lens[-1]}",
          flush=True)
    return out


rows = [json.loads(l) for l in DATA.open(encoding="utf-8")]
train_enc = encode(rows)


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        m = max(len(f["input_ids"]) for f in feats)
        return {
            "input_ids": torch.tensor(
                [f["input_ids"] + [self.pad_id] * (m - len(f["input_ids"])) for f in feats]),
            "attention_mask": torch.tensor(
                [[1] * len(f["input_ids"]) + [0] * (m - len(f["input_ids"])) for f in feats]),
            "pos": torch.tensor([f["pos"] for f in feats]),
            "tok_c": torch.tensor([f["tok_c"] for f in feats]),
            "tok_r": torch.tensor([f["tok_r"] for f in feats]),
            "beta": torch.tensor([f["beta"] for f in feats]),
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


class DPOTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        inputs = dict(inputs)
        pos = inputs.pop("pos")
        tok_c = inputs.pop("tok_c")
        tok_r = inputs.pop("tok_r")
        beta = inputs.pop("beta")
        idx = torch.arange(pos.size(0), device=pos.device)

        logits = model(**inputs).logits            # adapter ON  -> policy
        at = logits[idx, pos - 1]                  # position predicting letter
        pol = at[idx, tok_c] - at[idx, tok_r]

        with torch.no_grad(), model.disable_adapter():
            ref_logits = model(**inputs).logits    # adapter OFF -> reference
            rat = ref_logits[idx, pos - 1]
            ref = rat[idx, tok_c] - rat[idx, tok_r]

        loss = -F.logsigmoid(beta * (pol - ref)).mean()
        if model.training and self.state.global_step % 50 == 0:
            win = (pol > ref).float().mean().item()
            print(f"step={self.state.global_step} "
                  f"margin={(pol - ref).mean().item():.4f} win_rate={win:.3f} "
                  f"ref_acc={(ref > 0).float().mean().item():.3f}", flush=True)
        return loss


trainer = DPOTrainer(
    model=model,
    args=TrainingArguments(
        output_dir=str(CHECKPOINTS),
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=25,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=100,
        report_to=[],
        seed=42,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    ),
    train_dataset=train_enc,
    data_collator=Collator(tokenizer.pad_token_id),
)
trainer.train()
model.save_pretrained(str(ADAPTER))
tokenizer.save_pretrained(str(ADAPTER))
print(f"adapter saved -> {ADAPTER}", flush=True)
