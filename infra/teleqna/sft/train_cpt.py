#!/usr/bin/env python3
"""Continued pretraining (CPT) for the teleqna specialist.

Plain next-token LM on the error-targeted corpus (select_cpt_corpus.py):
the knowledge-injection channel that letter-level SFT/DPO does not have.
LoRA keeps the base frozen (drift insurance the SFT failure paid for), rank
is high (default 64) because knowledge needs capacity, and ~3% of the stream
is MCQ-formatted anchor rows rendered through the chat template so the
answer-interface does not drift while the adapter absorbs raw text.

Docs are tokenized once, joined with EOS, and greedy-packed into fixed
SEQ-length blocks — full-sequence loss, no padding.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import torch
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
CORPUS = Path(os.environ.get("CPT_CORPUS", ROOT / "data" / "cpt_corpus.jsonl"))
MCQ_ANCHORS = os.environ.get("MCQ_ANCHORS", "")   # train_v3-format jsonl
ANCHOR_EVERY = int(os.environ.get("ANCHOR_EVERY", "30"))  # 1 anchor per N docs
SEQ = int(os.environ.get("SEQ", "4096"))
LORA_R = int(os.environ.get("LORA_R", "64"))
BATCH = int(os.environ.get("BATCH", "8"))
ACCUM = int(os.environ.get("ACCUM", "4"))
EPOCHS = float(os.environ.get("EPOCHS", "1"))
LR = float(os.environ.get("LR", "1e-4"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "200"))
RUN = os.environ.get("RUN_NAME", f"cpt-r{LORA_R}")
set_seed(42)

ADAPTER = ROOT / "models" / f"teleqna-{RUN}-adapter"
CHECKPOINTS = ROOT / "models" / f"teleqna-{RUN}-checkpoints"
if ADAPTER.exists() and os.environ.get("ALLOW_OVERWRITE") != "1":
    raise SystemExit(f"{ADAPTER} exists; refusing overwrite")

tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
EOS = tokenizer.eos_token_id

anchors: list[str] = []
if MCQ_ANCHORS:
    rows = [json.loads(l) for l in open(MCQ_ANCHORS, encoding="utf-8")]
    random.Random(42).shuffle(rows)
    for r in rows:
        anchors.append(tokenizer.apply_chat_template(
            [{"role": "user", "content": r["prompt"]},
             {"role": "assistant", "content": r["completion"]}],
            tokenize=False, enable_thinking=False))
print(f"anchor pool: {len(anchors)}", flush=True)

docs = [json.loads(l)["text"] for l in CORPUS.open(encoding="utf-8")]
random.Random(42).shuffle(docs)   # decorrelate sources/topics within batches
if anchors:
    mixed: list[str] = []
    ai = 0
    for i, dtext in enumerate(docs):
        mixed.append(dtext)
        if (i + 1) % ANCHOR_EVERY == 0:
            mixed.append(anchors[ai % len(anchors)])
            ai += 1
    docs = mixed
    print(f"anchors interleaved: {ai}", flush=True)

print(f"docs to tokenize: {len(docs)}", flush=True)
stream: list[int] = []
BS = 256
for i in range(0, len(docs), BS):
    for ids in tokenizer(docs[i:i + BS], add_special_tokens=False)["input_ids"]:
        stream.extend(ids)
        stream.append(EOS)
    if (i // BS) % 20 == 0:
        print(f"tokenized {i}/{len(docs)} stream={len(stream)/1e6:.1f}M", flush=True)

n_blocks = len(stream) // SEQ
print(f"total tokens={len(stream)/1e6:.1f}M -> blocks={n_blocks} of {SEQ}", flush=True)
blocks = [{"input_ids": stream[i * SEQ:(i + 1) * SEQ]} for i in range(n_blocks)]
del stream


class Collator:
    def __call__(self, feats):
        x = torch.tensor([f["input_ids"] for f in feats])
        return {"input_ids": x, "attention_mask": torch.ones_like(x), "labels": x.clone()}


base = AutoModelForCausalLM.from_pretrained(
    BASE, dtype=torch.bfloat16, attn_implementation="sdpa",
    device_map=None, local_files_only=True).cuda()
base.config.use_cache = False
base.enable_input_require_grads()

model = get_peft_model(base, LoraConfig(
    r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.0, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=str(CHECKPOINTS),
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.02,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=25,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=100,
        report_to=[],
        seed=42,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    ),
    train_dataset=blocks,
    data_collator=Collator(),
)
trainer.train()
model.save_pretrained(str(ADAPTER))
tokenizer.save_pretrained(str(ADAPTER))
print(f"adapter saved -> {ADAPTER}", flush=True)
