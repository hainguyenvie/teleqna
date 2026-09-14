#!/usr/bin/env python3
"""Context distillation: teach the model to answer without the passage it needs.

Why this trainer exists rather than train_teleqna.py. Four campaigns have now
pushed facts into these weights with cross-entropy on text - distill_v1, CPT
campaign 1, four DPO recipes, and SFT on the benchmark's own explanations - and
the transfer rate was measured at 13%: the same facts bought +22.2 delivered in
context and +2.9 baked into weights (p=0.089). The published comparison on
identical source documents says cross-entropy is the weakest of the available
objectives, not merely a weak one: on NarrativeQA closed-book, next-token
training on the documents scores 17.9 Rouge-L against 24.5 for context
distillation and 27.3 once the corpus is synthetic, over a 12.5 base.

The objective here is the middle one. For each row the teacher reads

    [evidence window] + [question]  ->  [answer]

and the student reads only

    [question]  ->  [answer]

and the student is pulled toward the teacher's next-token distribution over the
answer tokens. The teacher is not a second model: it is this same base with the
LoRA adapter switched off, so the "teacher" is exactly what the student would be
if it still had the passage. That is what makes the gap being closed a gap in
retrieval-free recall and nothing else, and it is why no 122B needs to be held
in memory during training.

Two terms, and the mixing weight is the experiment:

  KL(teacher || student) over answer tokens   the distillation signal
  CE(student, answer)                          the ordinary SFT anchor

ALPHA=0 reduces this file to train_teleqna.py, which is the control arm. Run it.
A distillation result that is never compared against its own CE baseline on the
same data is not a result.

**Format damage is the failure mode to watch, not perplexity.** Exp B lost nine
percent of rows to unparsable output - unparsed went 10 -> 96 - and that damage
was three times the size of the knowledge signal it was chasing. Two defences
are wired in. Rows carrying no context contribute CE only, so the replay anchors
from make_replay_mix.py behave exactly as they did before. And the answer-token
KL never sees the prompt, so nothing pulls the model's instruction-following
toward the teacher's context-conditioned habits, which in the RAG probe included
answering "Not mentioned".

**Alignment.** The teacher's sequence is longer by the window, so positions do
not correspond from the left. Both sides are left-padded and the last MAXCOMP+1
columns are sliced off each, which makes the answer tokens line up from the
right whatever the prefix length. Left padding then makes position_ids wrong by
default, so they are passed explicitly - without that the model sees the pad run
as real positions and the teacher's distribution is quietly garbage.

**Not implemented, deliberately:** the hidden-state L1 term of Deep Context
Distillation, and the on-policy variant where the student is trained on its own
rollouts. Both are reported as further gains (OPCD adds 1.2-5.3 points over
off-policy and forgets less, which is the axis this repo has been hurt on). They
are the next two experiments, not this one; adding them now would leave three
changes tangled in one measurement.
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
DATA = Path(os.environ["TRAIN_DATA"])
LORA_R = int(os.environ.get("LORA_R", "64"))
MAXLEN = int(os.environ.get("MAXLEN", "2048"))
MAXCOMP = int(os.environ.get("MAXCOMP", "256"))
BATCH = int(os.environ.get("BATCH", "4"))
ACCUM = int(os.environ.get("ACCUM", "8"))
EPOCHS = float(os.environ.get("EPOCHS", "2"))
LR = float(os.environ.get("LR", "1e-4"))
ALPHA = float(os.environ.get("ALPHA", "0.7"))
TEMP = float(os.environ.get("TEMP", "1.0"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "200"))
SAVE_LIMIT = int(os.environ.get("SAVE_LIMIT", "3"))
RUN = os.environ.get("RUN_NAME", f"cd-a{ALPHA}-r{LORA_R}")
set_seed(42)

ADAPTER = ROOT / "models" / f"ctxdistill-{RUN}-adapter"
CHECKPOINTS = ROOT / "models" / f"ctxdistill-{RUN}-checkpoints"
if ADAPTER.exists() and os.environ.get("ALLOW_OVERWRITE") != "1":
    raise SystemExit(f"{ADAPTER} exists; refusing overwrite")

tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# Left padding is required for the right-alignment trick below, and it must be
# set before anything is encoded.
tokenizer.padding_side = "left"

CTX_TEMPLATE = ("Reference material.\n\n{context}\n\n---\n\n{prompt}")


def _assistant_suffix() -> str:
    """Whatever the template appends after the assistant's own words.

    Taken from a probe rather than assumed to be the eos token, because this
    family closes a turn with its own markers and a model trained without them
    never learns to stop.
    """
    SENT = "BODY"
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"},
         {"role": "assistant", "content": SENT}],
        tokenize=False, enable_thinking=False)
    i = full.find(SENT)
    if i < 0:
        raise SystemExit("cannot find the assistant body in the chat template")
    return full[i + len(SENT):]


ASSIST_SUFFIX = _assistant_suffix()


def encode_one(prompt: str, completion: str, context: str, name: str, i: int):
    """Token ids for the student and, when there is context, for the teacher.

    Built as generation_prompt + completion rather than by templating the whole
    two-turn conversation. Templating twice and asserting one is a prefix of the
    other is what the first version did, and it does not hold on this family:
    add_generation_prompt=True ends the prompt with an opened-and-closed thought
    channel, while rendering the assistant turn in full lays that region out
    differently, so the assertion fires on row 1 and nothing trains.

    Building the sequence by concatenation makes the prefix property true by
    construction instead of checked, and it makes the completion tokens
    byte-identical between the student and the teacher - which is the thing the
    KL term depends on, since the two sequences are compared position by
    position from the right.
    """
    def ids_for(p: str) -> tuple[list[int], list[int]]:
        pre = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        p_ids = tokenizer(pre, add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(completion + ASSIST_SUFFIX,
                          add_special_tokens=False)["input_ids"]
        return p_ids + c_ids, p_ids

    s_ids, s_pre = ids_for(prompt)
    n_comp = len(s_ids) - len(s_pre)
    if n_comp <= 0:
        raise SystemExit(f"{name}:{i} empty completion after templating")
    if n_comp > MAXCOMP:
        return None                      # answer longer than the aligned window
    if len(s_ids) > MAXLEN:
        return None

    rec = {"input_ids": s_ids, "n_comp": n_comp}
    if context:
        t_ids, t_pre = ids_for(CTX_TEMPLATE.format(context=context,
                                                   prompt=prompt))
        if len(t_ids) - len(t_pre) != n_comp or t_ids[-n_comp:] != s_ids[-n_comp:]:
            # The chat template put something different after the context. Rather
            # than align two sequences that do not actually share their tail,
            # drop the context and let the row train as a plain CE anchor.
            return rec
        if len(t_ids) <= MAXLEN:
            rec["teacher_ids"] = t_ids
    return rec


def encode(rows: list[dict], name: str) -> list[dict]:
    out, dropped, with_ctx, lens = [], 0, 0, []
    for i, row in enumerate(rows, 1):
        rec = encode_one(row["prompt"], row["completion"],
                         row.get("context", ""), name, i)
        if rec is None:
            dropped += 1
            continue
        with_ctx += "teacher_ids" in rec
        lens.append(len(rec["input_ids"]))
        out.append(rec)
    lens.sort()
    print(f"{name}: n={len(out)} dropped={dropped} with_context={with_ctx} "
          f"({100*with_ctx/max(len(out),1):.1f}%) student_tokens "
          f"min/med/p99/max={lens[0]}/{lens[len(lens)//2]}/"
          f"{lens[int(len(lens)*.99)]}/{lens[-1]}", flush=True)
    # Context only feeds the teacher pass, so its absence is a contradiction
    # solely when a KL term is actually being asked for. At ALPHA=0 this file is
    # a plain SFT trainer, and it is the only one whose masking is correct for
    # this base's chat template.
    if with_ctx == 0 and ALPHA > 0:
        raise SystemExit(f"{name}: no row carries context but ALPHA={ALPHA}; "
                         f"set ALPHA=0 for plain SFT")
    return out


rows = [json.loads(l) for l in DATA.open(encoding="utf-8")]
# Split on fact_id, never on row: expand_views.py emits ~7 views of one fact and
# putting two of them on opposite sides of the split would read as generalisation.
def key(r: dict) -> str:
    return str(r.get("fact_id") or r.get("sample_id"))


def h(r: dict) -> int:
    return hashlib.sha256(("evalsplit|" + key(r)).encode()).digest()[0]


eval_facts = {key(r) for r in rows if h(r) < 4}
eval_rows = [r for r in rows if key(r) in eval_facts][:2000]
train_rows = [r for r in rows if key(r) not in eval_facts]
print(f"train={len(train_rows)} eval={len(eval_rows)} "
      f"(split on fact_id, {len(eval_facts)} held facts)", flush=True)

train_enc = encode(train_rows, "train")
eval_enc = encode(eval_rows, "eval")


@dataclass
class Collator:
    """Left-pad student and teacher separately; keep them in one batch.

    Teacher rows are padded to the teacher max and student rows to the student
    max. They are never concatenated, only sliced from the right, so the two
    maxima need not agree.
    """
    pad_id: int

    def _pad(self, seqs: list[list[int]]):
        m = max(len(s) for s in seqs)
        ids = torch.tensor([[self.pad_id] * (m - len(s)) + s for s in seqs])
        att = torch.tensor([[0] * (m - len(s)) + [1] * len(s) for s in seqs])
        # Left padding breaks the default arange position_ids; derive them from
        # the mask so the pad run does not consume real positions.
        pos = (att.cumsum(-1) - 1).clamp_min(0)
        return ids, att, pos

    def __call__(self, feats):
        ids, att, pos = self._pad([f["input_ids"] for f in feats])
        n_comp = torch.tensor([f["n_comp"] for f in feats])
        batch = {"input_ids": ids, "attention_mask": att, "position_ids": pos,
                 "n_comp": n_comp}
        has = [f for f in feats if "teacher_ids" in f]
        batch["has_teacher"] = torch.tensor(["teacher_ids" in f for f in feats])
        if has:
            t_ids, t_att, t_pos = self._pad([f["teacher_ids"] for f in has])
            batch.update(teacher_input_ids=t_ids, teacher_attention_mask=t_att,
                         teacher_position_ids=t_pos)
        return batch


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


def forward_tail(m, width: int, **kw):
    """Forward, keeping only the logits the loss can actually reach.

    Only the last width+1 positions are ever sliced, but a full-sequence logits
    tensor is batch x len x 151936: at 4 x 2048 in bf16 that is 2.5 GiB per
    forward, thrown away immediately, twice per step. `logits_to_keep` asks the
    model to compute the tail alone. Not every architecture accepts it, so the
    full forward stays as a fallback rather than a hard requirement.
    """
    if getattr(forward_tail, "supported", True):
        try:
            return m(**kw, logits_to_keep=width + 1)
        except TypeError:
            forward_tail.supported = False
    return m(**kw)


def tail(logits: torch.Tensor, width: int) -> torch.Tensor:
    """The `width` positions whose predictions are the final `width` tokens.

    Correct whether the logits are full-length or already trimmed to width+1:
    the slice is anchored to the right end in both cases.

    Asserted rather than trusted. A sequence shorter than width+1 makes the
    negative start index clamp to zero and the slice comes back narrower, with
    no error - and a narrower tail is not a smaller version of the same thing,
    it is a different span, silently misaligning whatever is compared against
    it. Better to stop here than to train on it.
    """
    out = logits[:, -(width + 1):-1, :]
    if out.size(1) != width:
        raise RuntimeError(
            f"tail wanted {width} positions, sequence of length "
            f"{logits.size(1)} could only give {out.size(1)}")
    return out


def comp_mask(n_comp: torch.Tensor, width: int, device) -> torch.Tensor:
    """True where the predicted token belongs to the completion.

    Column j of the tail predicts the token `width - j` from the end, so the
    completion occupies the last n_comp columns.
    """
    idx = torch.arange(width, device=device).unsqueeze(0)
    return idx >= (width - n_comp.to(device).unsqueeze(1))


class CtxDistillTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        inputs = dict(inputs)
        n_comp = inputs.pop("n_comp")
        has_teacher = inputs.pop("has_teacher")
        t_ids = inputs.pop("teacher_input_ids", None)
        t_att = inputs.pop("teacher_attention_mask", None)
        t_pos = inputs.pop("teacher_position_ids", None)

        # Width comes from the completion region, not from the padded batch
        # length. Taking it from the student's padded length is what the first
        # version did and it breaks whenever the longest student row in a batch
        # has no teacher: the teacher tensor is padded over its own subset and
        # comes out shorter, so slicing it to the student's width silently
        # returns fewer columns and the KL compares two different spans. Only
        # the last n_comp columns are ever unmasked, so the largest n_comp in
        # the batch is all the width that is needed.
        width = min(MAXCOMP, int(n_comp.max().item()),
                    inputs["input_ids"].size(1) - 1)
        out = forward_tail(model, width, **inputs)
        s_tail = tail(out.logits, width)
        dev = s_tail.device
        mask = comp_mask(n_comp, width, dev)
        targets = inputs["input_ids"][:, -width:].to(dev)

        ce_tok = F.cross_entropy(
            s_tail.reshape(-1, s_tail.size(-1)).float(),
            targets.reshape(-1), reduction="none").view_as(targets)
        ce = ((ce_tok * mask).sum(1) / mask.sum(1).clamp_min(1)).mean()

        kl = torch.zeros((), device=dev, dtype=ce.dtype)
        n_t = int(has_teacher.sum())
        if t_ids is not None and n_t and ALPHA > 0:
            sel = has_teacher.to(dev)
            # A width both tensors can serve: wide enough for every teacher
            # row's completion, no wider than either sequence.
            w_kl = min(width, int(n_comp[has_teacher].max().item()),
                       t_ids.size(1) - 1)
            with torch.no_grad(), model.disable_adapter():
                t_out = forward_tail(model, w_kl,
                                     input_ids=t_ids.to(dev),
                                     attention_mask=t_att.to(dev),
                                     position_ids=t_pos.to(dev))
            t_tail = tail(t_out.logits, w_kl).float()
            s_sel = s_tail[sel][:, -w_kl:, :].float()
            m_sel = comp_mask(n_comp[has_teacher], w_kl, dev)
            p = F.softmax(t_tail / TEMP, dim=-1)
            logq = F.log_softmax(s_sel / TEMP, dim=-1)
            logp = F.log_softmax(t_tail / TEMP, dim=-1)
            kl_tok = (p * (logp - logq)).sum(-1)
            kl = ((kl_tok * m_sel).sum(1) / m_sel.sum(1).clamp_min(1)).mean()
            kl = kl * (TEMP ** 2)

        loss = ALPHA * kl + (1.0 - ALPHA) * ce
        if self.state.global_step % 25 == 0:
            print(f"step={self.state.global_step} ce={ce.item():.4f} "
                  f"kl={float(kl):.4f} ctx_rows={n_t}/{len(n_comp)}", flush=True)
        return (loss, out) if return_outputs else loss


class LearnGuard(TrainerCallback):
    """Refuse to run a training job that is not training anything."""

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


trainer = CtxDistillTrainer(
    model=model,
    args=TrainingArguments(
        output_dir=str(CHECKPOINTS),
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
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
        gradient_checkpointing=True,
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
print(f"adapter saved -> {ADAPTER}  (ALPHA={ALPHA}, TEMP={TEMP})", flush=True)
