#!/usr/bin/env python3
"""Letter-level DPO (single-token completions after 'ANSWER:'), torchrun DDP, full weights, frozen reference copy.
loss = -log sigmoid(beta * [(lp_w - ref_w) - (lp_l - ref_l)]); lp = log-prob of the chosen/rejected letter token at the
position right after the rendered prompt + 'ANSWER:'. Pushes the specific confusable wrong letter DOWN — the negative
signal SFT never gives. Usage: torchrun ... train_dpo_letter.py --pairs data/kit/dpo/pairs.jsonl --init CKPT --out DIR"""
import argparse, json, math, os, time, random, torch, torch.distributed as dist
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
ap = argparse.ArgumentParser(); ap.add_argument("--pairs", required=True); ap.add_argument("--init", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--beta", type=float, default=0.1); ap.add_argument("--lr", type=float, default=5e-7); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--acc", type=int, default=2)
ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--warm", type=int, default=30); ap.add_argument("--maxlen", type=int, default=768); ap.add_argument("--sft-mix", type=float, default=0.0, help="add this weight of NLL on the chosen letter (regulariser)"); a = ap.parse_args()
dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size(); lr_ = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr_)
def log(*x):
    if rank == 0: print(*x, flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.init, local_files_only=True)
rows = [json.loads(l) for l in open(R / a.pairs, encoding="utf-8")]
def enc(r):
    p = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False) + "ANSWER:"
    ids = tok(p, add_special_tokens=False)["input_ids"][-a.maxlen:]
    return ids, tok.encode(r["chosen"], add_special_tokens=False)[-1], tok.encode(r["rejected"], add_special_tokens=False)[-1]
data = [enc(r) for r in rows]; log(f"pairs {len(data):,}")
model = AutoModelForCausalLM.from_pretrained(a.init, dtype=torch.bfloat16, local_files_only=True).cuda(); model.gradient_checkpointing_enable(); model.train()
ref = AutoModelForCausalLM.from_pretrained(a.init, dtype=torch.bfloat16, local_files_only=True).cuda().eval()
for p in ref.parameters(): p.requires_grad_(False)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[lr_])
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
per_rank = len(data) // world; steps_ep = per_rank // (a.bs * a.acc); total = steps_ep * a.epochs
sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / a.warm, 1.0) * (0.5 * (1 + math.cos(math.pi * min(s / max(total, 1), 1.0)))))
log(f"{steps_ep} steps/epoch x {a.epochs}, effective batch {world*a.bs*a.acc} pairs")
PAD = tok.pad_token_id
def batch_logps(m, items):
    L = max(len(x[0]) for x in items); ids = torch.full((len(items), L), PAD, dtype=torch.long); att = torch.zeros((len(items), L), dtype=torch.long)
    for i, (p, _, _) in enumerate(items): ids[i, :len(p)] = torch.tensor(p); att[i, :len(p)] = 1
    ids, att = ids.cuda(), att.cuda(); last = att.sum(1) - 1
    logits = m(input_ids=ids, attention_mask=att).logits[torch.arange(len(items)), last]   # next-token logits after 'ANSWER:'
    lp = torch.log_softmax(logits.float(), dim=-1)
    w = torch.tensor([x[1] for x in items]).cuda(); l = torch.tensor([x[2] for x in items]).cuda()
    return lp[torch.arange(len(items)), w], lp[torch.arange(len(items)), l]
gstep = 0; t0 = time.time()
for ep in range(a.epochs):
    order = list(range(len(data))); random.Random(100 + ep).shuffle(order); mine = order[rank::world][:per_rank]
    for s in range(0, len(mine) - a.bs * a.acc + 1, a.bs * a.acc):
        for k in range(a.acc):
            items = [data[i] for i in mine[s + k * a.bs: s + (k + 1) * a.bs]]
            with torch.no_grad(): rw, rl = batch_logps(ref, items)
            pw, pl = batch_logps(model, items)
            loss = -torch.nn.functional.logsigmoid(a.beta * ((pw - rw) - (pl - rl))).mean()
            if a.sft_mix > 0: loss = loss + a.sft_mix * (-pw).mean()
            (loss / a.acc).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(set_to_none=True); gstep += 1
        if rank == 0 and gstep % 10 == 0:
            with torch.no_grad(): acc = ((pw - rw) > (pl - rl)).float().mean().item()
            print(f"ep{ep} step {gstep}/{total} loss {loss.item():.4f} pref-acc {acc:.2f} lr {sch.get_last_lr()[0]:.2e} {(time.time()-t0)/60:.1f}min", flush=True)
    if rank == 0:
        out = R / a.out / f"ep{ep+1}"; model.module.save_pretrained(out, safe_serialization=True); tok.save_pretrained(out); print("saved", out, flush=True)
    dist.barrier()
log("TRAIN_DONE")
