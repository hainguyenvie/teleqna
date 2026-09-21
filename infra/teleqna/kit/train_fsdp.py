#!/usr/bin/env python3
"""Optimised full-weight trainer (v2): FSDP2 (fully_shard per decoder layer, bf16 compute, fp32 sharded master params +
AdamW states), SDPA flash attention, Liger fused kernels when available (RMSNorm/RoPE/SwiGLU/fused linear-CE), fused
AdamW, weight decay 0.1 (CPT convention), gradient checkpointing optional, cosine to 10% with 2% warmup, big token
batches (micro-batch x acc x world x blk). Same pack format as train_tier1.py (ids/mask .npy blocks). Saves a full HF
checkpoint on rank 0 at the end of each epoch (and every --save-every steps) via a gathered state dict.
Usage: torchrun --nproc_per_node=8 train_fsdp.py --pack data/kit/big2k/pack --out models/kit/big3 --bs 4 --acc 8 --lr 3e-5"""
import argparse, json, math, os, time, numpy as np, torch, torch.distributed as dist
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"; BASE = str(Path.home() / "projects/_shared/models/Qwen3-8B")
ap = argparse.ArgumentParser(); ap.add_argument("--pack", required=True); ap.add_argument("--out", required=True); ap.add_argument("--init", default=BASE)
ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--bs", type=int, default=4); ap.add_argument("--acc", type=int, default=8); ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--wd", type=float, default=0.1); ap.add_argument("--warm-frac", type=float, default=0.02); ap.add_argument("--min-lr-frac", type=float, default=0.1)
ap.add_argument("--save-every", type=int, default=0); ap.add_argument("--no-ckpt", action="store_true", help="disable gradient checkpointing (FSDP frees memory; try first)")
ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke tests)"); ap.add_argument("--start-step", type=int, default=0, help="resume: skip the first N steps of the (deterministic) block order and advance the schedule; init should be the step-N checkpoint (optimizer moments are not restored)"); ap.add_argument("--no-liger", action="store_true"); a = ap.parse_args()
dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size(); lr_ = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr_)
def log(*x):
    if rank == 0: print(*x, flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
if not a.no_liger:
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3; apply_liger_kernel_to_qwen3(); log("liger kernels: on")
    except Exception as e: log("liger kernels: off", repr(e)[:80])
I = np.load(R / f"{a.pack}_ids.npy", mmap_mode="r"); M = np.load(R / f"{a.pack}_mask.npy", mmap_mode="r"); nblk, blk = I.shape
model = AutoModelForCausalLM.from_pretrained(a.init, dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa")
if not a.no_ckpt: model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.config.use_cache = False
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
model = model.float()   # fp32 master params, sharded; bf16 compute via the policy
for layer in model.model.layers: fully_shard(layer, mp_policy=mp)
fully_shard(model, mp_policy=mp)
model.cuda()
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95), fused=True)
per_rank = nblk // world; steps_ep = per_rank // (a.bs * a.acc); total = steps_ep * a.epochs; warm = max(1, int(total * a.warm_frac))
sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, 1.0) * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(s / max(total, 1), 1.0)))))
log(f"{nblk:,} blocks of {blk}, world {world}, tokens/step {world*a.bs*a.acc*blk:,}, {steps_ep} steps/epoch x {a.epochs}, warm {warm}, grad-ckpt {'off' if a.no_ckpt else 'on'}")
def save(tag):
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
    sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if rank == 0:
        out = R / a.out / tag; out.mkdir(parents=True, exist_ok=True)
        cpu = AutoModelForCausalLM.from_pretrained(a.init, dtype=torch.bfloat16, local_files_only=True)
        cpu.load_state_dict({k: v.to(torch.bfloat16) for k, v in sd.items()}, strict=True); cpu.save_pretrained(out, safe_serialization=True)
        AutoTokenizer.from_pretrained(BASE, local_files_only=True).save_pretrained(out); print(f"saved {out}", flush=True); del cpu
    del sd; dist.barrier()
gstep = 0; t0 = time.time(); model.train()
if a.start_step:
    for _ in range(a.start_step): sch.step()
    gstep = a.start_step; log(f"resuming at step {gstep} (lr {sch.get_last_lr()[0]:.2e})")
for ep in range(a.epochs):
    order = np.random.default_rng(1000 + ep).permutation(nblk); mine = order[rank::world][:per_rank]
    for s in range(0, len(mine) - a.bs * a.acc + 1, a.bs * a.acc):
        if s < a.start_step * a.bs * a.acc: continue
        for k in range(a.acc):
            sel = np.sort(mine[s + k * a.bs: s + (k + 1) * a.bs])
            x = torch.tensor(np.asarray(I[sel]), dtype=torch.long).cuda(); m = torch.tensor(np.asarray(M[sel]), dtype=torch.bool).cuda()
            y = x.clone(); y[~m] = -100
            model.set_requires_gradient_sync(k == a.acc - 1)
            out = model(input_ids=x, labels=y); (out.loss / a.acc).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(set_to_none=True); gstep += 1
        if rank == 0 and gstep % 25 == 0:
            tokps = (gstep - a.start_step) * world * a.bs * a.acc * blk / (time.time() - t0)
            print(f"ep{ep} step {gstep}/{total} loss {out.loss.item():.4f} lr {sch.get_last_lr()[0]:.2e} {tokps/1000:.1f}k tok/s mem {torch.cuda.max_memory_allocated()/2**30:.0f}G", flush=True)
        if a.save_every and gstep % a.save_every == 0: save(f"step{gstep}")
        if a.max_steps and gstep >= a.max_steps: break
    if not (a.max_steps and gstep >= a.max_steps): save(f"ep{ep+1}")
    else: break
log("TRAIN_DONE")
