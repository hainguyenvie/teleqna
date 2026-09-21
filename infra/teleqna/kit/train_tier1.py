#!/usr/bin/env python3
"""Full-weight CPT of Qwen3-8B on the tier-1 study-kit pack (run-3 trainer, re-pathed and parameterised).
torchrun DDP, one full replica per H200 (weights bf16 + fp32 AdamW ~96 GB, validated by the kswp full-weight arm).
lr 1e-5 cosine — the rate at which this trainer kept the ANSWER contract in runs 1-3 (unparsed 0)."""
import os, math, time, argparse
import numpy as np, torch
import torch.distributed as dist
from pathlib import Path
H = Path.home()
R = H / "projects/teleqna/runs/teleqna-8b"
BASE = str(H / "projects/_shared/models/Qwen3-8B")

def log(r, *a):
    if r == 0: print(*a, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="data/kit/tier1/pack"); ap.add_argument("--out", default="models/kit/tier1")
    ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--bs", type=int, default=2); ap.add_argument("--acc", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--warm", type=int, default=150); ap.add_argument("--init", default=BASE)
    ap.add_argument("--save-every", type=int, default=0, help="also save models/<out>/step<N> every N optimizer steps (early probes)")
    ap.add_argument("--ema-every", type=int, default=0, help="rank0 keeps an fp32 EMA of the weights on CPU, updated every N optimizer steps; saved as ep<N> (raw weights as ep<N>_raw)")
    ap.add_argument("--ema-decay", type=float, default=0.99)
    ap.add_argument("--train-only", default="all", help="all | mlp | attn : restrict updates to a parameter subset (knowledge lives in MLPs; attention frozen = less interference)")
    ap.add_argument("--layers", default="", help="optional inclusive layer range a-b to train (others frozen), e.g. 8-27")
    a = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    lr_ = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr_)
    I = np.load(R / f"{a.pack}_ids.npy", mmap_mode="r"); M = np.load(R / f"{a.pack}_mask.npy", mmap_mode="r")
    nblk, blk = I.shape
    log(rank, f"{nblk:,} blocks of {blk}, world {world}, effective batch {world*a.bs*a.acc} blocks")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(a.init, dtype=torch.bfloat16, local_files_only=True).cuda()
    model.gradient_checkpointing_enable(); model.train()
    if a.train_only != "all" or a.layers:
        import re as _re
        lo, hi = (int(x) for x in a.layers.split("-")) if a.layers else (0, 10**6)
        for n, p in model.named_parameters():
            m = _re.search(r"layers\.(\d+)\.", n); li = int(m.group(1)) if m else -1
            keep = (li >= 0 and lo <= li <= hi) and ((a.train_only == "all") or (a.train_only == "mlp" and ".mlp." in n) or (a.train_only == "attn" and ".self_attn." in n))
            p.requires_grad_(keep)
        if a.train_only != "all": model.enable_input_require_grads()
        nt = sum(p.numel() for p in model.parameters() if p.requires_grad); log(rank, f"trainable params {nt/1e9:.2f}B ({a.train_only}, layers {a.layers or 'all'})")
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[lr_], find_unused_parameters=False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    ema = {n: p.detach().float().cpu().clone() for n, p in model.module.named_parameters()} if (a.ema_every and rank == 0) else None
    steps_ep = math.ceil(nblk / (world * a.bs * a.acc)); total = steps_ep * a.epochs
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / a.warm, 1.0) * (0.5 * (1 + math.cos(math.pi * min(s / total, 1.0)))))
    log(rank, f"{steps_ep} steps/epoch x {a.epochs}")
    gstep = 0; t0 = time.time()
    for ep in range(a.epochs):
        order = np.random.default_rng(1000 + ep).permutation(nblk); mine = order[rank::world][: nblk // world]   # equal length on every rank, else the last allreduce hangs (vd pack 44,155 blocks)
        for s in range(0, len(mine) - a.bs * a.acc + 1, a.bs * a.acc):
            opt.zero_grad(set_to_none=True)
            for k in range(a.acc):
                sel = mine[s + k * a.bs: s + (k + 1) * a.bs]
                x = torch.tensor(np.asarray(I[sel]), dtype=torch.long).cuda(); m = torch.tensor(np.asarray(M[sel]), dtype=torch.bool).cuda()
                y = x.clone(); y[~m] = -100
                out = model(input_ids=x, labels=y); (out.loss / a.acc).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); gstep += 1
            if ema is not None and gstep % a.ema_every == 0:
                with torch.no_grad():
                    for n, p in model.module.named_parameters(): ema[n].mul_(a.ema_decay).add_(p.detach().float().cpu(), alpha=1 - a.ema_decay)
            if a.save_every and gstep % a.save_every == 0:
                dist.barrier()
                if rank == 0:
                    sd = R / f"{a.out}/step{gstep}"; sd.mkdir(parents=True, exist_ok=True)
                    model.module.save_pretrained(sd); AutoTokenizer.from_pretrained(BASE, local_files_only=True).save_pretrained(sd)
                    print(f"saved {sd}", flush=True)
                dist.barrier()
            if rank == 0 and gstep % 25 == 0:
                tokps = gstep * world * a.bs * a.acc * blk / (time.time() - t0)
                print(f"ep{ep} step {gstep}/{total} loss {out.loss.item():.4f} lr {sch.get_last_lr()[0]:.2e} {tokps/1000:.1f}k tok/s", flush=True)
        dist.barrier()
        if rank == 0:
            out_dir = R / f"{a.out}/ep{ep+1}"; out_dir.mkdir(parents=True, exist_ok=True)
            if ema is not None:   # save raw weights as ep<N>_raw, then load the EMA into the module and save it as ep<N>
                raw_dir = R / f"{a.out}/ep{ep+1}_raw"; raw_dir.mkdir(parents=True, exist_ok=True)
                model.module.save_pretrained(raw_dir); AutoTokenizer.from_pretrained(BASE, local_files_only=True).save_pretrained(raw_dir)
                raw_state = {n: p.detach().clone() for n, p in model.module.named_parameters()}
                with torch.no_grad():
                    for n, p in model.module.named_parameters(): p.copy_(ema[n].to(p.dtype))
                model.module.save_pretrained(out_dir); AutoTokenizer.from_pretrained(BASE, local_files_only=True).save_pretrained(out_dir)
                with torch.no_grad():
                    for n, p in model.module.named_parameters(): p.copy_(raw_state[n])
                del raw_state; print(f"saved EMA -> {out_dir}, raw -> {raw_dir}", flush=True)
            else:
                model.module.save_pretrained(out_dir); AutoTokenizer.from_pretrained(BASE, local_files_only=True).save_pretrained(out_dir)
                print(f"saved {out_dir}", flush=True)
        dist.barrier()
    log(rank, "TRAIN_DONE")
if __name__ == "__main__": main()
