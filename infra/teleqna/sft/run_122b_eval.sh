#!/usr/bin/env bash
# Serve Qwen3.5-122B-A10B in BF16 on node cards 1+5 and score TeleQnA.
#
# Runs on the login pod, not through a job runner. That pod turns out to have
# the full device set injected (/dev/nvidia0..7, nvidiactl, uvm) plus 256GB of
# cgroup memory and a 32GB /dev/shm, so it can host the engine directly. The
# alternative was queueing behind the 5h GRPO run on spare4's single-slot runner
# for no gain.
#
# BF16, not the 4-bit build, and the reason is specific to this benchmark rather
# than general principle. Our remaining errors are distractor discrimination —
# the probe puts +15.8pp there — so the losses live on thin margins between
# close options, which is exactly where quantisation noise lands. A 10B-active
# MoE is also a bad host for 4-bit: moe_intermediate_size is 1024, so each
# expert is small and has little redundancy to absorb rounding. And dev-1000's
# MDE is ~2.2pp, so giving away 1-2 points to the weights format would cost the
# signal we are trying to read.
#
# Memory: 250GB of weights over 2 cards is 125.1GB each against 143.77GB, so
# 0.95 utilisation leaves ~11.5GB a card. That is ample here only because
# max_model_len is 8192 rather than the model's native 262144, and because the
# config has num_key_value_heads=2 with most layers on linear attention — the
# KV cache is small. Do not raise the context window to "be safe"; it is the one
# knob that turns a comfortable fit into an OOM.
#
# The checkpoint is the multimodal wrapper: config.json declares
# architectures=['Qwen3_5MoeForConditionalGeneration'] with a vision_config, and
# the nightly registers that name (verified against ModelRegistry, alongside the
# Qwen3_5MoeForCausalLM and Qwen3_5MoeMTP siblings). TeleQnA is pure text, so the
# vision tower is weight we would pay for and never call — language_model_only
# skips loading it. It is a real EngineArgs field, not a CLI-only switch, so it
# rides through the scorer's --engine-kwargs rather than needing a forked entry
# point.
# First attempt died at engine init with
#   RuntimeError: Worker failed with error '[Errno 2] No such file or directory: 'ninja''
# and the message is misleading: ninja was already installed in the venv, both as
# a package and as venv/bin/ninja. The nightly JIT-compiles kernels at startup
# and looks the compiler up on PATH, but calling the interpreter by absolute path
# never activates the venv, so venv/bin was not on PATH and the lookup failed.
# Taking the error at face value and reinstalling ninja would have reported
# success and then failed identically. Put venv/bin ahead of PATH instead.
#
# Second attempt then died inside determine_available_memory with
#   Failed: Cuda error csrc/custom_all_reduce.cuh:164 'invalid argument'
#   Worker proc VllmWorker-1 died unexpectedly
# on the profiling forward pass — the first time TP actually all-reduces. It is
# not a topology problem: nvidia-smi reports NV18 between every pair and
# `topo -p2p r` is OK for all of them. Custom all-reduce goes further than P2P
# and shares buffers between the worker processes via CUDA IPC handles, and IPC
# is what containers restrict. Rather than chase the container's IPC settings,
# drop the backend: the dispatch order was ['CUSTOM','SYMM_MEM','PYNCCL'], so
# disabling CUSTOM falls through to NCCL over the same NVLink. SYMM_MEM goes off
# too because it shares buffers the same way and would fail the same way one
# step later. Custom all-reduce is a small-message latency optimisation; for
# 1,000-row batched scoring it is worth nothing measurable.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CARDS:-1,5}"
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
#
# Third attempt cleared the all-reduce and then hit
#   ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (466).
#   Each decode sequence requires one Mamba cache block
# This one is intrinsic to the architecture rather than to the environment. Most
# of this model's layers are linear attention, and a linear-attention layer
# carries a fixed-size recurrent state per in-flight sequence instead of a
# per-token KV cache. That state cannot be paged the way KV blocks can, so the
# block count is a hard ceiling on concurrent decodes, and vLLM's default
# max_num_seqs of 1024 sits well above the 466 that fit here. Raising the block
# count is not available: they come out of the same pool already set to 0.95.
#
# 256 rather than 448. The 466 figure is computed from whatever memory profiling
# happens to leave free, so it moves a little run to run, and a ceiling set just
# under it would fail again on a bad day — each failure costs a 250GB load. The
# throughput given up is small: at 256 concurrent thinking generations the
# bottleneck is decode length, not batch width.
ENGINE_KWARGS="${ENGINE_KWARGS:-{\"language_model_only\": true, \"disable_custom_all_reduce\": true, \"max_num_seqs\": 256\}}"
HOME_ROOT=/home/tensara
MODEL="${MODEL:-$HOME_ROOT/projects/telelogs/shared/models/Qwen3.5-122B-A10B}"
PY="${PY:-$HOME_ROOT/venv-vllm-nightly/bin/python}"
export PATH="$(dirname "$PY"):$PATH"
INFRA=$HOME_ROOT/projects/telelogs/runs/teleqna-sft/infra
DATA=$HOME_ROOT/projects/telelogs/runs/teleqna-sft/data
OUT="${OUT:-$HOME_ROOT/projects/telelogs/runs/teleqna-sft/results/vllm122b}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

# Both cards, not just the first. TP=2 dies on whichever card is short, and the
# error surfaces as a confusing NCCL failure rather than as an OOM.
IFS=',' read -ra IDX <<< "$CUDA_VISIBLE_DEVICES"
for c in "${IDX[@]}"; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$c")
  echo "card $c free=${free}MiB"
  [ "$free" -gt 135000 ] || { echo "ABORT: card $c has only ${free}MiB free"; exit 14; }
done

# Check the weights are all there before spending two card-loads on discovering
# they are not. The downloader has already produced a truncated shard once.
python3 - "$MODEL/.manifest" "$MODEL" <<'PY'
import os, sys
man, dest = sys.argv[1], sys.argv[2]
bad = []
for line in open(man):
    name, _, want = line.rstrip("\n").rpartition(" ")
    want = int(want)
    path = os.path.join(dest, name)
    have = os.path.getsize(path) if os.path.exists(path) else 0
    if want and have < want:
        bad.append((name, have, want))
if bad:
    print(f"ABORT: {len(bad)} files short, e.g. {bad[0]}")
    sys.exit(15)
print("weights complete")
PY

# otlite first, because it is what was asked for and it is the cheapest of the
# three. GSMA/ot-lite is a separate repo from ot-full rather than a config inside
# it; make_otlite.py converts its teleqna split and checks that all 1,000 rows
# match an ot-full row on question+choices with no label disagreement, so a
# number here is directly comparable with the ot-full run below.
#
# Worth stating plainly: 894 of these 1,000 rows live in the frozen held-out
# 9,000. Scoring them costs nothing today because the 122B is an off-the-shelf
# model that no part of this project has trained on or selected against — the
# freeze exists to stop us tuning our own checkpoints against that split. What it
# does spend is the right to later pick between the 122B and an 8B arm on the
# strength of this number alone, since that comparison would then be informed by
# held-out rows.
echo "#### otlite START $(date -Iseconds)"
"$PY" "$INFRA/eval_dev_vllm.py" \
  --base "$MODEL" --data "$DATA/otlite1000.jsonl" --out-dir "$OUT" \
  --tag otlite --with-base --thinking --tp 2 --gpu-mem 0.95 \
  --max-model-len 8192 ${ENGINE_KWARGS:+--engine-kwargs "$ENGINE_KWARGS"}
echo "#### otlite DONE $(date -Iseconds)"

echo "#### dev-1000 START $(date -Iseconds)"
"$PY" "$INFRA/eval_dev_vllm.py" \
  --base "$MODEL" --data "$DATA/dev1000.jsonl" --out-dir "$OUT" \
  --tag dev1000 --with-base --thinking --tp 2 --gpu-mem 0.95 \
  --max-model-len 8192 ${ENGINE_KWARGS:+--engine-kwargs "$ENGINE_KWARGS"}
echo "#### dev-1000 DONE $(date -Iseconds)"

# Only after dev-1000 has a number. dev-1000 is the one split with twelve arms
# already scored on this harness, so it is where a 122B result means something
# immediately; the full 10,000 is for the headline and costs ten times as much.
echo "#### ot-full 10000 START $(date -Iseconds)"
"$PY" "$INFRA/eval_dev_vllm.py" \
  --base "$MODEL" --data "$DATA/otfull10000.jsonl" --out-dir "$OUT" \
  --tag otfull --with-base --thinking --tp 2 --gpu-mem 0.95 \
  --max-model-len 8192 ${ENGINE_KWARGS:+--engine-kwargs "$ENGINE_KWARGS"}
echo "#### ALL DONE $(date -Iseconds)"
