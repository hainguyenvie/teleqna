#!/usr/bin/env bash
# UTR train relaunch (first attempt raced ens3 labels / HET gen for cards and OOMed). Pack exists; wait for the ens3 chain
# and merge sweep 4 to finish (serial queue), then train on all cards, eval, and mark UTR_CHAIN_DONE (appended to utr_chain.log).
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0
cd "$ROOT"; CARDS="${CARDS:-0,1,2,3,4,5,6,7}"; NP=$(echo "$CARDS" | tr "," "\n" | wc -l); INIT="${INIT:-models/kit/merge_f}"
until grep -q "ENS3_CHAIN_DONE\|ABORT" logs/ens3_chain.log 2>/dev/null; do sleep 120; done
until grep -q "MERGE SWEEP 4 END\|ABORT" logs/merge_sweep4.log 2>/dev/null; do sleep 120; done
for i in $(seq 1 720); do busy=0; for c in $(echo "$CARDS" | tr "," " "); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 30; done
sleep 20
echo "#### UTR TRAIN START (relaunch) init=$INIT $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=$CARDS timeout --foreground 43200 torchrun --nproc_per_node=$NP --master_port=29851 code/kit/train_fsdp.py --pack data/kit/utr/pack --out models/kit/utr --init "$INIT" --bs 4 --acc 2 --lr 5e-6 --warm-frac 0.05 --no-ckpt > logs/utr_train.log 2>&1
echo "#### UTR TRAIN DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/utr_train.log || { echo "ABORT utr train"; exit 1; }
CKPT=models/kit/utr/ep1 TAG=utr_ep1 CARDS=$(echo "$CARDS" | cut -d, -f1) bash jobs/eval_ckpt.sh > logs/eval_utr.log 2>&1; grep RESULT logs/eval_utr.log
echo "#### UTR_CHAIN_DONE $(date -Iseconds)"
