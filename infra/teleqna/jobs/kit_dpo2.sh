#!/usr/bin/env bash
# DPO round 2: same pairs (from vd3), 10x learning rate (3e-6) — round 1 at 5e-7 barely moved (loss ~0.69, pref-acc ~0.6).
# Cards 0 and 3 only (style generation holds 1,2). Eval on card 0.
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0
cd "$ROOT"; CARDS="${CARDS:-0,3}"; N=$(echo "$CARDS" | tr "," "\n" | wc -l); ACC=$(( 32 / (N * 2) )); LR="${LR:-3e-6}"; BETA="${BETA:-0.1}"; TAG="${TAG:-dpo2}"
for i in $(seq 1 480); do busy=0; for c in $(echo "$CARDS" | tr "," " "); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 30; done
echo "#### $TAG TRAIN START lr=$LR beta=$BETA cards=$CARDS $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=$CARDS timeout --foreground 43200 torchrun --nproc_per_node=$N --master_port=29712 code/kit/train_dpo_letter.py --pairs data/kit/dpo/pairs_dpo.jsonl --init models/kit/vd3/ep1 --out models/kit/$TAG --beta "$BETA" --lr "$LR" --bs 8 --acc $ACC --epochs 1 ${SFTMIX:+--sft-mix $SFTMIX} > logs/${TAG}_train.log 2>&1
echo "#### $TAG TRAIN DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/${TAG}_train.log || { tail -3 logs/${TAG}_train.log; exit 1; }
CKPT=models/kit/$TAG/ep1 TAG=${TAG}_ep1 CARDS=$(echo "$CARDS" | cut -d, -f1) bash jobs/eval_ckpt.sh > logs/eval_$TAG.log 2>&1; grep RESULT logs/eval_$TAG.log
echo "#### ${TAG}_CHAIN_DONE $(date -Iseconds)"
