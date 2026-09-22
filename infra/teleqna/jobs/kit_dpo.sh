#!/usr/bin/env bash
# Letter-DPO stage: pairs from the best checkpoint's own letter probabilities on synthetic MCQs (label = evidence-backed
# generator answer), then DPO from that checkpoint, eval. Usage: INIT=models/kit/vd3/ep1 bash jobs/kit_dpo.sh
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"; mkdir -p data/kit/dpo; CARDS="${CARDS:-0,1,2,3}"; N=$(echo "$CARDS" | tr "," "\n" | wc -l); ACC=$(( 32 / (N * 2) )); INIT="${INIT:?}"; TAG="${TAG:-dpo}"; BETA="${BETA:-0.1}"; LR="${LR:-5e-7}"
until grep -q "CM2_VD4_DONE\|rc=[1-9]" logs/cm2_vd4_chain.log 2>/dev/null; do sleep 120; done
for i in $(seq 1 480); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0); [ "$used" -le 1000 ] && break; sleep 30; done
echo "#### DPO PAIRS START init=$INIT $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=0 timeout --foreground 7200 "$PY" -u code/kit/dpo_pairs.py --ckpt "$INIT" --n 120000 --out data/kit/dpo/pairs_$TAG.jsonl > logs/dpo_pairs_$TAG.log 2>&1; grep -E "gated|pairs:|Error" logs/dpo_pairs_$TAG.log
grep -q DPO_PAIRS_DONE logs/dpo_pairs_$TAG.log || { echo "ABORT dpo pairs"; exit 1; }
for i in $(seq 1 480); do busy=0; for c in $(echo "$CARDS" | tr "," " "); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 30; done
echo "#### DPO TRAIN START beta=$BETA lr=$LR $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=$CARDS timeout --foreground 43200 torchrun --nproc_per_node=$N --master_port=29710 code/kit/train_dpo_letter.py --pairs data/kit/dpo/pairs_$TAG.jsonl --init "$INIT" --out models/kit/$TAG --beta "$BETA" --lr "$LR" --bs 8 --acc $ACC --epochs 1 > logs/${TAG}_train.log 2>&1
echo "#### DPO TRAIN DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/${TAG}_train.log || { tail -5 logs/${TAG}_train.log; exit 1; }
CKPT=models/kit/$TAG/ep1 TAG=${TAG}_ep1 CARDS=0 bash jobs/eval_ckpt.sh > logs/eval_$TAG.log 2>&1; grep RESULT logs/eval_$TAG.log
echo "#### ${TAG}_CHAIN_DONE $(date -Iseconds)"
