#!/usr/bin/env bash
# Style stage, train part only (pack already built): waits for the DPO-3 chain, then SFT on cards 0-3 from the best ckpt, eval, vd5.
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0
cd "$ROOT"; CARDS="${CARDS:-0,1,2,3}"; N=$(echo "$CARDS" | tr "," "\n" | wc -l); ACC=$(( 32 / (N * 2) ))
until grep -q "dpo3_CHAIN_DONE\|ABORT\|rc=[1-9]" logs/dpo3_chain.log 2>/dev/null; do sleep 60; done
read -r INIT SCORE < <(bash jobs/best_ckpt.sh); echo "init=$INIT ($SCORE)"
for i in $(seq 1 480); do busy=0; for c in $(echo "$CARDS" | tr "," " "); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 30; done
sleep 20
echo "#### STYLE TRAIN START init=$INIT $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=$CARDS timeout --foreground 43200 torchrun --nproc_per_node=$N --master_port=29721 code/kit/train_tier1.py --pack data/kit/style/pack --out models/kit/style --init "$INIT" --epochs 1 --bs 2 --acc $ACC --lr 5e-6 --warm 50 --train-only mlp > logs/style_train.log 2>&1
echo "#### STYLE TRAIN DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/style_train.log || { grep -E "Error|error" logs/style_train.log | head -3; exit 1; }
CKPT=models/kit/style/ep1 TAG=style_ep1 CARDS=0 bash jobs/eval_ckpt.sh > logs/eval_style.log 2>&1; grep RESULT logs/eval_style.log
CARDS=$CARDS ROUND=5 INIT=models/kit/style/ep1 N=100000 SEED=5 bash jobs/kit_vd_round.sh > logs/vd5_chain.log 2>&1; grep -E "RESULT|greedy" logs/vd5_chain.log
echo "#### STYLE_CHAIN_DONE $(date -Iseconds)"
