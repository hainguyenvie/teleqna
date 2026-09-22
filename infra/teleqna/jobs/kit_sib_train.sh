#!/usr/bin/env bash
# Sibling stage, train part only (pack exists): SFT on cards 0-3 from the best ckpt (EMA on), eval, vd6.
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0
cd "$ROOT"; CARDS="${CARDS:-0,1,2,3}"; NP=$(echo "$CARDS" | tr "," "\n" | wc -l); ACC=$(( 32 / (NP * 2) ))
read -r INIT SCORE < <(bash jobs/best_ckpt.sh); echo "init=$INIT ($SCORE)"
for i in $(seq 1 480); do busy=0; for c in $(echo "$CARDS" | tr "," " "); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 30; done
sleep 20; echo "#### SIB TRAIN START init=$INIT cards=$CARDS $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=$CARDS timeout --foreground 43200 torchrun --nproc_per_node=$NP --master_port=29751 code/kit/train_tier1.py --pack data/kit/sib/pack --out models/kit/sib --init "$INIT" --epochs 1 --bs 2 --acc $ACC --lr 5e-6 --warm 50 --ema-every 10 --ema-decay 0.99 > logs/sib_train.log 2>&1
echo "#### SIB TRAIN DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/sib_train.log || { grep -E "Error|error" logs/sib_train.log | grep -v "^\[" | head -3; exit 1; }
CKPT=models/kit/sib/ep1 TAG=sib_ep1 CARDS=$(echo "$CARDS" | cut -d, -f1) bash jobs/eval_ckpt.sh > logs/eval_sib.log 2>&1; grep RESULT logs/eval_sib.log
CKPT=models/kit/sib/ep1_raw TAG=sib_ep1raw CARDS=$(echo "$CARDS" | cut -d, -f2) bash jobs/eval_ckpt.sh > logs/eval_sib_raw.log 2>&1; grep RESULT logs/eval_sib_raw.log
CARDS=$CARDS ROUND=6 INIT=models/kit/sib/ep1 N=100000 SEED=6 bash jobs/kit_vd_round.sh > logs/vd6_chain.log 2>&1; grep -E "RESULT|greedy" logs/vd6_chain.log
echo "#### SIBLING_CHAIN_DONE $(date -Iseconds)"
