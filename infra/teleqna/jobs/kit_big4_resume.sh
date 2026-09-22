#!/usr/bin/env bash
# Resume big run 4 from models/kit/big4/step4000 (training was killed at step ~4380 by a mis-targeted kill on 18/09 23:58 UTC).
# Same pack, same deterministic block order, schedule advanced to step 4000; optimizer moments restart (lr already 3.9e-6).
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0
cd "$ROOT"; CARDS="${CARDS:-0,1,2,3,4,5,6,7}"; NP=$(echo "$CARDS" | tr "," "\n" | wc -l)
for i in $(seq 1 120); do busy=0; for c in $(echo "$CARDS" | tr "," " "); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 30; done
echo "#### TRAIN-BIG4 RESUME START from step4000 cards=$CARDS $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=$CARDS timeout --foreground 172800 torchrun --nproc_per_node=$NP --master_port=29822 code/kit/train_fsdp.py --pack data/kit/big4/pack --out models/kit/big4 --init models/kit/big4/step4000 --start-step 4000 --bs 4 --acc 8 --lr 3e-5 --save-every 500 --no-ckpt > logs/big4r_train.log 2>&1
echo "#### TRAIN-BIG4 DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/big4r_train.log || { echo "ABORT resume"; exit 1; }
EVC=$(echo "$CARDS" | cut -d, -f1)
(CKPT=models/kit/big4/step4500 TAG=big4_step4500 CARDS=1 bash jobs/eval_ckpt.sh > logs/eval_big4_step4500.log 2>&1; grep RESULT logs/eval_big4_step4500.log) &
CKPT=models/kit/big4/ep1 TAG=big4_ep1 CARDS=$EVC bash jobs/eval_ckpt.sh > logs/eval_big4_ep1.log 2>&1; grep RESULT logs/eval_big4_ep1.log; wait
CARDS=$CARDS ROUND=9 INIT=models/kit/big4/ep1 N=100000 SEED=9 bash jobs/kit_vd_round.sh > logs/vd9_chain.log 2>&1; grep -E "RESULT|greedy" logs/vd9_chain.log
echo "#### BIG4_CHAIN_DONE $(date -Iseconds)"
