#!/usr/bin/env bash
# Big run 3 on the optimised stack: FSDP2 trainer, block 2048 pack (kit + masked + QA x3 + sibling x2 + style), 8 cards,
# 0.5M tokens/step, lr 3e-5, wd 0.1, cosine to 10%. Steps: (0) 40-step save test + 1k-row sanity eval of that checkpoint;
# (1) full epoch with checkpoints every 500 steps (evaluated by kit_big3_evalloop.sh); (2) ep1 eval; (3) consistency round.
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0
cd "$ROOT"; CARDS="${CARDS:-0,1,2,3,4,5,6,7}"; NP=$(echo "$CARDS" | tr "," "\n" | wc -l); BS="${BS:-4}"; ACC="${ACC:-8}"; LR="${LR:-3e-5}"
until grep -q "PACK-2K DONE" logs/pack_big2k_chain.log 2>/dev/null && grep -q PACK_DONE logs/big2k_pack.log 2>/dev/null; do sleep 60; done
until grep -q "SMOKE END" logs/smoke_chain.log 2>/dev/null; do sleep 30; done
for i in $(seq 1 480); do busy=0; for c in $(echo "$CARDS" | tr "," " "); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 30; done
echo "#### BIG3 SAVE-TEST START $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=$CARDS timeout --foreground 1800 torchrun --nproc_per_node=$NP --master_port=29800 code/kit/train_fsdp.py --pack data/kit/big2k/pack --out models/kit/big3_savetest --bs $BS --acc $ACC --lr $LR --max-steps 40 --save-every 40 --no-ckpt > logs/big3_savetest.log 2>&1
grep -E "blocks of|step 25|saved|Error" logs/big3_savetest.log | grep -v "^\[rank[1-7]\]" | tail -4 | cut -c1-160
[ -f models/kit/big3_savetest/step40/model.safetensors ] || [ -f models/kit/big3_savetest/step40/model.safetensors.index.json ] || { echo "ABORT: save test produced no checkpoint"; exit 1; }
CUDA_VISIBLE_DEVICES=$(echo "$CARDS" | cut -d, -f1) timeout --foreground 1200 "$PY" -u code/eval_dev_vllm.py --base models/kit/big3_savetest/step40 --data data/eval/otfull1k.jsonl --out-dir results/landscape --tag big3_savetest_1k --with-base --max-new 32 --tp 1 --gpu-mem 0.85 --max-model-len 4096 > logs/eval_big3_savetest.log 2>&1; grep -E "acc=|Error" logs/eval_big3_savetest.log | tail -1
echo "#### TRAIN-BIG3 START cards=$CARDS bs=$BS acc=$ACC lr=$LR $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=$CARDS timeout --foreground 172800 torchrun --nproc_per_node=$NP --master_port=29801 code/kit/train_fsdp.py --pack data/kit/big2k/pack --out models/kit/big3 --bs $BS --acc $ACC --lr $LR --save-every 500 --no-ckpt > logs/big3_train.log 2>&1
echo "#### TRAIN-BIG3 DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/big3_train.log || exit 1
EVC=$(echo "$CARDS" | cut -d, -f1)
CKPT=models/kit/big3/ep1 TAG=big3_ep1 CARDS=$EVC bash jobs/eval_ckpt.sh > logs/eval_big3_ep1.log 2>&1; grep RESULT logs/eval_big3_ep1.log
CARDS=$CARDS ROUND=8 INIT=models/kit/big3/ep1 N=100000 SEED=8 bash jobs/kit_vd_round.sh > logs/vd8_chain.log 2>&1; grep -E "RESULT|greedy" logs/vd8_chain.log
echo "#### BIG3_CHAIN_DONE $(date -Iseconds)"
