#!/usr/bin/env bash
# Big run 2: big pack (2.15M blocks, all views + masked) + recall-QA x3 (self-contained, 114k windows) + sibling-contrast
# MCQs + style MCQs; EMA on; 8 cards; 1 epoch; checkpoints every 6000 steps. INIT chosen by the PIT result:
# qa_all/ep1 (QA-first) if PIT beat the mixed arm by >= 0.3, else base. Then a consistency round on the EMA checkpoint.
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0
cd "$ROOT"; CARDS="${CARDS:-0,1,2,3,4,5,6,7}"; NP=$(echo "$CARDS" | tr "," "\n" | wc -l); ACC=$(( 32 / (NP * 2) )); LR="${LR:-3e-5}"
until grep -q "SIB-REST GEN DONE" logs/sib_rest_chain.log 2>/dev/null && grep -q PACK_DONE logs/big2_pack.log 2>/dev/null; do sleep 120; done
until grep -q "PIT_CHAIN_DONE\|ABORT\|rc=[1-9]" logs/pit_chain.log 2>/dev/null; do sleep 300; done
fl() { grep RESULT "$1" 2>/dev/null | sed -E "s/.*first-letter ([0-9.]+) .*/\1/"; }
PIT=$(fl logs/eval_pit.log); MIX=$(fl logs/eval_recall.log); INIT="${INIT:-}"
if [ -z "$INIT" ]; then if [ -n "$PIT" ] && [ -n "$MIX" ] && [ "$(echo "$PIT $MIX" | awk '{print ($1 >= $2 + 0.3)}')" = 1 ]; then INIT=models/kit/qa_all/ep1; else INIT=$HOME/projects/_shared/models/Qwen3-8B; fi; fi
echo "PIT=$PIT mixed=$MIX -> INIT=$INIT"
for i in $(seq 1 2880); do busy=0; for c in $(echo "$CARDS" | tr "," " "); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 60; done
sleep 30; echo "#### TRAIN-BIG2 START cards=$CARDS nproc=$NP acc=$ACC lr=$LR init=$INIT $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=$CARDS timeout --foreground 259200 torchrun --nproc_per_node=$NP --master_port=29770 code/kit/train_tier1.py --pack data/kit/big2/pack --out models/kit/big2 --init "$INIT" --epochs 1 --bs 2 --acc $ACC --lr $LR --save-every 6000 > logs/big2_train.log 2>&1
echo "#### TRAIN-BIG2 DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/big2_train.log || exit 1
EVC=$(echo "$CARDS" | cut -d, -f1)
CKPT=models/kit/big2/ep1 TAG=big2_ep1 CARDS=$EVC bash jobs/eval_ckpt.sh > logs/eval_big2_ep1.log 2>&1; grep RESULT logs/eval_big2_ep1.log
CKPT=models/kit/big2/ep1_raw TAG=big2_ep1raw CARDS=$(echo "$CARDS" | cut -d, -f2) bash jobs/eval_ckpt.sh > logs/eval_big2_ep1raw.log 2>&1; grep RESULT logs/eval_big2_ep1raw.log
CARDS=$CARDS ROUND=7 INIT=models/kit/big2/ep1 N=100000 SEED=7 bash jobs/kit_vd_round.sh > logs/vd7_chain.log 2>&1; grep -E "RESULT|greedy" logs/vd7_chain.log
echo "#### BIG2_CHAIN_DONE $(date -Iseconds)"
