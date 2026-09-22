#!/usr/bin/env bash
# QA-first stage over ALL windows (PIT recipe, speculative): recall-QA rows for 114k windows (tier-1 + tiers 2-5), from BASE,
# lr 1e-5, 1 epoch, EMA on. Uses idle cards 4-7 so the sibling/PIT chains on 0-3 are not delayed. Output models/kit/qa_all.
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0
cd "$ROOT"; CARDS="${CARDS:-4,5,6,7}"; NP=$(echo "$CARDS" | tr "," "\n" | wc -l); ACC=$(( 32 / (NP * 2) ))
until grep -q "RERUN DONE" logs/recall_all_s1_chain.log 2>/dev/null && grep -q "RECALL-ALL GEN DONE" logs/recall_all_chain.log 2>/dev/null; do sleep 120; done
cat data/kit/recall/rows_s*.jsonl data/kit/recall_all/rows_s*.jsonl > data/kit/recall_all/rows_all.jsonl; wc -l data/kit/recall_all/rows_all.jsonl
"$PY" -u code/kit/pack_chat.py --rows data/kit/recall_all/rows_all.jsonl --out data/kit/recall_all/pack_qa > logs/qa_all_pack.log 2>&1; tail -2 logs/qa_all_pack.log
grep -q PACK_DONE logs/qa_all_pack.log || { echo "ABORT qa_all pack"; exit 1; }
for i in $(seq 1 480); do busy=0; for c in $(echo "$CARDS" | tr "," " "); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 30; done
echo "#### QA-ALL TRAIN START cards=$CARDS $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=$CARDS timeout --foreground 43200 torchrun --nproc_per_node=$NP --master_port=29760 code/kit/train_tier1.py --pack data/kit/recall_all/pack_qa --out models/kit/qa_all --epochs 1 --bs 2 --acc $ACC --lr 1e-5 --ema-every 10 --ema-decay 0.99 > logs/qa_all_train.log 2>&1
echo "#### QA-ALL TRAIN DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/qa_all_train.log || { tail -3 logs/qa_all_train.log; exit 1; }
CKPT=models/kit/qa_all/ep1 TAG=qa_all_ep1 CARDS=$(echo "$CARDS" | cut -d, -f1) bash jobs/eval_ckpt.sh > logs/eval_qa_all.log 2>&1; grep RESULT logs/eval_qa_all.log
CKPT=models/kit/qa_all/ep1_raw TAG=qa_all_ep1raw CARDS=$(echo "$CARDS" | cut -d, -f2) bash jobs/eval_ckpt.sh > logs/eval_qa_all_raw.log 2>&1; grep RESULT logs/eval_qa_all_raw.log
CUDA_VISIBLE_DEVICES=$(echo "$CARDS" | cut -d, -f1) "$PY" -u code/kit/recall_probe.py --ckpt models/kit/qa_all/ep1 --tag qa_all_ep1 --holdout "data/kit/recall*/holdout_s*.jsonl" > logs/recall_probe_qa_all.log 2>&1; grep recall logs/recall_probe_qa_all.log
echo "#### QA_ALL_CHAIN_DONE $(date -Iseconds)"
