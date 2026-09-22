#!/usr/bin/env bash
# PIT arm (Pre-Instruction-Tuning, arXiv 2402.12847): stage 1 = recall-QA rows ONLY from base (lr 1e-5), stage 2 = the
# tier-1 docs pack (pack_agree) from that checkpoint — QA before documents, vs. the mixed arm and the anchor-fix arm.
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0
cd "$ROOT"; CARDS="${CARDS:-0,1,2,3}"; NP=$(echo "$CARDS" | tr "," "\n" | wc -l); ACC=$(( 32 / (NP * 2) ))
until grep -q "SIBLING_CHAIN_DONE\|ABORT\|rc=[1-9]" logs/sibling_chain.log 2>/dev/null; do sleep 300; done
for i in $(seq 1 480); do busy=0; for c in $(echo "$CARDS" | tr "," " "); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 30; done
echo "#### PIT STAGE1 (QA only) START $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=$CARDS timeout --foreground 43200 torchrun --nproc_per_node=$NP --master_port=29740 code/kit/train_tier1.py --pack data/kit/recall/pack_qa --out models/kit/pit_qa --epochs 1 --bs 2 --acc $ACC --lr 1e-5 > logs/pit_qa_train.log 2>&1
grep -q TRAIN_DONE logs/pit_qa_train.log || { echo "ABORT pit stage1"; exit 1; }
CKPT=models/kit/pit_qa/ep1 TAG=pit_qa_ep1 CARDS=$(echo "$CARDS" | cut -d, -f1) bash jobs/eval_ckpt.sh > logs/eval_pit_qa.log 2>&1; grep RESULT logs/eval_pit_qa.log
for i in $(seq 1 480); do busy=0; for c in $(echo "$CARDS" | tr "," " "); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 30; done
echo "#### PIT STAGE2 (docs) START $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=$CARDS timeout --foreground 86400 torchrun --nproc_per_node=$NP --master_port=29741 code/kit/train_tier1.py --pack data/kit/tier1/pack_agree --out models/kit/pit --init models/kit/pit_qa/ep1 --epochs 1 --bs 2 --acc $ACC --lr 1e-5 > logs/pit_train.log 2>&1
echo "#### PIT TRAIN DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/pit_train.log || exit 1
CKPT=models/kit/pit/ep1 TAG=pit_ep1 CARDS=$(echo "$CARDS" | cut -d, -f1) bash jobs/eval_ckpt.sh > logs/eval_pit.log 2>&1; grep RESULT logs/eval_pit.log
CUDA_VISIBLE_DEVICES=$(echo "$CARDS" | cut -d, -f2) "$PY" -u code/kit/recall_probe.py --ckpt models/kit/pit/ep1 --tag pit_ep1 > logs/recall_probe_pit.log 2>&1; grep -E "recall" logs/recall_probe_pit.log
echo "#### PIT_CHAIN_DONE $(date -Iseconds)"
