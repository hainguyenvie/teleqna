#!/usr/bin/env bash
# Sibling-contrast stage: generate within-window sibling MCQs (2 cards, 60k windows), SFT (MLP-only) from the best ckpt,
# eval, then a consistency round. Cards 0-3 only. Runs after the recall arm; PIT waits for this.
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0
cd "$ROOT"; mkdir -p data/kit/sib; CARDS="${CARDS:-0,1,2,3}"; NP=$(echo "$CARDS" | tr "," "\n" | wc -l); ACC=$(( 32 / (NP * 2) ))
until grep -q "RECALL_CHAIN_DONE\|ABORT\|rc=[1-9]" logs/recall_chain.log 2>/dev/null; do sleep 120; done
for i in $(seq 1 480); do busy=0; for c in 0 1; do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 30; done
echo "#### SIB GEN START $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=0 timeout --foreground 10800 "$PY" -u code/kit/gen_sibling.py --shard 0 --nshards 2 --seed 1300 --sample 60000 > logs/sib_s0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 timeout --foreground 10800 "$PY" -u code/kit/gen_sibling.py --shard 1 --nshards 2 --seed 1301 --sample 60000 > logs/sib_s1.log 2>&1 &
wait; grep -h "sibling MCQs\|Traceback" logs/sib_s0.log logs/sib_s1.log | cut -c1-200
cat data/kit/sib/rows_s0.jsonl data/kit/sib/rows_s1.jsonl > data/kit/sib/rows.jsonl
"$PY" -u code/kit/pack_chat.py --rows data/kit/sib/rows.jsonl --out data/kit/sib/pack > logs/sib_pack.log 2>&1; tail -2 logs/sib_pack.log
grep -q PACK_DONE logs/sib_pack.log || { echo "ABORT sib pack"; exit 1; }
read -r INIT SCORE < <(bash jobs/best_ckpt.sh); echo "init=$INIT ($SCORE)"
for i in $(seq 1 480); do busy=0; for c in $(echo "$CARDS" | tr "," " "); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 30; done
sleep 20; echo "#### SIB TRAIN START init=$INIT $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=$CARDS timeout --foreground 43200 torchrun --nproc_per_node=$NP --master_port=29750 code/kit/train_tier1.py --pack data/kit/sib/pack --out models/kit/sib --init "$INIT" --epochs 1 --bs 2 --acc $ACC --lr 5e-6 --warm 50 > logs/sib_train.log 2>&1
echo "#### SIB TRAIN DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/sib_train.log || exit 1
CKPT=models/kit/sib/ep1 TAG=sib_ep1 CARDS=$(echo "$CARDS" | cut -d, -f1) bash jobs/eval_ckpt.sh > logs/eval_sib.log 2>&1; grep RESULT logs/eval_sib.log
CARDS=$CARDS ROUND=6 INIT=models/kit/sib/ep1 N=100000 SEED=6 bash jobs/kit_vd_round.sh > logs/vd6_chain.log 2>&1; grep -E "RESULT|greedy" logs/vd6_chain.log
echo "#### SIBLING_CHAIN_DONE $(date -Iseconds)"
