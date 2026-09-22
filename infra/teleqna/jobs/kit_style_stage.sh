#!/usr/bin/env bash
# Style-matched contrastive MCQs (question form like the test: What is/are, What does, no "Which of the following"),
# 60k windows on 2 cards, then a cm-style stage from the best checkpoint, then a consistency round. Runs after the DPO stage.
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"; mkdir -p data/kit/style; CARDS="${CARDS:-0,1,2,3}"; N=$(echo "$CARDS" | tr "," "\n" | wc -l); ACC=$(( 32 / (N * 2) ))
until grep -q "dpo_CHAIN_DONE\|ABORT\|rc=[1-9]" logs/dpo_chain.log 2>/dev/null; do sleep 120; done
POOL="data/kit/windows_keep.jsonl,data/kit/windows_rest.jsonl,data/kit/windows_tb_low.jsonl,data/kit/tb_deep_windows.jsonl,data/kit/tb_api_windows.jsonl"
for i in $(seq 1 480); do busy=0; for c in $(echo "$CARDS" | cut -d, -f2,3 | tr "," " "); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 30; done
echo "#### STYLE GEN START $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=$(echo "$CARDS" | cut -d, -f2) timeout --foreground 10800 "$PY" -u code/kit/gen_contrastive.py --windows-file data/kit/big/windows_all.jsonl --pool "$POOL" --out data/kit/style --shard 0 --nshards 2 --seed 1100 --style --sample 60000 --gpu-mem 0.90 > logs/style_s0.log 2>&1 &
CUDA_VISIBLE_DEVICES=$(echo "$CARDS" | cut -d, -f3) timeout --foreground 10800 "$PY" -u code/kit/gen_contrastive.py --windows-file data/kit/big/windows_all.jsonl --pool "$POOL" --out data/kit/style --shard 1 --nshards 2 --seed 1101 --style --sample 60000 --gpu-mem 0.90 > logs/style_s1.log 2>&1 &
wait; grep -h "cmcq gate\|Traceback" logs/style_s0.log logs/style_s1.log | cut -c1-160
read -r INIT SCORE < <(bash jobs/best_ckpt.sh); echo "init=$INIT ($SCORE)"
"$PY" -u code/kit/mcq_rows.py --globs "data/kit/style/views_s*.jsonl" --out data/kit/style/rows.jsonl --rot 2 > logs/style_rows.log 2>&1; tail -1 logs/style_rows.log
"$PY" -u code/kit/pack_chat.py --rows data/kit/style/rows.jsonl --out data/kit/style/pack > logs/style_pack.log 2>&1; tail -2 logs/style_pack.log
grep -q PACK_DONE logs/style_pack.log || { echo "ABORT style pack"; exit 1; }
for i in $(seq 1 480); do busy=0; for c in $(echo "$CARDS" | tr "," " "); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 30; done
echo "#### STYLE TRAIN START init=$INIT $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=$CARDS timeout --foreground 43200 torchrun --nproc_per_node=$N --master_port=29720 code/kit/train_tier1.py --pack data/kit/style/pack --out models/kit/style --init "$INIT" --epochs 1 --bs 2 --acc $ACC --lr 5e-6 --warm 50 --train-only mlp > logs/style_train.log 2>&1
echo "#### STYLE TRAIN DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/style_train.log || exit 1
CKPT=models/kit/style/ep1 TAG=style_ep1 CARDS=0 bash jobs/eval_ckpt.sh > logs/eval_style.log 2>&1; grep RESULT logs/eval_style.log
ROUND=5 INIT=models/kit/style/ep1 N=100000 SEED=5 bash jobs/kit_vd_round.sh > logs/vd5_chain.log 2>&1; grep -E "RESULT|greedy" logs/vd5_chain.log
echo "#### STYLE_CHAIN_DONE $(date -Iseconds)"
