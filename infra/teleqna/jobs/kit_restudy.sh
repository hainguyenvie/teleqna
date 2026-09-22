#!/usr/bin/env bash
# Focused re-study (label-free targeting): the tier-3 kit = trace-back windows of the questions where the 8B had margin < 0.9,
# 2 epochs from the best vd checkpoint at lr 1e-5 with agree-only anchors — does extra dose on the uncertain subset extract more?
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"; mkdir -p data/kit/restudy
LR="${LR:-1e-5}"; EPOCHS="${EPOCHS:-2}"
until grep -q "VD_CHAIN_DONE\|ABORT\|rc=[1-9]" logs/vd3_chain.log 2>/dev/null; do sleep 300; done
read -r INIT SCORE < <(bash jobs/best_ckpt.sh); echo "init=$INIT ($SCORE)"
[ -f data/kit/restudy/pack_ids.npy ] || { echo "#### RESTUDY PACK START $(date -Iseconds)"; "$PY" -u code/kit/pack_tier1.py --views "data/kit/tier3/views_s*.jsonl" --anchor "data/kit/tier3/anchor_selfreplay.jsonl" --chat-qa 0.5 --mcq-gold --out data/kit/restudy/pack > logs/restudy_pack.log 2>&1; grep -E "kit docs|anchors:|blocks of" logs/restudy_pack.log; }
grep -q PACK_DONE logs/restudy_pack.log || { echo "ABORT restudy pack"; exit 1; }
for i in $(seq 1 480); do busy=0; for c in 0 1 2 3 4 5 6 7; do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 30; done
echo "#### RESTUDY TRAIN START init=$INIT lr=$LR epochs=$EPOCHS $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout --foreground 86400 torchrun --nproc_per_node=8 --master_port=29690 code/kit/train_tier1.py --pack data/kit/restudy/pack --out models/kit/restudy --init "$INIT" --epochs "$EPOCHS" --bs 2 --acc 2 --lr "$LR" --save-every 6000 > logs/restudy_train.log 2>&1
echo "#### RESTUDY TRAIN DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/restudy_train.log || exit 1
for d in $(ls -d models/kit/restudy/ep* models/kit/restudy/step* 2>/dev/null); do t=restudy_$(basename $d); CKPT=$d TAG=$t CARDS=0 bash jobs/eval_ckpt.sh > logs/eval_$t.log 2>&1; grep RESULT logs/eval_$t.log; done
echo "#### RESTUDY_CHAIN_DONE $(date -Iseconds)"
