#!/usr/bin/env bash
# The two arms, sized from what the smoke run actually measured rather than
# from the guess in the plan.
#
# The smoke reported student sequences of median 108 tokens, p99 283, max 294 -
# against a MAXLEN of 2048. Padding to 2048 and running BATCH=4 puts a handful
# of short rows on a card that can hold two orders of magnitude more, and the
# measured 2.69 s/step at an effective batch of 8 works out to fourteen hours
# per arm. The sequences are short, so the fix is width, not patience:
#
#   MAXLEN 2048 -> 1024   still twice the longest student row, and the teacher
#                         rows carry an evidence span on top so the headroom
#                         goes to them, not to padding
#   BATCH 4 -> 16         same effective batch, a quarter of the steps
#   ACCUM 8 -> 2
#   EPOCHS 2 -> 1         the question is whether KL beats CE, and a second
#                         epoch answers the same question at twice the price
#
# The effective batch stays 32 so the learning rate does not have to move.
#
# Both arms read the identical file and differ only in ALPHA. That is the whole
# experiment: ALPHA=0 is plain SFT on this data, ALPHA=0.7 adds the KL against
# the same weights reading the evidence. Anything the data does, it does to
# both.
set -uo pipefail
HOME_ROOT=/home/tensara
R="$HOME_ROOT/projects/telelogs/runs/teleqna-sft"
PY="$HOME_ROOT/venv-vllm-nightly/bin/python"
MODEL="$HOME_ROOT/projects/telelogs/shared/models/OTel-2.0-31B-IT"
STATUS="$HOME_ROOT/PIPELINE_STATUS.txt"
export PATH="$HOME_ROOT/venv-vllm-nightly/bin:$PATH"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROJ_ROOT="$R" BASE_MODEL="$MODEL"
cd "$R"
say () { echo "[$(date -Iseconds)] $*" | tee -a "$STATUS"; }

wait_for_card () {
  local want=$1 skip=${2:-none} t=0
  while true; do
    for c in 4 6 0 1 7 5 2 3; do
      [ "$c" = "$skip" ] && continue
      local f; f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$c" 2>/dev/null || echo 0)
      [ "$f" -gt "$want" ] && { echo "$c"; return 0; }
    done
    t=$((t+1)); [ $t -gt 240 ] && return 1; sleep 60
  done
}

COMMON=(MAXLEN=1024 MAXCOMP=256 BATCH=16 ACCUM=2 EPOCHS=1 LR=1e-4 LORA_R=64
        SAVE_STEPS=1000 SAVE_LIMIT=1 ALLOW_OVERWRITE=1)
say "=== arms start: $(wc -l < data/synth/train_mix_100k.jsonl) rows, ${COMMON[*]} ==="

CA=$(wait_for_card 100000) || { say "no card for arm CE"; exit 1; }
CB=$(wait_for_card 100000 "$CA") || { say "no card for arm KD"; exit 1; }
say "arm CE (ALPHA=0.0) card $CA | arm KD (ALPHA=0.7) card $CB"

env "${COMMON[@]}" CUDA_VISIBLE_DEVICES=$CA \
  TRAIN_DATA="$R/data/synth/train_mix_100k.jsonl" RUN_NAME=armCE ALPHA=0.0 \
  nohup $PY infra/train_ctxdistill.py > "$HOME_ROOT/arm_ce.log" 2>&1 &
PA=$!
env "${COMMON[@]}" CUDA_VISIBLE_DEVICES=$CB \
  TRAIN_DATA="$R/data/synth/train_mix_100k.jsonl" RUN_NAME=armKD ALPHA=0.7 \
  nohup $PY infra/train_ctxdistill.py > "$HOME_ROOT/arm_kd.log" 2>&1 &
PB=$!
wait $PA; RA=$?
wait $PB; RB=$?
say "arm CE rc=$RA | arm KD rc=$RB"
[ $RA -ne 0 ] && tail -25 "$HOME_ROOT/arm_ce.log" >> "$STATUS"
[ $RB -ne 0 ] && tail -25 "$HOME_ROOT/arm_kd.log" >> "$STATUS"

# Both adapters and the untrained base scored off one model load. The base is
# not optional: the anchor ratio had to drop to 1:6.5, so both arms may sit
# below where they started, and only the base says whether that happened.
say "scoring dev1000 no-think"
EC=$(wait_for_card 100000) || { say "no card for eval"; exit 1; }
ADAPT=()
[ -d "$R/models/ctxdistill-armCE-adapter" ] && ADAPT+=(--adapter "armCE=$R/models/ctxdistill-armCE-adapter")
[ -d "$R/models/ctxdistill-armKD-adapter" ] && ADAPT+=(--adapter "armKD=$R/models/ctxdistill-armKD-adapter")
CUDA_VISIBLE_DEVICES=$EC $PY infra/eval_dev_vllm.py \
  --base "$MODEL" --data data/dev1000.jsonl \
  --out-dir results/ctxdistill --tag arms --with-base "${ADAPT[@]}" \
  --max-new 512 --tp 1 --gpu-mem 0.90 --max-model-len 4096 \
  --max-lora-rank 64 > "$HOME_ROOT/eval_arms.log" 2>&1 || say "eval FAILED"
say "=== ARMS DONE ==="
grep -hoE '"(tag|accuracy|acc|n)": *[^,}]*' "$R"/results/ctxdistill/*.json 2>/dev/null | tail -30 >> "$STATUS"
