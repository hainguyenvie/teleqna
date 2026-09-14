#!/usr/bin/env bash
# One sharded sampling arm on a named prompt set. Generalises
# run_passk_otfull.sh so the reasoning arm can be launched over one set of
# cards while its greedy control runs on another at the same time -- the two
# have to come from the same weights and the same parser or the comparison
# with the 79.39 already on file means nothing.
#
# Env: DATA (basename under data/), OUT (basename under results/),
#      CARDS, K, TEMP, MAXTOK.
set -uo pipefail
HOME_ROOT=/home/tensara
SFT="$HOME_ROOT/projects/telelogs/runs/teleqna-sft"
PY="${PY:-$HOME_ROOT/venv-vllm-nightly/bin/python}"
MODEL="${MODEL:-$HOME_ROOT/projects/telelogs/shared/models/OTel-2.0-31B-IT}"
DATA="$SFT/data/${DATA:?set DATA}"
OUTDIR="$SFT/results/${OUT:?set OUT}"
LOGDIR="$SFT/logs/${OUT}"
CARDS="${CARDS:-0 2 3 4 6 7}"
K="${K:-8}"
TEMP="${TEMP:-1.0}"
MAXTOK="${MAXTOK:-4096}"
MIN_FREE_MIB="${MIN_FREE_MIB:-120000}"

export PATH="$(dirname "$PY"):$PATH"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTDIR" "$LOGDIR"
[ -s "$DATA" ] || { echo "ABORT: no $DATA"; exit 2; }

claimed=()
for c in $CARDS; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$c")
  if [ "$free" -lt "$MIN_FREE_MIB" ]; then
    echo "skip card $c: only ${free}MiB free"; continue
  fi
  claimed+=("$c")
done
N=${#claimed[@]}
[ "$N" -gt 0 ] || { echo "ABORT: no free card"; exit 14; }
echo "$OUT: claiming $N cards: ${claimed[*]}  k=$K T=$TEMP"

for i in "${!claimed[@]}"; do
  c=${claimed[$i]}
  CUDA_VISIBLE_DEVICES="$c" setsid nohup "$PY" "$SFT/infra/passk_shard.py" \
    --model "$MODEL" --data "$DATA" \
    --out "$OUTDIR/shard${i}of${N}.jsonl" \
    -k "$K" --shard "$i" --nshards "$N" --temperature "$TEMP" \
    --max-tokens "$MAXTOK" --max-model-len 6144 --gpu-mem 0.85 \
    > "$LOGDIR/shard${i}_card${c}.log" 2>&1 &
  echo "  card $c -> shard $i/$N pid=$!"
  sleep 5
done
echo "$N" > "$OUTDIR/nshards"
echo "$OUT: launched $N shards $(date -Iseconds)"
