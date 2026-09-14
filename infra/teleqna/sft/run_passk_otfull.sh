#!/usr/bin/env bash
# Stage 0: the sampled-trace ceiling of OTel-2.0-31B-IT on ot-full, sharded
# across every idle card.
#
# This is the number the whole next campaign is gated on. pass@8 minus pass@1
# is the headroom that any method which reweights branches the policy already
# produces -- rejection sampling, GRPO -- is allowed to convert. If it is small,
# the "the knowledge is in there, it just picks wrong" story is false and the
# only remaining lever is putting knowledge in, which has failed three times.
#
# No-think only. The greedy think arm on ot-lite parses 52/1000 strict against
# 732 for no-think; measuring a ceiling through a broken output channel would
# measure the channel.
#
# Cards are claimed one at a time and each is checked free immediately before
# its shard starts, so a card that someone else takes between the survey and
# the launch is skipped rather than fought over. Shards are independent files;
# losing one costs that shard, not the run.
set -uo pipefail
HOME_ROOT=/home/tensara
SFT="$HOME_ROOT/projects/telelogs/runs/teleqna-sft"
PY="${PY:-$HOME_ROOT/venv-vllm-nightly/bin/python}"
MODEL="${MODEL:-$HOME_ROOT/projects/telelogs/shared/models/OTel-2.0-31B-IT}"
DATA="$SFT/data/otfull_passk.jsonl"
OUTDIR="$SFT/results/passk_otfull_base"
LOGDIR="$SFT/logs/passk"
CARDS="${CARDS:-0 2 3 4 6 7}"
K="${K:-8}"
MIN_FREE_MIB="${MIN_FREE_MIB:-120000}"

export PATH="$(dirname "$PY"):$PATH"     # worker JIT looks ninja up on PATH
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTDIR" "$LOGDIR"

# The prompt shape has to be the harness's, byte for byte, or pass@1 is not
# comparable with the greedy numbers already on file.
[ -s "$DATA" ] || "$PY" "$SFT/infra/make_passk_set.py" \
  --data "$SFT/data/otfull10000.jsonl" --out "$DATA"

# An orphaned EngineCore holds every byte it reserved and reads as a busy card.
for pid in $(pgrep -f "VLLM::EngineCore" 2>/dev/null); do
  [ "$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')" = "1" ] || continue
  echo "reaping orphaned EngineCore pid=$pid"; kill -9 "$pid" 2>/dev/null
done

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
echo "claiming $N cards: ${claimed[*]}"

for i in "${!claimed[@]}"; do
  c=${claimed[$i]}
  log="$LOGDIR/shard${i}_card${c}.log"
  CUDA_VISIBLE_DEVICES="$c" setsid nohup "$PY" "$SFT/infra/passk_shard.py" \
    --model "$MODEL" --data "$DATA" \
    --out "$OUTDIR/shard${i}of${N}.jsonl" \
    -k "$K" --shard "$i" --nshards "$N" \
    --max-tokens 4096 --max-model-len 6144 --gpu-mem 0.85 \
    > "$log" 2>&1 &
  echo "card $c -> shard $i/$N  log=$log  pid=$!"
  sleep 5
done
echo "$N" > "$OUTDIR/nshards"
echo "launched $N shards $(date -Iseconds)"
