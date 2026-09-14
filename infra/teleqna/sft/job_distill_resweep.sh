#!/usr/bin/env bash
# Re-measure the distill1 checkpoints under budgets that can actually hold this
# model's output.
#
# The first sweep scored 1.1-4.6% — below the 21.89% random floor, which is only
# reachable by systematically failing to emit a parsable line. It did not measure
# damage. eval_dev.py gives the no-think arm 8 new tokens, which is right for a
# base model that answers "ANSWER: C" and stops, and hopeless for one trained to
# state the governing spec fact and eliminate three distractors first. The model
# never reaches its own answer line.
#
# Three arms, and the comparison only means something if the base is measured
# the same way, so the base is re-run at each budget too:
#   think        the deployed mode, 6000 tokens          bar: base 75.10
#   nothink-256  same prompt, room to finish             bar: base at 256, run here
# Checkpoints are sampled, not swept: 25 / 100 / 236 covers the early, middle and
# final dose, and each thinking eval over 1,000 rows is not cheap.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
BASE=/workspace/telelogs-base
ROOT=/workspace/teleqna-spare4
INFRA="$BASE/runs/teleqna-sft/infra"
PYT="$BASE/venvs/venv-train/bin/python"
MODEL="$BASE/shared/hf-cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
DEV="$BASE/runs/teleqna-sft/data/dev1000.jsonl"
RUN="${RUN:-distill1}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n '1p')
echo "node card 0 free=${free_mib}MiB"
[ "$free_mib" -gt 100000 ] || { echo "ABORT: node card 0 is not free"; exit 14; }

run_eval () {  # name, adapter-args..., extra eval flags
  local out="$ROOT/results/$1.json"; shift
  if [ -e "$out" ]; then echo "skip $out"; return; fi
  timeout 10800 "$PYT" "$INFRA/eval_dev.py" --base "$MODEL" --batch 32 \
    --data "$DEV" --out "$out" "$@" || { echo "FAILED $out"; return; }
  "$PYT" - "$out" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))["summary"]
print(f"  -> {sys.argv[1].split('/')[-1]:44s} acc={s['accuracy']:.4f} "
      f"unparsed={s['unparsed']:4d} think={s['thinking']} max_new={s['max_new']}",
      flush=True)
PY
}

echo "#### base references at the budgets this model needs $(date -Iseconds)"
run_eval "dev1000_base_nothink256" --max-new 256

for step in 25 100 236; do
  CK="$ROOT/models/teleqna-$RUN-checkpoints/checkpoint-$step"
  [ -d "$CK" ] || { echo "missing $CK"; continue; }
  echo "#### $RUN step $step $(date -Iseconds)"
  run_eval "dev1000_${RUN}_step${step}_nothink256" --adapter "$CK" --max-new 256
  run_eval "dev1000_${RUN}_step${step}_think"      --adapter "$CK" --thinking
done
echo "#### RESWEEP DONE $(date -Iseconds)"
