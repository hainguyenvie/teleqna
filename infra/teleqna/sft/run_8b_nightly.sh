#!/usr/bin/env bash
# Score the bare Qwen3-8B on the nightly vLLM, from the login pod, card 0.
#
# Purpose is to make the 122B comparison legal. Every 8B arm was measured on
# vLLM 0.11.0; the 122B cannot be, because 0.11.0 does not register its
# architecture. Comparing them is therefore model+stack against model+stack,
# which this project forbids. The 122B cannot move down to 0.11.0, so the 8B
# moves up to the nightly, once, and the gap is read from there.
#
# This was first queued as spare4 job 94 and died instantly with rc127:
#   timeout: failed to run command '/home/tensara/venv-vllm-nightly/bin/python'
# The nightly venv lives in the login pod's home, outside the shared mount that
# spare4 sees as /workspace — the shared venvs are all under
# ~/projects/telelogs/venvs/, and this one was built in the wrong place. Copying
# a venv into the mount would not help either: its absolute paths are baked in.
# So it runs where the venv is.
#
# Single arm, no adapter sweep. The measured A/A floor for a multi-arm sweep
# (byte-identical adapter scored twice) was 0.50pp, and this number is going to
# be subtracted from another one, so it should be as clean as possible.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CARDS:-0}"
HOME_ROOT=/home/tensara
PY="$HOME_ROOT/venv-vllm-nightly/bin/python"
export PATH="$(dirname "$PY"):$PATH"     # worker JIT looks ninja up on PATH
SFT="$HOME_ROOT/projects/telelogs/runs/teleqna-sft"
INFRA="$SFT/infra"
BASE="$HOME_ROOT/projects/telelogs/shared/hf-cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
OUT="${OUT:-$SFT/results/vllm122b}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

for i in $(seq 1 120); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")
  [ "$free" -gt 45000 ] && break
  [ "$i" = 1 ] && echo "card $CUDA_VISIBLE_DEVICES busy (${free}MiB) — waiting $(date -Iseconds)"
  sleep 20
done
[ "$free" -gt 45000 ] || { echo "ABORT: card $CUDA_VISIBLE_DEVICES stuck at ${free}MiB"; exit 14; }
echo "card $CUDA_VISIBLE_DEVICES free=${free}MiB  $(date -Iseconds)"

echo "#### 8B base on nightly START $(date -Iseconds)"
"$PY" "$INFRA/eval_dev_vllm.py" \
  --base "$BASE" --data "$SFT/data/dev1000.jsonl" \
  --out-dir "$OUT" --tag nightly8b --with-base --thinking --gpu-mem 0.85
echo "#### 8B base on nightly DONE $(date -Iseconds)"

python3 - "$OUT/nightly8b_base_think.json" <<'PY'
import json, sys
got = json.load(open(sys.argv[1]))["summary"]["accuracy"]
old = 0.7520   # base thinking, vLLM 0.11.0, card 4
d = (got - old) * 100
print(f"8B base: nightly {got:.4f}  vs  0.11.0 {old:.4f}   stack delta = {d:+.2f}pp")
print("READ:", "stacks agree within the 0.50pp A/A floor — read the 122B against 75.20 directly"
      if abs(d) <= 0.5 else
      "stack effect exceeds the noise floor — subtract THIS number, not 75.20")
PY
