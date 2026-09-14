#!/usr/bin/env bash
# Score the bare Qwen3-8B on the *nightly* vLLM, so the 122B gap is stack-matched.
#
# Every 8B arm in this project was measured on vLLM 0.11.0. The 122B cannot be:
# 0.11.0 does not register Qwen3_5MoeForConditionalGeneration, which is why the
# nightly venv exists at all. So the obvious comparison — 122B against our best
# 8B — is model+stack against model+stack, exactly the cross-stack read this
# project forbids after being burned by one.
#
# It cannot be fixed by moving the 122B down; it will not load on 0.11.0. It can
# be fixed by moving the 8B up, which is this job. One number, base thinking,
# same dev-1000, same scorer, on the nightly. Then:
#
#   122B(nightly) - 8B_base(nightly)   is a clean model delta
#   8B_base(nightly) - 8B_base(0.11.0) is the stack delta itself, worth knowing
#                                      because it prices every future comparison
#
# Card 0, which the GRPO sweep and job 93 have just released. Cards 1 and 5 are
# the 122B's ot-full run and must not be touched.
#
# Adapters are deliberately not swept here. A multi-arm sweep varies batch
# composition between arms, and the measured A/A floor (identical adapter scored
# twice in one sweep) was 0.50pp. This number wants to be as clean as the
# reference it will be subtracted from, so it is a single arm.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
HOME_ROOT=/home/tensara
BASE_ROOT=/workspace/telelogs-base
ROOT=/workspace/teleqna-spare4
SFT="$BASE_ROOT/runs/teleqna-sft"
INFRA="$SFT/infra"
PY="$HOME_ROOT/venv-vllm-nightly/bin/python"
export PATH="$(dirname "$PY"):$PATH"
BASE="$BASE_ROOT/shared/hf-cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for i in $(seq 1 240); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n '1p')
  [ "$free" -gt 45000 ] && break
  [ "$i" = 1 ] && echo "card 0 busy (${free}MiB) — waiting $(date -Iseconds)"
  sleep 30
done
[ "$free" -gt 45000 ] || { echo "ABORT: card 0 still busy"; exit 14; }
echo "card 0 free=${free}MiB  $(date -Iseconds)"

echo "#### 8B base on nightly START $(date -Iseconds)"
timeout 7200 "$PY" "$INFRA/eval_dev_vllm.py" \
  --base "$BASE" --data "$SFT/data/dev1000.jsonl" \
  --out-dir "$ROOT/results/vllm" --tag nightly \
  --with-base --thinking --gpu-mem 0.85
echo "#### 8B base on nightly DONE $(date -Iseconds)"

python3 - "$ROOT/results/vllm/nightly_base_think.json" <<'PY'
import json, sys
got = json.load(open(sys.argv[1]))["summary"]["accuracy"]
old = 0.7520   # base thinking, vLLM 0.11.0, card 4
print(f"8B base: nightly {got:.4f}  vs  0.11.0 {old:.4f}   stack delta = {(got-old)*100:+.2f}pp")
# 0.50pp is this stack's measured A/A floor; below that there is no stack effect
# worth carrying, and 122B-vs-8B can be read straight off the nightly numbers.
print("READ:", "stacks agree within the noise floor — compare 122B against 75.20 directly"
      if abs(got - old) * 100 <= 0.5 else
      "stack effect exceeds the noise floor — subtract THIS number, not 75.20")
PY
