#!/usr/bin/env bash
# The leakage sweep: the answer key in context, buried to varying depths.
#
# NOTHING HERE IS SUBMITTABLE. Every arm is fed the benchmark's own
# `explanation` field. These numbers exist to set the discount on honest ones,
# exactly as the "fit on all 10,000, score on the same 10,000" run does for
# prompt tuning.
#
# Runs on the login pod, not through a job runner. Both runners are occupied
# (teleqna-sft is generating the Tele-Data set on card 4, spare4 is five hours
# into GRPO on card 0) and this sweep is 40 minutes of a card that is sitting
# idle. The login pod has the full device set injected and a working nightly
# vLLM, which is how run_122b_eval.sh reaches cards 1 and 5.
#
# **This is a different serving stack from every stored arm** — vLLM 0.26 rc
# here against 0.11.0 for base/dpo3/grpo/distill/ceiling. Cross-stack deltas are
# meaningless, which is why the first two arms re-measure base and the K=1
# ceiling on this stack. Read the sweep against those, never against 75.20.
#
# Card 1, alone. Cards 1+5 are the pair reserved for the 122B, whose weights are
# at 232GB of ~250GB — so this takes one card and finishes before that lands.
#
# Card selection is dynamic, per arm. This node is shared and its free memory
# moves on a minute scale: cards 1 and 5 went 143GB -> 23GB mid-sweep when the
# 122B engine came up, and card 4 keeps ~107GB held by the orphaned vLLM child
# of a job that was killed (no exec into that pod, so it cannot be reaped). A
# card chosen once at the top is a card that is gone by arm three.
set -euo pipefail
HOME_ROOT=/home/tensara
MODEL="$HOME_ROOT/projects/telelogs/shared/hf-cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
PY="$HOME_ROOT/venv-vllm-nightly/bin/python"
INFRA="$HOME_ROOT/projects/telelogs/runs/teleqna-sft/infra"
DATA="$HOME_ROOT/projects/telelogs/runs/teleqna-sft/data"
OUT="$HOME_ROOT/projects/telelogs/runs/teleqna-sft/results/haystack"
# The venv's bin must be on PATH, not just its python. vLLM's compile step
# shells out to `ninja`, and invoking the interpreter by absolute path leaves
# PATH pointing at the system dirs where ninja does not exist — the engine then
# dies with a bare FileNotFoundError from a worker subprocess, three frames
# below anything that names the real problem.
export PATH="$HOME_ROOT/venv-vllm-nightly/bin:$PATH"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# The cgroup gives this pod a CPU quota; torch's default OMP pool sizes itself
# to the host's core count and then thrashes against it.
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0)
NEED_MIB="${NEED_MIB:-28000}"     # 8B weights are 16.3GB; the rest is KV cache

# Writes "<index> <free_mib>" for the emptiest card, or nothing if none clears
# NEED_MIB. Cards named in $AVOID are skipped: 0 and 4 are this project's own
# GRPO and generation work, and taking them would preempt the long-running jobs
# this measurement is meant to inform.
pick_card() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | tr -d ' ' | awk -F, -v need="$NEED_MIB" -v avoid="${AVOID:-0,4}" '
        BEGIN { split(avoid, a, ","); for (i in a) skip[a[i]] = 1 }
        !skip[$1] && $2 >= need && $2 > best { best = $2; idx = $1 }
        END { if (best) print idx, best }'
}

# base and expl re-measure the A/A references on this stack; the rest are the
# sweep. 8192 is enough for every arm: the deepest carries ~1,800 tokens of
# notes under a 6,000-token thinking budget, and a window we never fill is KV
# cache reserved out of the only resource that is scarce here.
for arm in base expl hay4 hay16 hay64 miss64; do
  case "$arm" in
    base) f="$DATA/dev1000.jsonl" ;;
    expl) f="$DATA/dev1000_expl.jsonl" ;;
    *)    f="$DATA/dev1000_${arm}.jsonl" ;;
  esac
  [ -s "$f" ] || { echo "SKIP $arm: missing $f"; continue; }
  if [ -s "$OUT/${arm}_base_think.json" ]; then echo "skip $arm (done)"; continue; fi

  for try in $(seq 1 60); do
    read -r CARD FREE <<< "$(pick_card)"
    [ -n "${CARD:-}" ] && break
    [ "$try" = 1 ] && echo "no card with ${NEED_MIB}MiB free — waiting $(date -Iseconds)"
    sleep 30
  done
  [ -n "${CARD:-}" ] || { echo "ABORT: no usable card for $arm"; exit 14; }
  # Take at most ~45% of a card even when more is free: this node is shared and
  # a 1,000-row eval does not need 100GB of KV cache.
  GM=$(python3 -c "print(round(max(0.18, min(0.45, ($FREE - 6000) / $TOTAL)), 2))")
  export CUDA_VISIBLE_DEVICES="$CARD"
  echo "#### $arm START $(date -Iseconds)  card=$CARD free=${FREE}MiB gpu_mem=$GM"
  set +e
  timeout 5400 "$PY" "$INFRA/eval_dev_vllm.py" \
    --base "$MODEL" --data "$f" --out-dir "$OUT" --tag "$arm" \
    --with-base --thinking --gpu-mem "$GM" --max-model-len 8192
  rc=$?
  set -e
  [ $rc -ne 0 ] && echo "#### $arm FAILED rc=$rc $(date -Iseconds)"
  echo "#### $arm DONE $(date -Iseconds)"
done
echo "#### ALL DONE $(date -Iseconds)"
