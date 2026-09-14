#!/usr/bin/env bash
# A second, independent judge on the 3,451 rows carrying arm G's label error.
#
# The 122B is a different model family from the 31B, so its disagreement is
# informative in a way a second sample from the same model is not. Three engine
# settings are not optional here, each learned from a 250GB crash:
# nightly vLLM + language_model_only, disable_custom_all_reduce, and the venv on
# PATH so the worker JIT can find ninja.
#
# GPU_MEM is 0.70, not the 0.92 the earlier 122B runs used. This is a shared
# machine and other tenants are holding 30-40GB on half the cards; at 0.92 vLLM
# demands 128.6 GiB free per card and aborts during engine init. 4 x 0.70 x 140
# = 391 GiB against 250 GiB of weights still leaves ample KV.
#
# The wait loop checks the memory the engine will ACTUALLY ask for. The previous
# version used a flat 100GB threshold, which passed on cards holding 103 GiB free
# and then died at init wanting 128.6 -- the check has to track the setting.
set -uo pipefail
export CUDA_VISIBLE_DEVICES="${CARDS:-2,3,4,6}"
GPU_MEM="${GPU_MEM:-0.70}"
HOME_ROOT=/home/tensara
PY="$HOME_ROOT/venv-vllm-nightly/bin/python"
export PATH="$(dirname "$PY"):$PATH"
SFT="$HOME_ROOT/projects/telelogs/runs/teleqna-sft"
MODEL="$HOME_ROOT/projects/telelogs/shared/models/Qwen3.5-122B-A10B"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for pid in $(pgrep -f "VLLM::EngineCore" 2>/dev/null || true); do
  [ "$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')" = "1" ] || continue
  echo "reaping orphaned EngineCore pid=$pid"; kill -9 "$pid" 2>/dev/null || true; sleep 8
done

IFS=',' read -ra IDX <<< "$CUDA_VISIBLE_DEVICES"
# +10GB of headroom on top of what the engine demands. The neighbours on this
# box grow: the previous attempt passed the check at 101,381MiB free and then
# died at init 40 seconds later when that card had drifted to 99,340MiB.
need=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${IDX[0]}" \
       | awk -v m="$GPU_MEM" '{printf "%d", $1*m + 10240}')
echo "engine demands $(echo "$need" | awk '{print $1-10240}')MiB; waiting for ${need}MiB free on each of $CUDA_VISIBLE_DEVICES"
for i in $(seq 1 360); do
  ok=1; state=""
  for c in "${IDX[@]}"; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$c")
    state="$state $c:${free}"
    [ "$free" -ge "$need" ] || ok=0
  done
  [ "$ok" = 1 ] && break
  [ $((i % 15)) -eq 1 ] && echo "waiting for cards ($state) $(date -Iseconds)"
  sleep 20
done
[ "$ok" = 1 ] || { echo "ABORT: cards never freed ($state)"; exit 14; }
echo "cards ok ($state) $(date -Iseconds)"

echo "#### 122B JUDGE START $(date -Iseconds)"
"$PY" -u "$SFT/infra/eval_dev_vllm.py" \
  --base "$MODEL" --data "$SFT/data/uncertain_rag8_strong.jsonl" \
  --out-dir "$SFT/results/landscape" --tag uncertain_122b_ragstrong8 --with-base \
  --max-new 512 --tp 4 --gpu-mem "$GPU_MEM" --max-model-len 32768 \
  --engine-kwargs '{"language_model_only": true, "disable_custom_all_reduce": true, "max_num_seqs": 128}'
rc=$?
echo "#### 122B JUDGE DONE rc=$rc $(date -Iseconds)"
[ "$rc" = 0 ] || exit "$rc"

"$PY" -u "$SFT/infra/reparse.py" \
  --result "$SFT/results/landscape/uncertain_122b_ragstrong8_base_nothink512.json" \
  --out "$SFT/results/landscape/uncertain_122b_ragstrong8_reparsed.json"
echo "#### 122B REPARSE DONE $(date -Iseconds)"
