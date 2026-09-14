#!/usr/bin/env bash
# The 12,672 windows only the 31B needs: pulled by the 541 rows it gets wrong
# that the 122B gets right. Same teacher, same gate, same prompt as gen-rest -
# the only difference is which rows chose the windows.
#
# Worth generating even though the 122B answers these correctly. The student is
# the 31B, and a fact the teacher already knows is exactly the fact the student
# is missing; targeting the teacher's errors would train for the wrong gap.
set -euo pipefail
HOME_ROOT=/home/tensara
ROOT="$HOME_ROOT/projects/telelogs/runs/teleqna-sft"
PY="$HOME_ROOT/venv-vllm-nightly/bin/python"
MODEL="$HOME_ROOT/projects/telelogs/shared/models/Qwen3.5-122B-A10B"
export PATH="$HOME_ROOT/venv-vllm-nightly/bin:$PATH"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CARDS="${CARDS:-4,6}"
IFS=',' read -ra IDX <<< "$CARDS"
for c in "${IDX[@]}"; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$c")
  echo "card $c free=${free}MiB"
  [ "$free" -gt 135000 ] || { echo "ABORT: card $c has only ${free}MiB free"; exit 14; }
done
export CUDA_VISIBLE_DEVICES="$CARDS"

mkdir -p "$ROOT/data/synth"
echo "#### gen-extra START $(date -Iseconds)"
"$PY" "$ROOT/infra/gen_targeted_qa.py" \
  --chunks "$ROOT/data/chunks_extra31b.jsonl" \
  --test "$ROOT/data/otfull10000.jsonl" \
  --out "$ROOT/data/synth/extra31b_items.jsonl" \
  --model "$MODEL" \
  --mode targeted \
  --max-targets 4 \
  --max-chunk-chars 6000 \
  --tp 2 --gpu-mem 0.94 --max-model-len 4096 --max-tokens 1200 \
  --engine-kwargs '{"language_model_only": true, "disable_custom_all_reduce": true, "max_num_seqs": 64}'
echo "#### gen-extra DONE rc=$? $(date -Iseconds)"
