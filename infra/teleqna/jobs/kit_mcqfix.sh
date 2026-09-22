#!/usr/bin/env bash
# Regenerate only the mcq view for both shards (sequentially on one card) after the parse crash.
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CARDS:-5}"
cd "$ROOT"
for S in 0 1; do
  echo "#### MCQFIX shard $S START $(date -Iseconds)"
  timeout --foreground 7200 "$PY" -u code/kit/gen_kit.py --only-mcq --shard $S --nshards 2 --seed $S 2>&1 | grep -vE "it/s\]|Processed prompts"
  grep -q "KIT_GEN_DONE" <(tail -c 2000 /dev/null) || true
done
python3 - <<'PY'
import json, collections
for s in (0, 1):
    c = collections.Counter(json.loads(l)["view"] for l in open(f"data/kit/tier1/views_s{s}.jsonl"))
    print("shard", s, dict(c))
PY
echo "MCQFIX_DONE $(date -Iseconds)"
