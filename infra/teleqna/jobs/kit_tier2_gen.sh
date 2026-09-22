#!/usr/bin/env bash
# Tier-2 generation: the windows NOT in the tier-1 keep set (so tier1 ∪ tier2 = all 51,637 strong-RAG windows).
# Usage: CARDS=7 SHARD=0 NSHARDS=4 GPUMEM=0.45 bash jobs/kit_tier2_gen.sh
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CARDS:?}"
cd "$ROOT"; mkdir -p data/kit/tier2
[ -f data/kit/windows_rest.jsonl ] || python3 - <<'PY'
import json
keep={json.loads(l)["win_id"] for l in open("data/kit/windows_keep.jsonl")}
n=0
with open("data/kit/windows_rest.jsonl","w",encoding="utf-8") as fh:
    for l in open("data/eg2/windows.jsonl",encoding="utf-8"):
        w=json.loads(l)
        if w["win_id"] in keep: continue
        fh.write(json.dumps(w,ensure_ascii=False)+"\n"); n+=1
print("rest windows",n)
PY
echo "#### TIER2 shard ${SHARD:?}/${NSHARDS:?} START card=$CUDA_VISIBLE_DEVICES $(date -Iseconds)"
timeout --foreground 172800 "$PY" -u code/kit/gen_kit.py --windows-file data/kit/windows_rest.jsonl --out data/kit/tier2 --shard "$SHARD" --nshards "$NSHARDS" --seed "$((100 + SHARD))" --gpu-mem "${GPUMEM:-0.45}" 2>&1 | grep -vE "it/s\]|Processed prompts"
echo "#### TIER2 shard $SHARD DONE rc=$? $(date -Iseconds)"
