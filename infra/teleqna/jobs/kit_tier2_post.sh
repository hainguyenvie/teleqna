#!/usr/bin/env bash
# Tier-2 post: wait for generation shards 1-3, self-replay anchor for tier-2 MCQs (one card), then pack
# kit = tier1 views + tier2 views, factview K=30 override for tier-1 windows, anchors tier1+tier2, replay 10%.
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CARDS:-6}"
cd "$ROOT"
while [ "$(grep -l "TIER2 shard . DONE rc=0" logs/tier2_s1.log logs/tier2_s2.log logs/tier2_s3.log 2>/dev/null | wc -l)" -lt 3 ]; do sleep 120; done
echo "#### SELFREPLAY-T2 START $(date -Iseconds)"
timeout --foreground 7200 "$PY" -u code/kit/selfreplay_tier1.py "data/kit/tier2/views_s*.jsonl" data/kit/tier2/anchor_selfreplay.jsonl > logs/tier2_selfreplay.log 2>&1
grep -q SELFREPLAY_DONE logs/tier2_selfreplay.log || { echo "ABORT selfreplay-t2"; exit 1; }; grep "anchor rows" logs/tier2_selfreplay.log
echo "#### PACK-T2 START $(date -Iseconds)"
"$PY" -u code/kit/pack_tier1.py --views "data/kit/tier1/views_s*.jsonl,data/kit/tier2/views_s*.jsonl" --factview-override "data/kit/tier1_fv30/views_s*.jsonl" \
  --anchor "data/kit/tier1/anchor_selfreplay.jsonl,data/kit/tier2/anchor_selfreplay.jsonl" --out data/kit/tier2/pack > logs/tier2_pack.log 2>&1
grep -q PACK_DONE logs/tier2_pack.log || { echo "ABORT pack-t2"; tail -3 logs/tier2_pack.log; exit 1; }
grep -E "override|kit docs|scrubbed|dropped|replay|anchor|blocks of" logs/tier2_pack.log
echo "#### POST-T2 DONE $(date -Iseconds)"
