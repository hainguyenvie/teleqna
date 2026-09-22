#!/usr/bin/env bash
# After the plain tier-2 pack: build the mixed pack (same kit + chat-format qa 50% + gold MCQs in harness format).
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
cd "$ROOT"
until grep -q "POST-T2 DONE" logs/tier2_post.log 2>/dev/null; do grep -q ABORT logs/tier2_post.log 2>/dev/null && { echo "ABORT: post-t2 failed"; exit 1; }; sleep 120; done
echo "#### PACK-MIXED START $(date -Iseconds)"
OMP_NUM_THREADS=8 "$PY" -u code/kit/pack_tier1.py --views "data/kit/tier1/views_s*.jsonl,data/kit/tier2/views_s*.jsonl" --factview-override "data/kit/tier1_fv30/views_s*.jsonl" \
  --anchor "data/kit/tier1/anchor_selfreplay.jsonl,data/kit/tier2/anchor_selfreplay.jsonl" --chat-qa 0.5 --mcq-gold --out data/kit/tier2/pack_mixed > logs/tier2_mixed_pack.log 2>&1
grep -q PACK_DONE logs/tier2_mixed_pack.log || { echo "ABORT pack-mixed"; tail -3 logs/tier2_mixed_pack.log; exit 1; }
grep -E "override|kit docs|chat-format|anchors:|^anchor |replay|blocks of" logs/tier2_mixed_pack.log
echo "#### PACK-MIXED DONE $(date -Iseconds)"
