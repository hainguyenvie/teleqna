#!/usr/bin/env bash
# Big-run post-processing: agree-only self-replay anchors for tiers 3, 4 and 5 (Wikipedia windows) (card 7 after tier-4 gen), then two packs:
#   data/kit/big/pack         = tier-1 (fv30) + tier-2 + tier-3 + tier-4 views, chat-qa 50%, mcq-gold, anchors of all tiers
#   data/kit/big/pack_masked  = the same + masked-reconstruction rows (K=1) over every window
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"; mkdir -p data/kit/big
until grep -q "TIER3 shard 0 DONE rc=0" logs/tier3_s0.log 2>/dev/null && grep -q "TIER3 shard 1 DONE rc=0" logs/tier3_s1.log 2>/dev/null && grep -q "TIER4 DONE rc=0" logs/tier4_s0.log 2>/dev/null && grep -q "TIER5 DONE rc=0" logs/tier5_s0.log 2>/dev/null; do sleep 120; done
for i in $(seq 1 720); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7); [ "$used" -le 1000 ] && break; sleep 30; done
for t in tier3 tier4 tier5; do
  echo "#### SELFREPLAY-$t START $(date -Iseconds)"
  CUDA_VISIBLE_DEVICES=7 GPUMEM=0.90 timeout --foreground 14400 "$PY" -u code/kit/selfreplay_tier1.py "data/kit/$t/views_s*.jsonl" data/kit/$t/anchor_selfreplay.jsonl > logs/${t}_selfreplay.log 2>&1
  grep -q SELFREPLAY_DONE logs/${t}_selfreplay.log || { echo "ABORT selfreplay-$t"; tail -3 logs/${t}_selfreplay.log; exit 1; }; grep "anchor rows" logs/${t}_selfreplay.log
done
echo "#### PACK-BIG START $(date -Iseconds)"
"$PY" -u code/kit/pack_tier1.py --views "data/kit/tier1/views_s*.jsonl,data/kit/tier2/views_s*.jsonl,data/kit/tier3/views_s*.jsonl,data/kit/tier4/views_s*.jsonl,data/kit/tier5/views_s*.jsonl" \
  --factview-override "data/kit/tier1_fv30/views_s*.jsonl" \
  --anchor "data/kit/tier1/anchor_selfreplay.jsonl,data/kit/tier2/anchor_selfreplay.jsonl,data/kit/tier3/anchor_selfreplay.jsonl,data/kit/tier4/anchor_selfreplay.jsonl,data/kit/tier5/anchor_selfreplay.jsonl" \
  --chat-qa 0.5 --mcq-gold --out data/kit/big/pack > logs/big_pack.log 2>&1
grep -q PACK_DONE logs/big_pack.log || { echo "ABORT pack-big"; tail -3 logs/big_pack.log; exit 1; }
grep -E "override|kit docs|chat-format|anchors:|^anchor |replay|blocks of" logs/big_pack.log
cat data/kit/windows_keep.jsonl data/kit/windows_rest.jsonl data/kit/windows_tb_low.jsonl data/kit/tb_deep_windows.jsonl data/kit/tb_api_windows.jsonl > data/kit/big/windows_all.jsonl; wc -l data/kit/big/windows_all.jsonl
"$PY" -u code/kit/pack_masked.py --windows data/kit/big/windows_all.jsonl --k 1 --base-pack data/kit/big/pack --out data/kit/big/pack_masked --seed 7 > logs/big_pack_masked.log 2>&1
grep -q PACK_DONE logs/big_pack_masked.log || { echo "ABORT pack-big-masked"; tail -3 logs/big_pack_masked.log; exit 1; }; grep -E "masked rows|blocks" logs/big_pack_masked.log
echo "#### BIG-POST DONE $(date -Iseconds)"
