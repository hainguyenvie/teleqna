#!/usr/bin/env bash
# After both generation shards finish: self-replay anchor (one card) then pack (CPU).
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CARDS:?}"
cd "$ROOT"
until grep -q "MCQFIX_DONE" logs/tier1_mcqfix.log 2>/dev/null; do sleep 120; done
[ "$(grep -c KIT_GEN_DONE logs/tier1_mcqfix.log)" -ge 2 ] || { echo "ABORT: mcq regeneration did not finish cleanly"; exit 1; }
echo "#### SELFREPLAY START card=$CUDA_VISIBLE_DEVICES $(date -Iseconds)"
timeout --foreground 7200 "$PY" -u code/kit/selfreplay_tier1.py > logs/tier1_selfreplay.log 2>&1
grep -q SELFREPLAY_DONE logs/tier1_selfreplay.log || { echo "ABORT selfreplay"; tail -3 logs/tier1_selfreplay.log; exit 1; }
echo "#### T2GATE START $(date -Iseconds)"
"$PY" -u code/kit/check_tier1_t2.py > logs/tier1_gate.log 2>&1 || { echo "ABORT gate build"; exit 1; }
for n in verbatim rewrite factview all; do
  OUT=results/landscape/t1gate_${n}_q3_8b_base_nothink512.json; [ -f "$OUT" ] && continue
  timeout --foreground 3600 "$PY" -u code/eval_dev_vllm.py --base "$HOME/projects/_shared/models/Qwen3-8B" --data data/eval/t1gate_$n.jsonl \
    --out-dir results/landscape --tag t1gate_${n}_q3_8b --with-base --max-new 512 --tp 1 --gpu-mem 0.90 --max-model-len 16384 > logs/eval_t1gate_$n.log 2>&1
done
"$PY" code/kit/score_tier1_gate.py | tee -a logs/tier1_gate.log
grep -q GATE_PASS logs/tier1_gate.log || { echo "ABORT: T2 gate failed — not training on this kit"; exit 1; }
echo "#### T2GATE PASS $(date -Iseconds)"
echo "#### PACK START $(date -Iseconds)"
"$PY" -u code/kit/pack_tier1.py > logs/tier1_pack.log 2>&1
grep -q PACK_DONE logs/tier1_pack.log || { echo "ABORT pack"; tail -3 logs/tier1_pack.log; exit 1; }
grep -E "kit docs|replay|anchor|blocks" logs/tier1_pack.log
echo "#### POST DONE $(date -Iseconds)"
