#!/usr/bin/env bash
# Gate the Tele-Data grounded items, then expand the survivors into views.
#
# Queued behind 85-grounded-std-v2.sh: the runner takes one job at a time, so
# this starts when generation finishes and reads whatever is in the file then.
#
# The gate is unchanged from the run that produced distill_v1 — it was never the
# problem. It drops, in order: evidence that is not a verbatim span of its own
# chunk, answers not contained in that evidence, tautologies, document
# boilerplate, duplicates, anything overlapping the test set, and anything the
# student already answers closed-book (there is nothing to teach it). Last run:
# 21,687 in, 7,718 kept, 35.6%.
#
# The expansion afterwards is the part that is new, and it is the second of the
# three diagnosed defects. distill_v1 put each fact in the set once, in one
# surface form, with one option order — against a literature that puts the
# requirement at ~1000 exposures in varied contexts. expand_views.py turns each
# survivor into ~7 rows: three option permutations of the harness-format MCQ,
# plus recall, cloze, reverse and statement views.
set -euo pipefail
ROOT=/workspace/teleqna-sft
BASE_ROOT=/workspace/telelogs-base
PYV="$BASE_ROOT/venvs/venv/bin/python"     # vllm 0.11.0
MODEL="$BASE_ROOT/shared/hf-cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
CHUNKS="$ROOT/data/chunks_teledata_std.jsonl"
ITEMS="$ROOT/results/grounded_std_v2.jsonl"
KEPT="$ROOT/results/grounded_std_v2_kept.jsonl"
VIEWS="$ROOT/data/train_std_v2_views.jsonl"
TEST="$BASE_ROOT/runs/bench4/teleqna/data/test.jsonl"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
[ -s "$ITEMS" ] || { echo "ABORT: no generated items at $ITEMS"; exit 15; }
echo "items to gate: $(wc -l < "$ITEMS")"

TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sed -n '1p')
for i in $(seq 1 60); do
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n '1p')
  [ "$free_mib" -gt 45000 ] && break
  [ "$i" = 1 ] && echo "card has only ${free_mib}MiB free — waiting $(date -Iseconds)"
  sleep 30
done
[ "$free_mib" -gt 45000 ] || { echo "ABORT: under 45GB free on the card"; exit 14; }
GPU_MEM=$(python3 -c "print(round(max(0.25, min(0.85, ($free_mib - 8000) / $TOTAL_MIB)), 2))")

echo "#### gate START $(date -Iseconds)  gpu_mem=$GPU_MEM"
timeout 28800 "$PYV" "$ROOT/infra/validate_grounded_qa.py" \
  --items "$ITEMS" --chunks "$CHUNKS" --test "$TEST" \
  --out "$KEPT" --model "$MODEL" -k 4 --gpu-mem "$GPU_MEM"
echo "#### gate DONE $(date -Iseconds)  kept=$(wc -l < "$KEPT")"

# CPU only, seconds — but it lives here so the set is never half-built.
echo "#### expand START $(date -Iseconds)"
python3 "$ROOT/infra/expand_views.py" --items "$KEPT" --out "$VIEWS" \
  --perms 3 --views qa,cloze,reverse,statement --nota-rate 0.06
echo "#### expand DONE $(date -Iseconds)  rows=$(wc -l < "$VIEWS")"

# Coverage of the thing we are actually going to train on, against the
# benchmark. distill_v1 reached 105/1,999 Standards specifications rows at
# >=80%; the chunk pool this came from reaches 1,350. The gate and the
# generator both throw rows away, so the number that matters is measured here,
# at the end, not inferred from the corpus.
echo "#### coverage START $(date -Iseconds)"
python3 "$ROOT/infra/measure_otel_coverage.py" --sft "$VIEWS" --test "$TEST" \
  --field prompt --field completion \
  --out "$ROOT/results/teledata/coverage_train_std_v2.json"
echo "#### ALL DONE $(date -Iseconds)"
