#!/usr/bin/env bash
# Phase 0.3 — audit Tele-Data before a single token of it is trained on.
#
# Tele-Data (AliMaatouk/Tele-Data, ~2.5B tokens) is the only public corpus that
# covers the half of TeleQnA our 3GPP corpus cannot reach: 4,500 Research
# publications + 2,000 Research overview rows, whose source documents are
# arXiv papers and surveys and which carry no provenance tag at all.
#
# Two questions, in this order, and the first is blocking:
#
#   1. contamination. AdaptKey-Nemotron-30b shipped ~9,986/10,000 test rows in
#      its published training corpus. Tele-Data was built by the group that
#      also built TeleQnA's sibling benchmark, so overlap is plausible on
#      purpose rather than by accident. 8-gram shingle scan, then the verbatim
#      second stage, per shard.
#   2. coverage. The same rare-term measurement already run on the OTel SFT set
#      (mean 0.746, 2,231 rows fully covered) and on our own distill_v1 (mean
#      0.318, 139 rows fully covered). This is what decides whether the corpus
#      can teach the research half at all.
#
# CPU only — runs on the login pod, no GPU, no job runner.
set -uo pipefail
ROOT=/home/tensara/projects/telelogs
CORPUS=$ROOT/shared/corpora/tele-data
INFRA=$ROOT/runs/teleqna-sft/infra
TEST=$ROOT/runs/bench4/teleqna/data/test.jsonl
OUT=$ROOT/runs/teleqna-sft/results/teledata
mkdir -p "$OUT"

for shard in standard wiki arxiv web; do
  f="$CORPUS/$shard/$shard.jsonl"
  if [ ! -s "$f" ]; then echo "== $shard: not downloaded yet, skipping"; continue; fi
  # Re-runnable while the remaining shards are still coming down the wire.
  if [ -s "$OUT/coverage_${shard}_w400.json" ]; then
    echo "== $shard: already audited, skipping"; continue
  fi
  # A shard still being written would give a silently partial answer, so the
  # download log is checked rather than the file's mere existence.
  echo "== $shard  $(du -h "$f" | cut -f1)  $(date -Iseconds)"

  python3 "$INFRA/scan_contamination.py" "$f" \
    --target "teleqna=$TEST:question" --field content --k 8 \
    --out "$OUT/scan_$shard.json" > "$OUT/scan_$shard.log" 2>&1
  echo "   scan rc=$? -> $OUT/scan_$shard.json"

  python3 "$INFRA/verify_contamination.py" "$f" \
    --test "$TEST" --field content \
    --out "$OUT/verify_$shard.json" > "$OUT/verify_$shard.log" 2>&1
  echo "   verify rc=$? -> $OUT/verify_$shard.json"

  # Two coverage passes. The whole-record one is comparable with nothing else
  # here (a full arXiv paper is not an SFT record); the 400-word window is the
  # one to read, because it asks whether the fact sits in a passage a retriever
  # could return or a generator could be handed.
  python3 "$INFRA/measure_otel_coverage.py" --sft "$f" --test "$TEST" \
    --field content --out "$OUT/coverage_$shard.json" \
    > "$OUT/coverage_$shard.log" 2>&1
  echo "   coverage(record) rc=$? -> $OUT/coverage_$shard.json"

  python3 "$INFRA/measure_otel_coverage.py" --sft "$f" --test "$TEST" \
    --field content --window 400 --out "$OUT/coverage_${shard}_w400.json" \
    > "$OUT/coverage_${shard}_w400.log" 2>&1
  echo "   coverage(w400)   rc=$? -> $OUT/coverage_${shard}_w400.json"
done
echo "== ALL DONE $(date -Iseconds)"
