#!/usr/bin/env bash
# Extend the evidence gate to the rows it has never seen.
#
# The margin router only escalated 2,730 rows, so the gate is structurally blind
# to every row the model is CONFIDENT about -- including all 497 C_hard rows,
# which is precisely the slice arm V hardened by -2.21. Retrieval is CPU work,
# so this runs while the cards are busy with arm E.
set -uo pipefail
HOME_ROOT=/home/tensara
PY="$HOME_ROOT/venv-vllm-nightly/bin/python"
SFT="$HOME_ROOT/projects/telelogs/runs/teleqna-sft"
C="$HOME_ROOT/projects/telelogs/shared/corpora/tele-data"

echo "#### BLIND SET BUILD $(date -Iseconds)"
"$PY" -u "$SFT/infra/make_blind_set.py"

echo "#### RETRIEVE START $(date -Iseconds)"
"$PY" -u "$SFT/infra/retrieve_ctx.py" \
  --test "$SFT/data/blind6202.jsonl" \
  --corpus "$C/standard/standard.jsonl" --corpus "$C/arxiv/arxiv.jsonl" \
  --corpus "$C/wiki/wiki.jsonl" --corpus "$C/web/web.jsonl" \
  --out "$SFT/data/blind6202_rag16_strong.jsonl" \
  --report "$SFT/results/rag/retr_blind_strong.json" \
  -k 16 --top-docs 300 --win 800 --stride 400 --with-options
echo "#### RETRIEVE DONE rc=$? $(date -Iseconds)"

# k=16 overran 40k tokens on the escalated set; windows come back rank-ordered so
# the first 8 are the best 8, and that fits 32k with room to spare.
"$PY" -u "$SFT/infra/trim_ctx.py" \
  --src "$SFT/data/blind6202_rag16_strong.jsonl" \
  --dst "$SFT/data/blind6202_rag8_strong.jsonl"
echo "#### TRIM DONE $(date -Iseconds)"
