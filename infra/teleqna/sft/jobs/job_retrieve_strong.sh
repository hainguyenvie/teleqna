#!/usr/bin/env bash
# Legitimate retrieval, strong config, on the 2,730 rows the cascade escalates.
# What changes against the deployed otfull_rag8 (top_docs 20, win 400, k 8, no
# options, answer-in-context 37.2%):
#   --with-options   the harness hands the options to the system, so using them
#                    in the query is legitimate; it inflates the answer-in-ctx
#                    diagnostic, which is why accuracy is the number that counts
#   --top-docs 300   what the oracle-query sweep used when it reached 95% recall
#   --win 800        the fact needs to survive inside one window
# No --explanations anywhere: that flag puts the answer key in the query.
set -uo pipefail
HOME_ROOT=/home/tensara
PY="$HOME_ROOT/venv-vllm-nightly/bin/python"
SFT="$HOME_ROOT/projects/telelogs/runs/teleqna-sft"
C="$HOME_ROOT/projects/telelogs/shared/corpora/tele-data"
OPT="${OPT:---with-options}"; TAG="${TAG:-strong}"
"$PY" -u "$SFT/infra/retrieve_ctx.py" \
  --test "$SFT/data/escalate2730.jsonl" \
  --corpus "$C/standard/standard.jsonl" --corpus "$C/arxiv/arxiv.jsonl" \
  --corpus "$C/wiki/wiki.jsonl" --corpus "$C/web/web.jsonl" \
  --out "$SFT/data/escalate2730_rag16_$TAG.jsonl" \
  --report "$SFT/results/rag/retr_escalate_$TAG.json" \
  -k 16 --top-docs 300 --win 800 --stride 400 $OPT
echo "#### RETRIEVE DONE rc=$? $(date -Iseconds)"
