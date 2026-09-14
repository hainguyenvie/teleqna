#!/usr/bin/env bash
# Evidence for the last 1,068 rows: the holdout, which was deliberately excluded
# from the blind retrieval so transfer stayed measurable.
#
# Running this closes the only gap in full coverage. It also permanently ends the
# holdout's usefulness -- once these rows are labelled and trained on, no clean
# transfer measurement is left. That is a real cost and it is why arm G (with the
# holdout intact) stays on record as the honest number; this arm exists to answer
# a different question, which is what the leaderboard would actually see.
set -uo pipefail
HOME_ROOT=/home/tensara
PY="$HOME_ROOT/venv-vllm-nightly/bin/python"
export PATH="$(dirname "$PY"):$PATH"
SFT="$HOME_ROOT/projects/telelogs/runs/teleqna-sft"
C="$HOME_ROOT/projects/telelogs/shared/corpora/tele-data"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

"$PY" - <<'PY'
import json, os
ROOT = os.path.expanduser("~/projects/telelogs/runs/teleqna-sft")
CANON = os.path.expanduser("~/projects/telelogs/runs/bench4/teleqna/data/test.jsonl")
ids = set(json.load(open(f"{ROOT}/results/landscape/no_evidence_ids.json")))
n = 0
with open(f"{ROOT}/data/holdout1068.jsonl", "w") as fh:
    for l in open(CANON):
        r = json.loads(l)
        if r["sample_id"] in ids:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
print(f"{n} rows -> data/holdout1068.jsonl")
assert n == len(ids)
PY

echo "#### RETRIEVE START $(date -Iseconds)"
"$PY" -u "$SFT/infra/retrieve_ctx.py" \
  --test "$SFT/data/holdout1068.jsonl" \
  --corpus "$C/standard/standard.jsonl" --corpus "$C/arxiv/arxiv.jsonl" \
  --corpus "$C/wiki/wiki.jsonl" --corpus "$C/web/web.jsonl" \
  --out "$SFT/data/holdout1068_rag16_strong.jsonl" \
  --report "$SFT/results/rag/retr_holdout_strong.json" \
  -k 16 --top-docs 300 --win 800 --stride 400 --with-options
echo "#### RETRIEVE DONE rc=$? $(date -Iseconds)"
"$PY" -u "$SFT/infra/trim_ctx.py" \
  --src "$SFT/data/holdout1068_rag16_strong.jsonl" \
  --dst "$SFT/data/holdout1068_rag8_strong.jsonl"
echo "#### TRIM DONE $(date -Iseconds)"
