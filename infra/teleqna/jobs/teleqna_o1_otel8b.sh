#!/bin/bash
# OTel-LLM-8.3B-IT closed-book on all 10,000 rows, prompt byte-identical to the
# Qwen3-8B arm (teleqna_b0_nothink.sh). Same template, same temperature, same
# max-tokens, same permuted control. The only thing that changes is the weights,
# so the delta is attributable.
#
# What the comparison is against:
#   random baseline                     21.89%
#   distractor-shape heuristic          39.72%   <- the real floor
#   Qwen3-8B ours, no-think             71.95%   (b0_nothink)
#   qwen3-8b leaderboard                74.50%
#   gpt-5 leaderboard                   83.80%
#   OTel-LLM-8.3B-QnA leaderboard       91.20%   <- sibling, weights not released
#   OTel-2.0-LLM-31B-IT leaderboard     91.70%   <- column SOTA
#
# -IT is the instruction-tuned sibling of the -QnA variant, trained on the same
# public 606k-record RAG SFT set. That set is 0.03% multiple choice, so this also
# tests whether grounded open-ended QA transfers to the MCQ contract at all.
# A low parse rate here is a finding, not a bug — check parse_failures in the
# summary before reading the score.
set -euo pipefail
ROOT=/workspace/telelogs-bench4/teleqna
PY=/workspace/telelogs-bench4/dspy/.venv/bin/python
cd "$ROOT/code"

export TELEQNA_MODEL=OTel-LLM-8.3B-IT
# Served from the bench4 GPU pod on hgx046 (hgx45 and its otel45 service are
# gone — the node was reclaimed on 2026-08-05).
export VLLM_CHAT_URL=http://telelogs-bench4-vllm:8000/v1/chat/completions

# Smoke first. This model's SFT set is 0.03% multiple choice, so it may simply
# not emit "ANSWER: X" — and a 10,000-row run that ends in a parse-failure score
# teaches nothing it could not have taught on 100 rows. Abort above 20% failures
# and leave the raw responses for inspection rather than burning the full run.
#
# The smoke dir is wiped first: run_baseline.py resumes finished sample_ids, so
# a smoke that failed on an old parser would otherwise replay its old verdict
# forever. (The wipe has to happen here, inside the pod — the files are
# root-owned and the login host cannot remove them.) The full runs below are
# NOT wiped: resume is exactly what we want there.
rm -rf "$ROOT/results/o1_otel8b_smoke"
"$PY" run_baseline.py \
  --data "$ROOT/data/test.jsonl" \
  --out "$ROOT/results/o1_otel8b_smoke" \
  --limit 100 --workers 16 --max-tokens 32 --temperature 0.0

"$PY" - "$ROOT/results/o1_otel8b_smoke/summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
n, bad = s["total"], s["parse_failures"]
print(f"smoke: n={n} accuracy={s['accuracy']} parse_failures={bad} ({bad/n:.1%})")
if bad / n > 0.20:
    sys.exit("ABORT: the model is not honouring the ANSWER: contract; "
             "inspect results/o1_otel8b_smoke/results.jsonl before the full run")
PY

"$PY" run_baseline.py \
  --data "$ROOT/data/test.jsonl" \
  --out "$ROOT/results/o1_otel8b" \
  --workers 32 --max-tokens 32 --temperature 0.0

"$PY" run_baseline.py \
  --data "$ROOT/data/test.jsonl" \
  --out "$ROOT/results/o1_otel8b_perm" \
  --permute \
  --workers 32 --max-tokens 32 --temperature 0.0
