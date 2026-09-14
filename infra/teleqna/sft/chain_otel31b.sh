#!/usr/bin/env bash
# Wait for verified OTel-2.0-31B-IT weights, then score otlite in both chat
# modes at once — thinking on card 1, no-think on card 5 — and go straight on
# to ot-full in whichever mode wins.
#
# Both modes, because which one is right here is not knowable in advance. Gemma
# 4's chat template does implement enable_thinking (it injects a <|think|> token
# into a synthesised system turn), so the flag is live rather than ignored. But
# OTel 2.0 is a post-train on SDG-expanded telecom text, and if that data
# carried no thinking traces the tuning may have trained the behaviour out.
# Guessing wrong costs a full re-run; running both costs one idle card for the
# ~15 minutes no-think takes.
#
# No gemma-4-31B-it control arm: dropped at the user's call. Recording what that
# costs so nobody re-derives it later — OTel 2.0 is a post-train of Gemma 4
# 31B-IT, so its score is Gemma-4's own knowledge plus whatever the 440B telecom
# tokens added, and without the base measured on the same set those two terms
# cannot be separated. What comes out below is therefore OTel-2.0's standing
# against our own models, which is a fair question on its own, and not evidence
# about whether the telecom post-training worked.
#
# This waits on fetch_chunked.py's DONE marker rather than on file sizes. The
# previous version waited on the curl fetcher's marker, which reported success
# on size alone — and 11 of 14 shards were *over* the published size, because
# `curl -C -` resumes through a 302 onto a CDN edge that sometimes answers 200
# and gets the whole file appended to the partial one. Size-complete is not
# correct; sha256 against the Hub's lfs.oid is.
set -uo pipefail
HOME_ROOT=/home/tensara
INFRA="$HOME_ROOT/projects/telelogs/runs/teleqna-sft/infra"
RESULTS="$HOME_ROOT/projects/telelogs/runs/teleqna-sft/results/otel31b"
L="$HOME_ROOT"

for i in $(seq 1 720); do
  grep -q "^#### FETCH DONE" "$L/fetch_chunked_otel.log" 2>/dev/null && break
  grep -q "^#### FETCH FAILED" "$L/fetch_chunked_otel.log" 2>/dev/null && {
    echo "ABORT: fetch reported FAILED"; exit 12; }
  [ "$i" = 1 ] && echo "waiting for verified weights $(date -Iseconds)"
  sleep 30
done
grep -q "^#### FETCH DONE" "$L/fetch_chunked_otel.log" || {
  echo "ABORT: weights never verified"; exit 12; }
echo "weights verified against published sha256 $(date -Iseconds)"

CARDS=1 MODE=think   SET=otlite1000 TAG=otlite_think \
  nohup "$INFRA/run_otel31b.sh" > "$L/otel31b_otlite_think.log"   2>&1 &
P1=$!
CARDS=5 MODE=nothink SET=otlite1000 TAG=otlite_nothink \
  nohup "$INFRA/run_otel31b.sh" > "$L/otel31b_otlite_nothink.log" 2>&1 &
P2=$!
echo "OTel: think pid $P1 (card 1), nothink pid $P2 (card 5)  $(date -Iseconds)"

wait $P2; echo "#### OTel nothink finished rc=$? $(date -Iseconds)"
wait $P1; echo "#### OTel think   finished rc=$? $(date -Iseconds)"

# ot-full straight after, in whichever mode otlite picked, so the 10,000-row
# number lands without waiting on a round trip through me. Ties go to thinking:
# on every model measured here it has been the stronger mode, and otlite is only
# 1,000 rows so a tie is well inside its own noise.
BEST=$(python3 -c '
import json, pathlib
d = pathlib.Path("'"$RESULTS"'")
def acc(name):
    p = d / name
    return json.load(open(p))["summary"]["accuracy"] if p.exists() else -1
t = acc("otlite_think_base_think.json")
n = acc("otlite_nothink_base_nothink512.json")
print("think" if t >= n else "nothink")
')
echo "#### otlite winner: $BEST $(date -Iseconds)"

CARDS=1 MODE="$BEST" SET=otfull10000 TAG="otfull_$BEST" \
  "$INFRA/run_otel31b.sh" > "$L/otel31b_otfull_$BEST.log" 2>&1
echo "#### OTel ot-full finished rc=$? $(date -Iseconds)"
echo "#### CHAIN DONE $(date -Iseconds)"
