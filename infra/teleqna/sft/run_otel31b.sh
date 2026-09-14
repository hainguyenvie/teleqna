#!/usr/bin/env bash
# Score OTel-2.0-LLM-31B-IT on a TeleQnA set. One card, one mode, one arm.
#
# What this model is, since it matters for reading the number:
# a post-train of google/gemma-4-31B-it on ~440B telecom tokens that Red Hat's
# SDG Hub expanded out of ~15B raw GSMA/Open-Telco tokens (3GPP, ETSI, ITU,
# O-RAN, TM Forum, CAMARA). Apache 2.0, ungated, no benchmark numbers on the
# card. So its score is Gemma-4-31B's knowledge plus whatever the telecom
# post-train added, and only the second term is news. The control is this same
# script with MODEL=.../gemma-4-31B-it OUT=.../results/gemma4-31b — identical
# prompts, scorer, card and token budget, so the difference is the post-train
# and nothing else.
#
# The -IT suffix is a warning, not a detail. In OTel 1.0 the -IT weights scored
# 60.87 on this benchmark against Qwen3-8B's 71.95, while the sibling that
# actually carried MCQ (-QnA, 91.2 claimed) was never released. Same family,
# same suffix, so a low number here would be a repeat, not a surprise.
#
# TP=1 on purpose. 64GB of BF16 weights fit in one H200's 143GB with room for
# KV, and Gemma 4 is cheap on KV anyway: 50 of its 60 layers are sliding-window
# at 1024, only 10 are full attention. Staying on one card also skips the
# custom-all-reduce IPC failure that cost the 122B a load cycle to diagnose.
#
# MODE picks the chat-template branch. Gemma 4's template does implement
# enable_thinking (it injects a <|think|> token into a synthesised system turn),
# so --thinking is meaningful here rather than silently ignored — but the OTel
# post-train may not have kept the behaviour, which is why both modes get run.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CARDS:-1}"
HOME_ROOT=/home/tensara
PY="${PY:-$HOME_ROOT/venv-vllm-nightly/bin/python}"
export PATH="$(dirname "$PY"):$PATH"     # worker JIT looks ninja up on PATH
SFT="$HOME_ROOT/projects/telelogs/runs/teleqna-sft"
INFRA="$SFT/infra"
DATA="$SFT/data"
MODEL="${MODEL:-$HOME_ROOT/projects/telelogs/shared/models/OTel-2.0-31B-IT}"
OUT="${OUT:-$SFT/results/otel31b}"
SET="${SET:-otlite1000}"
MODE="${MODE:-think}"
TAG="${TAG:-${SET%%[0-9]*}_$MODE}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

# The weights must be all there. A short shard reads as a corrupt-checkpoint
# stack trace 200 seconds into a load, which is an expensive way to learn that
# the download was still running.
"$PY" - "$MODEL" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
idx = root / "model.safetensors.index.json"
if not idx.exists():
    sys.exit("ABORT: no index json — download has not finished")
want = sorted({v for v in json.loads(idx.read_text())["weight_map"].values()})
missing = [f for f in want if not (root / f).exists()]
if missing:
    sys.exit(f"ABORT: {len(missing)}/{len(want)} shards missing, first {missing[0]}")
part = [f for f in want if (root / f).stat().st_size < 1_000_000]
if part:
    sys.exit(f"ABORT: {len(part)} shards look truncated, first {part[0]}")
print(f"weights ok: {len(want)} shards, "
      f"{sum((root/f).stat().st_size for f in want)/1e9:.1f} GB")
PY

# Reap orphaned engines first. Killing a vLLM run by its parent pid leaves the
# EngineCore child alive, reparented to init, still holding every byte it had
# reserved -- one of those squatted 131.9 GB of card 1 for four hours while the
# thinking arm sat in the wait loop below and then aborted, and the ot-full run
# queued behind it aborted too. The test is narrow on purpose: ppid 1 means the
# parent is already gone, so nothing that is still supervised can match.
for pid in $(pgrep -f "VLLM::EngineCore" 2>/dev/null); do
  [ "$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')" = "1" ] || continue
  echo "reaping orphaned EngineCore pid=$pid $(date -Iseconds)"
  kill -9 "$pid" 2>/dev/null
  sleep 8
done

# Wait rather than abort: this is normally queued behind another eval that is
# still holding the card, and coming back in an hour to find it gave up at
# second one would waste the whole window.
for i in $(seq 1 360); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")
  [ "$free" -gt 100000 ] && break
  [ "$i" = 1 ] && echo "card $CUDA_VISIBLE_DEVICES busy (${free}MiB) — waiting $(date -Iseconds)"
  sleep 20
done
[ "$free" -gt 100000 ] || { echo "ABORT: card $CUDA_VISIBLE_DEVICES stuck at ${free}MiB"; exit 14; }
echo "card $CUDA_VISIBLE_DEVICES free=${free}MiB  $(date -Iseconds)"

THINK_FLAG=(); [ "$MODE" = "think" ] && THINK_FLAG=(--thinking)

# 16000, not the 6000 default. The 122B lost 31 of 1000 otlite rows to
# truncation, and every one of them was a genuine overrun rather than format
# drift — re-running those alone moved otlite from 81.50 to 82.80. Paying for
# the headroom up front is cheaper than a retry pass, and costs nothing on rows
# that stop early.
echo "#### OTel-31B $SET $MODE START $(date -Iseconds)"
"$PY" "$INFRA/eval_dev_vllm.py" \
  --base "$MODEL" --data "$DATA/$SET.jsonl" --out-dir "$OUT" \
  --tag "$TAG" --with-base "${THINK_FLAG[@]}" \
  --max-new "$( [ "$MODE" = think ] && echo 16000 || echo 512 )" \
  --tp 1 --gpu-mem 0.90 --max-model-len "$( [ "$MODE" = think ] && echo 20480 || echo 4096 )"
echo "#### OTel-31B $SET $MODE DONE $(date -Iseconds)"

"$PY" - "$OUT/${TAG}_base_$( [ "$MODE" = think ] && echo think || echo nothink512 ).json" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
if not p.exists():
    cand = sorted(p.parent.glob(p.name.split("_base")[0] + "_base*.json"))
    if not cand:
        sys.exit(f"no result file matching {p}")
    p = cand[-1]
d = json.load(open(p))
s, rows = d["summary"], d["results"]
n = len(rows)
strict = sum(r["correct"] for r in rows)
lenient = sum(r["correct_lenient"] for r in rows)
unp = [r for r in rows if not r["parsed"]]
print(f"\n=== {p.name}")
print(f"official (strict) : {strict}/{n} = {strict/n:.4f}")
print(f"lenient parse     : {lenient}/{n} = {lenient/n:.4f}")
print(f"unparsed          : {len(unp)}")
# Distinguish the two reasons a row can go unparsed, because they have opposite
# fixes: an overrun wants a bigger token budget, a model that answers "A) NOMA"
# instead of "ANSWER: A" wants nothing at all — the official harness has no
# bare-letter fallback and that is the score. OTel 1.0 did exactly this.
if unp:
    trunc = sum(1 for r in unp if len(r["completion"]) > 12000)
    print(f"  of which look truncated : {trunc}")
    print(f"  of which look drifted   : {len(unp)-trunc}")
    rec = sum(1 for r in unp if r["parsed_lenient"])
    print(f"  recoverable by lenient  : {rec}")
print("\n  %-26s %5s %7s" % ("subject", "n", "acc"))
for k, v in sorted(s.get("by_subject", {}).items(), key=lambda kv: -kv[1]["n"]):
    print("  %-26s %5d %7.4f" % (k, v["n"], v["acc"]))
print("\nreferences on this same 1000-row otlite set:")
print("  Qwen3.5-122B-A10B thinking   0.8280")
print("  Qwen3-8B base thinking       (dev-1000: 0.7550 nightly)")
print("noise floor: 0.50pp A/A. Read against gemma-4-31B-it, not against these.")
PY
