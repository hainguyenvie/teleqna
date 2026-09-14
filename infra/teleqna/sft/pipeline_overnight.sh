#!/usr/bin/env bash
# Unattended run: wait out the four GPU jobs, build the training set, prove the
# trainer works on a handful of steps, then run the two arms and score them.
#
# The arms are the whole point. ALPHA=0 is plain cross-entropy on exactly the
# same rows; ALPHA=0.7 adds the KL against the same model reading the evidence.
# One variable. If the difference does not clear the 0.50pp A/A noise floor,
# the distillation hypothesis is dead and the answer is "the data was the win,
# the loss was not" - which is worth knowing by morning either way.
#
# Everything writes to a status file. A run nobody is watching has to say what
# it did without being asked.
set -uo pipefail

HOME_ROOT=/home/tensara
R="$HOME_ROOT/projects/telelogs/runs/teleqna-sft"
PY="$HOME_ROOT/venv-vllm-nightly/bin/python"
MODEL="$HOME_ROOT/projects/telelogs/shared/models/OTel-2.0-31B-IT"
STATUS="$HOME_ROOT/PIPELINE_STATUS.txt"
export PATH="$HOME_ROOT/venv-vllm-nightly/bin:$PATH"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROJ_ROOT="$R" BASE_MODEL="$MODEL"
cd "$R"

say () { echo "[$(date -Iseconds)] $*" | tee -a "$STATUS"; }
die () { say "FAILED: $*"; say "pipeline stopped"; exit 1; }

# A free card, or wait for one. Other people share this box, so a card that is
# free now may not be in ten minutes, and a hard-coded index would abort the
# whole overnight run over a transient.
wait_for_card () {
  local want_free=$1 skip=${2:-none} tries=0
  while true; do
    for c in 0 1 4 6 5 7 2 3; do
      [ "$c" = "$skip" ] && continue
      local f
      f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$c" 2>/dev/null || echo 0)
      if [ "$f" -gt "$want_free" ]; then echo "$c"; return 0; fi
    done
    tries=$((tries+1))
    [ $tries -gt 240 ] && return 1     # 4h of waiting is a stuck box, not a busy one
    sleep 60
  done
}

say "=== pipeline start ==="

# ---- 1. wait for the four jobs already running -----------------------------
say "waiting for mining and probe jobs"
while pgrep -f "infra/mine_hard_negatives.py" >/dev/null \
   || pgrep -f "infra/validate_grounded_qa.py" >/dev/null; do sleep 60; done
say "all four finished"
for f in rest_novel extra31b_novel mined_neg_rest mined_neg_extra; do
  if [ -f "$R/data/synth/$f.jsonl" ]; then
    say "  $f: $(wc -l < "$R/data/synth/$f.jsonl") rows"
  else
    say "  $f: MISSING"
  fi
done

# ---- 2. merge the two filtered pools ---------------------------------------
say "building the item set"
$PY infra/build_train_set.py \
  --novel data/synth/rest_novel.jsonl \
  --novel data/synth/extra31b_novel.jsonl \
  --mined data/synth/mined_neg_rest.jsonl \
  --mined data/synth/mined_neg_extra.jsonl \
  --out data/synth/train_items.jsonl \
  --report results/rag/train_items.json >> "$STATUS" 2>&1 || die "build_train_set"

# ---- 3. views, context, replay ---------------------------------------------
say "expanding views"
$PY infra/expand_views.py \
  --items data/synth/train_items.jsonl \
  --out data/synth/views.jsonl >> "$STATUS" 2>&1 || die "expand_views"

say "attaching teacher context"
$PY infra/add_context.py \
  --views data/synth/views.jsonl \
  --items data/synth/train_items.jsonl \
  --out data/synth/views_ctx.jsonl >> "$STATUS" 2>&1 || die "add_context"

say "mixing replay anchors"
$PY infra/make_replay_mix.py \
  --knowledge data/synth/views_ctx.jsonl \
  --anchor data/anchor_mcq.jsonl \
  --ratio 1.0 \
  --out data/synth/train_mix.jsonl >> "$STATUS" 2>&1 || die "make_replay_mix"

# ---- 4. subsample for the measurement arms ---------------------------------
# The full mix is ~1M rows and two epochs of it is a day per arm. The question
# tonight is whether KL beats CE at all, and that is answered at a size both
# arms can finish overnight. Cut on fact_id: two views of one fact on opposite
# sides of the split would read as generalisation and inflate the eval.
say "subsampling to the measurement size"
$PY - <<'PYEOF' >> "$STATUS" 2>&1 || die "subsample"
import hashlib, json, pathlib
R = pathlib.Path("/home/tensara/projects/telelogs/runs/teleqna-sft")
src = R/"data/synth/train_mix.jsonl"
dst = R/"data/synth/train_mix_100k.jsonl"
TARGET = 100_000
rows = [json.loads(l) for l in src.open()]
def fid(r): return str(r.get("fact_id") or r.get("sample_id"))
def bucket(r): return hashlib.sha256(("sub|"+fid(r)).encode()).digest()[0]
keep, cut = [], 256
for c in range(1, 257):
    sel = [r for r in rows if bucket(r) < c]
    if len(sel) >= TARGET:
        keep, cut = sel, c
        break
else:
    keep = rows
with dst.open("w") as fh:
    for r in keep:
        fh.write(json.dumps(r, ensure_ascii=False)+"\n")
print(f"subsample: {len(keep)}/{len(rows)} rows, {len({fid(r) for r in keep})} facts, cut={cut}/256")
PYEOF

# ---- 5. smoke test: this trainer has never executed -------------------------
# Twelve steps on four hundred rows. It costs four minutes and it is the only
# thing standing between a typo and an empty morning. Checked with ALPHA=0.7 so
# the KL path and the disable_adapter() teacher pass both actually run - a
# smoke test at ALPHA=0 would exercise none of the code that is new here.
say "smoke test"
head -400 data/synth/train_mix_100k.jsonl > data/synth/smoke.jsonl
SMOKE_CARD=$(wait_for_card 100000) || die "no card for the smoke test"
say "  smoke on card $SMOKE_CARD"
CUDA_VISIBLE_DEVICES=$SMOKE_CARD TRAIN_DATA="$R/data/synth/smoke.jsonl" \
  RUN_NAME=smoke ALPHA=0.7 EPOCHS=1 BATCH=2 ACCUM=4 SAVE_STEPS=1000 \
  ALLOW_OVERWRITE=1 \
  $PY infra/train_ctxdistill.py >> "$HOME_ROOT/smoke.log" 2>&1 \
  || { tail -40 "$HOME_ROOT/smoke.log" >> "$STATUS"; die "smoke test - trainer does not run"; }
say "smoke test passed"

# ---- 6. the two arms, in parallel ------------------------------------------
CARD_A=$(wait_for_card 100000) || die "no card for arm CE"
CARD_B=$(wait_for_card 100000 "$CARD_A") || die "no card for arm KD"
say "arm CE (ALPHA=0) on card $CARD_A, arm KD (ALPHA=0.7) on card $CARD_B"

CUDA_VISIBLE_DEVICES=$CARD_A TRAIN_DATA="$R/data/synth/train_mix_100k.jsonl" \
  RUN_NAME=armCE ALPHA=0.0 EPOCHS=2 ALLOW_OVERWRITE=1 \
  nohup $PY infra/train_ctxdistill.py > "$HOME_ROOT/arm_ce.log" 2>&1 &
PID_A=$!
CUDA_VISIBLE_DEVICES=$CARD_B TRAIN_DATA="$R/data/synth/train_mix_100k.jsonl" \
  RUN_NAME=armKD ALPHA=0.7 EPOCHS=2 ALLOW_OVERWRITE=1 \
  nohup $PY infra/train_ctxdistill.py > "$HOME_ROOT/arm_kd.log" 2>&1 &
PID_B=$!
wait $PID_A; RC_A=$?
wait $PID_B; RC_B=$?
say "arm CE rc=$RC_A, arm KD rc=$RC_B"
[ $RC_A -ne 0 ] && tail -30 "$HOME_ROOT/arm_ce.log" >> "$STATUS"
[ $RC_B -ne 0 ] && tail -30 "$HOME_ROOT/arm_kd.log" >> "$STATUS"

# ---- 7. score both, plus the untrained base as the reference ---------------
# All three in one process off one model load, and no --thinking: this model's
# thinking mode scores 5.20% because the format collapses, so a thinking eval
# would measure the scaffold rather than the adapters.
say "scoring both arms on dev1000, no-think"
EVAL_CARD=$(wait_for_card 100000) || die "no card for eval"
ADAPT=()
[ -d "$R/models/ctxdistill-armCE-adapter" ] && ADAPT+=(--adapter "armCE=$R/models/ctxdistill-armCE-adapter")
[ -d "$R/models/ctxdistill-armKD-adapter" ] && ADAPT+=(--adapter "armKD=$R/models/ctxdistill-armKD-adapter")
CUDA_VISIBLE_DEVICES=$EVAL_CARD $PY infra/eval_dev_vllm.py \
  --base "$MODEL" --data data/dev1000.jsonl \
  --out-dir results/ctxdistill --tag arms \
  --with-base "${ADAPT[@]}" \
  --max-new 512 --tp 1 --gpu-mem 0.90 --max-model-len 4096 \
  --max-lora-rank 64 >> "$HOME_ROOT/eval_arms.log" 2>&1 || say "eval FAILED, see eval_arms.log"

say "=== pipeline done ==="
grep -hE "accuracy|acc|summary" "$HOME_ROOT/eval_arms.log" | tail -20 >> "$STATUS"
ls -la "$R/results/ctxdistill/" >> "$STATUS" 2>&1
