#!/usr/bin/env bash
# Resume from the failure. views_ctx.jsonl was built correctly - 509,402 rows,
# every one of them joined to an evidence span - so everything up to the mix
# stands and only the mix onward re-runs.
#
# The bug was an ordering mistake: replay anchors were mixed at 1:1 against the
# full 509k views and only subsampled afterwards, so it asked for 509,402
# anchors from a pool of 14,000. Subsampling first turns the ratio into
# something the pool can actually satisfy.
#
# The ratio ends up near 1:6 rather than the 1:1 the plan called for. That is a
# real deviation and it thins the protection against drifting off the 7,285 rows
# the model already answers correctly. It does not bias tonight's comparison -
# both arms train on the identical mix, so whatever the anchors do, they do to
# both - but it does mean the absolute scores may sit lower than a properly
# anchored run, which is why the untrained base is scored alongside them.
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

wait_for_card () {
  local want_free=$1 skip=${2:-none} tries=0
  while true; do
    for c in 4 6 0 1 7 5 2 3; do
      [ "$c" = "$skip" ] && continue
      local f
      f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$c" 2>/dev/null || echo 0)
      if [ "$f" -gt "$want_free" ]; then echo "$c"; return 0; fi
    done
    tries=$((tries+1)); [ $tries -gt 240 ] && return 1
    sleep 60
  done
}

say "=== resume: subsample then mix ==="

# ---- subsample the knowledge rows, cutting on fact_id ----------------------
$PY - <<'PYEOF' >> "$STATUS" 2>&1 || die "subsample"
import hashlib, json, pathlib
R = pathlib.Path("/home/tensara/projects/telelogs/runs/teleqna-sft")
src, dst = R/"data/synth/views_ctx.jsonl", R/"data/synth/views_sub.jsonl"
TARGET = 86_000
rows = [json.loads(l) for l in src.open()]
def fid(r): return str(r.get("fact_id") or r.get("sample_id"))
def b(r): return hashlib.sha256(("sub|"+fid(r)).encode()).digest()[0]
keep = rows
for c in range(1, 257):
    sel = [r for r in rows if b(r) < c]
    if len(sel) >= TARGET:
        keep = sel; break
with dst.open("w") as fh:
    for r in keep:
        fh.write(json.dumps(r, ensure_ascii=False)+"\n")
print(f"subsample: {len(keep)}/{len(rows)} rows, {len({fid(r) for r in keep})} facts")
PYEOF

KNOW=$(wc -l < data/synth/views_sub.jsonl)
POOL=$(wc -l < data/anchor_mcq.jsonl)
# Largest ratio the pool can actually serve, with a little slack so a rounding
# difference inside the mixer cannot re-trip the same SystemExit.
RATIO=$($PY -c "print(min(1.0, ($POOL*0.97)/$KNOW))")
say "knowledge=$KNOW anchors=$POOL ratio=$RATIO"

$PY infra/make_replay_mix.py \
  --knowledge data/synth/views_sub.jsonl \
  --anchor data/anchor_mcq.jsonl \
  --ratio "$RATIO" \
  --out data/synth/train_mix_100k.jsonl >> "$STATUS" 2>&1 || die "make_replay_mix"
say "mix rows: $(wc -l < data/synth/train_mix_100k.jsonl)"

# ---- smoke test: the trainer has still never executed ----------------------
say "smoke test"
head -400 data/synth/train_mix_100k.jsonl > data/synth/smoke.jsonl
SC=$(wait_for_card 100000) || die "no card for smoke test"
say "  smoke on card $SC"
CUDA_VISIBLE_DEVICES=$SC TRAIN_DATA="$R/data/synth/smoke.jsonl" \
  RUN_NAME=smoke ALPHA=0.7 EPOCHS=1 BATCH=2 ACCUM=4 SAVE_STEPS=1000 \
  ALLOW_OVERWRITE=1 \
  $PY infra/train_ctxdistill.py > "$HOME_ROOT/smoke.log" 2>&1 \
  || { tail -40 "$HOME_ROOT/smoke.log" >> "$STATUS"; die "smoke test"; }
say "smoke test passed"
grep -E "train=|with_context|ALIGN|loss" "$HOME_ROOT/smoke.log" | tail -6 >> "$STATUS"

# ---- the two arms ----------------------------------------------------------
CA=$(wait_for_card 100000) || die "no card for arm CE"
CB=$(wait_for_card 100000 "$CA") || die "no card for arm KD"
say "arm CE (ALPHA=0) card $CA | arm KD (ALPHA=0.7) card $CB"
CUDA_VISIBLE_DEVICES=$CA TRAIN_DATA="$R/data/synth/train_mix_100k.jsonl" \
  RUN_NAME=armCE ALPHA=0.0 EPOCHS=2 ALLOW_OVERWRITE=1 \
  nohup $PY infra/train_ctxdistill.py > "$HOME_ROOT/arm_ce.log" 2>&1 &
PA=$!
CUDA_VISIBLE_DEVICES=$CB TRAIN_DATA="$R/data/synth/train_mix_100k.jsonl" \
  RUN_NAME=armKD ALPHA=0.7 EPOCHS=2 ALLOW_OVERWRITE=1 \
  nohup $PY infra/train_ctxdistill.py > "$HOME_ROOT/arm_kd.log" 2>&1 &
PB=$!
wait $PA; RA=$?; wait $PB; RB=$?
say "arm CE rc=$RA | arm KD rc=$RB"
[ $RA -ne 0 ] && tail -30 "$HOME_ROOT/arm_ce.log" >> "$STATUS"
[ $RB -ne 0 ] && tail -30 "$HOME_ROOT/arm_kd.log" >> "$STATUS"

# ---- score both arms and the untrained base off one model load -------------
say "scoring on dev1000, no-think"
EC=$(wait_for_card 100000) || die "no card for eval"
ADAPT=()
[ -d "$R/models/ctxdistill-armCE-adapter" ] && ADAPT+=(--adapter "armCE=$R/models/ctxdistill-armCE-adapter")
[ -d "$R/models/ctxdistill-armKD-adapter" ] && ADAPT+=(--adapter "armKD=$R/models/ctxdistill-armKD-adapter")
CUDA_VISIBLE_DEVICES=$EC $PY infra/eval_dev_vllm.py \
  --base "$MODEL" --data data/dev1000.jsonl \
  --out-dir results/ctxdistill --tag arms --with-base "${ADAPT[@]}" \
  --max-new 512 --tp 1 --gpu-mem 0.90 --max-model-len 4096 \
  --max-lora-rank 64 > "$HOME_ROOT/eval_arms.log" 2>&1 || say "eval FAILED"
say "=== pipeline done ==="
ls -la "$R/results/ctxdistill/" >> "$STATUS" 2>&1
