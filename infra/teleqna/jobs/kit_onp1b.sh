#!/usr/bin/env bash
# On-policy variant b (proper "partially mastered"): only judge-verified questions with p(gold) >= 0.4, weight = round(4*p)
# (no floor), + consistent-right vote rows + All-prior; SFT lr 3e-6 from wise_u3. Reuses onp1's sampling + judge outputs.
# Then WiSE interpolation wise_u3 <-> onp1 (0.7/0.3) for reference. Waits until the cards are really idle.
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0
cd "$ROOT"; CARDS="${CARDS:-0,1,2,3}"; NP=$(echo "$CARDS" | tr "," "\n" | wc -l); INIT=models/kit/wise_u3; TAG=onp1b; PMIN="${PMIN:-0.4}"
"$PY" - "$PMIN" <<'PY'
import json, random, sys, collections
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"; rng = random.Random(0); pmin = float(sys.argv[1])
part = [json.loads(l) for l in open(R / "data/kit/onp1/rows_partial_verified.jsonl", encoding="utf-8")]
rows = []; cnt = collections.Counter()
for r in part:
    if r["pgold"] < pmin: cnt["dropped"] += 1; continue
    k = round(4 * r["pgold"]); cnt[k] += 1
    for _ in range(k): rows.append(dict(prompt=r["prompt"], completion=r["completion"]))
npart = len(rows)
easy = [json.loads(l) for l in open(R / "data/kit/onp1/rows_easy.jsonl", encoding="utf-8")]; rows += [dict(prompt=r["prompt"], completion=r["completion"]) for r in easy]
beh = [json.loads(l) for f in ("data/kit/beh/rows_s0.jsonl", "data/kit/beh/rows_s1.jsonl") for l in open(R / f, encoding="utf-8")]
def last_letter(p): return chr(65 + len([x for x in p.split("\n") if len(x) > 2 and x[1] == ")"]) - 1)
allc = [r for r in beh if "All of the above" in r["prompt"] and r["completion"].strip().endswith(last_letter(r["prompt"]))]
one = [r for r in beh if "All of the above" in r["prompt"] and not r["completion"].strip().endswith(last_letter(r["prompt"]))]
rng.shuffle(one); allp = [dict(prompt=r["prompt"], completion=r["completion"]) for r in allc * 4 + one[: max(1, len(allc) // 4)]]; rows += allp
rng.shuffle(rows); (R / "data/kit/onp1b").mkdir(exist_ok=True)
with open(R / "data/kit/onp1b/rows.jsonl", "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"onp1b: partial p>={pmin}: weighted rows {npart:,} (counts {dict(cnt)}) + easy {len(easy):,} + All-prior {len(allp):,} = {len(rows):,}")
PY
"$PY" -u code/kit/pack_chat.py --rows data/kit/$TAG/rows.jsonl --out data/kit/$TAG/pack --blk 2048 > logs/${TAG}_pack.log 2>&1; tail -2 logs/${TAG}_pack.log | head -1
grep -q PACK_DONE logs/${TAG}_pack.log || { echo "ABORT $TAG pack"; exit 1; }
for i in $(seq 1 1440); do busy=0; for c in $(echo "$CARDS" | tr "," " "); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 30; done; sleep 20
echo "#### $TAG TRAIN START init=$INIT cards=$CARDS $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=$CARDS timeout --foreground 43200 torchrun --nproc_per_node=$NP --master_port=29895 code/kit/train_fsdp.py --pack data/kit/$TAG/pack --out models/kit/$TAG --init "$INIT" --bs 4 --acc 2 --lr 3e-6 --warm-frac 0.05 --no-ckpt > logs/${TAG}_train.log 2>&1
echo "#### $TAG TRAIN DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/${TAG}_train.log || exit 1
CKPT=models/kit/$TAG/ep1 TAG=${TAG}_ep1 CARDS=$(echo "$CARDS" | cut -d, -f1) bash jobs/eval_ckpt.sh > logs/eval_$TAG.log 2>&1; grep RESULT logs/eval_$TAG.log
BASE=$HOME/projects/_shared/models/Qwen3-8B
$PY code/kit/ties_merge.py --base $BASE --ckpts models/kit/wise_u3,models/kit/onp1/ep1 --out models/kit/wise_o3 --plain --weights 0.7,0.3 > logs/wise_o3.log 2>&1
CKPT=models/kit/wise_o3 TAG=wise_o3 CARDS=$(echo "$CARDS" | cut -d, -f2) GPUMEM=0.85 MEMWAIT=100000 bash jobs/eval_ckpt.sh > logs/eval_wise_o3.log 2>&1; grep RESULT logs/eval_wise_o3.log
echo "#### ONP1B_CHAIN_DONE $(date -Iseconds)"
