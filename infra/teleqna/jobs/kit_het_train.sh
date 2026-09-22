#!/usr/bin/env bash
# HET stage: SFT on the high-exposure self-contained QA rows (4 styles) of the question-relevant windows, mixed with the
# UTR hard/easy MCQ rows (format + hard facts), from the best checkpoint available at launch time. 1 epoch, lr 5e-6, FSDP.
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0
cd "$ROOT"; CARDS="${CARDS:-0,1,2,3,4,5,6,7}"; NP=$(echo "$CARDS" | tr "," "\n" | wc -l); EPOCHS="${EPOCHS:-1}"; TAG="${TAG:-het}"; export NOHARD="${NOHARD:-}"
until grep -q "HET_GEN_DONE" logs/het_gen_chain.log 2>/dev/null; do sleep 120; done
until grep -q "UTR_CHAIN_DONE\|ABORT" logs/utr_chain.log 2>/dev/null; do sleep 120; done
INIT="${INIT:-$("$PY" - <<'PY'
import json, glob, os
R = os.path.expanduser("~/projects/teleqna/runs/teleqna-8b")
cands = {"merge_f": "models/kit/merge_f", "utr_ep1": "models/kit/utr/ep1", "vd9_ep1": "models/kit/vd9/ep1", "ens3_ep1": "models/kit/ens3/ep1", "merge_j": "models/kit/merge_j", "merge_k": "models/kit/merge_k", "merge_l": "models/kit/merge_l", "big4_soup": "models/kit/big4_soup"}
best = ("models/kit/merge_f", 0)
for tag, p in cands.items():
    f = f"{R}/results/landscape/{tag}_otfull10000_base_nothink512.json"
    if not (os.path.exists(f) and os.path.exists(f"{R}/{p}/config.json")): continue
    d = json.load(open(f)); d = d["results"] if isinstance(d, dict) and "results" in d else d
    if isinstance(d, dict): d = list(d.values())
    acc = sum(1 for x in d if isinstance(x, dict) and x.get("correct")) / 100
    if acc > best[1]: best = (p, acc)
print(best[0])
PY
)}"
"$PY" - "$TAG" <<'PY'
import json, random, sys, os
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"; rng = random.Random(0); tag = sys.argv[1]
rows = []
for l in open(R / "data/kit/het/rows_all.jsonl", encoding="utf-8"):
    r = json.loads(l); rows.append(dict(prompt=r["prompt"], completion=r["completion"]))
nq = len(rows)
for f in (["data/kit/utr/rows_easy.jsonl"] if os.environ.get("NOHARD") else ["data/kit/utr/rows_hard.jsonl", "data/kit/utr/rows_easy.jsonl"]):
    for l in open(R / f, encoding="utf-8"):
        r = json.loads(l); rows.append(dict(prompt=r["prompt"], completion=r["completion"]))
# All-of-the-above prior rows (synthetic MCQ never has an All gold; UTR without them collapsed All-gold 89.8 -> 47.4)
beh = [json.loads(l) for f in ("data/kit/beh/rows_s0.jsonl", "data/kit/beh/rows_s1.jsonl") for l in open(R / f, encoding="utf-8")]
def last_letter(p): return chr(65 + len([x for x in p.split("\n") if len(x) > 2 and x[1] == ")"]) - 1)
allc = [r for r in beh if "All of the above" in r["prompt"] and r["completion"].strip().endswith(last_letter(r["prompt"]))]
one = [r for r in beh if "All of the above" in r["prompt"] and not r["completion"].strip().endswith(last_letter(r["prompt"]))]
rng.shuffle(one); one = one[: max(1, len(allc) // 4)]; nall = 0
for r in allc * 4 + one: rows.append(dict(prompt=r["prompt"], completion=r["completion"])); nall += 1
rng.shuffle(rows); (R / f"data/kit/{tag}").mkdir(exist_ok=True)
with open(R / f"data/kit/{tag}/rows.jsonl", "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{tag} rows: HET QA {nq:,} + UTR MCQ {len(rows)-nq-nall:,} + All-prior {nall:,} = {len(rows):,}")
PY
"$PY" -u code/kit/pack_chat.py --rows data/kit/$TAG/rows.jsonl --out data/kit/$TAG/pack --blk 2048 > logs/${TAG}_pack.log 2>&1; tail -2 logs/${TAG}_pack.log | head -1
grep -q PACK_DONE logs/${TAG}_pack.log || { echo "ABORT $TAG pack"; exit 1; }
for i in $(seq 1 720); do busy=0; for c in $(echo "$CARDS" | tr "," " "); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 30; done
echo "#### $TAG TRAIN START init=$INIT epochs=$EPOCHS $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=$CARDS timeout --foreground 43200 torchrun --nproc_per_node=$NP --master_port=29860 code/kit/train_fsdp.py --pack data/kit/$TAG/pack --out models/kit/$TAG --init "$INIT" --epochs $EPOCHS --bs 4 --acc 2 --lr 5e-6 --warm-frac 0.05 --no-ckpt > logs/${TAG}_train.log 2>&1
echo "#### $TAG TRAIN DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/${TAG}_train.log || exit 1
for e in $(seq 1 $EPOCHS); do CKPT=models/kit/$TAG/ep$e TAG=${TAG}_ep$e CARDS=$(echo "$CARDS" | cut -d, -f1) bash jobs/eval_ckpt.sh > logs/eval_${TAG}_ep$e.log 2>&1; grep RESULT logs/eval_${TAG}_ep$e.log; done
CUDA_VISIBLE_DEVICES=$(echo "$CARDS" | cut -d, -f1) "$PY" -u code/kit/recall_probe.py --ckpt models/kit/$TAG/ep$EPOCHS --tag ${TAG}_ep$EPOCHS > logs/recall_${TAG}.log 2>&1; grep recall logs/recall_${TAG}.log
echo "#### ${TAG^^}_CHAIN_DONE $(date -Iseconds)"
