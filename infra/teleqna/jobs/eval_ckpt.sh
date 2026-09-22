#!/usr/bin/env bash
# Score one checkpoint on ot-full 10k + rotated choices, on card 7, waiting until the card has room.
# Usage: CKPT=models/kit/tier1/ep1 TAG=kit1_ep1 bash jobs/eval_ckpt.sh
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CARDS:-7}"
cd "$ROOT"
for set in otfull10000 otfull_rot1; do
  OUT=results/landscape/${TAG}_${set}_base_nothink512.json; [ -f "$OUT" ] && { echo "skip $set"; continue; }
  for i in $(seq 1 120); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $CUDA_VISIBLE_DEVICES); [ "$used" -le ${MEMWAIT:-85000} ] && break; sleep 30; done
  echo "#### EVAL ${TAG}_${set} START $(date -Iseconds)"
  timeout --foreground 5400 "$PY" -u code/eval_dev_vllm.py --base "$CKPT" --data "data/eval/$set.jsonl" --out-dir results/landscape --tag "${TAG}_${set}" --with-base --max-new 512 --tp 1 --gpu-mem ${GPUMEM:-0.40} --max-model-len 4096 > logs/eval_${TAG}_${set}.log 2>&1
  echo "#### EVAL ${TAG}_${set} DONE rc=$? $(date -Iseconds)"
done
"$PY" - <<PY
import json
def L(p):
    d=json.load(open(p)); d=d["results"] if isinstance(d,dict) and "results" in d else d
    if isinstance(d,dict): d=list(d.values())
    return {x["sample_id"]:x for x in d if isinstance(x,dict) and "sample_id" in x}
B=L("results/landscape/otfull_q3_8b_base_nothink512.json"); E=L("results/landscape/${TAG}_otfull10000_base_nothink512.json"); P=L("results/landscape/${TAG}_otfull_rot1_base_nothink512.json")
ids=list(B); acc=lambda D: sum(D[q]["correct"] for q in ids)/len(ids)*100
fx=sum((not B[q]["correct"]) and E[q]["correct"] for q in ids); br=sum(B[q]["correct"] and not E[q]["correct"] for q in ids); un=sum((E[q].get("parsed") or "")=="" for q in ids)
import re
T10={r["sample_id"]:r for r in map(json.loads,open("data/eval/otfull10000.jsonl"))}; TR={r["sample_id"]:r for r in map(json.loads,open("data/eval/otfull_rot1.jsonl"))}
def fl(D,T):
    ok=0
    for q,x in D.items():
        m=re.search(r"ANSWER\s*:\s*([A-E])", x.get("completion",""))
        ok+= bool(m) and (ord(m.group(1))-65)==T[q]["answer"]
    return ok/len(D)*100
print(f"RESULT ${TAG}: base {acc(B):.2f} -> {acc(E):.2f} ({acc(E)-acc(B):+.2f})  fixed {fx} broke {br} unparsed {un} | rotated-choices {acc(P):.2f} | first-letter {fl(E,T10):.2f} / rot1 {fl(P,TR):.2f}")
PY
echo "#### EVAL_CKPT_DONE ${TAG}"
