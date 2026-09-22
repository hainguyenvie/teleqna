#!/usr/bin/env bash
# UTR stage: SFT from merge_f on (a) gold rows of the HARD synthetic MCQs (uncertain / consistently wrong vs kit gold,
# evidence-gated), (b) existing recall-QA + sibling MCQ rows of those windows, (c) consistent-right vote rows for stability.
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0
cd "$ROOT"; CARDS="${CARDS:-0,1,2,3,4,5,6,7}"; NP=$(echo "$CARDS" | tr "," "\n" | wc -l); INIT="${INIT:-models/kit/merge_f}"
until grep -q "UTR_DONE" logs/utr_select.log 2>/dev/null; do sleep 60; done
"$PY" - <<'PY'
import json, random, collections, glob, re
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"; rng = random.Random(0)
TEMPLATE = ("Answer the following multiple choice question. The entire content of your response should be of the following "
            "format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.\n\n{question}\n\n{choices}")
hard = [json.loads(l) for l in open(R / "data/kit/utr/rows_hard.jsonl", encoding="utf-8")]
easy = [json.loads(l) for l in open(R / "data/kit/utr/rows_easy.jsonl", encoding="utf-8")]
wins = set(open(R / "data/kit/utr/hard_wins.txt").read().split())
rq = []
for l in open(R / "data/kit/recall_all/rows_all.jsonl", encoding="utf-8"):
    i = l.find('"win_id": "'); w = l[i + 11:l.find('"', i + 11)]
    if w in wins: r = json.loads(l); rq.append(dict(prompt=r["prompt"], completion=r["completion"]))
sib = []
for f in glob.glob(str(R / "data/kit/sib/views_s?.jsonl")) + glob.glob(str(R / "data/kit/sib_rest/views_s?.jsonl")):
    for l in open(f, encoding="utf-8"):
        if '"ok": true' not in l: continue
        i = l.find('"win_id": "'); w = l[i + 11:l.find('"', i + 11)]
        if w not in wins: continue
        try: m = json.loads(json.loads(l)["text"])
        except Exception: continue
        if not (isinstance(m, dict) and isinstance(m.get("options"), list) and 4 <= len(m["options"]) <= 5 and isinstance(m.get("answer"), int)): continue
        n = len(m["options"])
        for s in range(n):
            ch = [m["options"][(i + s) % n] for i in range(n)]
            sib.append(dict(prompt=TEMPLATE.format(letters=",".join(chr(65 + i) for i in range(n)), question=m["q"], choices="\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(ch))), completion=f"ANSWER: {chr(65 + (m['answer'] - s) % n)}"))
rows = [dict(prompt=r["prompt"], completion=r["completion"]) for r in hard] + [dict(prompt=r["prompt"], completion=r["completion"]) for r in easy] + rq + sib
rng.shuffle(rows)
with open(R / "data/kit/utr/rows.jsonl", "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"utr rows: hard {len(hard):,} easy {len(easy):,} recall-QA {len(rq):,} sibling {len(sib):,} -> {len(rows):,}")
PY
"$PY" -u code/kit/pack_chat.py --rows data/kit/utr/rows.jsonl --out data/kit/utr/pack --blk 2048 > logs/utr_pack.log 2>&1; tail -2 logs/utr_pack.log | head -1
grep -q PACK_DONE logs/utr_pack.log || { echo "ABORT utr pack"; exit 1; }
until grep -q "BIG4_CHAIN_DONE\|ABORT\|rc=[1-9]" logs/big4r_chain.log 2>/dev/null; do sleep 120; done
for i in $(seq 1 600); do busy=0; for c in $(echo "$CARDS" | tr "," " "); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 30; done
echo "#### UTR TRAIN START init=$INIT $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=$CARDS timeout --foreground 43200 torchrun --nproc_per_node=$NP --master_port=29850 code/kit/train_fsdp.py --pack data/kit/utr/pack --out models/kit/utr --init "$INIT" --bs 4 --acc 2 --lr 5e-6 --warm-frac 0.05 --no-ckpt > logs/utr_train.log 2>&1
echo "#### UTR TRAIN DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/utr_train.log || exit 1
CKPT=models/kit/utr/ep1 TAG=utr_ep1 CARDS=$(echo "$CARDS" | cut -d, -f1) bash jobs/eval_ckpt.sh > logs/eval_utr.log 2>&1; grep RESULT logs/eval_utr.log
echo "#### UTR_CHAIN_DONE $(date -Iseconds)"
