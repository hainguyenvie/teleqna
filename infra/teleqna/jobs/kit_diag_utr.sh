#!/usr/bin/env bash
# Diagnose the UTR regression (74.0 from merge_f 81.8): short SFT from merge_f on each component alone, eval.
# a) hard gold rows only; b) recall-QA rows of the hard windows only; c) sibling rows of the hard windows only. Runs after HET.
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0
cd "$ROOT"; CARDS="${CARDS:-0,1,2,3,4,5,6,7}"; NP=$(echo "$CARDS" | tr "," "\n" | wc -l); INIT=models/kit/merge_f
"$PY" - <<'PY'
import json, glob
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
TEMPLATE = ("Answer the following multiple choice question. The entire content of your response should be of the following "
            "format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.\n\n{question}\n\n{choices}")
wins = set(open(R / "data/kit/utr/hard_wins.txt").read().split())
(R / "data/kit/diag").mkdir(exist_ok=True)
with open(R / "data/kit/diag/rows_hard.jsonl", "w", encoding="utf-8") as f:
    for l in open(R / "data/kit/utr/rows_hard.jsonl", encoding="utf-8"):
        r = json.loads(l); f.write(json.dumps(dict(prompt=r["prompt"], completion=r["completion"]), ensure_ascii=False) + "\n")
with open(R / "data/kit/diag/rows_rq.jsonl", "w", encoding="utf-8") as f:
    for l in open(R / "data/kit/recall_all/rows_all.jsonl", encoding="utf-8"):
        i = l.find('"win_id": "'); w = l[i + 11:l.find('"', i + 11)]
        if w in wins: r = json.loads(l); f.write(json.dumps(dict(prompt=r["prompt"], completion=r["completion"]), ensure_ascii=False) + "\n")
n = 0
with open(R / "data/kit/diag/rows_sib.jsonl", "w", encoding="utf-8") as f:
    for fn in glob.glob(str(R / "data/kit/sib/views_s?.jsonl")) + glob.glob(str(R / "data/kit/sib_rest/views_s?.jsonl")):
        for l in open(fn, encoding="utf-8"):
            if '"ok": true' not in l: continue
            i = l.find('"win_id": "'); w = l[i + 11:l.find('"', i + 11)]
            if w not in wins: continue
            try: m = json.loads(json.loads(l)["text"])
            except Exception: continue
            if not (isinstance(m, dict) and isinstance(m.get("options"), list) and 4 <= len(m["options"]) <= 5 and isinstance(m.get("answer"), int)): continue
            k = len(m["options"])
            for s in range(k):
                ch = [m["options"][(j + s) % k] for j in range(k)]
                f.write(json.dumps(dict(prompt=TEMPLATE.format(letters=",".join(chr(65 + j) for j in range(k)), question=m["q"], choices="\n".join(f"{chr(65+j)}) {c}" for j, c in enumerate(ch))), completion=f"ANSWER: {chr(65 + (m['answer'] - s) % k)}"), ensure_ascii=False) + "\n"); n += 1
print("diag rows written; sib", n)
PY
until grep -q "HET_CHAIN_DONE\|ABORT" logs/het_train_chain.log 2>/dev/null; do sleep 120; done
for part in hard rq sib; do
  "$PY" -u code/kit/pack_chat.py --rows data/kit/diag/rows_$part.jsonl --out data/kit/diag/pack_$part --blk 2048 > logs/diag_pack_$part.log 2>&1; tail -2 logs/diag_pack_$part.log | head -1
  for i in $(seq 1 720); do busy=0; for c in $(echo "$CARDS" | tr "," " "); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 30; done; sleep 20
  echo "#### DIAG $part TRAIN START $(date -Iseconds)"
  CUDA_VISIBLE_DEVICES=$CARDS timeout --foreground 14400 torchrun --nproc_per_node=$NP --master_port=29870 code/kit/train_fsdp.py --pack data/kit/diag/pack_$part --out models/kit/diag_$part --init "$INIT" --bs 4 --acc 2 --lr 5e-6 --warm-frac 0.05 --no-ckpt > logs/diag_train_$part.log 2>&1
  echo "#### DIAG $part TRAIN DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/diag_train_$part.log || continue
  CKPT=models/kit/diag_$part/ep1 TAG=diag_${part}_ep1 CARDS=$(echo "$CARDS" | cut -d, -f1) bash jobs/eval_ckpt.sh > logs/eval_diag_$part.log 2>&1; grep RESULT logs/eval_diag_$part.log
done
echo "#### DIAG_DONE $(date -Iseconds)"
