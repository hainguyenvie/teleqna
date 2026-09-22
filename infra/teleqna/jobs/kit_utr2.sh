#!/usr/bin/env bash
# UTR-2: teach the judge-verified hard facts (31B agrees with kit gold) without breaking the model.
# arm A = verified-hard + easy vote rows + All-prior (hard ~31%); arm B = A + recall-QA of the hard windows (hard ~16%).
# Both from soup_het (82.04 / rot1 81.33), 1 epoch, lr 3e-6, FSDP 8 cards, sequential. Waits for the 31B judging.
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0
cd "$ROOT"; CARDS="${CARDS:-0,1,2,3,4,5,6,7}"; NP=$(echo "$CARDS" | tr "," "\n" | wc -l); INIT="${INIT:-models/kit/soup_het}"; LR="${LR:-3e-6}"
until grep -q "HARD_JUDGE_DONE" logs/hard_judge_chain.log 2>/dev/null; do sleep 120; done
"$PY" - <<'PY'
import json, random
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"; rng = random.Random(0)
def rows(f): return [dict(prompt=(r := json.loads(l))["prompt"], completion=r["completion"]) for l in open(R / f, encoding="utf-8")]
hard = rows("data/kit/utr/rows_hard_verified.jsonl"); easy = rows("data/kit/utr/rows_easy.jsonl")
beh = [json.loads(l) for f in ("data/kit/beh/rows_s0.jsonl", "data/kit/beh/rows_s1.jsonl") for l in open(R / f, encoding="utf-8")]
def last_letter(p): return chr(65 + len([x for x in p.split("\n") if len(x) > 2 and x[1] == ")"]) - 1)
allc = [r for r in beh if "All of the above" in r["prompt"] and r["completion"].strip().endswith(last_letter(r["prompt"]))]
one = [r for r in beh if "All of the above" in r["prompt"] and not r["completion"].strip().endswith(last_letter(r["prompt"]))]
rng.shuffle(one); allp = [dict(prompt=r["prompt"], completion=r["completion"]) for r in allc * 4 + one[: max(1, len(allc) // 4)]]
wins = set(open(R / "data/kit/utr/hard_wins.txt").read().split()); rq = []
for l in open(R / "data/kit/recall_all/rows_all.jsonl", encoding="utf-8"):
    i = l.find('"win_id": "'); w = l[i + 11:l.find('"', i + 11)]
    if w in wins: r = json.loads(l); rq.append(dict(prompt=r["prompt"], completion=r["completion"]))
for tag, parts in (("utr2a", [hard, easy, allp]), ("utr2b", [hard, easy, allp, rq])):
    rs = [r for p in parts for r in p]; rng.shuffle(rs); (R / f"data/kit/{tag}").mkdir(exist_ok=True)
    with open(R / f"data/kit/{tag}/rows.jsonl", "w", encoding="utf-8") as f:
        for r in rs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{tag}: hard-verified {len(hard):,} easy {len(easy):,} All-prior {len(allp):,}" + (f" recall-QA {len(rq):,}" if len(parts) == 4 else "") + f" -> {len(rs):,} (hard {100*len(hard)/len(rs):.0f}%)")
PY
for tag in utr2a utr2b; do
  "$PY" -u code/kit/pack_chat.py --rows data/kit/$tag/rows.jsonl --out data/kit/$tag/pack --blk 2048 > logs/${tag}_pack.log 2>&1; tail -2 logs/${tag}_pack.log | head -1
  grep -q PACK_DONE logs/${tag}_pack.log || { echo "ABORT $tag pack"; continue; }
  for i in $(seq 1 720); do busy=0; for c in $(echo "$CARDS" | tr "," " "); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 30; done; sleep 20
  echo "#### $tag TRAIN START init=$INIT lr=$LR $(date -Iseconds)"
  CUDA_VISIBLE_DEVICES=$CARDS timeout --foreground 43200 torchrun --nproc_per_node=$NP --master_port=29880 code/kit/train_fsdp.py --pack data/kit/$tag/pack --out models/kit/$tag --init "$INIT" --bs 4 --acc 2 --lr $LR --warm-frac 0.05 --no-ckpt > logs/${tag}_train.log 2>&1
  echo "#### $tag TRAIN DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/${tag}_train.log || continue
  CKPT=models/kit/$tag/ep1 TAG=${tag}_ep1 CARDS=$(echo "$CARDS" | cut -d, -f1) bash jobs/eval_ckpt.sh > logs/eval_$tag.log 2>&1; grep RESULT logs/eval_$tag.log
done
echo "#### UTR2_CHAIN_DONE $(date -Iseconds)"
