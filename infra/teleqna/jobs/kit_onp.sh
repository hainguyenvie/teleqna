#!/usr/bin/env bash
# On-policy distillation round (teacher = OTel-31B reading the source window; student = closed-book checkpoint):
# 1) student samples 300k synthetic MCQs (4 rot x 4 samples) -> p(gold) per question (utr_select, already launched as $RUN);
# 2) 31B judges the PARTIALLY-KNOWN questions (0.01 <= p(gold) <= 0.99), keep judge == gold;
# 3) rows duplicated proportional to p(gold) (GRPO-equivalent weighting for one-token answers) + consistent-right vote rows
#    + All-prior rows; SFT lr 3e-6 from the student; eval. Usage: RUN=data/kit/onp1 INIT=models/kit/wise_u3 TAG=onp1 bash jobs/kit_onp.sh
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0
cd "$ROOT"; RUN="${RUN:-data/kit/onp1}"; INIT="${INIT:-models/kit/wise_u3}"; TAG="${TAG:-onp1}"; JCARDS="${JCARDS:-2,3,4,5,6,7}"; CARDS="${CARDS:-0,1,2,3,4,5,6,7}"; NP=$(echo "$CARDS" | tr "," "\n" | wc -l); LR="${LR:-3e-6}"; DUP="${DUP:-4}"
until grep -q "UTR_DONE" logs/${TAG}_select.log 2>/dev/null; do sleep 60; done
NJ=$(echo "$JCARDS" | tr "," "\n" | wc -l); echo "#### $TAG JUDGE START cards=$JCARDS $(date -Iseconds)"
s=0; for c in $(echo "$JCARDS" | tr "," " "); do CUDA_VISIBLE_DEVICES=$c nohup "$PY" -u code/kit/onp_judge.py --run $RUN --shard $s --nshards $NJ > logs/${TAG}_judge_s$s.log 2>&1 & s=$((s+1)); sleep 3; done; wait
grep -h "verified" logs/${TAG}_judge_s*.log | cut -c1-200; cat $RUN/rows_partial_verified_s*.jsonl > $RUN/rows_partial_verified.jsonl
"$PY" - "$RUN" "$TAG" "$DUP" <<'PY'
import json, random, sys, collections
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"; run, tag, dup = sys.argv[1], sys.argv[2], int(sys.argv[3]); rng = random.Random(0)
part = [json.loads(l) for l in open(R / run / "rows_partial_verified.jsonl", encoding="utf-8")]
rows = []; cnt = collections.Counter()
for r in part:
    k = max(1, round(dup * r["pgold"])); cnt[k] += 1
    for _ in range(k): rows.append(dict(prompt=r["prompt"], completion=r["completion"]))
npart = len(rows)
easy = [json.loads(l) for l in open(R / run / "rows_easy.jsonl", encoding="utf-8")]; rows += [dict(prompt=r["prompt"], completion=r["completion"]) for r in easy]
beh = [json.loads(l) for f in ("data/kit/beh/rows_s0.jsonl", "data/kit/beh/rows_s1.jsonl") for l in open(R / f, encoding="utf-8")]
def last_letter(p): return chr(65 + len([x for x in p.split("\n") if len(x) > 2 and x[1] == ")"]) - 1)
allc = [r for r in beh if "All of the above" in r["prompt"] and r["completion"].strip().endswith(last_letter(r["prompt"]))]
one = [r for r in beh if "All of the above" in r["prompt"] and not r["completion"].strip().endswith(last_letter(r["prompt"]))]
rng.shuffle(one); allp = [dict(prompt=r["prompt"], completion=r["completion"]) for r in allc * 4 + one[: max(1, len(allc) // 4)]]; rows += allp
rng.shuffle(rows); (R / f"data/kit/{tag}").mkdir(exist_ok=True)
with open(R / f"data/kit/{tag}/rows.jsonl", "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{tag}: partial-verified rows {len(part):,} -> weighted {npart:,} (dup counts {dict(cnt)}) + easy {len(easy):,} + All-prior {len(allp):,} = {len(rows):,}")
PY
"$PY" -u code/kit/pack_chat.py --rows data/kit/$TAG/rows.jsonl --out data/kit/$TAG/pack --blk 2048 > logs/${TAG}_pack.log 2>&1; tail -2 logs/${TAG}_pack.log | head -1
grep -q PACK_DONE logs/${TAG}_pack.log || { echo "ABORT $TAG pack"; exit 1; }
for i in $(seq 1 720); do busy=0; for c in $(echo "$CARDS" | tr "," " "); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 30; done; sleep 20
echo "#### $TAG TRAIN START init=$INIT lr=$LR $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=$CARDS timeout --foreground 43200 torchrun --nproc_per_node=$NP --master_port=29890 code/kit/train_fsdp.py --pack data/kit/$TAG/pack --out models/kit/$TAG --init "$INIT" --bs 4 --acc 2 --lr $LR --warm-frac 0.05 --no-ckpt > logs/${TAG}_train.log 2>&1
echo "#### $TAG TRAIN DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/${TAG}_train.log || exit 1
CKPT=models/kit/$TAG/ep1 TAG=${TAG}_ep1 CARDS=$(echo "$CARDS" | cut -d, -f1) bash jobs/eval_ckpt.sh > logs/eval_$TAG.log 2>&1; grep RESULT logs/eval_$TAG.log
echo "#### ${TAG^^}_CHAIN_DONE $(date -Iseconds)"
