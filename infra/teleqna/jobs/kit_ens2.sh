#!/usr/bin/env bash
# ens round 2: init = merge_a (0.4 vd6 / 0.6 vd8 = 81.53), teachers vd8 + vd6 (pooled votes), same debiasing caps and
# All-prior rows; SFT lr 5e-6 (FSDP trainer, 8 cards). Runs after ens round 1, before big run 4 (big4 waits for cards).
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0
cd "$ROOT"; mkdir -p data/kit/ens2; CARDS="${CARDS:-0,1,2,3,4,5,6,7}"; NP=$(echo "$CARDS" | tr "," "\n" | wc -l); INIT="${INIT:-models/kit/merge_a}"
until grep -q "ENS_CHAIN_DONE\|ABORT\|rc=[1-9]" logs/ens_resume.log 2>/dev/null; do sleep 120; done
for i in $(seq 1 480); do busy=0; for c in $(echo "$CARDS" | tr "," " "); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 30; done
echo "#### ENS2 LABELS START init=$INIT teachers=vd8+vd6 $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=$(echo "$CARDS" | cut -d, -f1) timeout --foreground 14400 "$PY" -u code/kit/vote_distill.py --ckpt models/kit/vd8/ep1 --ckpt2 models/kit/vd6/ep1 --n 120000 --seed 10 --longest-frac 0.33 --overlap-frac 0.39 --out data/kit/ens2/labels.jsonl > logs/ens2_labels.log 2>&1; grep -E "gated|voted|longest|kept|Error" logs/ens2_labels.log
grep -q VD_DONE logs/ens2_labels.log || { echo "ABORT ens2 labels"; exit 1; }
"$PY" - <<'PY'
import json, random
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"; rng = random.Random(1)
rows = [l for l in open(R / "data/kit/ens2/labels.jsonl", encoding="utf-8")]
beh = [json.loads(l) for f in ("data/kit/beh/rows_s0.jsonl", "data/kit/beh/rows_s1.jsonl") for l in open(R / f, encoding="utf-8")]
def last_letter(p): return chr(65 + len([x for x in p.split("\n") if len(x) > 2 and x[1] == ")"]) - 1)
allc = [r for r in beh if "All of the above" in r["prompt"] and r["completion"].strip().endswith(last_letter(r["prompt"]))]
one = [r for r in beh if "All of the above" in r["prompt"] and not r["completion"].strip().endswith(last_letter(r["prompt"]))]
rng.shuffle(one); one = one[: max(1, len(allc) // 4)]
with open(R / "data/kit/ens2/rows.jsonl", "w", encoding="utf-8") as f:
    f.writelines(rows)
    for r in allc + one: f.write(json.dumps(dict(prompt=r["prompt"], completion=r["completion"]), ensure_ascii=False) + "\n")
print(f"ens2 rows {len(rows):,} + All-correct {len(allc):,} + All-distractor {len(one):,}")
PY
"$PY" -u code/kit/pack_chat.py --rows data/kit/ens2/rows.jsonl --out data/kit/ens2/pack --blk 2048 > logs/ens2_pack.log 2>&1; tail -2 logs/ens2_pack.log | head -1
grep -q PACK_DONE logs/ens2_pack.log || { echo "ABORT ens2 pack"; exit 1; }
echo "#### ENS2 TRAIN START init=$INIT $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=$CARDS timeout --foreground 43200 torchrun --nproc_per_node=$NP --master_port=29830 code/kit/train_fsdp.py --pack data/kit/ens2/pack --out models/kit/ens2 --init "$INIT" --bs 4 --acc 2 --lr 5e-6 --warm-frac 0.05 --no-ckpt > logs/ens2_train.log 2>&1
echo "#### ENS2 TRAIN DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/ens2_train.log || exit 1
CKPT=models/kit/ens2/ep1 TAG=ens2_ep1 CARDS=$(echo "$CARDS" | cut -d, -f1) bash jobs/eval_ckpt.sh > logs/eval_ens2.log 2>&1; grep RESULT logs/eval_ens2.log
echo "#### ENS2_CHAIN_DONE $(date -Iseconds)"
