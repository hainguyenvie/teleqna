#!/usr/bin/env bash
# Ensemble-vote distillation after big run 3: labels = pooled votes of the best big3 checkpoint AND vd6 (two lineages,
# union 85.7 vs each ~80) on 120k synthetic MCQs (kit mcq + cmcq + sibling), length-bias capped (gold-longest 33%),
# plus All-of-the-above prior rows (beh 'all' : 'one' = 4:1). SFT (FSDP trainer, lr 5e-6) from the best big3 ckpt, eval.
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0
cd "$ROOT"; mkdir -p data/kit/ens; CARDS="${CARDS:-0,1,2,3,4,5,6,7}"; NP=$(echo "$CARDS" | tr "," "\n" | wc -l)
until grep -q "BIG3_CHAIN_DONE\|ABORT\|rc=[1-9]" logs/big3_chain.log 2>/dev/null; do sleep 300; done
until grep -q "BIG3 SOUP END" logs/big3_soup.log 2>/dev/null; do sleep 120; done
read -r INIT SCORE < <(bash jobs/best_ckpt.sh); echo "init=$INIT ($SCORE)"
for i in $(seq 1 480); do busy=0; for c in $(echo "$CARDS" | tr "," " "); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 30; done
echo "#### ENS LABELS START init=$INIT + vd6 $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=$(echo "$CARDS" | cut -d, -f1) timeout --foreground 14400 "$PY" -u code/kit/vote_distill.py --ckpt "$INIT" --ckpt2 models/kit/vd6/ep1 --n 120000 --seed 9 --longest-frac 0.33 --overlap-frac 0.39 --out data/kit/ens/labels.jsonl > logs/ens_labels.log 2>&1; grep -E "gated|voted|longest|kept|Error" logs/ens_labels.log
grep -q VD_DONE logs/ens_labels.log || { echo "ABORT ens labels"; exit 1; }
"$PY" - <<'PY'
import json, random
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"; rng = random.Random(0)
rows = [l for l in open(R / "data/kit/ens/labels.jsonl", encoding="utf-8")]
beh = [json.loads(l) for f in ("data/kit/beh/rows_s0.jsonl", "data/kit/beh/rows_s1.jsonl") for l in open(R / f, encoding="utf-8")]
def last_letter(p): return chr(65 + len([x for x in p.split("\n") if len(x) > 2 and x[1] == ")"]) - 1)
allc = [r for r in beh if "All of the above" in r["prompt"] and r["completion"].strip().endswith(last_letter(r["prompt"]))]
one = [r for r in beh if "All of the above" in r["prompt"] and not r["completion"].strip().endswith(last_letter(r["prompt"]))]
rng.shuffle(one); one = one[: max(1, len(allc) // 4)]
with open(R / "data/kit/ens/rows.jsonl", "w", encoding="utf-8") as f:
    f.writelines(rows)
    for r in allc + one: f.write(json.dumps(dict(prompt=r["prompt"], completion=r["completion"]), ensure_ascii=False) + "\n")
print(f"ens rows {len(rows):,} + All-correct {len(allc):,} + All-distractor {len(one):,}")
PY
"$PY" -u code/kit/pack_chat.py --rows data/kit/ens/rows.jsonl --out data/kit/ens/pack --blk 2048 > logs/ens_pack.log 2>&1; tail -2 logs/ens_pack.log
grep -q PACK_DONE logs/ens_pack.log || { echo "ABORT ens pack"; exit 1; }
echo "#### ENS TRAIN START init=$INIT $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=$CARDS timeout --foreground 43200 torchrun --nproc_per_node=$NP --master_port=29810 code/kit/train_fsdp.py --pack data/kit/ens/pack --out models/kit/ens --init "$INIT" --bs 4 --acc 2 --lr 5e-6 --warm-frac 0.05 --no-ckpt > logs/ens_train.log 2>&1
echo "#### ENS TRAIN DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/ens_train.log || exit 1
CKPT=models/kit/ens/ep1 TAG=ens_ep1 CARDS=$(echo "$CARDS" | cut -d, -f1) bash jobs/eval_ckpt.sh > logs/eval_ens.log 2>&1; grep RESULT logs/eval_ens.log
echo "#### ENS_CHAIN_DONE $(date -Iseconds)"
