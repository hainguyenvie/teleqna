#!/usr/bin/env bash
# On card 2 after the anchor-fix arm (before the 5e-5 arm): OTel-31B recitations for uncovered rows, then 8B eval with them in context.
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=2
cd "$ROOT"
until grep -q "TIER1AGREE_CHAIN_DONE\|ABORT" logs/tier1agree_chain.log 2>/dev/null; do sleep 120; done
for i in $(seq 1 240); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2); [ "$used" -le 1000 ] && break; sleep 30; done
echo "#### OTEL RECITE START $(date -Iseconds)"
timeout --foreground 3600 "$PY" -u code/kit/otel_recite.py > logs/otel_recite.log 2>&1; grep -E "recited|Traceback" logs/otel_recite.log
timeout --foreground 3600 "$PY" -u code/eval_dev_vllm.py --base "$HOME/projects/_shared/models/Qwen3-8B" --data data/eval/otel_recite_eval.jsonl --out-dir results/landscape --tag otel_recite_q3_8b --with-base --max-new 512 --tp 1 --gpu-mem 0.85 --max-model-len 8192 > logs/eval_otel_recite.log 2>&1
"$PY" - <<'PY'
import json
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
d = json.load(open(R / "results/landscape/otel_recite_q3_8b_base_nothink512.json")); d = d["results"] if isinstance(d, dict) and "results" in d else d
if isinstance(d, dict): d = list(d.values())
res = {x["sample_id"]: bool(x["correct"]) for x in d}
rows = [json.loads(l) for l in open(R / "data/eval/otel_recite_eval.jsonl")]
base = {}
b = json.load(open(R / "results/landscape/otfull_q3_8b_base_nothink512.json")); b = b["results"] if isinstance(b, dict) and "results" in b else b
if isinstance(b, dict): b = list(b.values())
base = {x["sample_id"]: bool(x["correct"]) for x in b}
n = len(rows); ok = [r for r in rows if r["gate_ok"]]
print(f"uncovered rows {n}: base correct {sum(base[r['sample_id']] for r in rows)} | with OTel recitation correct {sum(res[r['sample_id']] for r in rows)} | gate-kept {len(ok)}: correct {sum(res[r['sample_id']] for r in ok)}")
print("RECITE_EVAL_DONE")
PY
echo "#### OTEL RECITE END $(date -Iseconds)"
