#!/usr/bin/env bash
# Card 2 after the OTel recitation: dense rerank of deep BM25 candidates, 8B eval, coverage delta for the uncovered set.
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=2
cd "$ROOT"
until [ -f $HOME/projects/_shared/models/Qwen3-Embedding-4B/.done ]; do sleep 60; done
for i in $(seq 1 240); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2); [ "$used" -le 100000 ] && break; sleep 30; done
echo "#### DENSE RERANK START $(date -Iseconds)"
timeout --foreground 3600 "$PY" -u code/kit/dense_rerank_deep.py > logs/dense_rerank_deep.log 2>&1; grep -E "reranked|Traceback" logs/dense_rerank_deep.log
timeout --foreground 3600 "$PY" -u code/eval_dev_vllm.py --base "$HOME/projects/_shared/models/Qwen3-8B" --data data/eval/tb_deep8.jsonl --out-dir results/landscape --tag tb_deep8_q3_8b --with-base --max-new 512 --tp 1 --gpu-mem 0.28 --max-model-len 16384 > logs/eval_tb_deep8.log 2>&1
"$PY" - <<'PY'
import json
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
def L(p):
    d = json.load(open(p)); d = d["results"] if isinstance(d, dict) and "results" in d else d
    if isinstance(d, dict): d = list(d.values())
    return {x["sample_id"]: bool(x["correct"]) for x in d}
D = L(R / "results/landscape/tb_deep8_q3_8b_base_nothink512.json"); O = L(R / "results/landscape/otel_recite_q3_8b_base_nothink512.json")
ids = list(D); print(f"uncovered {len(ids)}: dense-deep-8 correct {sum(D[q] for q in ids)} | OTel recitation correct {sum(O.get(q, False) for q in ids)} | either {sum(D[q] or O.get(q, False) for q in ids)}")
print("DEEP_EVAL_DONE")
PY
echo "#### DENSE RERANK END $(date -Iseconds)"
