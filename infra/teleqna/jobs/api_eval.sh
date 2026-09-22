#!/usr/bin/env bash
# Wikipedia-API windows for the uncovered questions: build windows + eval set, 8B eval (card 0), summary vs base/deep/control.
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${CARDS:-0}
cd "$ROOT"
echo "#### API EVAL START $(date -Iseconds)"
"$PY" -u code/kit/api_windows.py 2>&1 | grep -v Warning
timeout --foreground 3600 "$PY" -u code/eval_dev_vllm.py --base "$HOME/projects/_shared/models/Qwen3-8B" --data data/eval/tb_api8.jsonl --out-dir results/landscape --tag tb_api8_q3_8b --with-base --max-new 512 --tp 1 --gpu-mem 0.45 --max-model-len 16384 > logs/eval_tb_api8.log 2>&1
grep -E "Error|acc" logs/eval_tb_api8.log | tail -2
"$PY" - <<'PY'
import json, collections
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
def L(p):
    d = json.load(open(p)); d = d["results"] if isinstance(d, dict) and "results" in d else d
    if isinstance(d, dict): d = list(d.values())
    return {x["sample_id"]: bool(x["correct"]) for x in d}
A = L(R / "results/landscape/tb_api8_q3_8b_base_nothink512.json"); C = L(R / "results/landscape/tb_deep8ctrl_q3_8b_base_nothink512.json")
D = L(R / "results/landscape/tb_deep8_q3_8b_base_nothink512.json"); D.update(L(R / "results/landscape/tb_deep8p2_q3_8b_base_nothink512.json"))
O = L(R / "results/landscape/otel_recite_q3_8b_base_nothink512.json")
U = {json.loads(l)["sample_id"]: json.loads(l) for l in open(R / "data/kit/uncovered_functional.jsonl", encoding="utf-8")}
ids = [q for q in U if q in A]; bw = [q for q in ids if not U[q]["base_correct"]]; br = [q for q in ids if U[q]["base_correct"]]
print(f"api-8 evaluated {len(ids)} | base-wrong {len(bw)}: api fixes {sum(A[q] for q in bw)} control {sum(C[q] for q in bw)} deep {sum(D[q] for q in bw)} otel {sum(O.get(q,False) for q in bw)} api∪deep∪otel {sum(A[q] or D[q] or O.get(q,False) for q in bw)} | base-right {len(br)}: api keeps {sum(A[q] for q in br)} control {sum(C[q] for q in br)}")
c = collections.defaultdict(lambda: [0, 0])
for q in bw: c[U[q]["tag"]][0] += 1; c[U[q]["tag"]][1] += A[q]
print("api fixes by tag (base-wrong):", {k: f"{v[1]}/{v[0]}" for k, v in sorted(c.items(), key=lambda kv: -kv[1][0])})
print("API_EVAL_DONE")
PY
echo "#### API EVAL END $(date -Iseconds)"
