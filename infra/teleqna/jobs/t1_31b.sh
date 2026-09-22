#!/usr/bin/env bash
# Source-coverage ceiling with a STRONGER reader: OTel-31B answers 1,500 test rows with the RAG-8 windows in context
# (base 8B got 83.75 on the full set). If 31B reads the same windows to ~90+, coverage was under-measured by the 8B's reading.
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=${CARDS:-3}
cd "$ROOT"
until grep -q "RECALL_CHAIN_DONE\|ABORT\|rc=[1-9]" logs/recall_chain.log 2>/dev/null; do sleep 60; done
for i in $(seq 1 480); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $CUDA_VISIBLE_DEVICES); [ "$used" -le 1000 ] && break; sleep 20; done
python3 - <<'PY'
import json,random
rows=[l for l in open("data/eval/otfull_rag8_strong.jsonl")]; random.Random(11).shuffle(rows); open("data/eval/otfull_rag8_1500.jsonl","w").writelines(rows[:1500])
PY
echo "#### T1-31B START $(date -Iseconds)"
timeout --foreground 5400 "$PY" -u code/eval_dev_vllm.py --base "$HOME/projects/_shared/models/OTel-2.0-31B-IT" --data data/eval/otfull_rag8_1500.jsonl --out-dir results/landscape --tag t1_otel31b_1500 --with-base --max-new 64 --tp 1 --gpu-mem 0.90 --max-model-len 24576 > logs/t1_31b.log 2>&1
grep -E "acc=|Error" logs/t1_31b.log | tail -2
python3 - <<'PY'
import json,re
T={r["sample_id"]:r for r in map(json.loads,open("data/eval/otfull10000.jsonl"))}
ids={json.loads(l)["sample_id"] for l in open("data/eval/otfull_rag8_1500.jsonl")}
def FL(p):
    d=json.load(open(p)); d=d.get("results",d) if isinstance(d,dict) else d; d=list(d.values()) if isinstance(d,dict) else d
    out={}
    for x in d:
        if x["sample_id"] not in ids: continue
        m=re.search(r"ANSWER\s*:\s*([A-E])", x["completion"].split("</think>")[-1]); out[x["sample_id"]]=bool(m) and (ord(m.group(1))-65)==T[x["sample_id"]]["answer"]
    return out
A=FL("results/landscape/t1_otel31b_1500_base_nothink512.json"); B=FL("results/landscape/ragstrong_q3_8b_base_nothink512.json")
print(f"same 1500 rows, RAG-8 in context: base-8B {sum(B[q] for q in A)/len(A)*100:.1f} | OTel-31B {sum(A.values())/len(A)*100:.1f} | union {sum(A[q] or B[q] for q in A)/len(A)*100:.1f}")
NF=json.load(open("results/kit/never_flick.json")); nv=[q for q in A if q in set(NF["never"])]
print(f"never-correct rows in sample {len(nv)}: 31B+RAG right {sum(A[q] for q in nv)} | base+RAG right {sum(B[q] for q in nv)}")
PY
echo "#### T1-31B END $(date -Iseconds)"
