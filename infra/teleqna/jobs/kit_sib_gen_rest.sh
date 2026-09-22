#!/usr/bin/env bash
# Sibling-contrast MCQs for the ~54k windows not covered by the first 60k sample (same shuffle seed 4321 -> complement).
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"; mkdir -p data/kit/sib_rest; CARDS="${CARDS:-2,3,4,5,6,7}"; NP=$(echo "$CARDS" | tr "," "\n" | wc -l)
python3 - <<'PY'
import json,random
wins=[json.loads(l) for l in open("data/kit/big/windows_all.jsonl")]; random.Random(4321).shuffle(wins)
rest=wins[60000:]; open("data/kit/windows_sib_rest.jsonl","w").writelines(json.dumps(w,ensure_ascii=False)+"\n" for w in rest); print("rest windows",len(rest))
PY
echo "#### SIB-REST GEN START cards=$CARDS $(date -Iseconds)"
i=0; for c in $(echo "$CARDS" | tr "," " "); do CUDA_VISIBLE_DEVICES=$c timeout --foreground 10800 "$PY" -u code/kit/gen_sibling.py --windows-file data/kit/windows_sib_rest.jsonl --out data/kit/sib_rest --shard $i --nshards $NP --seed $((1500+i)) > logs/sib_rest_s$i.log 2>&1 & i=$((i+1)); done; wait
grep -h "sibling MCQs\|Traceback" logs/sib_rest_s*.log | cut -c1-160
cat data/kit/sib/rows_s*.jsonl data/kit/sib_rest/rows_s*.jsonl > data/kit/sib_rest/rows_all.jsonl; wc -l data/kit/sib_rest/rows_all.jsonl
"$PY" -u code/kit/pack_chat.py --rows data/kit/sib_rest/rows_all.jsonl --out data/kit/sib_rest/pack_all > logs/sib_all_pack.log 2>&1; tail -2 logs/sib_all_pack.log
"$PY" -u code/kit/pack_concat.py --packs data/kit/big/pack_masked,data/kit/recall_all/pack_qa,data/kit/recall_all/pack_qa,data/kit/recall_all/pack_qa,data/kit/sib_rest/pack_all,data/kit/sib_rest/pack_all,data/kit/style/pack --out data/kit/big2/pack --seed 3 > logs/big2_pack.log 2>&1; tail -1 logs/big2_pack.log
echo "#### SIB-REST GEN DONE $(date -Iseconds)"
