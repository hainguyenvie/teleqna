#!/usr/bin/env bash
# On-policy round 2: if onp1 improved on wise_u3, re-sample with onp1/ep1 and repeat (student moves -> new partially-known set).
cd ~/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python; export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1
until grep -q "ONP1_CHAIN_DONE\|ABORT" logs/onp1_chain.log 2>/dev/null; do sleep 120; done
acc() { "$PY" - "$1" <<'PY'
import json, sys, os
f = os.path.expanduser(f"~/projects/teleqna/runs/teleqna-8b/results/landscape/{sys.argv[1]}_otfull10000_base_nothink512.json")
d = json.load(open(f)); d = d["results"] if isinstance(d, dict) and "results" in d else d
if isinstance(d, dict): d = list(d.values())
print(f"{100*sum(1 for x in d if isinstance(x, dict) and x.get('correct'))/len(d):.2f}")
PY
}
A1=$(acc onp1_ep1 2>/dev/null || echo 0); A0=$(acc wise_u3); echo "onp1_ep1=$A1 wise_u3=$A0"
if "$PY" -c "import sys; sys.exit(0 if float('$A1') > float('$A0') else 1)"; then
  echo "#### ROUND 2 from onp1/ep1 $(date -Iseconds)"
  CUDA_VISIBLE_DEVICES=0 "$PY" -u code/kit/utr_select.py --ckpt models/kit/onp1/ep1 --n 300000 --seed 8 --gpu-mem 0.85 --out data/kit/onp2 --easy-keep 80000 > logs/onp2_select.log 2>&1
  RUN=data/kit/onp2 INIT=models/kit/onp1/ep1 TAG=onp2 bash jobs/kit_onp.sh
else
  echo "#### ROUND 2 skipped (no improvement) $(date -Iseconds)"
fi
