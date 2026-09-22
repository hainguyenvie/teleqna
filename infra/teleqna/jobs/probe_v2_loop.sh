#!/usr/bin/env bash
# Re-run the v2 probe for every new checkpoint (tier1, gf, r2) until all arms are done. Card 7.
cd $HOME/projects/teleqna/runs/teleqna-8b
PY=$HOME/venv-vllm-nightly/bin/python; export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=${CARDS:-2}
while true; do
  for d in $(ls -d models/kit/tier2/step* models/kit/tier2/ep* models/kit/tier1_agree/ep* models/kit/tier1_lr5e5/ep* 2>/dev/null); do
    arm=$(basename $(dirname $d)); tag="$(basename $d)"; [ "$arm" != tier1 ] && tag="${arm}_${tag}"
    [ -f "results/kit/v2/probe_$tag.json" ] && continue
    [ -f "$d/config.json" ] && [ -f "$d/model.safetensors.index.json" -o -f "$d/model.safetensors" ] || continue
    sleep 60
    "$PY" -u code/kit/probe_logprob.py --ckpt "$d" --tag "$tag" --probes data/kit/tier1/probes_v2.json --out results/kit/v2 > logs/probe_v2_$tag.log 2>&1
    echo "#### PROBE-V2 $tag $(date -Iseconds)"; "$PY" code/kit/probe_report.py results/kit/v2
  done
  grep -q "TIER2_CHAIN_DONE" logs/tier2_chain.log 2>/dev/null && [ -f results/kit/v2/probe_tier2_ep1.json ] && break
  sleep 600
done
echo "PROBE_V2_LOOP_DONE"
