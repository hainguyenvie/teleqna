#!/usr/bin/env bash
# Evaluate each re-study step checkpoint as it appears (card 0 shared with training, gpu-mem 0.28); keep all checkpoints.
cd ~/projects/teleqna/runs/teleqna-8b
while true; do
  for d in $(ls -d models/kit/restudy/step* 2>/dev/null | sort -t p -k 3 -n); do
    tag="restudy_$(basename $d)"; [ -f "results/landscape/${tag}_otfull_rot1_base_nothink512.json" ] && continue
    [ -f "$d/config.json" ] && { [ -f "$d/model.safetensors" ] || [ -f "$d/model.safetensors.index.json" ]; } || continue
    sleep 120; CKPT=$d TAG=$tag CARDS=0 GPUMEM=0.28 MEMWAIT=100000 bash jobs/eval_ckpt.sh > logs/eval_$tag.log 2>&1; grep RESULT logs/eval_$tag.log
  done
  grep -q "RESTUDY TRAIN DONE" logs/restudy_chain.log 2>/dev/null && break; sleep 600
done
echo "RESTUDY_EVALLOOP_DONE"
