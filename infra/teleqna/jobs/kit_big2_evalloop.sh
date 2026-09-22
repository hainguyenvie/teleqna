#!/usr/bin/env bash
# Evaluate every big-run step checkpoint on ot-full 10k + rot1 as it appears (card 0, shared with the training at gpu-mem 0.28),
# then delete it unless it is the best so far by the first-letter metric (stop-after-letter serving) (disk: 322 GB free, 16 GB per checkpoint).
cd $HOME/projects/teleqna/runs/teleqna-8b
until grep -q "TRAIN-BIG2 START" logs/big2_chain.log 2>/dev/null; do sleep 300; done
while true; do
  for d in $(ls -d models/kit/big2/step* 2>/dev/null | sort -t p -k 3 -n); do
    tag="big2_$(basename $d)"; [ -f "results/landscape/${tag}_otfull_rot1_base_nothink512.json" ] && continue
    [ -f "$d/config.json" ] && { [ -f "$d/model.safetensors" ] || [ -f "$d/model.safetensors.index.json" ]; } || continue
    sleep 120; CKPT=$d TAG=$tag CARDS=0 GPUMEM=0.28 MEMWAIT=100000 bash jobs/eval_ckpt.sh > logs/eval_$tag.log 2>&1; grep RESULT logs/eval_$tag.log
    # disk: keep only the best-scoring step checkpoint so far (16 GB each); ep1 is saved separately by the trainer
    acc=$(grep RESULT logs/eval_$tag.log | sed -E "s/.*first-letter ([0-9.]+) .*/\1/"); best=$(cat models/kit/big2/BEST 2>/dev/null || echo "0 none")
    if [ -n "$acc" ] && [ "$(echo "$acc ${best%% *}" | awk "{print (\$1 > \$2)}")" = 1 ]; then old=${best#* }; [ -d "models/kit/big2/$old" ] && rm -rf "models/kit/big/$old"; echo "$acc $(basename $d)" > models/kit/big2/BEST; else rm -rf "$d"; fi
  done
  grep -q "TRAIN-BIG2 DONE" logs/big2_chain.log 2>/dev/null && break; sleep 600
done
echo "BIG2_EVALLOOP_DONE"
