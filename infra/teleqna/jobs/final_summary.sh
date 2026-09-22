#!/usr/bin/env bash
# After the last queued stage: collect every RESULT line and the best checkpoint into logs/FINAL_SUMMARY.txt.
cd ~/projects/teleqna/runs/teleqna-8b
until grep -q "VD_CHAIN_DONE round=4\|ABORT\|rc=[1-9]" logs/vd4_chain.log 2>/dev/null; do sleep 300; done
{ echo "#### FINAL SUMMARY $(date -Iseconds)"; for t in vd vd2 cm num beh vd3 cm_mlp cm2 vd4; do grep -h RESULT logs/eval_$t.log 2>/dev/null; done; grep -h RESULT logs/eval_restudy_*.log 2>/dev/null; grep -hE "greedy" logs/vote_vd.log logs/vote_vd2.log logs/vote_cm.log logs/vote_vd3.log logs/vote_vd4.log 2>/dev/null; echo "BEST: $(bash jobs/best_ckpt.sh)"; } > logs/FINAL_SUMMARY.txt 2>&1
cat logs/FINAL_SUMMARY.txt
