#!/usr/bin/env bash
# Stop the re-study arm early (chain, eval loop, torchrun workers) and the final-summary waiter.
pkill -f "^bash jobs/kit_restudy.sh"; pkill -f "^bash jobs/restudy_evalloop.sh"; pkill -f "^bash jobs/final_summary.sh"; sleep 1
pkill -f "train_tier1.py --pack data/kit/restudy/pack"; sleep 8
pkill -9 -f "train_tier1.py --pack data/kit/restudy/pack" 2>/dev/null; true
