#!/usr/bin/env bash
# Stop the masked arm (chain shell + its torchrun workers). Pattern lives here so the ssh command line never matches itself.
pkill -f "^bash jobs/kit_tier1_masked.sh"; sleep 1
pkill -f "train_tier1.py --pack data/kit/tier1/pack_agree_masked"; sleep 5
pkill -9 -f "train_tier1.py --pack data/kit/tier1/pack_agree_masked" 2>/dev/null; true
