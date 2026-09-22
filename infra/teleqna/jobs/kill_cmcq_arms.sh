#!/usr/bin/env bash
pkill -f "^bash jobs/kit_cmcq_arms.sh"; sleep 1
pkill -f "pack_tier1.py --views .* --out data/kit/tier1/pack_arm"; sleep 2
pkill -f "train_tier1.py --pack data/kit/tier1/pack_arm"; true
