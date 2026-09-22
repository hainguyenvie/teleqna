#!/usr/bin/env bash
pkill -f "^bash jobs/kit_big2_train.sh"; sleep 1; pkill -f "train_tier1.py --pack data/kit/big2/pack"; sleep 8; pkill -9 -f "train_tier1.py --pack data/kit/big2/pack" 2>/dev/null; true
