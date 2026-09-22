#!/usr/bin/env bash
# vd round 3 = consistency pass on top of the contrastive stage (cm/ep1: recover 1181 but broke 491) — runs after the behaviour stage.
cd ~/projects/teleqna/runs/teleqna-8b
until grep -q "BEH_CHAIN_DONE\|ABORT" logs/beh_chain.log 2>/dev/null; do sleep 300; done
ROUND=3 INIT=models/kit/cm/ep1 N=100000 SEED=3 bash jobs/kit_vd_round.sh
