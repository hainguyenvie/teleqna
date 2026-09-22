#!/usr/bin/env bash
# After the LR sweep frees the cards: run tier-2 generation shards 1-3 in parallel on cards 1,3,4 (full memory).
cd $HOME/projects/teleqna/runs/teleqna-8b
until grep -q "TIER1LR_CHAIN_DONE\|ABORT" logs/tier1lr_chain.log 2>/dev/null; do sleep 300; done
for i in $(seq 1 1440); do busy=0; for c in 1 3 4; do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 60; done
(CARDS=1 SHARD=1 NSHARDS=4 GPUMEM=0.90 nohup bash jobs/kit_tier2_gen.sh > logs/tier2_s1.log 2>&1 </dev/null &)
(CARDS=3 SHARD=2 NSHARDS=4 GPUMEM=0.90 nohup bash jobs/kit_tier2_gen.sh > logs/tier2_s2.log 2>&1 </dev/null &)
(CARDS=4 SHARD=3 NSHARDS=4 GPUMEM=0.90 nohup bash jobs/kit_tier2_gen.sh > logs/tier2_s3.log 2>&1 </dev/null &)
echo "#### TIER2 shards 1-3 launched $(date -Iseconds)"
