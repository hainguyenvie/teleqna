#!/usr/bin/env bash
# One-shot status of every chain in this track (run on the server).
cd $HOME/projects/teleqna/runs/teleqna-8b; L=logs
echo "== $(date -u) =="; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr "\n" " "; echo
st() { local name=$1 log=$2 pat=$3; if [ -f "$L/$log" ]; then printf "%-22s %s\n" "$name" "$(grep -E "$pat" "$L/$log" 2>/dev/null | tail -1 | cut -c1-140)"; else printf "%-22s (no log)\n" "$name"; fi; }
st tier2-train tier2_train.log "step [0-9]*00/|TRAIN_DONE|Traceback"
st tier2-chain tier2_chain.log "#### |RESULT|ABORT"
st anchor-fix tier1agree_chain.log "#### |RESULT|ABORT"
st anchor-fix-train tier1agree_train.log "step [0-9]*00/|TRAIN_DONE"
st lr5e5 tier1lr5_chain.log "#### |RESULT|ABORT"
st otel-recite otel_recite_chain.log "#### |uncovered|Traceback"
st deep-rerank deep_rerank_chain.log "#### |uncovered|Traceback"
st deep-bm25 traceback_deep.log "indexing|saved|candidates|DEEP_DONE|Traceback"
st tier3-gen-s0 tier3_s0.log "^#### |^facts:|^factview:|^register|^gate"
st tier3-gen-s1 tier3_s1.log "^#### |^facts:|^factview:|^register|^gate"
st probe-v2 probe_v2_loop.log "#### |Traceback"
echo "-- running chains:"; pgrep -fa "bash jobs/" | grep -v pgrep | awk '{print "   "$3}' | sort | uniq -c
