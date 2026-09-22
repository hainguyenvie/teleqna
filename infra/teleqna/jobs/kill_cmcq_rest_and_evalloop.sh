#!/usr/bin/env bash
# Stop the shared-card contrastive generation (resumable) and the eval loop (to relaunch with the first-letter BEST metric).
pkill -f "^bash jobs/kit_cmcq_gen_rest.sh"; pkill -f "gen_contrastive.py --windows-file data/kit/windows_rest_all.jsonl"; pkill -f "^bash jobs/kit_big_evalloop.sh"; sleep 3
pkill -9 -f "gen_contrastive.py --windows-file data/kit/windows_rest_all.jsonl" 2>/dev/null; true
