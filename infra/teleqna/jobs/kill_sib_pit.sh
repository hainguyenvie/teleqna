#!/usr/bin/env bash
pkill -f "^bash jobs/kit_sibling_stage.sh"; pkill -f "^bash jobs/kit_recall_pit.sh"; sleep 1; pkill -f "gen_sibling.py --shard"; true
