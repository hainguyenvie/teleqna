#!/usr/bin/env bash
cd $HOME/projects/teleqna/runs/teleqna-8b
until grep -q "DL_DONE" logs/dl_corpora.log 2>/dev/null; do sleep 300; done
echo "#### TRACEBACK START $(date -Iseconds)"
OMP_NUM_THREADS=16 $HOME/venv-vllm-nightly/bin/python -u code/kit/traceback_index.py --k 16 > logs/traceback.log 2>&1
grep -E "chunks|indexing|gold text|TRACEBACK_DONE|Error|Traceback" logs/traceback.log | tail -8
echo "#### TRACEBACK END $(date -Iseconds)"
