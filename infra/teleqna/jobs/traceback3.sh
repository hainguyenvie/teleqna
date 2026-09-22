#!/usr/bin/env bash
cd $HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
echo "#### TRACEBACK pass3 START $(date -Iseconds)"
OMP_NUM_THREADS=16 $PY -u code/kit/traceback_index.py --k 16 --skip-build > logs/traceback3.log 2>&1
tr "\r" "\n" < logs/traceback3.log | grep -E "indexing|gold text|TRACEBACK_DONE|Traceback|Error" | tail -4
echo "#### COVERAGE pass3 $(date -Iseconds)"; $PY code/kit/coverage_report.py | tee logs/coverage_pass3.log | tail -25
echo "#### TRACEBACK3 END $(date -Iseconds)"
