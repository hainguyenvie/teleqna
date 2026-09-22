#!/usr/bin/env bash
cd $HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
until grep -q "TRACEBACK END" logs/traceback_chain.log 2>/dev/null; do sleep 300; done
echo "#### COVERAGE pass1 $(date -Iseconds)"; $PY code/kit/coverage_report.py | tee logs/coverage_pass1.log | tail -25
cp data/kit/tb_windows.jsonl data/kit/tb_windows.pass1.jsonl
echo "#### STORE FILTER $(date -Iseconds)"; $PY code/kit/store_filter.py
echo "#### TRACEBACK pass2 START $(date -Iseconds)"
OMP_NUM_THREADS=16 $PY -u code/kit/traceback_index.py --k 16 --skip-build > logs/traceback2.log 2>&1
grep -E "indexing|gold text|TRACEBACK_DONE|Traceback" logs/traceback2.log | tail -4
echo "#### COVERAGE pass2 $(date -Iseconds)"; $PY code/kit/coverage_report.py | tee logs/coverage_pass2.log | tail -25
echo "#### TRACEBACK2 END $(date -Iseconds)"
