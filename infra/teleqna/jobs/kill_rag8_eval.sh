#!/usr/bin/env bash
pkill -f "eval_dev_vllm.py --base models/kit/tier2/ep1 --data data/eval/otfull_rag8_strong.jsonl"; sleep 3
pkill -9 -f "eval_dev_vllm.py --base models/kit/tier2/ep1 --data data/eval/otfull_rag8_strong.jsonl" 2>/dev/null; true
