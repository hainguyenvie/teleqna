#!/usr/bin/env bash
# Order the arm-D pool with KaLM over all 10,000 rows, and write the reranked set.
#
# Under the no-RAG plan this is not a serving component. Its job is to put the
# window that actually carries the fact near the front, so the generator reads 8
# windows instead of 32 for the same coverage: the depth histogram says 87.9% of
# the addressable rows are already reachable within 8 of the BM25 order, and the
# 1,000-row probe put KaLM 2.5 points above BM25 at the same k. Four times less
# teacher time for the same facts is the whole return here; the accuracy of the
# retriever itself never reaches the leaderboard.
#
# Card is chosen by probing BEFORE exporting CUDA_VISIBLE_DEVICES - exporting
# first renumbers the devices and nvidia-smi -i then reports the wrong card, or
# no card at all.
set -euo pipefail
HOME_ROOT=/home/tensara
ROOT="$HOME_ROOT/projects/telelogs/runs/teleqna-sft"
PY="$HOME_ROOT/venv-vllm-nightly/bin/python"
MODEL="$HOME_ROOT/projects/telelogs/shared/models/KaLM-Embedding-Gemma3-12B-2511"

# Card 0 is left out deliberately, not because it is busy - it is kept free for
# the other people on this box. 1, 4 and 6 are equivalent for a single-device
# embedder, so there is nothing to weigh.
PICK=""
for c in 1 4 6; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$c" 2>/dev/null || echo 0)
  echo "card $c free=${free}MiB"
  if [ "$free" -gt 100000 ] && [ -z "$PICK" ]; then PICK="$c"; fi
done
[ -n "$PICK" ] || { echo "ABORT: no free card among 0,1,4,6"; exit 14; }
echo "using card $PICK"
export CUDA_VISIBLE_DEVICES="$PICK"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "#### dense-full START $(date -Iseconds)"
"$PY" "$ROOT/infra/dense_rerank.py" \
  --rag "$ROOT/data/otfull_rag32_expl.jsonl" \
  --model "$MODEL" \
  --explanations "$ROOT/data/TeleQnA_src.json" \
  --out "$ROOT/data/otfull_dense8.jsonl" \
  --report "$ROOT/results/rag/dense_rerank_k8_full.json" \
  -k 8 --batch 96 --maxlen 1024
# batch 96, not the 16 the 1,000-row probe used: at 16 the 12B embedder held
# 32 GiB of a 143 GiB card and would have taken two hours for 200,241 windows.
# Nothing about the ranking changes with batch size - the pooling is per-example
# and the padding is on the left - so this is throughput only.
echo "#### dense-full DONE rc=$? $(date -Iseconds)"
