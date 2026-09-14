#!/usr/bin/env bash
# Wait for the arXiv shard to land, then finish the Tele-Data audit.
#
# The arXiv shard is the one the whole decision rests on — it is the only
# candidate source for the 6,500 research rows — and it is also the slowest to
# download (the HF endpoint keeps timing out and resuming). The audit script
# skips shards it has already measured, so running it again is cheap and safe.
ROOT=/home/tensara/projects/telelogs
for i in $(seq 1 480); do
  if [ -s "$ROOT/shared/corpora/tele-data/arxiv/arxiv.jsonl" ] \
     && ! pgrep -f snapshot_download >/dev/null; then
    echo "arxiv ready $(date -Iseconds)"
    cd "$ROOT/runs/teleqna-sft" && exec bash infra/job_teledata_audit.sh
  fi
  sleep 60
done
echo "TIMEOUT waiting for arxiv $(date -Iseconds)"
