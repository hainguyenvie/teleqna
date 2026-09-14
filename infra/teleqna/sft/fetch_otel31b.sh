#!/usr/bin/env bash
# Pull OTel-2.0-LLM-31B-IT, then its base google/gemma-4-31B-it.
#
# Three downloaders failed here before this one, and the useful part is why,
# because two of them looked like slow networks rather than bugs:
#
#   hf download --max-workers 16   moved 0.87 GB in 34 minutes (~0.4 MB/s)
#     while an isolated curl to the same shard URL ran at 11.5 MB/s. Sixteen
#     concurrent xet-bridge streams were fighting each other.
#
#   curl -C - --retry-all-errors   shrank the directory from 1.92 GB to 0.6 GB.
#     Not because resume is unsupported -- it is, the CDN answers 206 and a
#     hand-run resume added 244 MB in 35s -- but because --retry-all-errors
#     restarts the transfer on any failure and -o truncates on the way in.
#
#   curl -sfL, no retry at all   completed exactly 3 shards of 14 and left the
#     other 11 at 4 MB to 361 MB, three passes running. That is the real fault
#     underneath the other two: the connection drops partway through a multi-GB
#     transfer, reliably, and one-shot curl simply gives up where it stopped.
#
# So the shape that works is resume plus an outer loop: each attempt keeps
# whatever arrived and asks for the rest, and a file is finished when its size
# matches what the Hub API reports -- not when curl exits 0, which it did while
# leaving 4 MB on disk. The drops are per-connection rather than a bandwidth cap,
# so the fix for throughput is more streams, not fewer -- see the xargs -P below.
set -uo pipefail
DEST=$HOME/projects/telelogs/shared/models
TRIES=150         # a drop costs one attempt and each attempt keeps its bytes,
                  # so this bounds stalls, not progress. Sized from the observed
                  # drop pattern: connections die after roughly 30s at ~12 MB/s,
                  # about 360 MB, so a 4.9 GB shard needs ~14 resumes on a good
                  # day and the ceiling has to sit well clear of that.

fetch_one() {     # fetch_one <repo> <dir> <path> <want-bytes>
  repo=$1; dir=$2; path=$3; want=$4
  for t in $(seq 1 "$TRIES"); do
    got=$( [ -f "$dir/$path" ] && stat -c %s "$dir/$path" || echo 0 )
    [ "$got" = "$want" ] && { [ "$t" = 1 ] || echo "  ok $path (${t} attempts)"; return 0; }
    curl -sL -C - --create-dirs -o "$dir/$path" \
      "https://huggingface.co/$repo/resolve/main/$path" >/dev/null 2>&1
    new=$( [ -f "$dir/$path" ] && stat -c %s "$dir/$path" || echo 0 )
    # Back off only when an attempt gained nothing. Drops here are constant
    # rather than exceptional, so a long sleep on every reconnect would cost
    # more than the drops do.
    [ "$new" = "$got" ] && sleep 2
  done
  echo "  GIVE UP $path after $TRIES attempts"; return 1
}
export -f fetch_one
export TRIES

get() {           # get <repo> <destdir>
  repo=$1; dir=$2
  mkdir -p "$dir"
  echo "#### $repo START $(date -Iseconds)"
  curl -sL "https://huggingface.co/api/models/$repo/tree/main?recursive=1" > "$dir/.tree.json"
  python3 - "$dir" <<'PY'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
files = [(f["path"], f["size"]) for f in json.loads((d / ".tree.json").read_text())
         if f.get("type") == "file"]
(d / ".want.tsv").write_text("".join(f"{p}\t{s}\n" for p, s in files))
print(f"  {len(files)} files, {sum(s for _, s in files)/1e9:.1f} GB")
PY
  # Biggest first: they are the ones that drop, and starting them early means
  # the long tail is not a few small files waiting behind one 5 GB shard.
  #
  # Twelve streams, not four. Every connection dies after about 30 seconds no
  # matter how few are open, so this is not a bandwidth cap being shared out --
  # it is a per-connection lifetime, and the only way to keep the pipe full is
  # to have more connections in flight than are dying at any moment. Four
  # streams measured 5 MiB/s aggregate against the 12 MiB/s a single healthy
  # stream reaches, which is the reconnect gap showing up as idle time; twelve
  # measured 14 MiB/s. Twenty-four measured 10, so the gain is not monotonic --
  # somewhere past a dozen streams a real aggregate ceiling takes over and the
  # extra connections only add reconnect overhead. Twelve is the measured best.
  sort -k2 -n -r "$dir/.want.tsv" \
    | xargs -P 12 -n 2 bash -c 'fetch_one "$0" "$1" "$2" "$3"' "$repo" "$dir"

  python3 - "$dir" <<'PY'
import pathlib, sys
d = pathlib.Path(sys.argv[1])
bad = [(p, int(s), (d / p).stat().st_size if (d / p).exists() else -1)
       for p, s in (l.rsplit("\t", 1) for l in (d / ".want.tsv").read_text().splitlines())
       if not (d / p).exists() or (d / p).stat().st_size != int(s)]
for p, w, g in bad:
    print(f"  BAD {p}: want {w} got {g}")
sys.exit(f"INCOMPLETE: {len(bad)} file(s)" if bad else 0)
PY
  rc=$?
  [ $rc = 0 ] && echo "  all files match the Hub-reported sizes"
  echo "#### $repo $( [ $rc = 0 ] && echo DONE || echo FAILED ) $(date -Iseconds)"
  du -sh "$dir"
  return $rc
}

# Sequential, so the 31B lands first and the base downloads underneath its eval
# instead of competing with it. The base is not optional: OTel 2.0 is a
# post-train of Gemma 4 31B-IT, so its score is Gemma-4's own knowledge plus
# whatever 440B telecom tokens added, and only the second term is news.
get farbodtavakkoli/OTel-2.0-LLM-31B-IT "$DEST/OTel-2.0-31B-IT"
get google/gemma-4-31B-it              "$DEST/gemma-4-31B-it"
echo "#### ALL DONE $(date -Iseconds)"
