#!/usr/bin/env python3
"""Check downloaded weights against the sha256 the Hub publishes, not their size.

Size equality is what the fetch loop uses to decide a file is finished, and it
is the right test *there* — it is cheap and it catches the failure that actually
happens, a dropped connection leaving 4 MB of a 5 GB shard. It is not sufficient
here, for one concrete reason: at 10:22 a hand-run `curl -C -` was pointed at
model-00013 while the fetch loop already had that shard assigned. Two writers
appending at independently-tracked offsets can land on the right total length
with interleaved bytes inside, and safetensors carries no internal checksum, so
that file would load as a tensor of noise rather than raise anything.

The Hub reports each LFS file's sha256 as `lfs.oid` in the tree API, so the
check is exact rather than heuristic. Anything that fails is deleted, which
makes the fetch script pick it up again on the next run — the two scripts
compose without needing to know about each other.

Usage: verify_shards.py <model-dir> [--delete-bad]
"""
import hashlib
import json
import pathlib
import sys

d = pathlib.Path(sys.argv[1])
delete = "--delete-bad" in sys.argv
tree = json.loads((d / ".tree.json").read_text())

bad, checked, skipped = [], 0, []
for f in tree:
    if f.get("type") != "file":
        continue
    p = d / f["path"]
    oid = (f.get("lfs") or {}).get("oid")
    if not oid:
        skipped.append(f["path"])          # small non-LFS files: config, tokenizer
        continue
    if not p.exists():
        bad.append((f["path"], "missing"))
        continue
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    checked += 1
    if h.hexdigest() != oid:
        bad.append((f["path"], f"sha256 {h.hexdigest()[:12]} != {oid[:12]}"))
        print(f"  BAD  {f['path']}: {bad[-1][1]}", flush=True)
    else:
        print(f"  ok   {f['path']}", flush=True)

print(f"\nchecked {checked} LFS files, {len(skipped)} small files not hashed "
      f"(no lfs oid published for them)")
if bad:
    for name, why in bad:
        if delete and (d / name).exists():
            (d / name).unlink()
    print(f"FAIL: {len(bad)} file(s) corrupt"
          + (" — deleted, re-run the fetch script" if delete else ""))
    sys.exit(1)
print("PASS: every LFS file matches its published sha256")
