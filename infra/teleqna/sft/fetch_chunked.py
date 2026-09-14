#!/usr/bin/env python3
"""Fetch HF weights in fixed byte ranges, verifying every chunk and every file.

Why not curl -C -, which is the obvious answer and was tried first: resume asks
for `Range: bytes=N-`, huggingface.co answers 302, and the CDN edge behind it
does not reliably honour the header it gets handed. When it answers 206 the
resume works — a hand-run test added 244 MB that way. When it answers 200 curl
appends the *whole file* to the partial one, and the result is a shard that is
too big rather than too small: 11 of 14 here, over by 2 KB to 127 MB. Nothing
downstream would catch that. A "is the file at least the expected size" test
passes, safetensors carries no internal checksum, and the model loads into
tensors of noise.

So this asks for explicit ranges instead of open-ended ones, and checks the only
two things that can be checked: every chunk is exactly as long as the range that
was requested (a 200-instead-of-206 chunk is the full file and fails on length),
and every assembled file matches the sha256 the Hub publishes as lfs.oid.

Chunk size is set against the observed failure mode rather than for throughput.
Connections here die after roughly 30 seconds at ~12 MB/s, so 128 MB lands near
10 seconds and mostly completes inside the window; a dropped chunk costs one
chunk rather than a multi-GB shard. Twelve workers, measured best on this link —
four gave 5 MiB/s aggregate, twelve 14, twenty-four back down to 10.

Usage: fetch_chunked.py <repo> <dir> [--all]
       default repairs only files whose size or sha256 is wrong; --all refetches
"""
import concurrent.futures as cf
import hashlib
import json
import pathlib
import subprocess
import sys
import urllib.request

CHUNK = 128 << 20
WORKERS = 12
TRIES = 60

repo, root = sys.argv[1], pathlib.Path(sys.argv[2])
refetch_all = "--all" in sys.argv
root.mkdir(parents=True, exist_ok=True)

api = f"https://huggingface.co/api/models/{repo}/tree/main?recursive=1"
tree = json.loads(urllib.request.urlopen(api).read())
files = [(f["path"], f["size"], (f.get("lfs") or {}).get("oid"))
         for f in tree if f.get("type") == "file"]
print(f"{len(files)} files, {sum(s for _, s, _ in files)/1e9:.1f} GB", flush=True)


def sha256(p):
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def need(path, size, oid):
    p = root / path
    if refetch_all or not p.exists() or p.stat().st_size != size:
        return True
    if oid and sha256(p) != oid:
        print(f"  sha mismatch, will refetch: {path}", flush=True)
        return True
    return False


def get_chunk(args):
    path, idx, start, end = args
    part = root / f"{path}.part{idx:05d}"
    want = end - start + 1
    for _ in range(TRIES):
        if part.exists() and part.stat().st_size == want:
            return True
        subprocess.run(
            ["curl", "-sL", "-r", f"{start}-{end}", "--create-dirs",
             "-o", str(part), f"https://huggingface.co/{repo}/resolve/main/{path}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # A chunk that comes back the wrong length is the 200-instead-of-206
        # case (the whole file) or a mid-flight drop. Both are simply retried;
        # nothing is ever appended to a file that already has bytes in it.
        if part.exists() and part.stat().st_size != want:
            part.unlink()
    print(f"  GIVE UP {path} chunk {idx}", flush=True)
    return False


todo = [(p, s, o) for p, s, o in files if need(p, s, o)]
print(f"{len(todo)} file(s) to fetch, "
      f"{sum(s for _, s, _ in todo)/1e9:.1f} GB", flush=True)

jobs = []
for path, size, _ in todo:
    (root / path).unlink(missing_ok=True)
    for i, start in enumerate(range(0, size, CHUNK)):
        jobs.append((path, i, start, min(start + CHUNK, size) - 1))
print(f"{len(jobs)} chunks of {CHUNK >> 20} MB", flush=True)

with cf.ThreadPoolExecutor(WORKERS) as ex:
    done = 0
    for ok in ex.map(get_chunk, jobs):
        done += 1
        if done % 25 == 0:
            print(f"  {done}/{len(jobs)} chunks", flush=True)

bad = []
for path, size, oid in todo:
    parts = sorted(root.glob(f"{path}.part*"))
    with (root / path).open("wb") as out:
        for q in parts:
            out.write(q.read_bytes())
            q.unlink()
    got = (root / path).stat().st_size
    if got != size:
        bad.append((path, f"size {got} != {size}"))
    elif oid and sha256(root / path) != oid:
        bad.append((path, "sha256 mismatch"))
    else:
        print(f"  ok {path}", flush=True)

for path, why in bad:
    print(f"  BAD {path}: {why}", flush=True)
print("#### FETCH " + ("FAILED" if bad else "DONE"), flush=True)
sys.exit(1 if bad else 0)
