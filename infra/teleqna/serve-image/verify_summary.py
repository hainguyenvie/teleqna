#!/usr/bin/env python3
"""In lại điểm của lần `verify` gần nhất từ log Inspect, kèm phân tách theo môn."""
import sys, json, zipfile, collections
from pathlib import Path
d = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/verify_teleqna")
logs = sorted(d.glob("*.eval"), key=lambda p: p.stat().st_mtime)
if not logs: print(f"không thấy log .eval trong {d}"); sys.exit(1)
z = zipfile.ZipFile(logs[-1])
names = z.namelist()
hdr = json.loads(z.read("header.json")) if "header.json" in names else {}
res = hdr.get("results") or {}
for s in res.get("scores", []):
    for k, v in (s.get("metrics") or {}).items():
        print(f"{s.get('name','score')}.{k}: {v.get('value')}")
print(f"samples: {res.get('completed_samples')}/{res.get('total_samples')}")
by = collections.Counter(); tot = collections.Counter()
for n in names:
    if not n.startswith("samples/"): continue
    try: smp = json.loads(z.read(n))
    except Exception: continue
    sub = (smp.get("metadata") or {}).get("subject") or "?"
    tot[sub] += 1
    sc = (smp.get("scores") or {}).get("choice") or {}
    if sc.get("value") in ("C", 1, 1.0, True): by[sub] += 1
for sub in sorted(tot):
    print(f"  {sub:28s} {100*by[sub]/tot[sub]:5.1f}%  ({by[sub]}/{tot[sub]})")
