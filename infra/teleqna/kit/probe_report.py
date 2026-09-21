#!/usr/bin/env python3
"""DiD table across checkpoints from results/kit/probe_*.json (base = probe_base.json)."""
import json, glob, sys
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
D = sys.argv[1] if len(sys.argv) > 1 else "results/kit"
b = json.load(open(R / f"{D}/probe_base.json"))
print(f"{'ckpt':12s} " + " ".join(f"{d+'_DiD':>9s}" for d in ("mem", "sem", "qa")) + f" {'em_tr':>6s} {'em_ho':>6s} {'emDiD':>6s}")
for f in sorted(glob.glob(str(R / f"{D}/probe_*.json"))):
    tag = f.split("probe_")[-1][:-5]; r = json.load(open(f))
    did = [(r[f"{d}_train"] - b[f"{d}_train"]) - (r[f"{d}_heldout"] - b[f"{d}_heldout"]) for d in ("mem", "sem", "qa")]
    print(f"{tag:12s} " + " ".join(f"{x:+9.3f}" for x in did) + f" {r['em_train']:6.1f} {r['em_heldout']:6.1f} {(r['em_train']-b['em_train'])-(r['em_heldout']-b['em_heldout']):+6.1f}")
