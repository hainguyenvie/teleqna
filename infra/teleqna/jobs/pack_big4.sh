#!/usr/bin/env bash
# Big run 4 pack (block 2048): debiased MCQ views (gold-longest capped to 33%) for kit + mcq-gold, style (debiased) x1,
# sibling x2, recall-QA x3, masked, plus All-of-the-above prior rows (beh 'all' x4, 'one' x1 = 4:1). CPU only.
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python; export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=16
cd "$ROOT"; mkdir -p data/kit/big4; echo "#### PACK-4 START $(date -Iseconds)"
"$PY" -u code/kit/pack_tier1.py --blk 2048 --views "data/kit/tier1/views_s*.debiased.jsonl,data/kit/tier2/views_s*.debiased.jsonl,data/kit/tier3/views_s*.debiased.jsonl,data/kit/tier4/views_s*.debiased.jsonl,data/kit/tier5/views_s*.debiased.jsonl" \
  --factview-override "data/kit/tier1_fv30/views_s*.jsonl" \
  --anchor "data/kit/tier1/anchor_selfreplay.jsonl,data/kit/tier2/anchor_selfreplay.jsonl,data/kit/tier3/anchor_selfreplay.jsonl,data/kit/tier4/anchor_selfreplay.jsonl,data/kit/tier5/anchor_selfreplay.jsonl" \
  --chat-qa 0.5 --mcq-gold --out data/kit/big4/pack_kit > logs/big4_pack_kit.log 2>&1; grep -E "blocks of|anchors:|mcq-gold" logs/big4_pack_kit.log | cut -c1-160
"$PY" -u code/kit/mcq_rows.py --globs "data/kit/style/views_s*.debiased.jsonl" --out data/kit/big4/style_rows.jsonl --rot 2 > logs/big4_style_rows.log 2>&1; tail -1 logs/big4_style_rows.log
"$PY" - <<'PY'
import json, random
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"; rng = random.Random(1)
beh = [json.loads(l) for f in ("data/kit/beh/rows_s0.jsonl", "data/kit/beh/rows_s1.jsonl") for l in open(R / f, encoding="utf-8")]
def last_letter(p): return chr(65 + len([x for x in p.split("\n") if len(x) > 2 and x[1] == ")"]) - 1)
allc = [r for r in beh if "All of the above" in r["prompt"] and r["completion"].strip().endswith(last_letter(r["prompt"]))]
one = [r for r in beh if "All of the above" in r["prompt"] and not r["completion"].strip().endswith(last_letter(r["prompt"]))]
rng.shuffle(one); one = one[: len(allc)]   # 'all' x4 + 'one' x1 = 4:1
with open(R / "data/kit/big4/all_rows.jsonl", "w", encoding="utf-8") as f:
    for _ in range(4):
        for r in allc: f.write(json.dumps(dict(prompt=r["prompt"], completion=r["completion"]), ensure_ascii=False) + "\n")
    for r in one: f.write(json.dumps(dict(prompt=r["prompt"], completion=r["completion"]), ensure_ascii=False) + "\n")
print(f"All-prior rows: all x4 = {4*len(allc):,}, one = {len(one):,}")
PY
for spec in "data/kit/big4/style_rows.jsonl:style" "data/kit/big4/all_rows.jsonl:allp"; do f=${spec%%:*}; t=${spec##*:}; "$PY" -u code/kit/pack_chat.py --blk 2048 --rows $f --out data/kit/big4/pack_$t > logs/big4_pack_$t.log 2>&1; tail -2 logs/big4_pack_$t.log | head -1 | cut -c1-120; done
"$PY" -u code/kit/pack_concat.py --packs data/kit/big4/pack_kit,data/kit/big2k/pack_masked,data/kit/big2k/pack_qa,data/kit/big2k/pack_qa,data/kit/big2k/pack_qa,data/kit/big2k/pack_sib,data/kit/big2k/pack_sib,data/kit/big4/pack_style,data/kit/big4/pack_allp --out data/kit/big4/pack --seed 5 > logs/big4_pack.log 2>&1; tail -2 logs/big4_pack.log | head -1
echo "#### PACK-4 DONE $(date -Iseconds)"
