#!/usr/bin/env python3
"""Error classification, corrected, plus the question that decides the next arm.

Fix to the first version: features comparing the CHOSEN option to the gold one
(near-synonym, length ratio) fire trivially on every correct row, because there
the chosen option IS the gold one. Their "error rate" column was meaningless.
They are now reported as a breakdown WITHIN the wrong rows, and only
item-intrinsic features (catch-all present, numeric, negated stem) keep a real
denominator.

New and more useful: on the rows arm E trained to FLIP, did the model obey the
gate and the gate was wrong, or did it ignore the gate? That separates taught
error from unlearned error, and it is what decides whether gate purity is worth
more work -- the wide-vs-high comparison said purity 70% vs 81% was a wash, and
this checks that from the other side.
"""
import json, os, re, collections

ROOT = os.path.expanduser("~/projects/telelogs/runs/teleqna-sft")
R = f"{ROOT}/results"
CANON = os.path.expanduser("~/projects/telelogs/runs/bench4/teleqna/data/test.jsonl")

ref = {}
for l in open(CANON):
    r = json.loads(l)
    ch = r["choices"]
    r["choices"] = eval(ch) if isinstance(ch, str) else ch
    ref[r["sample_id"]] = r

def load(p):
    d = json.load(open(p))
    return {r["sample_id"]: r for r in d["results"]}

arm = load(f"{R}/landscape/otfull_armEwide_armEwide_nothink512.json")
base = load(f"{R}/landscape/otfull_armEwide_base_nothink512.json")

taught = {}
for l in open(f"{ROOT}/data/train/eligible/armE_wide.jsonl"):
    r = json.loads(l)
    taught[r["sample_id"]] = (r["tier"], r["completion"].rstrip()[-1])

TOKEN = re.compile(r"[a-z0-9][a-z0-9.\-]{2,}")
STOP = set("the a an of to in for and or is are be by on with as at from that this it its".split())
def toks(s):
    return {w for w in TOKEN.findall(s.lower()) if w not in STOP}

CATCHALL = re.compile(r"\b(all|none|both) of the (above|these)\b", re.I)
NUMERIC = re.compile(r"\d")
NEG = re.compile(r"\b(not|never|cannot|except|incorrect|false)\b", re.I)

ids = sorted(set(arm) & set(base))
wrong = [s for s in ids if not arm[s]["correct"]]
gold = {s: chr(65 + int(ref[s]["answer"])) for s in ids}

print(f"arm E wrong on {len(wrong)}/{len(ids)} = {len(wrong)/len(ids)*100:.1f}%\n")

print("=== item-intrinsic shape (real denominators) ===")
print("  %-28s %7s %7s %9s" % ("feature", "wrong", "all", "err rate"))
base_rate = len(wrong) / len(ids) * 100
for name, fn in [
    ("catch-all option present", lambda s: any(CATCHALL.search(c) for c in ref[s]["choices"])),
    ("numeric / quantity item", lambda s: sum(1 for c in ref[s]["choices"] if NUMERIC.search(c))
                                          >= max(2, len(ref[s]["choices"]) - 1)),
    ("negated stem", lambda s: bool(NEG.search(ref[s]["question"]))),
    ("5 options", lambda s: len(ref[s]["choices"]) == 5),
    ("4 options", lambda s: len(ref[s]["choices"]) == 4),
]:
    all_n = [s for s in ids if fn(s)]
    w = [s for s in all_n if not arm[s]["correct"]]
    if all_n:
        print("  %-28s %7d %7d %8.1f%%  (%+.1f vs %.1f baseline)"
              % (name, len(w), len(all_n), len(w) / len(all_n) * 100,
                 len(w) / len(all_n) * 100 - base_rate, base_rate))

print("\n=== how the wrong answer relates to the gold one (wrong rows only) ===")
rel = collections.Counter()
for s in wrong:
    ch = ref[s]["choices"]
    got = arm[s]["parsed"]
    if not got or not (0 <= ord(got) - 65 < len(ch)):
        rel["unparsed / out of range"] += 1
        continue
    g, c = ch[ord(gold[s]) - 65], ch[ord(got) - 65]
    tg, tc = toks(g), toks(c)
    j = len(tg & tc) / len(tg | tc) if (tg or tc) else 0
    if j >= 0.4:
        rel["near-synonym of gold (Jaccard>=0.40)"] += 1
    elif j >= 0.15:
        rel["partial overlap (0.15-0.40)"] += 1
    else:
        rel["lexically unrelated to gold"] += 1
for k, v in rel.most_common():
    print("  %-42s %5d  %5.1f%% of errors" % (k, v, v / len(wrong) * 100))

print("\n=== taught error vs unlearned error, on the 855 flip rows ===")
fl = [s for s in ids if taught.get(s, ("", ""))[0] == "flip"]
flw = [s for s in fl if not arm[s]["correct"]]
obeyed_wrong = [s for s in flw if arm[s]["parsed"] == taught[s][1]]
label_was_wrong = [s for s in fl if taught[s][1] != gold[s]]
print(f"  flip rows                                {len(fl):5d}")
print(f"  of them the gate's label was WRONG       {len(label_was_wrong):5d} "
      f"({len(label_was_wrong)/len(fl)*100:.1f}%)")
print(f"  arm E wrong on                           {len(flw):5d}")
print(f"    ...because it obeyed a wrong label     {len(obeyed_wrong):5d} "
      f"({len(obeyed_wrong)/len(flw)*100:.1f}% of the flip errors)")
print(f"    ...because it did not learn the label  {len(flw)-len(obeyed_wrong):5d}")
obeyed_all = [s for s in fl if arm[s]["parsed"] == taught[s][1]]
print(f"  obeys the taught letter on               {len(obeyed_all)/len(fl)*100:.1f}% of flip rows")
right_label = [s for s in fl if taught[s][1] == gold[s]]
got_right = [s for s in right_label if arm[s]["correct"]]
print(f"  where the label was RIGHT, arm E is right on "
      f"{len(got_right)}/{len(right_label)} = {len(got_right)/len(right_label)*100:.1f}%")

print("\n=== same question for the retain tier ===")
rt = [s for s in ids if taught.get(s, ("", ""))[0] == "retain"]
rtw = [s for s in rt if not arm[s]["correct"]]
rt_label_wrong = [s for s in rt if taught[s][1] != gold[s]]
rt_obeyed_wrong = [s for s in rtw if arm[s]["parsed"] == taught[s][1]]
print(f"  retain rows {len(rt)}, gate label wrong on {len(rt_label_wrong)} "
      f"({len(rt_label_wrong)/len(rt)*100:.1f}%)")
print(f"  arm E wrong on {len(rtw)}, of which obeyed a wrong label {len(rt_obeyed_wrong)} "
      f"({len(rt_obeyed_wrong)/len(rtw)*100:.1f}%)")
