#!/usr/bin/env python3
"""Cross-tabulate every dev-1000 arm by subject, and price per-subject routing.

The five subjects do not respond alike and never have. Thinking bought +3.80 on
Lexicon and +0.35 on Standards specifications. The sampled profile put Lexicon at
pass@1 88.5 / pass@8 98.0 with a 16% trainable band, against Standards
specifications at 61.4 / 77.5 with a 34% band. The GRPO smoke gained overall
while losing 2.5pp on Standards specifications. A single checkpoint has to be a
compromise across all five; routing does not.

Two numbers matter and they are very different:

  oracle routing   pick each subject's best arm using this very table. This is
                   selection on the data being scored, and with n as low as 50
                   (Lexicon) the max over k arms is biased upward by roughly
                   sigma*sqrt(2*ln k). Reported with that bias estimated, never
                   as an achievable score.
  honest routing   the same policy scored under leave-one-out: for each subject
                   pick the winner using every OTHER subject's ranking... which
                   is not possible, since arms are subject-specific by
                   construction. So instead the bias term is subtracted
                   explicitly and the residual is what remains to believe.

Routing also needs a router. The harness hands the model the question and the
options, never the `subject` field, so a real deployment must classify the
question text. That accuracy is measured separately (classify_subject.py) and
caps whatever this table promises.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

SUBJECTS = ["Lexicon", "Research overview", "Research publications",
            "Standards overview", "Standards specifications"]


def load(dirs: list[str]) -> dict[str, dict]:
    arms = {}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            try:
                j = json.load(open(f))
            except Exception:
                continue
            s = j.get("summary") if isinstance(j, dict) else None
            if not isinstance(s, dict) or "by_subject" not in s:
                continue
            if s.get("total") != 1000:
                continue
            by = s["by_subject"]
            if not all(k in by for k in SUBJECTS):
                continue
            arms[os.path.basename(f)[:-5]] = {
                "overall": s["accuracy"],
                "by": {k: by[k]["acc"] for k in SUBJECTS},
                "n": {k: by[k]["n"] for k in SUBJECTS},
            }
    return arms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True)
    ap.add_argument("--include", default="",
                    help="comma-separated substrings; empty = all arms")
    args = ap.parse_args()

    arms = load(args.dir)
    if args.include:
        want = [w for w in args.include.split(",") if w]
        arms = {k: v for k, v in arms.items() if any(w in k for w in want)}
    if not arms:
        raise SystemExit("no comparable dev-1000 arms found")

    names = sorted(arms, key=lambda k: -arms[k]["overall"])
    n = arms[names[0]]["n"]

    width = max(len(x) for x in names) + 2
    head = "".join(f"{s[:13]:>15s}" for s in SUBJECTS)
    print(f"{'arm':<{width}}{'overall':>9}{head}")
    print(f"{'':<{width}}{'':>9}" + "".join(f"{'n=' + str(n[s]):>15s}" for s in SUBJECTS))
    for k in names:
        row = "".join(f"{arms[k]['by'][s]*100:>15.2f}" for s in SUBJECTS)
        print(f"{k:<{width}}{arms[k]['overall']*100:>9.2f}{row}")

    best_single = names[0]
    print(f"\nbest single arm: {best_single} = {arms[best_single]['overall']*100:.2f}")

    routed, picks, bias_total = 0.0, {}, 0.0
    k_arms = len(arms)
    for s in SUBJECTS:
        win = max(arms, key=lambda a: arms[a]["by"][s])
        acc = arms[win]["by"][s]
        picks[s] = (win, acc)
        routed += acc * n[s]
        # Upward bias of a maximum over k noisy estimates, each with the
        # binomial standard error at this subject's n.
        se = math.sqrt(max(acc * (1 - acc), 1e-9) / n[s])
        bias_total += se * math.sqrt(2 * math.log(max(k_arms, 2))) * n[s]
    routed /= sum(n.values())
    bias = bias_total / sum(n.values())

    print("\nper-subject winners (selected ON this table — biased):")
    for s in SUBJECTS:
        w, a = picks[s]
        print(f"  {s:<28s} {a*100:6.2f}  <- {w}")

    print(f"\noracle routing            {routed*100:6.2f}")
    print(f"selection bias estimate   {bias*100:6.2f}  "
          f"(max over {k_arms} arms, per-subject n)")
    print(f"bias-corrected routing    {(routed - bias)*100:6.2f}")
    print(f"gain over best single     "
          f"{(routed - bias - arms[best_single]['overall'])*100:+6.2f}")
    print("\nA router still has to classify the question: the harness never "
          "passes `subject` to the model, so this is an upper bound that a real "
          "deployment reaches only with a perfect classifier.")


if __name__ == "__main__":
    main()
