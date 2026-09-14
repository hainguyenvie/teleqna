#!/usr/bin/env python3
"""Why does the model still miss, with the passage in front of it?

466 rows are wrong closed-book, carry the answer inside one retrieved window,
and are STILL wrong when that window is in the prompt. Those are the rows no
amount of extra corpus can buy, so what they actually are decides what the next
training stage has to be. Three stories fit the same summary number and they
prescribe opposite fixes:

  distraction        the window argues FOR a wrong option more than for gold.
                     Retrieval pulled a passage about a neighbouring concept.
                     Fix: retrieval, or a reader trained to weigh evidence.

  hard negative      gold and the chosen option are both supported about
                     equally; the model cannot separate two close claims.
                     Fix: contrastive training on hard negatives.

  conviction         the model answers exactly what it answered closed-book.
                     The passage changed nothing - it was not read.
                     Fix: on-policy, or the prompt/format, not more facts.

plus two that are not model errors at all - unparsable output, and a broken
gold label - which have to be counted before the other three mean anything.

The separating measurement is per-option support: what fraction of an option's
rare terms appear in the window that was shown. If the chosen option scores
higher than gold, the passage genuinely pointed the wrong way and calling that
a discrimination failure would be wrong. Rare terms rather than all terms
because "the", "network" and "procedure" appear in every window and would drown
the signal that separates two options.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

STRICT = re.compile(r"(?i)^ANSWER\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)", re.MULTILINE)
LOOSE = re.compile(r"(?i)ANSWER\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)")
TOKEN = re.compile(r"[a-z0-9][a-z0-9.\-]{2,}")
RULE = "\n\n---\n\n"
MARKER = re.compile(r"(?m)^\[(\d+)\]\s")

# Copied verbatim from retrieve_ctx.py. Widening it here would silently change
# what counts as a rare term and make the buckets incomparable with the
# retrieval numbers everything else in this campaign is quoted against.
STOP = {
    "the", "and", "for", "that", "with", "this", "are", "was", "which", "from",
    "has", "have", "not", "can", "may", "shall", "will", "its", "their", "than",
    "when", "what", "where", "how", "why", "who", "does", "did", "any", "all",
    "one", "two", "following", "above", "below", "used", "use", "using", "such",
    "purpose", "main", "key", "type", "types", "based", "into", "other", "correct",
    "answer", "option", "because", "refers", "statement", "true", "false",
}


def official(completion: str) -> str:
    m = STRICT.findall(completion or "") or LOOSE.findall(completion or "")
    return m[-1].strip().rstrip(".").upper() if m else ""


def terms(text: str) -> set[str]:
    return {t for t in TOKEN.findall((text or "").lower()) if t not in STOP}


def windows(question: str) -> str:
    """The retrieved block only, with the restated question cut off the end."""
    return question.split(RULE)[0] if RULE in question else question


def support(option: str, ctx_terms: set[str]) -> float:
    t = terms(option)
    return len(t & ctx_terms) / len(t) if t else 0.0


def load_eval(path: Path, field: str) -> dict:
    j = json.load(path.open(encoding="utf-8"))
    res = j["results"] if isinstance(j, dict) else j
    return {r["sample_id"]: (r.get(field) or "") for r in res}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--base-field", default="completion")
    ap.add_argument("--ctx", type=Path, required=True)
    ap.add_argument("--ctx-field", default="completion")
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--rag", type=Path, required=True,
                    help="the jsonl actually shown to the ctx arm")
    ap.add_argument("--audit", type=Path, help="label_audit_alwayswrong.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dump", type=int, default=25)
    ap.add_argument("--pairs", type=Path,
                    help="write every discrimination-bucket row as a hard "
                         "negative pair, gold vs the option actually chosen")
    ap.add_argument("--margin", type=float, default=0.10,
                    help="support gap below which two options count as tied")
    ap.add_argument("--dup", type=float, default=0.5,
                    help="option-to-option jaccard above which the two say "
                         "nearly the same thing")
    args = ap.parse_args()

    ground = {}
    for line in args.rows.open(encoding="utf-8"):
        r = json.loads(line)
        ground[r["sample_id"]] = r

    gold, choices, raw_q = {}, {}, {}
    for line in args.gold.open(encoding="utf-8"):
        r = json.loads(line)
        gold[r["sample_id"]] = chr(65 + int(r["answer"]))
        choices[r["sample_id"]] = r["choices"]
        raw_q[r["sample_id"]] = r.get("question", "")

    ctx_text = {}
    for line in args.rag.open(encoding="utf-8"):
        r = json.loads(line)
        ctx_text[r["sample_id"]] = windows(r.get("question", ""))

    broken = set()
    if args.audit and args.audit.exists():
        a = json.load(args.audit.open(encoding="utf-8"))
        cand = a if isinstance(a, list) else (a.get("items") or a.get("rows") or [])
        for r in cand:
            sid = r.get("sample_id") if isinstance(r, dict) else r
            if sid:
                broken.add(sid)

    base = load_eval(args.base, args.base_field)
    ctxc = load_eval(args.ctx, args.ctx_field)

    bucket = collections.Counter()
    by_sub = collections.defaultdict(collections.Counter)
    n_choices = collections.Counter()
    gaps, jacs, samples, pairs = [], [], [], []
    DISCRIM = {"window_supports_both", "near_duplicate_options",
               "discrimination_gold_favoured"}
    # A zero support gap is ambiguous and the ambiguity decides the diagnosis:
    # both options well grounded in the window means the model had the evidence
    # and could not use it, while both at zero means the window it was shown
    # does not carry the answer at all. Groundability was measured over 32
    # windows and the arm was only shown 8, so the second case is not
    # hypothetical - it is whatever fraction of these rows sits at depth >= 8.
    s_gold_by_bucket = collections.defaultdict(list)
    depth_by_bucket = collections.defaultdict(list)
    still_wrong = 0

    for sid, gr in ground.items():
        g = gold.get(sid)
        if g is None or not gr.get("single"):
            continue
        b, c = base.get(sid), ctxc.get(sid)
        if b is None or c is None:
            continue
        valid = {chr(65 + i) for i in range(len(choices[sid]))}
        pb, pc = official(b), official(c)
        if (pb in valid) and pb == g:
            continue                      # right closed-book: not addressable
        if (pc in valid) and pc == g:
            continue                      # context fixed it: one of the 858
        still_wrong += 1

        sub = gr.get("subject", "?")
        n_choices[len(choices[sid])] += 1

        if sid in broken:
            kind = "broken_label"
        elif pc not in valid:
            kind = "unparsable"
        else:
            ct = terms(ctx_text.get(sid, ""))
            g_txt, p_txt = choices[sid][ord(g) - 65], choices[sid][ord(pc) - 65]
            s_gold = support(g_txt, ct)
            s_pick = support(p_txt, ct)
            gaps.append(round(s_pick - s_gold, 3))
            # Order matters and the first version got it backwards. Checking
            # "the answer did not change when the passage arrived" first swept
            # every hard negative into a conviction bucket, because a pair of
            # near-identical options produces a stable wrong answer too.
            # Stability separates nothing. What separates is how close the two
            # options are TO EACH OTHER, and whether the window supports both
            # equally - a window that fully supports gold and the pick alike
            # cannot discriminate them no matter how carefully it is read.
            tg, tp = terms(g_txt), terms(p_txt)
            jac = len(tg & tp) / len(tg | tp) if (tg | tp) else 0.0
            jacs.append(round(jac, 3))
            if jac >= args.dup:
                kind = "near_duplicate_options"
            elif s_gold >= 0.8 and s_pick >= 0.8:
                kind = "window_supports_both"
            elif s_pick > s_gold + args.margin:
                kind = "distraction_ctx_favours_wrong"
            elif s_gold > s_pick + args.margin:
                kind = ("conviction_evidence_ignored" if pc == pb
                        else "discrimination_gold_favoured")
            else:
                kind = "tied_low_support"
            if len(samples) < args.dump:
                samples.append({
                    "sample_id": sid, "subject": sub, "kind": kind,
                    "gold": g, "pred_base": pb, "pred_ctx": pc,
                    "support_gold": round(s_gold, 3), "support_pred": round(s_pick, 3),
                    "option_jaccard": round(jac, 3),
                    "gold_text": choices[sid][ord(g) - 65][:160],
                    "pred_text": choices[sid][ord(pc) - 65][:160],
                })
            s_gold_by_bucket[kind].append(round(s_gold, 3))
            if kind in DISCRIM:
                pairs.append({
                    "sample_id": sid, "subject": sub, "bucket": kind,
                    "question": raw_q.get(sid, ""),
                    "gold_letter": g, "gold_text": g_txt,
                    "hard_negative_letter": pc, "hard_negative_text": p_txt,
                    "option_jaccard": round(jac, 3),
                    "support_gold": round(s_gold, 3),
                    "support_negative": round(s_pick, 3),
                    "stable_closed_book": pc == pb,
                })
        depth_by_bucket[kind].append(int(gr.get("best_window", -1)))
        bucket[kind] += 1
        by_sub[sub][kind] += 1

    gaps.sort()
    report = {
        "still_wrong_with_context": still_wrong,
        "buckets": dict(bucket.most_common()),
        "buckets_pct": {k: round(v / max(still_wrong, 1), 4)
                        for k, v in bucket.most_common()},
        "n_choices_hist": dict(sorted(n_choices.items())),
        "option_similarity_gold_vs_pred": {
            "n": len(jacs),
            "median": sorted(jacs)[len(jacs)//2] if jacs else None,
            "frac_ge_0.5": round(sum(1 for x in jacs if x >= 0.5)/len(jacs), 4) if jacs else None,
            "frac_ge_0.3": round(sum(1 for x in jacs if x >= 0.3)/len(jacs), 4) if jacs else None,
        },
        "support_gap_pred_minus_gold": {
            "n": len(gaps),
            "p10": gaps[len(gaps) // 10] if gaps else None,
            "median": gaps[len(gaps) // 2] if gaps else None,
            "p90": gaps[len(gaps) * 9 // 10] if gaps else None,
            "frac_positive": round(sum(1 for x in gaps if x > 0) / len(gaps), 4)
            if gaps else None,
        },
        "evidence_actually_shown": {
            k: {
                "n": len(v),
                "median_support_gold": sorted(v)[len(v) // 2],
                "frac_gold_unsupported_lt_0.2": round(
                    sum(1 for x in v if x < 0.2) / len(v), 4),
                "frac_gold_well_supported_ge_0.6": round(
                    sum(1 for x in v if x >= 0.6) / len(v), 4),
            } for k, v in sorted(s_gold_by_bucket.items()) if v},
        "depth_beyond_the_8_shown": {
            k: {
                "n": len(v),
                "frac_best_window_ge_8": round(
                    sum(1 for x in v if x >= 8) / len(v), 4),
                "median_best_window": sorted(v)[len(v) // 2],
            } for k, v in sorted(depth_by_bucket.items()) if v},
        "by_subject": {s: dict(c) for s, c in sorted(
            by_sub.items(), key=lambda kv: -sum(kv[1].values()))},
        "margin": args.margin,
        "note": "conviction_unmoved is checked before the support test on "
                "purpose: if the answer is byte-identical to the closed-book "
                "one, what the passage supports is beside the point.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    if args.pairs:
        args.pairs.parent.mkdir(parents=True, exist_ok=True)
        with args.pairs.open("w", encoding="utf-8") as fh:
            for r in pairs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("hard negative pairs:", len(pairs), "->", args.pairs)
    random.Random(0).shuffle(samples)
    args.out.with_suffix(".samples.json").write_text(json.dumps(samples, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "by_subject"}, indent=1))


if __name__ == "__main__":
    main()
