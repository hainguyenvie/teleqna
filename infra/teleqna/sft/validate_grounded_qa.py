#!/usr/bin/env python3
"""The double gate: keep only the items that are grounded AND that the student
cannot already answer.

Every previous training set on this track was built from what a generator
produced, filtered for form. Form was never the problem. SFT on that pool cost
3.6 points at its gentlest dose and CPT on the same specifications was negative
at all nine checkpoints, and the reason is visible in the pass@8 profile: 60.7%
of dev rows are answered correctly in all eight sampled traces. Training on
material the model already knows spends capacity to relearn it and drifts the
policy for nothing.

So each item passes four checks, in increasing cost order:

  1. grounded    the `evidence` span is a real substring of its chunk, and the
                 answer's content words appear inside that span. No model
                 opinion involved — this is string work.
  2. clean       question tail not present in the real 10,000, not a duplicate
                 of another item. Same guard make_grpo_set.py uses.
  3. answerable  the student, WITH the chunk in context, reproduces the answer.
                 If it cannot, the question is not really supported by the text
                 and the generator hallucinated it.
  4. NOT known   the student, closed-book, fails it in every one of k samples —
                 first without thinking (cheap, drops most), then with thinking
                 on the survivors. Anything it can already reach is discarded.

What survives is, by construction, the intersection of "provable from a public
3GPP document" and "outside the model's reach today" — the same population as
the 160 dev rows where all eight traces miss.

Answer equivalence is deliberately mechanical: normalised containment or token
F1 above --f1. Answers are capped at 12 words upstream precisely so this can be
checked rather than judged.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

_WS = re.compile(r"\s+")
_TAG = re.compile(r"\[[^\]]*\]")
_WORD = re.compile(r"[a-z0-9][a-z0-9.\-]+")
_PUNCT = re.compile(r"[^\w\s.\-]")
STOP = {"the", "a", "an", "of", "to", "in", "is", "are", "for", "and", "or",
        "be", "by", "on", "at", "it", "as", "that", "this", "with", "from",
        "what", "which", "does", "do", "used", "use", "when", "how"}

# Trivia about the documents themselves rather than about telecom. The first
# pilot proved this needs an explicit filter: the "student cannot answer it"
# gate selects for obscurity, and 3GPP process trivia is maximally obscure while
# being worth nothing on a benchmark that asks about protocols. Every one of the
# first three survivors was of this kind.
BOILERPLATE = re.compile(
    r"(?i)\b(3gpp portal|change request|\bCRs?\b|work item|version number"
    r"|(first|second|third) digit|rapporteur|\bTSG\b|copyright|approval process"
    r"|specification numbering|document (history|status|version)"
    r"|release description|editorial change|present document|drafting rule"
    r"|reference list|contact address)\b")

# Table-of-contents leader dots: an evidence span made of them states nothing.
TOC = re.compile(r"\.{4,}")

WITH_CTX = """Use the excerpt to answer.

<spec_excerpt>
{text}
</spec_excerpt>

Question: {question}

Reply with the shortest factual span that answers it. No explanation."""

CLOSED = """Question: {question}

Reply with the shortest factual span that answers it. No explanation."""


def norm(s: str) -> str:
    return _WS.sub(" ", _PUNCT.sub(" ", (s or "").lower())).strip()


def toks(s: str) -> list[str]:
    return [w for w in norm(s).split() if w not in STOP]


def norm_tail(q: str, n: int = 6) -> str:
    w = _WORD.findall(_WS.sub(" ", _TAG.sub(" ", q)).strip().lower())
    return " ".join(w[-n:])


def f1(pred: str, gold: str) -> float:
    p, g = toks(pred), toks(gold)
    if not p or not g:
        return 0.0
    common = collections.Counter(p) & collections.Counter(g)
    same = sum(common.values())
    if not same:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def matches(pred: str, gold: str, thr: float) -> bool:
    np_, ng = norm(pred), norm(gold)
    if ng and ng in np_:
        return True
    return f1(pred, gold) >= thr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--chunks", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True,
                    help="the real 10,000, used only as a contamination guard")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("-k", type=int, default=4, help="closed-book samples")
    ap.add_argument("--f1", type=float, default=0.6)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    chunks = {}
    for line in args.chunks.open(encoding="utf-8"):
        r = json.loads(line)
        chunks[r["chunk_id"]] = r["text"]
    test_tails = {norm_tail(json.loads(l)["question"])
                  for l in args.test.open(encoding="utf-8")}
    print(f"{len(chunks)} chunks, {len(test_tails)} test tails indexed", flush=True)

    items = [json.loads(l) for l in args.items.open(encoding="utf-8")]
    if args.limit:
        items = items[:args.limit]
    total = len(items)
    drops: collections.Counter = collections.Counter()

    # ---- gate 1+2: grounded and clean. String work only, no model. ----------
    seen: set[str] = set()
    stage1 = []
    for it in items:
        text = chunks.get(it["chunk_id"])
        if text is None:
            drops["no_chunk"] += 1
            continue
        if norm(it["evidence"]) not in norm(text):
            drops["evidence_not_verbatim"] += 1
            continue
        ans_words = set(toks(it["answer"]))
        if ans_words and not (ans_words & set(toks(it["evidence"]))):
            drops["answer_not_in_evidence"] += 1
            continue
        # A question that already contains its own answer teaches nothing and
        # still passes the closed-book gate, because the model is being asked to
        # repeat words it was just given. "What is the name of the service for
        # remote identification of UAS?" -> "Remote Identification of UAS".
        if ans_words and ans_words <= set(toks(it["question"])):
            drops["tautology"] += 1
            continue
        if BOILERPLATE.search(it["question"]) or BOILERPLATE.search(it["evidence"]):
            drops["document_boilerplate"] += 1
            continue
        if TOC.search(it["evidence"]):
            drops["toc_evidence"] += 1
            continue
        tail = norm_tail(it["question"])
        if tail in test_tails:
            drops["TEST_OVERLAP"] += 1
            continue
        if tail in seen:
            drops["dup"] += 1
            continue
        seen.add(tail)
        stage1.append(it)
    print(f"gate 1+2 grounded/clean: {len(stage1)}/{total}  {dict(drops)}", flush=True)
    if not stage1:
        return

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=8192, enforce_eager=False)

    def ask(prompts: list[str], n: int, thinking: bool, max_tokens: int):
        chats = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True,
                                         enable_thinking=thinking) for p in prompts]
        sp = SamplingParams(n=n, temperature=1.0 if n > 1 else 0.0,
                            top_p=0.95, top_k=20, max_tokens=max_tokens, seed=42)
        return llm.generate(chats, sp)

    def final_span(text: str) -> str:
        # thinking rollouts carry the trace; the answer is what follows it
        return (text or "").split("</think>")[-1].strip()

    # ---- gate 3: answerable WITH the document -------------------------------
    outs = ask([WITH_CTX.format(text=chunks[it["chunk_id"]][:9000],
                                question=it["question"]) for it in stage1],
               n=1, thinking=False, max_tokens=64)
    stage3 = []
    for it, o in zip(stage1, outs):
        if matches(final_span(o.outputs[0].text), it["answer"], args.f1):
            stage3.append(it)
        else:
            drops["not_answerable_with_context"] += 1
    print(f"gate 3 answerable with context: {len(stage3)}/{len(stage1)}", flush=True)
    if not stage3:
        return

    # ---- gate 4a: closed-book, no thinking. Cheap, drops the bulk. ----------
    outs = ask([CLOSED.format(question=it["question"]) for it in stage3],
               n=args.k, thinking=False, max_tokens=64)
    stage4a = []
    for it, o in zip(stage3, outs):
        hits = sum(matches(final_span(c.text), it["answer"], args.f1)
                   for c in o.outputs)
        if hits == 0:
            stage4a.append(it)
        else:
            drops["known_closed_book_nothink"] += 1
    print(f"gate 4a unknown without thinking: {len(stage4a)}/{len(stage3)}", flush=True)
    if not stage4a:
        return

    # ---- gate 4b: closed-book WITH thinking, on the survivors only ----------
    outs = ask([CLOSED.format(question=it["question"]) for it in stage4a],
               n=args.k, thinking=True, max_tokens=1024)
    kept = []
    for it, o in zip(stage4a, outs):
        hits = sum(matches(final_span(c.text), it["answer"], args.f1)
                   for c in o.outputs)
        if hits == 0:
            kept.append(it)
        else:
            drops["known_closed_book_thinking"] += 1
    print(f"gate 4b unknown with thinking: {len(kept)}/{len(stage4a)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for it in kept:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    report = {
        "in": total, "kept": len(kept),
        "keep_rate": round(len(kept) / total, 4) if total else 0.0,
        "drops": dict(drops),
        "by_spec": dict(collections.Counter(it.get("spec") for it in kept).most_common(25)),
    }
    args.out.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
