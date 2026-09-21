#!/usr/bin/env python3
"""Tier-1 study-kit generator (PLAN_CLOSED_BOOK_8B §6.2, recipe fixed by the 14/09 pilots).

Unit = strong-RAG window from data/kit/windows_keep.jsonl. Generator = Qwen3-8B (pilot: its own
rewrites are read best by itself, 68% of RAG vs 52% for OTel-31B, and 2.7x faster).
Per window:
  facts      JSON facts with a verbatim answer span (gate: span in source, not in question)
  factviews  each gated fact rewritten N ways, answer span verbatim in every line (K-sweep mechanism)
  register   whole-window rewrites in R registers (gate: atom coverage >= 0.7 vs source)
  qa         12 Q/A + why (gate: answer verbatim in source)         -> prompt-distillation stage
  mcq        4 harness-style MCQs with near-miss distractors (gate: evidence in source) -> anchors
Answer key / explanation never enter any prompt. Resumable per shard: windows already in the output are skipped.
"""
import json, re, argparse, unicodedata, collections
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
GEN = str(Path.home() / "projects/_shared/models/Qwen3-8B")
ATOM = re.compile(r"\b(?:TS\s?\d+\.\d+|TR\s?\d+\.\d+|Rel-?\d+|\d+(?:\.\d+)?\s?(?:ms|s|dB|dBm|GHz|MHz|kHz|Mbps|Gbps|bit|bits|bytes|%)|[A-Z]{3,7}|\d{2,4})\b")
STOP = {"the","and","for","this","that","with","not","are","was","its","can","may","shall","from"}
META = re.compile(r"(?i)\b(the|this)\s+(document|excerpt|passage|text|section)\s+(does not|doesn't|is silent|lacks|omits|provides no|has no)|not (explicitly )?(listed|stated|specified|mentioned|described|defined|provided|given) in (the|this) (document|excerpt|passage|text|section)")
def norm(s): return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s or "").lower()).strip()
def atoms_ok(text, srcn, th=0.7):
    at = {x for x in ATOM.findall(text) if x.lower() not in STOP and len(x) >= 2}
    return True if not at else sum(1 for x in at if norm(x) in srcn) / len(at) >= th
def jl(text):
    out = []
    for line in text.splitlines():
        line = line.strip().strip(",")
        if line.startswith("{") and line.endswith("}"):
            try: out.append(json.loads(line))
            except Exception: pass
    return out

FACTS = ("EXCERPT:\n{doc}\n\nList EVERY atomic fact in the excerpt. For each fact give: \"fact\" — one standalone sentence with "
         "the subject named explicitly; \"answer\" — the short distinctive answer span (a value, identifier, term, name) copied "
         "VERBATIM from the excerpt; \"q\" — a question whose answer is exactly that span and which does not contain the span. "
         "Reply with one JSON object per line, nothing else:\n{{\"fact\": \"...\", \"answer\": \"...\", \"q\": \"...\"}}")
VIEWS = ("Here is a fact from a telecom specification or paper:\n\n{fact}\n\nRewrite it {n} different ways. Vary sentence structure, "
         "voice, word order, level of detail and register: plain statement, definition, passive, inverted, a question with its answer, "
         "a short note, a spec-style clause, a tutorial sentence, a comparison, a consequence. Each rewriting is standalone and names "
         "the subject explicitly. HARD REQUIREMENT: the exact string \"{answer}\" must appear verbatim in every rewriting. "
         "Reply with exactly {n} lines numbered \"1. \" ... and nothing else.")
REG = {
 "textbook": "Rewrite the excerpt as a section of an engineering textbook for someone who has never read it: full prose, well organised, self-contained.",
 "notes": "Write terse study notes an engineer would keep from this excerpt: one complete standalone sentence per fact, subject named explicitly, exact values kept. One sentence per line, no bullets.",
 "tutorial": "Explain the content of the excerpt step by step to a junior engineer, as a tutorial, keeping every technical detail exact.",
 "faq": "Turn the excerpt into a FAQ: 8-12 questions an engineer would ask, each followed by a precise answer taken from the excerpt.",
 "clause": "Rewrite the excerpt as formal normative specification text (shall/may/should), one requirement per sentence, exact identifiers and values.",
 "summary_detail": "First summarise the excerpt in two sentences, then restate every detail of it in full, grouped by topic.",
 "walkthrough": "Describe the procedure, mechanism or concept in the excerpt as a narrative walkthrough of what happens and why, with every parameter and condition named exactly.",
 "cheatsheet": "Produce an exam cheat-sheet of the excerpt: every definition, number, identifier, condition and relationship, phrased as complete sentences.",
}
REG_TAIL = (" HARD REQUIREMENTS: keep EVERY concrete value, identifier, parameter name, message/procedure name, condition and reference "
            "exactly as in the excerpt; add nothing that is not in the excerpt; never comment on what the excerpt lacks; no preamble, no markdown headings.")
QA = ("EXCERPT:\n{doc}\n\nWrite 12 diverse exam questions about this excerpt, spanning definitions, values, conditions, procedures and "
      "relationships. Each answer must be short (<= 12 words) and must appear VERBATIM in the excerpt. Also give a one-sentence "
      "explanation. One JSON object per line, nothing else:\n{{\"q\": \"...\", \"a\": \"...\", \"why\": \"...\"}}")
MCQ = ("EXCERPT:\n{doc}\n\nWrite 4 multiple-choice questions in the style of a telecom certification exam, each with 4 or 5 options and "
       "exactly one correct option. Distractors must be plausible near-misses (a neighbouring value, a sibling procedure, an adjacent "
       "release, a related but different entity). Do not use 'All of the above' or 'None of the above'. Give the verbatim excerpt "
       "sentence that supports the correct option as \"evidence\". One JSON object per line, nothing else:\n"
       "{{\"q\": \"...\", \"options\": [\"...\", \"...\", \"...\", \"...\"], \"answer\": 0, \"evidence\": \"...\"}}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--nviews", type=int, default=10); ap.add_argument("--registers", default=",".join(REG))
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--out", default="data/kit/tier1"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-mcq", action="store_true", help="regenerate only the mcq view for all windows of the shard (strips old mcq rows)")
    ap.add_argument("--windows-file", default="data/kit/windows_keep.jsonl"); ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--only-factview", default="", help="path of an existing views file: reuse its gated facts, regenerate factview with --nviews into --out")
    a = ap.parse_args()
    outdir = R / a.out; outdir.mkdir(parents=True, exist_ok=True); outp = outdir / f"views_s{a.shard}.jsonl"
    done = set()
    if a.only_mcq and outp.exists():
        tmp = outp.with_suffix(".nomcq"); n_strip = 0
        with open(tmp, "w", encoding="utf-8") as fh:
            for l in open(outp, encoding="utf-8"):
                if '"view": "mcq"' in l: n_strip += 1; continue
                fh.write(l)
        tmp.replace(outp); print(f"stripped {n_strip} old mcq rows", flush=True)
    if outp.exists() and not a.only_mcq:
        for l in open(outp, encoding="utf-8"):
            try: done.add(json.loads(l)["win_id"])
            except Exception: pass
    wins = [json.loads(l) for l in open(R / a.windows_file, encoding="utf-8")]
    wins = [w for i, w in enumerate(wins) if i % a.nshards == a.shard and w["win_id"] not in done]
    if a.limit: wins = wins[:a.limit]
    print(f"shard {a.shard}/{a.nshards}: {len(wins)} windows to do ({len(done)} already done)", flush=True)
    if not wins: print("KIT_GEN_DONE"); return
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(GEN, local_files_only=True)
    llm = LLM(model=GEN, tensor_parallel_size=1, gpu_memory_utilization=a.gpu_mem, max_model_len=8192, dtype="bfloat16", seed=a.seed)
    rend = lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    def D(t):
        ids = tok(t, add_special_tokens=False)["input_ids"]; return tok.decode(ids[:3000]) if len(ids) > 3000 else t
    docs = [D(w["text"]) for w in wins]; srcn = [norm(d) for d in docs]
    out = open(outp, "a", encoding="utf-8"); st = collections.Counter(); ntok = collections.Counter()
    def emit(i, view, text, ok, **kw):
        st[view, ok] += 1
        if ok: ntok[view] += int(len(text.split()) * 1.35)
        out.write(json.dumps(dict(win_id=wins[i]["win_id"], view=view, text=text, ok=ok, **kw), ensure_ascii=False) + "\n")
    def run(prompts, temp, mx): return [o.outputs[0].text.strip() for o in llm.generate([rend(p) for p in prompts], SamplingParams(temperature=temp, top_p=0.95, max_tokens=mx))]

    if a.only_factview:
        byw = collections.defaultdict(list)
        for l in open(R / a.only_factview, encoding="utf-8"):
            r = json.loads(l)
            if r["view"] == "facts" and r["ok"]: byw[r["win_id"]].append(json.loads(r["text"]))
        jobs, meta = [], []
        for i, w in enumerate(wins):
            for j, r in enumerate(byw.get(w["win_id"], [])[:20]): jobs.append(VIEWS.format(fact=r["fact"], answer=r["answer"], n=a.nviews)); meta.append((i, j, r["answer"]))
        print(f"factview x{a.nviews}: {len(jobs):,} facts", flush=True)
        for (i, j, ans), t_ in zip(meta, run(jobs, 0.9, 2600)):
            seen = set()
            for l in [re.sub(r"^\s*\d+[.)]\s*", "", x).strip() for x in t_.splitlines()]:
                if len(l) < 20 or norm(l) in seen: continue
                seen.add(norm(l)); emit(i, "factview", l, norm(ans) in norm(l), fact=j)
        out.close(); print("factview:", {f"{k[0]}:{k[1]}": v for k, v in sorted(st.items())}, " tokens", dict(ntok), flush=True); print("KIT_GEN_DONE", flush=True); return
    if a.only_mcq:
        for i, t_ in enumerate(run([MCQ.format(doc=d) for d in docs], 0.5, 1500)):
            for r in jl(t_):
                if all(k in r for k in ("q", "options", "answer", "evidence")) and isinstance(r["options"], list) and 4 <= len(r["options"]) <= 5 \
                        and all(isinstance(x, str) and x.strip() for x in r["options"]) and isinstance(r["q"], str) and isinstance(r["evidence"], str) \
                        and isinstance(r["answer"], int) and 0 <= r["answer"] < len(r["options"]):
                    ok = norm(r["evidence"])[:80] in srcn[i] and len({norm(x) for x in r["options"]}) == len(r["options"])
                    emit(i, "mcq", json.dumps(r, ensure_ascii=False), ok)
        out.close(); print("mcq:", {f"{k[0]}:{k[1]}": v for k, v in sorted(st.items())}, flush=True); print("KIT_GEN_DONE", flush=True); return
    # verbatim
    for i, d in enumerate(docs): emit(i, "verbatim", d, True)
    # facts
    facts = collections.defaultdict(list)
    for i, t in enumerate(run([FACTS.format(doc=d) for d in docs], 0.3, 1600)):
        for r in jl(t):
            if all(k in r for k in ("fact", "answer", "q")) and isinstance(r["answer"], str):
                ok = 2 <= len(r["answer"]) <= 80 and norm(r["answer"]) in srcn[i] and norm(r["answer"]) not in norm(r["q"])
                emit(i, "facts", json.dumps(r, ensure_ascii=False), ok)
                if ok: facts[i].append(r)
    print(f"facts: {dict((k[1], v) for k, v in st.items() if k[0]=='facts')}", flush=True)
    # fact views (K-sweep mechanism)
    jobs, meta = [], []
    for i, fl in facts.items():
        for j, r in enumerate(fl[:20]): jobs.append(VIEWS.format(fact=r["fact"], answer=r["answer"], n=a.nviews)); meta.append((i, j, r["answer"]))
    for (i, j, ans), t in zip(meta, run(jobs, 0.8, 900)):
        seen = set(); lines = [re.sub(r"^\s*\d+[.)]\s*", "", l).strip() for l in t.splitlines()]
        for l in lines:
            if len(l) < 20 or norm(l) in seen: continue
            seen.add(norm(l)); emit(i, "factview", l, norm(ans) in norm(l), fact=j)
    print(f"factview: {dict((k[1], v) for k, v in st.items() if k[0]=='factview')}", flush=True)
    # register rewrites
    for reg in a.registers.split(","):
        for i, t in enumerate(run([f"EXCERPT:\n{d}\n\n{REG[reg]}{REG_TAIL} Write the text only." for d in docs], 0.7, 1500)):
            t = re.sub(r"\*\*|__|`|^#+\s*", "", t, flags=re.M).strip()
            ok = len(t) > 200 and atoms_ok(t, srcn[i]) and not META.search(t)
            emit(i, "register", t, ok, register=reg)
        print(f"register {reg}: {dict((k[1], v) for k, v in st.items() if k[0]=='register')}", flush=True)
    # qa
    for i, t in enumerate(run([QA.format(doc=d) for d in docs], 0.5, 1400)):
        for r in jl(t):
            if all(k in r for k in ("q", "a", "why")): emit(i, "qa", json.dumps(r, ensure_ascii=False), norm(str(r["a"])) in srcn[i])
    # mcq
    for i, t in enumerate(run([MCQ.format(doc=d) for d in docs], 0.5, 1500)):
        for r in jl(t):
            if all(k in r for k in ("q", "options", "answer", "evidence")) and isinstance(r["options"], list) and 4 <= len(r["options"]) <= 5 \
                    and all(isinstance(x, str) and x.strip() for x in r["options"]) and isinstance(r["q"], str) and isinstance(r["evidence"], str) \
                    and isinstance(r["answer"], int) and 0 <= r["answer"] < len(r["options"]):
                ok = norm(r["evidence"])[:80] in srcn[i] and len({norm(x) for x in r["options"]}) == len(r["options"])
                emit(i, "mcq", json.dumps(r, ensure_ascii=False), ok)
    out.close()
    src = sum(len(d.split()) * 1.35 for d in docs)
    print("gate:", {f"{k[0]}:{k[1]}": v for k, v in sorted(st.items())}, flush=True)
    print("tokens kept per view:", dict(ntok), " amplification total:", round(sum(ntok.values()) / src, 2), flush=True)
    print("KIT_GEN_DONE", flush=True)
if __name__ == "__main__": main()
