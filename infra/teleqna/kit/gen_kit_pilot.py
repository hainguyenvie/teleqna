#!/usr/bin/env python3
"""Study-kit pilot: generate six views per strong-RAG window, to measure which views keep the
knowledge (PLAN_CLOSED_BOOK_8B §5.2b). Unit = window (~1.2k tokens), generator = OTel-2.0-31B-IT.

Sample: N TARGET questions of run 3, ALL 8 of their strong-RAG windows (so the reference ceiling
is the 8-window RAG score on the same questions). Answer key never enters any prompt.
Outputs data/kit/pilot_views.jsonl: {win_id, view, text, ok} — `ok` is the string gate result.
"""
import json, re, random, argparse, unicodedata, collections, itertools
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
GEN_DEFAULT = str(Path.home() / "projects/_shared/models/OTel-2.0-31B-IT")
def norm(s): return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s or "").lower()).strip()

V = {}
V["textbook"] = ("Below is an excerpt from a telecom standard or research paper.\n\nEXCERPT:\n{doc}\n\n"
    "Rewrite the excerpt as a section of an engineering textbook for someone who has never read it: full "
    "prose, well organised, self-contained. HARD REQUIREMENTS: keep EVERY concrete value, identifier, "
    "parameter name, message/procedure name, condition and reference exactly as in the excerpt; add "
    "nothing that is not in the excerpt; no preamble, no markdown headings. Write the text only.", 0.7, 1400)
V["notes"] = ("EXCERPT:\n{doc}\n\n"
    "Write terse study notes an engineer would keep from this excerpt: one line per distinct fact, each "
    "line a complete standalone sentence that names its subject explicitly (no pronouns, no 'it') and "
    "states the exact value / identifier / condition from the excerpt. Cover every fact in the excerpt. "
    "No preamble, no bullets or numbering, one sentence per line.", 0.5, 1200)
V["facts"] = ("EXCERPT:\n{doc}\n\n"
    "List EVERY atomic fact in the excerpt. For each fact give: \"fact\" — one standalone sentence with the "
    "subject named explicitly; \"answer\" — the short distinctive answer span (a value, identifier, term, "
    "name) copied VERBATIM from the excerpt; \"q\" — a question whose answer is exactly that span and "
    "which does not contain the span. Reply with one JSON object per line, nothing else:\n"
    "{{\"fact\": \"...\", \"answer\": \"...\", \"q\": \"...\"}}", 0.3, 1600)
V["qa"] = ("EXCERPT:\n{doc}\n\n"
    "Write 12 diverse exam questions about this excerpt, spanning definitions, values, conditions, "
    "procedures and relationships. Each answer must be short (<= 12 words) and must appear VERBATIM in "
    "the excerpt. Also give a one-sentence explanation. One JSON object per line, nothing else:\n"
    "{{\"q\": \"...\", \"a\": \"...\", \"why\": \"...\"}}", 0.5, 1400)
V["mcq"] = ("EXCERPT:\n{doc}\n\n"
    "Write 4 multiple-choice questions in the style of a telecom certification exam, each with 4 or 5 "
    "options and exactly one correct option. Distractors must be plausible near-misses (a neighbouring "
    "value, a sibling procedure, an adjacent release, a related but different entity), not obviously "
    "wrong. Do not use 'All of the above' or 'None of the above'. Give the verbatim excerpt sentence that "
    "supports the correct option as \"evidence\". One JSON object per line, nothing else:\n"
    "{{\"q\": \"...\", \"options\": [\"...\", \"...\", \"...\", \"...\"], \"answer\": 0, \"evidence\": \"...\"}}", 0.5, 1500)
DENSE = ("PASSAGE:\n{doc}\n\nRewrite this passage in different words as one standalone explanatory paragraph for "
         "an engineer who has not read it. HARD REQUIREMENTS: keep EVERY value, identifier, parameter, message/"
         "procedure name, condition, exception and list item from the passage; do not summarise, shorten or omit "
         "anything; name subjects explicitly instead of pronouns; add nothing that is not in the passage; no "
         "preamble, no markdown. Write the paragraph only.")
def segments(text, tok, size=260):
    ids = tok(text, add_special_tokens=False)["input_ids"]; out = []; i = 0
    while i < len(ids):
        j = min(len(ids), i + size)
        if j < len(ids):
            piece = tok.decode(ids[i:j]); k = max(piece.rfind(". "), piece.rfind(".\n"))
            if k > len(piece) * 0.5: j = i + len(tok(piece[:k + 1], add_special_tokens=False)["input_ids"])
        out.append(tok.decode(ids[i:j])); i = j
    return out
ENT = ("EXCERPT:\n{doc}\n\nList the salient entities of this excerpt: procedures, messages, information "
       "elements, parameters and their values, network functions, interfaces, referenced specs. Reply with "
       "ONLY a JSON list of 8 to 20 short strings.")
REL = ("EXCERPT:\n{doc}\n\nExplain precisely how {e1} and {e2} relate according to the excerpt: what one "
       "requires, constrains, carries or triggers in the other. 3-6 factual sentences, concrete values and "
       "identifiers as given. No preamble.")

def jl(text):
    out = []
    for line in text.splitlines():
        line = line.strip().strip(",")
        if line.startswith("{") and line.endswith("}"):
            try: out.append(json.loads(line))
            except Exception: pass
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=300); ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--rel-pairs", type=int, default=6)
    ap.add_argument("--gen", default=GEN_DEFAULT); ap.add_argument("--views", default="all", help="comma list or all")
    ap.add_argument("--tag", default="pilot"); a = ap.parse_args(); GEN = a.gen
    VIEWS = list(V) + ["relations"] if a.views == "all" else a.views.split(",")
    G = json.load(open(R / "data/eg3/groups.json"))
    rng = random.Random(a.seed); qs = sorted(G["target"]); rng.shuffle(qs); qs = qs[:a.n]
    qset = set(qs)
    wins = []
    for l in open(R / "data/eg2/windows.jsonl", encoding="utf-8"):
        w = json.loads(l)
        if qset & set(w["for_q"]): wins.append(w)
    (R / "data/kit").mkdir(exist_ok=True)
    json.dump({"questions": qs, "windows": [w["win_id"] for w in wins]}, open(R / f"data/kit/{a.tag}_sample.json", "w"))
    print(f"pilot: {len(qs)} questions, {len(wins)} windows, ~{sum(len(w['text'].split()) for w in wins)*1.35/1e6:.2f}M source tokens", flush=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(GEN, local_files_only=True)
    llm = LLM(model=GEN, tensor_parallel_size=1, gpu_memory_utilization=0.90, max_model_len=8192, dtype="bfloat16")
    _kw = {"enable_thinking": False} if "qwen" in GEN.lower() else {}
    rend = lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, **_kw)
    def D(t):
        ids = tok(t, add_special_tokens=False)["input_ids"]
        return tok.decode(ids[:3000]) if len(ids) > 3000 else t
    docs = [D(w["text"]) for w in wins]; srcn = [norm(d) for d in docs]
    out = open(R / f"data/kit/{a.tag}_views.jsonl", "w", encoding="utf-8"); st = collections.Counter(); ntok = collections.Counter()
    def emit(i, view, text, ok=True):
        st[view, ok] += 1; ntok[view] += int(len(text.split()) * 1.35)
        out.write(json.dumps(dict(win_id=wins[i]["win_id"], view=view, text=text, ok=ok), ensure_ascii=False) + "\n")

    for view, (tpl, temp, mx) in V.items():
        if view not in VIEWS: continue
        outs = llm.generate([rend(tpl.format(doc=d)) for d in docs], SamplingParams(temperature=temp, top_p=0.95, max_tokens=mx))
        for i, o in enumerate(outs):
            t = o.outputs[0].text.strip()
            if view in ("textbook", "notes"):
                emit(i, view, t, len(t) > 200)
            elif view == "facts":
                for r in jl(t):
                    if all(k in r for k in ("fact", "answer", "q")):
                        ok = norm(r["answer"]) in srcn[i] and norm(r["answer"]) not in norm(r["q"])
                        emit(i, "facts", json.dumps(r, ensure_ascii=False), ok)
            elif view == "qa":
                for r in jl(t):
                    if all(k in r for k in ("q", "a", "why")):
                        emit(i, "qa", json.dumps(r, ensure_ascii=False), norm(r["a"]) in srcn[i])
            elif view == "mcq":
                for r in jl(t):
                    if all(k in r for k in ("q", "options", "answer", "evidence")) and isinstance(r["options"], list) \
                            and 4 <= len(r["options"]) <= 5 and isinstance(r["answer"], int) and 0 <= r["answer"] < len(r["options"]):
                        ok = norm(r["evidence"])[:80] in srcn[i] and len({norm(x) for x in r["options"]}) == len(r["options"])
                        emit(i, "mcq", json.dumps(r, ensure_ascii=False), ok)
        print(f"{view}: {dict((k[1], v) for k, v in st.items() if k[0] == view)}  ~{ntok[view]:,} tok", flush=True)

    if "dense" in VIEWS:
        ATOM = re.compile(r"\b(?:TS\s?\d+\.\d+|TR\s?\d+\.\d+|Rel-?\d+|\d+(?:\.\d+)?\s?(?:ms|s|dB|dBm|GHz|MHz|kHz|Mbps|Gbps|bit|bits|bytes|%)|[A-Z]{3,7}|\d{2,4})\b")
        jobs, meta = [], []
        for i, d in enumerate(docs):
            for sg in segments(d, tok):
                if len(sg.split()) < 15: continue
                jobs.append(rend(DENSE.format(doc=sg))); meta.append((i, sg))
        print(f"dense: {len(jobs):,} segments ({len(jobs)/max(len(docs),1):.1f}/window)", flush=True)
        so = llm.generate(jobs, SamplingParams(temperature=0.7, top_p=0.95, max_tokens=520))
        per = collections.defaultdict(list); bad = 0
        for (i, sg), o in zip(meta, so):
            r = o.outputs[0].text.strip(); atoms = {x for x in ATOM.findall(sg) if len(x) >= 2}
            cov = sum(1 for x in atoms if norm(x) in norm(r)) / len(atoms) if atoms else 1.0
            if cov < 0.8 or len(r) < 60: bad += 1; per[i].append(sg)      # fall back to the source segment
            else: per[i].append(r)
        for i in per: emit(i, "dense", "\n".join(per[i]), True)
        print(f"dense: coverage-failed segments {bad:,}/{len(jobs):,} (source kept)  ~{ntok['dense']:,} tok", flush=True)
    # relations (EntiGraph control)
    eo = [] if "relations" not in VIEWS else llm.generate([rend(ENT.format(doc=d)) for d in docs], SamplingParams(temperature=0.2, max_tokens=400))
    jobs, meta = [], []
    for i, o in enumerate(eo):
        m = re.search(r"\[.*?\]", o.outputs[0].text, re.S); e = []
        if m:
            try: e = [str(x).strip() for x in json.loads(m.group(0)) if 2 < len(str(x).strip()) < 70]
            except Exception: e = []
        pr = list(itertools.combinations(list(dict.fromkeys(e))[:20], 2)); random.Random(i).shuffle(pr)
        for e1, e2 in pr[:a.rel_pairs]: jobs.append(rend(REL.format(doc=docs[i], e1=e1, e2=e2))); meta.append(i)
    ro = [] if not jobs else llm.generate(jobs, SamplingParams(temperature=0.8, top_p=0.95, max_tokens=320))
    for i, o in zip(meta, ro): emit(i, "relations", o.outputs[0].text.strip(), len(o.outputs[0].text) > 150)
    print(f"relations: {dict((k[1], v) for k, v in st.items() if k[0] == 'relations')}  ~{ntok['relations']:,} tok", flush=True)
    out.close()
    src = sum(len(d.split()) * 1.35 for d in docs)
    print("amplification per view:", {v: round(ntok[v] / src, 2) for v in ntok}, flush=True)
    print("KIT_PILOT_DONE", flush=True)
if __name__ == "__main__": main()
