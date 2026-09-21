#!/usr/bin/env python3
"""High-exposure targeted study (HET): many more DIVERSE self-contained recall views per question-relevant window
(windows_keep + windows_rest = the RAG/trace-back windows of the test questions; selecting source documents by the test
questions is allowed, the answer key is never used). Styles: direct (QA), cloze (fill-in-the-blank statements),
reverse (value/property -> entity, 'which spec/standard/entity'), numeric (numbers, units, ranges, counts). Same gates as
gen_recall_qa: answer verbatim in window, question free of context words, >=2 content words shared with the window.
Rows {prompt, completion, win_id, style}; 5% held out per window for the recall probe."""
import json, re, random, argparse, collections, unicodedata
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"; GEN = str(Path.home() / "projects/_shared/models/Qwen3-8B")
def norm(s): return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s or "").lower()).strip()
CTX = re.compile(r"\b(the|this) (paper|excerpt|document|study|authors?|section|passage|text|article|work|proposed (scheme|method|approach|framework))\b|\bin this\b|\bproposed\b|\bwe\b", re.I)
COMMON = ("Rules:\n- Every item must be SELF-CONTAINED: name the specific entity, standard, procedure, parameter or acronym it is about, so a reader who has not seen the passage knows what is asked. "
          "Never refer to 'the paper', 'the excerpt', 'this document', 'the authors', 'the proposed scheme'.\n- The answer is a short phrase (1-8 words) copied verbatim from the passage.\n"
          "- Cover DIFFERENT facts; do not repeat a fact.\nOne JSON object per line, nothing else: {{\"q\": \"...\", \"a\": \"...\"}}")
STYLES = {
 "direct": "PASSAGE:\n{doc}\n\nWrite {k} exam-style question/answer pairs that test the facts in the PASSAGE. Vary the phrasing ('What is', 'Which', 'How many', 'What does', 'When', 'Where', 'Who'). " + COMMON,
 "cloze": "PASSAGE:\n{doc}\n\nWrite {k} fill-in-the-blank items from the PASSAGE. Each 'q' is a complete factual sentence with ONE key term, value or name replaced by ____, rewritten so it is self-contained (name the entity/standard it is about); 'a' is the removed span, verbatim. " + COMMON,
 "reverse": "PASSAGE:\n{doc}\n\nWrite {k} reverse-direction question/answer pairs from the PASSAGE: give a property, value, role, definition or behaviour and ask WHICH entity / standard / procedure / parameter / message / field / algorithm has it ('Which 3GPP message carries ...?', 'Which parameter controls ...?', 'What is the name of the entity that ...?'). " + COMMON,
 "numeric": "PASSAGE:\n{doc}\n\nWrite {k} question/answer pairs from the PASSAGE whose answers are NUMBERS, ranges, units, counts, sizes, durations, frequencies, versions or release numbers (e.g. 'How many ...', 'What is the maximum ...', 'What is the value of ...', 'In which release ...'). If the passage has fewer numeric facts, ask about the ones it has and stop. " + COMMON,
}
ap = argparse.ArgumentParser(); ap.add_argument("--windows-file", default="data/kit/het/windows.jsonl"); ap.add_argument("--out", default="data/kit/het"); ap.add_argument("--style", default="direct")
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1); ap.add_argument("--k", type=int, default=20)
ap.add_argument("--gpu-mem", type=float, default=0.90); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--holdout", type=float, default=0.05); a = ap.parse_args()
rng = random.Random(a.seed * 100 + a.shard); outdir = R / a.out; outdir.mkdir(parents=True, exist_ok=True)
wins = [json.loads(l) for l in open(R / a.windows_file, encoding="utf-8")]; wins = [w for i, w in enumerate(wins) if i % a.nshards == a.shard]
if a.style == "numeric": wins = [w for w in wins if len(re.findall(r"\d", w["text"])) >= 8]
print(f"{a.style} shard {a.shard}/{a.nshards}: {len(wins):,} windows", flush=True)
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(GEN); llm = LLM(model=GEN, tensor_parallel_size=1, gpu_memory_utilization=a.gpu_mem, max_model_len=8192, dtype="bfloat16", seed=a.seed * 100 + a.shard)
prompts = [tok.apply_chat_template([{"role": "user", "content": STYLES[a.style].format(doc=w["text"][:6000], k=a.k)}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for w in wins]
outs = [o.outputs[0].text for o in llm.generate(prompts, SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1600, seed=a.seed * 100 + a.shard))]
st = collections.Counter(); ntr = nho = 0
with open(outdir / f"rows_{a.style}_s{a.shard}.jsonl", "w", encoding="utf-8") as ftr, open(outdir / f"holdout_{a.style}_s{a.shard}.jsonl", "w", encoding="utf-8") as fho:
    for w, o in zip(wins, outs):
        src = norm(w["text"]); words = set(t for t in re.findall(r"[a-z0-9][a-z0-9\-\.]{3,}", src)); rows = []; seen = set()
        for line in o.splitlines():
            line = line.strip()
            if not line.startswith("{"): continue
            try: r = json.loads(line)
            except Exception: st["badjson"] += 1; continue
            q, ans = r.get("q"), r.get("a")
            if not (isinstance(q, str) and isinstance(ans, str) and 3 <= len(q.split()) <= 45 and 1 <= len(ans.split()) <= 10): st["bad"] += 1; continue
            if CTX.search(q): st["context-dep"] += 1; continue
            qa = norm(ans)
            if qa not in src: st["ans-not-verbatim"] += 1; continue
            if a.style == "cloze" and "____" not in q: st["no-blank"] += 1; continue
            if a.style == "numeric" and not re.search(r"\d", ans): st["not-numeric"] += 1; continue
            if len([t for t in re.findall(r"[a-z0-9][a-z0-9\-\.]{3,}", norm(q)) if t in words]) < 2: st["no-entity"] += 1; continue
            if qa in seen: st["dup-answer"] += 1; continue
            seen.add(qa); p = ("Fill in the blank: " + q.strip()) if a.style == "cloze" else q.strip()
            rows.append(dict(prompt=p, completion=ans.strip(), win_id=w["win_id"], style=a.style))
        rng.shuffle(rows); nh = int(len(rows) * a.holdout)
        for r in rows[:nh]: fho.write(json.dumps(r, ensure_ascii=False) + "\n"); nho += 1
        for r in rows[nh:]: ftr.write(json.dumps(r, ensure_ascii=False) + "\n"); ntr += 1
        st["kept"] += len(rows)
print(f"{a.style} shard {a.shard}: train rows {ntr:,} holdout rows {nho:,}; gate {dict(st)}"); print("HET_GEN_DONE")
