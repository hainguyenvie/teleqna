#!/usr/bin/env python3
"""Recall-QA view (hypothesis B: teach *recall*, not just recognition). For each window the 8B writes ~20 SELF-CONTAINED
QA pairs: the question names the entity / standard / parameter so it is answerable without the passage; answer is a
1-8 word span copied from the passage; ~25% asked in the reverse direction (value -> entity). Gates: answer verbatim in
window; question free of context words ("the paper/excerpt/document/study/authors/this section"); question shares >=2
content words (>=4 chars) with the window. Output chat rows {prompt, completion, win_id}; 10% of rows per window are
written to a held-out file for the recall probe (never trained)."""
import json, re, random, argparse, collections, unicodedata
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"; GEN = str(Path.home() / "projects/_shared/models/Qwen3-8B")
def norm(s): return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s or "").lower()).strip()
CTX = re.compile(r"\b(the|this) (paper|excerpt|document|study|authors?|section|passage|text|article|work|proposed (scheme|method|approach|framework))\b|\bin this\b|\bproposed\b|\bwe\b", re.I)
PROMPT = ("PASSAGE:\n{doc}\n\nWrite {k} exam-style question/answer pairs that test the facts in the PASSAGE. Rules:\n"
          "- Each question must be SELF-CONTAINED: name the specific entity, standard, procedure, parameter or acronym it is about, so a reader who has not seen the passage still knows what is asked. "
          "Never refer to 'the paper', 'the excerpt', 'this document', 'the authors', 'the proposed scheme'.\n"
          "- The answer is a short phrase (1-8 words) copied verbatim from the passage.\n"
          "- Cover different facts; about a quarter of the questions should ask in the reverse direction (given the value/property, ask which entity has it).\n"
          "- Vary the phrasing: 'What is ...', 'What does ... ', 'Which ... ', 'How many ...', 'When ...', 'What are ...'.\n"
          "One JSON object per line, nothing else: {{\"q\": \"...\", \"a\": \"...\"}}")
ap = argparse.ArgumentParser(); ap.add_argument("--windows-file", default="data/kit/windows_keep.jsonl"); ap.add_argument("--out", default="data/kit/recall")
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1); ap.add_argument("--k", type=int, default=20)
ap.add_argument("--gpu-mem", type=float, default=0.90); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--holdout", type=float, default=0.1); a = ap.parse_args()
rng = random.Random(a.seed); outdir = R / a.out; outdir.mkdir(parents=True, exist_ok=True)
wins = [json.loads(l) for l in open(R / a.windows_file, encoding="utf-8")]; wins = [w for i, w in enumerate(wins) if i % a.nshards == a.shard]
print(f"shard {a.shard}/{a.nshards}: {len(wins):,} windows", flush=True)
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(GEN); llm = LLM(model=GEN, tensor_parallel_size=1, gpu_memory_utilization=a.gpu_mem, max_model_len=8192, dtype="bfloat16", seed=a.seed)
prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(doc=w["text"][:6000], k=a.k)}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for w in wins]
outs = [o.outputs[0].text for o in llm.generate(prompts, SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1600))]
st = collections.Counter(); ntr = nho = 0
with open(outdir / f"rows_s{a.shard}.jsonl", "w", encoding="utf-8") as ftr, open(outdir / f"holdout_s{a.shard}.jsonl", "w", encoding="utf-8") as fho:
    for w, o in zip(wins, outs):
        src = norm(w["text"]); words = set(t for t in re.findall(r"[a-z0-9][a-z0-9\-\.]{3,}", src)); rows = []
        for line in o.splitlines():
            line = line.strip()
            if not line.startswith("{"): continue
            try: r = json.loads(line)
            except Exception: st["badjson"] += 1; continue
            q, ans = r.get("q"), r.get("a")
            if not (isinstance(q, str) and isinstance(ans, str) and 3 <= len(q.split()) <= 40 and 1 <= len(ans.split()) <= 10): st["bad"] += 1; continue
            if CTX.search(q): st["context-dep"] += 1; continue
            qa = norm(ans)
            if qa not in src: st["ans-not-verbatim"] += 1; continue
            if len([t for t in re.findall(r"[a-z0-9][a-z0-9\-\.]{3,}", norm(q)) if t in words]) < 2: st["no-entity"] += 1; continue
            rows.append(dict(prompt=q.strip(), completion=ans.strip(), win_id=w["win_id"]))
        rng.shuffle(rows); nh = int(len(rows) * a.holdout)
        for r in rows[:nh]: fho.write(json.dumps(r, ensure_ascii=False) + "\n"); nho += 1
        for r in rows[nh:]: ftr.write(json.dumps(r, ensure_ascii=False) + "\n"); ntr += 1
        st["kept"] += len(rows)
print(f"train rows {ntr:,} holdout rows {nho:,}; gate {dict(st)}"); print("RECALL_GEN_DONE")
