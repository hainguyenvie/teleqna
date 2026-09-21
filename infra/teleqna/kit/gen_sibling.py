#!/usr/bin/env python3
"""Sibling-contrast view (binding fix): the never-correct core is confidently wrong on the *sibling* of the gold item that
sits in the same passage (EDGE-1 vs EDGE-3, state N8 vs N9, first- vs second-order Markov, 10 cm vs 1.5 m, queuing theory
vs network calculus). The 8B enumerates groups of confusable sibling items inside ONE window and writes questions whose
answer is one sibling and whose options are the other siblings, all copied verbatim from the window. Gate: every option
verbatim in the window, evidence sentence verbatim, >=3 options, distinct. Rows: chat MCQ (harness template, shuffled)."""
import json, re, random, argparse, collections, unicodedata
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"; GEN = str(Path.home() / "projects/_shared/models/Qwen3-8B")
def norm(s): return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s or "").lower()).strip()
TEMPLATE = ("Answer the following multiple choice question. The entire content of your response should be of the following "
            "format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.\n\n{question}\n\n{choices}")
PROMPT = ("PASSAGE:\n{doc}\n\nStep 1: list groups of SIBLING items that appear in this passage and are easy to confuse with each other — "
          "e.g. numbered states or procedures, interfaces/reference points (EDGE-1/EDGE-3), parameters and their values (10 cm vs 1.5 m), "
          "named methods or theories listed side by side, message names, timers, releases. Use the exact wording of the passage.\n"
          "Step 2: for each group write 1-2 short direct questions (What is / Which / What does / How many ...) whose correct answer is ONE sibling "
          "and whose other options are the OTHER siblings of the same group (3-5 options total, all copied verbatim from the passage), plus the "
          "verbatim passage sentence that decides it. The question must name the entity/context so it is answerable without the passage.\n"
          "Output one JSON object per line, nothing else:\n{{\"q\": \"...\", \"options\": [\"...\", \"...\", \"...\"], \"answer\": 0, \"evidence\": \"...\"}}")
ap = argparse.ArgumentParser(); ap.add_argument("--windows-file", default="data/kit/big/windows_all.jsonl"); ap.add_argument("--out", default="data/kit/sib"); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--sample", type=int, default=0); ap.add_argument("--gpu-mem", type=float, default=0.90); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
rng = random.Random(a.seed); outdir = R / a.out; outdir.mkdir(parents=True, exist_ok=True)
wins = [json.loads(l) for l in open(R / a.windows_file, encoding="utf-8")]
if a.sample: random.Random(4321).shuffle(wins); wins = wins[:a.sample]
wins = [w for i, w in enumerate(wins) if i % a.nshards == a.shard]; print(f"shard {a.shard}/{a.nshards}: {len(wins):,} windows", flush=True)
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(GEN); llm = LLM(model=GEN, tensor_parallel_size=1, gpu_memory_utilization=a.gpu_mem, max_model_len=8192, dtype="bfloat16", seed=a.seed)
prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(doc=w["text"][:6000])}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for w in wins]
outs = [o.outputs[0].text for o in llm.generate(prompts, SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1400))]
st = collections.Counter(); n = 0
with open(outdir / f"rows_s{a.shard}.jsonl", "w", encoding="utf-8") as f, open(outdir / f"views_s{a.shard}.jsonl", "w", encoding="utf-8") as fv:
    for w, o in zip(wins, outs):
        src = norm(w["text"])
        for line in o.splitlines():
            line = line.strip()
            if not line.startswith("{"): continue
            try: r = json.loads(line)
            except Exception: st["badjson"] += 1; continue
            q, opts, ans, ev = r.get("q"), r.get("options"), r.get("answer"), r.get("evidence")
            if not (isinstance(q, str) and isinstance(opts, list) and 3 <= len(opts) <= 5 and all(isinstance(x, str) and x.strip() for x in opts) and isinstance(ans, int) and 0 <= ans < len(opts) and isinstance(ev, str)): st["bad"] += 1; continue
            if len({norm(x) for x in opts}) != len(opts): st["dup-opt"] += 1; continue
            if not all(norm(x) in src for x in opts): st["opt-not-verbatim"] += 1; continue
            if norm(ev)[:80] not in src: st["ev-not-verbatim"] += 1; continue
            st["kept"] += 1; n += 1
            fv.write(json.dumps(dict(win_id=w["win_id"], view="mcq", text=json.dumps(dict(q=q, options=opts, answer=ans, evidence=ev), ensure_ascii=False), ok=True, sibling=True), ensure_ascii=False) + "\n")
            for s in rng.sample(range(len(opts)), min(2, len(opts))):
                ch = [opts[(i + s) % len(opts)] for i in range(len(opts))]; letter = chr(65 + (ans - s) % len(opts))
                u = TEMPLATE.format(letters=",".join(chr(65 + i) for i in range(len(opts))), question=q.strip(), choices="\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(ch)))
                f.write(json.dumps(dict(prompt=u, completion=f"ANSWER: {letter}"), ensure_ascii=False) + "\n")
print(f"sibling MCQs kept {n:,}; gate {dict(st)}"); print("SIB_DONE")
