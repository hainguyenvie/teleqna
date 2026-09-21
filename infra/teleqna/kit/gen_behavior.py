#!/usr/bin/env python3
"""Behaviour view for question categories (label-free, from windows only): per window the 8B writes
  (a) an MCQ whose correct option is "All of the above" (3 statements each supported by the excerpt),
  (b) an MCQ where "All of the above" is a distractor (1 supported statement + 2 near-miss false statements),
  (c) a NOT/EXCEPT MCQ (3 supported statements + 1 false; the false one is the answer).
Gate: every statement claimed as supported must have a verbatim evidence sentence in the excerpt. Rows are emitted as
chat rows {prompt, completion} (harness template; "All of the above" fixed as the last option; other options shuffled)."""
import json, re, random, argparse, collections, unicodedata
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"; GEN = str(Path.home() / "projects/_shared/models/Qwen3-8B")
def norm(s): return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s or "").lower()).strip()
TEMPLATE = ("Answer the following multiple choice question. The entire content of your response should be of the following "
            "format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.\n\n{question}\n\n{choices}")
PROMPT = ("EXCERPT:\n{doc}\n\nWrite three exam items about the EXCERPT as JSON objects, one per line, nothing else:\n"
          "1. {{\"kind\": \"all\", \"q\": \"...\", \"true\": [\"stmt1\", \"stmt2\", \"stmt3\"], \"evidence\": [\"verbatim excerpt sentence for stmt1\", \"...\", \"...\"]}}  "
          "— a question for which ALL THREE statements are correct answers according to the excerpt (the exam option will be 'All of the above').\n"
          "2. {{\"kind\": \"one\", \"q\": \"...\", \"true\": [\"stmt\"], \"false\": [\"plausible but wrong stmt\", \"another wrong stmt\"], \"evidence\": [\"verbatim sentence for the true stmt\"]}}  "
          "— exactly one correct statement; the wrong ones must be near-misses (neighbouring value, sibling entity, adjacent procedure) that the excerpt does NOT support.\n"
          "3. {{\"kind\": \"not\", \"q\": \"Which of the following is NOT ...?\", \"true\": [\"stmt1\", \"stmt2\", \"stmt3\"], \"false\": [\"the one wrong stmt\"], \"evidence\": [\"verbatim sentence for stmt1\", \"...\", \"...\"]}}  "
          "— three statements supported by the excerpt and one that is not; the wrong one is the answer.\n"
          "Statements must be self-contained (name the entity), 5-25 words, no option letters.")
ap = argparse.ArgumentParser(); ap.add_argument("--windows", default="data/kit/big/windows_all.jsonl"); ap.add_argument("--out", default="data/kit/beh"); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--sample", type=int, default=20000); ap.add_argument("--gpu-mem", type=float, default=0.90); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
rng = random.Random(a.seed); outdir = R / a.out; outdir.mkdir(parents=True, exist_ok=True); outp = outdir / f"rows_s{a.shard}.jsonl"
wins = [json.loads(l) for l in open(R / a.windows, encoding="utf-8")]; random.Random(123).shuffle(wins); wins = wins[:a.sample]
wins = [w for i, w in enumerate(wins) if i % a.nshards == a.shard]
print(f"shard {a.shard}/{a.nshards}: {len(wins):,} windows", flush=True)
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(GEN); llm = LLM(model=GEN, tensor_parallel_size=1, gpu_memory_utilization=a.gpu_mem, max_model_len=8192, dtype="bfloat16", seed=a.seed)
prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(doc=w["text"][:6000])}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for w in wins]
outs = [o.outputs[0].text for o in llm.generate(prompts, SamplingParams(temperature=0.7, top_p=0.95, max_tokens=900))]
def supported(stmts, evs, src): return len(evs) >= len(stmts) and all(isinstance(e, str) and len(norm(e)) >= 30 and norm(e)[:80] in src for e in evs[:len(stmts)])
def row(q, options, answer_idx):
    n = len(options); u = TEMPLATE.format(letters=",".join(chr(65 + i) for i in range(n)), question=q, choices="\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(options)))
    return json.dumps(dict(prompt=u, completion=f"ANSWER: {chr(65 + answer_idx)}"), ensure_ascii=False) + "\n"
st = collections.Counter(); n = 0
with open(outp, "w", encoding="utf-8") as f:
    for w, o in zip(wins, outs):
        src = norm(w["text"])
        for line in o.splitlines():
            line = line.strip()
            if not line.startswith("{"): continue
            try: r = json.loads(line)
            except Exception: st["badjson"] += 1; continue
            k = r.get("kind"); tr = r.get("true") or []; fa = r.get("false") or []; ev = r.get("evidence") or []; q = r.get("q")
            if not (isinstance(q, str) and len(q) > 15 and all(isinstance(x, str) and x.strip() for x in tr + fa)): st["bad"] += 1; continue
            if k == "all" and len(tr) == 3 and supported(tr, ev, src):
                opts = tr[:]; rng.shuffle(opts); opts.append("All of the above"); f.write(row(q, opts, len(opts) - 1)); st["all"] += 1; n += 1
            elif k == "one" and len(tr) == 1 and len(fa) >= 2 and supported(tr, ev, src) and all(norm(x)[:60] not in src for x in fa):
                opts = [tr[0]] + fa[:2]; rng.shuffle(opts); ans = opts.index(tr[0]); opts.append("All of the above"); f.write(row(q, opts, ans)); st["one"] += 1; n += 1
            elif k == "not" and len(tr) == 3 and len(fa) >= 1 and supported(tr, ev, src) and norm(fa[0])[:60] not in src and re.search(r"\bNOT\b|\bEXCEPT\b", q):
                opts = tr[:3] + [fa[0]]; rng.shuffle(opts); f.write(row(q, opts, opts.index(fa[0]))); st["not"] += 1; n += 1
            else: st["gated"] += 1
print(f"behaviour rows {n:,}; {dict(st)}"); print("BEH_DONE")
