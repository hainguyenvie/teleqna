#!/usr/bin/env python3
"""Contrastive MCQ view: for each window, MCQs whose correct option is supported by the TARGET window and whose
distractors are drawn from NEIGHBOUR windows (other windows retrieved for the same test question = the near-miss
passages the 8B confuses, cf. failure analysis). Generator Qwen3-8B, evidence gate (verbatim sentence in target),
rows written with view "mcq" (same fields as the kit mcq view) into --out/views_s{shard}.jsonl so pack_tier1
--mcq-gold and selfreplay pick them up unchanged. Question text only ever comes from the generator."""
import argparse, json, random, re, unicodedata, collections
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"; GEN = str(Path.home() / "projects/_shared/models/Qwen3-8B")
def norm(s): return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s or "").lower()).strip()
PROMPT = ("TARGET EXCERPT:\n{doc}\n\nNEIGHBOUR EXCERPTS (related passages retrieved for the same topic; use them ONLY to build distractors):\n{nb}\n\n"
          "Write 4 multiple-choice questions in the style of a telecom certification exam about the TARGET excerpt, each with 4 options and exactly one "
          "correct option. The correct option must be supported by the TARGET excerpt. Each distractor must be a statement, value or entity that comes "
          "from a NEIGHBOUR excerpt (true there, wrong for the TARGET) or a near-miss of the correct option. Do not use 'All of the above' or "
          "'None of the above'. Give the verbatim TARGET sentence that supports the correct option as \"evidence\". One JSON object per line, nothing else:\n"
          "{{\"q\": \"...\", \"options\": [\"...\", \"...\", \"...\", \"...\"], \"answer\": 0, \"evidence\": \"...\"}}")
ap = argparse.ArgumentParser(); ap.add_argument("--windows-file", default="data/kit/windows_keep.jsonl"); ap.add_argument("--pool", default="data/kit/windows_keep.jsonl,data/kit/windows_rest.jsonl")
ap.add_argument("--out", default="data/kit/tier1c"); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--gpu-mem", type=float, default=0.90); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--nnb", type=int, default=3)
ap.add_argument("--style", action="store_true", help="match the test question style: short direct questions (What is/are, What does, Which, How), no 'Which of the following'"); ap.add_argument("--sample", type=int, default=0); a = ap.parse_args()
rng = random.Random(a.seed); outdir = R / a.out; outdir.mkdir(parents=True, exist_ok=True); outp = outdir / f"views_s{a.shard}.jsonl"
done = set()
if outp.exists():
    for l in open(outp, encoding="utf-8"):
        try: done.add(json.loads(l)["win_id"])
        except Exception: pass
pool = {}; byq = collections.defaultdict(list)
for f in a.pool.split(","):
    for l in open(R / f, encoding="utf-8"):
        w = json.loads(l); pool[w["win_id"]] = w
        for q in w.get("for_q", []): byq[q].append(w["win_id"])
wins = [json.loads(l) for l in open(R / a.windows_file, encoding="utf-8")]
if a.sample: random.Random(321).shuffle(wins); wins = wins[:a.sample]
wins = [w for i, w in enumerate(wins) if i % a.nshards == a.shard and w["win_id"] not in done]
print(f"shard {a.shard}/{a.nshards}: {len(wins)} windows to do ({len(done)} done)", flush=True)
if not wins: print("CMCQ_DONE"); raise SystemExit
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(GEN); llm = LLM(model=GEN, tensor_parallel_size=1, gpu_memory_utilization=a.gpu_mem, max_model_len=8192, dtype="bfloat16", seed=a.seed)
def rend(p): return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
prompts, meta = [], []
for w in wins:
    nbs = [x for q in w.get("for_q", []) for x in byq.get(q, []) if x != w["win_id"]]
    nbs = list(dict.fromkeys(nbs)); rng.shuffle(nbs); nbs = nbs[:a.nnb]
    if not nbs: continue
    nb = "\n\n".join(f"[{i+1}] {pool[x]['text'][:1500]}" for i, x in enumerate(nbs))
    pr = PROMPT.format(doc=w["text"][:6000], nb=nb)
    if a.style: pr += ("\n\nQUESTION STYLE (important): write each question the way a telecom exam asks about a passage — short and direct (8-14 words), "
                       "starting with 'What is', 'What are', 'What does', 'Which', 'How', 'When' or 'Why'; name the entity/context in the question; "
                       "NEVER start with 'Which of the following'. Options: concise noun phrases or short statements (2-12 words).")
    prompts.append(pr); meta.append((w, nbs))
outs = [o.outputs[0].text.strip() for o in llm.generate([rend(p) for p in prompts], SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1200))]
st = collections.Counter()
with open(outp, "a", encoding="utf-8") as out:
    for (w, nbs), o in zip(meta, outs):
        srcn = norm(w["text"])
        for line in o.splitlines():
            line = line.strip()
            if not line.startswith("{"): continue
            try: r = json.loads(line)
            except Exception: st["badjson"] += 1; continue
            ok = all(k in r for k in ("q", "options", "answer", "evidence")) and isinstance(r["options"], list) and 4 <= len(r["options"]) <= 5 \
                 and all(isinstance(x, str) and x.strip() for x in r["options"]) and isinstance(r["q"], str) and isinstance(r["evidence"], str) \
                 and isinstance(r["answer"], int) and 0 <= r["answer"] < len(r["options"]) \
                 and norm(r["evidence"])[:80] in srcn and len({norm(x) for x in r["options"]}) == len(r["options"])
            st[ok] += 1
            out.write(json.dumps(dict(win_id=w["win_id"], view="mcq", text=json.dumps(r, ensure_ascii=False), ok=ok, contrastive=nbs), ensure_ascii=False) + "\n")
print("cmcq gate:", dict(st), flush=True); print("CMCQ_DONE")
