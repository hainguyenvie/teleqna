#!/usr/bin/env python3
"""Vote distillation (label-free consistency stage): take gated synthetic MCQs from the kit views (tier-1 mcq, contrastive
mcq, tiers 2-5 mcq), let the checkpoint answer each under R choice rotations x K samples, majority-vote in original index
space, keep questions with vote share >= --min-share, and emit chat rows for every rotation whose target is the voted
answer in that rotation's lettering, terminated with <|im_end|> (loss on the letter AND on stopping -> fixes the tail
drift). The generator's own answer is never used; the label is the model's self-consistent answer."""
import json, re, glob, random, argparse, collections
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
TEMPLATE = ("Answer the following multiple choice question. The entire content of your response should be of the following "
            "format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.\n\n{question}\n\n{choices}")
LOOSE = re.compile(r"(?i)ANSWER\s*:\s*([A-E])")
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--n", type=int, default=60000); ap.add_argument("--rot", type=int, default=4)
ap.add_argument("--k", type=int, default=4); ap.add_argument("--min-share", type=float, default=0.6); ap.add_argument("--out", default="data/kit/vd/labels.jsonl")
ap.add_argument("--gpu-mem", type=float, default=0.85); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--dry", action="store_true")
ap.add_argument("--ckpt2", default="", help="extra checkpoint(s), comma-separated, whose votes are pooled with the first (ensemble distillation across lineages)")
ap.add_argument("--overlap-frac", type=float, default=0.0, help="if >0, cap the share of questions whose voted answer is the option with the most question-word overlap (lexical shortcut control; test prior ~0.39 among rows with a clear max)")
ap.add_argument("--longest-frac", type=float, default=0.0, help="if >0, subsample questions whose voted answer is the longest option down to this fraction (length-bias control; test prior 0.33)")
a = ap.parse_args()
rng = random.Random(a.seed)
files = sorted(glob.glob(str(R / "data/kit/tier1/views_s*.jsonl")) + glob.glob(str(R / "data/kit/tier1c/views_s*.jsonl")) + glob.glob(str(R / "data/kit/tier[2345]/views_s*.jsonl")))
mcqs = []
for f in files:
    for l in open(f, encoding="utf-8"):
        if '"view": "mcq"' not in l or '"ok": true' not in l: continue
        r = json.loads(l)
        try: m = json.loads(r["text"])
        except Exception: continue
        if isinstance(m, dict) and isinstance(m.get("options"), list) and 4 <= len(m["options"]) <= 5 and isinstance(m.get("q"), str): mcqs.append(dict(q=m["q"], choices=m["options"], src=f.split("/")[-2]))
print(f"gated mcq rows {len(mcqs):,} from {len(files)} files; by tier {collections.Counter(x['src'] for x in mcqs)}", flush=True)
rng.shuffle(mcqs); mcqs = mcqs[:a.n]
def user(r, shift):
    n = len(r["choices"]); ch = [r["choices"][(i + shift) % n] for i in range(n)]
    return TEMPLATE.format(letters=",".join(chr(65 + i) for i in range(n)), question=r["q"], choices="\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(ch)))
votes = [collections.Counter() for _ in mcqs]
if a.dry:
    for i, r in enumerate(mcqs): votes[i][rng.randrange(len(r["choices"]))] += a.rot * a.k
else:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    import gc, torch
    for ck in [a.ckpt] + ([c for c in a.ckpt2.split(",") if c] if a.ckpt2 else []):
        tok = AutoTokenizer.from_pretrained(ck, local_files_only=True); llm = LLM(model=ck, gpu_memory_utilization=a.gpu_mem, max_model_len=2048, dtype="bfloat16")
        prompts, meta = [], []
        for i, r in enumerate(mcqs):
            for s in range(min(a.rot, len(r["choices"]))):
                prompts.append(tok.apply_chat_template([{"role": "user", "content": user(r, s)}], tokenize=False, add_generation_prompt=True, enable_thinking=False)); meta.append((i, s))
        outs = llm.generate(prompts, SamplingParams(temperature=0.7, top_p=0.95, max_tokens=8, n=a.k))
        for (i, s), o in zip(meta, outs):
            n = len(mcqs[i]["choices"])
            for out in o.outputs:
                m = LOOSE.findall(out.text)
                if m:
                    g = ord(m[-1].upper()) - 65
                    if g < n: votes[i][(g + s) % n] += 1
        print(f"voted with {ck}", flush=True); del llm; gc.collect(); torch.cuda.empty_cache()
kept = 0; st = collections.Counter()
Path(R / a.out).parent.mkdir(parents=True, exist_ok=True)
keep_longest = None
if a.longest_frac > 0:   # length-bias control: cap the share of questions whose voted answer is the longest option
    L = [i for i, r in enumerate(mcqs) if sum(votes[i].values()) and votes[i].most_common(1)[0][0] == max(range(len(r["choices"])), key=lambda j: len(r["choices"][j]))]
    NL = [i for i, r in enumerate(mcqs) if sum(votes[i].values()) and i not in set(L)]
    target = int(a.longest_frac / (1 - a.longest_frac) * len(NL)); rng.shuffle(L); keep_longest = set(L[:target]); print(f"longest-answer questions {len(L):,} -> keep {len(keep_longest):,} (others {len(NL):,})", flush=True)
STOP = set("the a an of in to and or for is are what which does do by on with as at from that this it be can its their".split())
def toks(s): return set(t for t in re.sub(r"[^a-z0-9 ]", " ", s.lower()).split() if t not in STOP and len(t) > 2)
def maxov(r):
    qt = toks(r["q"]); ov = [len(qt & toks(o)) for o in r["choices"]]; mx = max(ov)
    return None if mx == 0 or ov.count(mx) > 1 else ov.index(mx)
keep_ov = None
if a.overlap_frac > 0:
    O = [i for i, r in enumerate(mcqs) if sum(votes[i].values()) and maxov(r) is not None and votes[i].most_common(1)[0][0] == maxov(r)]
    NO = [i for i, r in enumerate(mcqs) if sum(votes[i].values()) and i not in set(O)]
    target = int(a.overlap_frac / (1 - a.overlap_frac) * len(NO)); rng.shuffle(O); keep_ov = set(O[:target]); print(f"max-overlap-answer questions {len(O):,} -> keep {len(keep_ov):,} (others {len(NO):,})", flush=True)
with open(R / a.out, "w", encoding="utf-8") as f:
    for i, r in enumerate(mcqs):
        v = votes[i]; tot = sum(v.values())
        if not tot: st["novote"] += 1; continue
        ans, c = v.most_common(1)[0]; share = c / tot
        if share < a.min_share: st["lowshare"] += 1; continue
        if keep_longest is not None and ans == max(range(len(r["choices"])), key=lambda j: len(r["choices"][j])) and i not in keep_longest: st["longest-capped"] += 1; continue
        if keep_ov is not None and maxov(r) == ans and i not in keep_ov: st["overlap-capped"] += 1; continue
        kept += 1; n = len(r["choices"])
        for s in range(min(a.rot, n)):
            letter = chr(65 + (ans - s) % n)   # displayed position of original index `ans` under shift s
            f.write(json.dumps(dict(prompt=user(r, s), completion=f"ANSWER: {letter}", share=share, src=r["src"]), ensure_ascii=False) + "\n")
print(f"kept {kept:,}/{len(mcqs):,} questions (share>={a.min_share}), rows {kept*a.rot:,}, dropped {dict(st)}"); print("VD_DONE")
