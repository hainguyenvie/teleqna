#!/usr/bin/env python3
"""Cleaner held-out for the early probe: facts from windows of questions that have NO window in the kit
(no topical spill-over from kept windows of the same question). Facts written by Qwen3-8B with the same
prompt/gate as the kit. Then builds data/kit/tier1/probes_v2.json = train facts (pilot, kept windows) + these."""
import json, re, random, unicodedata, collections
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"; GEN = str(Path.home() / "projects/_shared/models/Qwen3-8B")
def norm(s): return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s or "").lower()).strip()
FACTS = ("EXCERPT:\n{doc}\n\nList EVERY atomic fact in the excerpt. For each fact give: \"fact\" — one standalone sentence with "
         "the subject named explicitly; \"answer\" — the short distinctive answer span (a value, identifier, term, name) copied "
         "VERBATIM from the excerpt; \"q\" — a question whose answer is exactly that span and which does not contain the span. "
         "Reply with one JSON object per line, nothing else:\n{{\"fact\": \"...\", \"answer\": \"...\", \"q\": \"...\"}}")
keep = {json.loads(l)["win_id"] for l in open(R / "data/kit/windows_keep.jsonl", encoding="utf-8")}
q_kept = set(); wins = []
for l in open(R / "data/eg2/windows.jsonl", encoding="utf-8"):
    w = json.loads(l); wins.append(w)
    if w["win_id"] in keep: q_kept.update(w["for_q"])
cand = [w for w in wins if w["win_id"] not in keep and not (set(w["for_q"]) & q_kept)]
random.Random(5).shuffle(cand); cand = cand[:500]
print(f"questions with a kept window: {len(q_kept)}; candidate clean windows {len(cand)}", flush=True)
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(GEN, local_files_only=True)
llm = LLM(model=GEN, gpu_memory_utilization=0.70, max_model_len=8192, dtype="bfloat16")
rend = lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
def D(t):
    ids = tok(t, add_special_tokens=False)["input_ids"]; return tok.decode(ids[:3000]) if len(ids) > 3000 else t
docs = [D(w["text"]) for w in cand]
outs = llm.generate([rend(FACTS.format(doc=d)) for d in docs], SamplingParams(temperature=0.3, top_p=0.95, max_tokens=1600))
probes = []
for w, d, o in zip(cand, docs, outs):
    srcn = norm(d)
    for line in o.outputs[0].text.splitlines():
        line = line.strip().strip(",")
        if not (line.startswith("{") and line.endswith("}")): continue
        try: f = json.loads(line)
        except Exception: continue
        if not all(k in f for k in ("fact", "answer", "q")) or not isinstance(f["answer"], str): continue
        span = f["answer"]
        if not (2 <= len(span) <= 80) or norm(span) not in srcn or norm(span) in norm(f["q"]): continue
        i = d.lower().find(span.lower()); j = f["fact"].lower().find(span.lower())
        if i < 0 or j < 10: continue
        st = max(d.rfind(". ", 0, i), d.rfind("\n", 0, i)) + 1; mem = d[st:i].strip()
        if len(mem) < 15: continue
        probes.append(dict(win_id=w["win_id"], group="heldout", span=span, mem=mem, sem=f["fact"][:j].strip(), qa=f"Question: {f['q']}\nAnswer:"))
random.Random(7).shuffle(probes)
old = json.load(open(R / "data/kit/tier1/probes.json")); train = [p for p in old if p["group"] == "train"]
n = min(1500, len(probes)); out = train[:n] + probes[:n]
json.dump(out, open(R / "data/kit/tier1/probes_v2.json", "w"), ensure_ascii=False)
print(f"probes_v2: train {n} heldout(clean) {n} (available {len(probes)})"); print("HELDOUT_V2_DONE", flush=True)
