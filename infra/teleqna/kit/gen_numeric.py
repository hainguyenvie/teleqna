#!/usr/bin/env python3
"""Numeric-fact drill view: sentences of the windows that carry a number/unit/identifier (timers, bands, sizes, TS
numbers, hex codes) -> the 8B writes short Q/A pairs whose answer is the exact value; gate = the answer string occurs
verbatim in the sentence. Output: chat rows {prompt, completion} for pack_chat.py (completion = value, ended by
<|im_end|> at packing). Numeric gold rows are the weakest bucket (vd 63.5% vs 78.9% overall)."""
import json, re, random, argparse, collections, unicodedata
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"; GEN = str(Path.home() / "projects/_shared/models/Qwen3-8B")
NUM = re.compile(r"(\d+(?:[.,]\d+)?\s?(?:ms|µs|us|ns|MHz|GHz|kHz|Hz|dBm|dBi|dB|bits?|bytes?|octets?|%|seconds?|minutes?|hours?|km|cm|mm|m|W|mW|V|mA|Mbps|Gbps|kbps|bps|slots?|symbols?|subframes?|frames?|PRBs?|RBs?|CCEs?|°C|degrees?)\b|0x[0-9A-Fa-f]{2,}|\bTS\s?\d{2}\.\d{3}\b|\bRelease\s?\d{1,2}\b|\b\d{3,5}\s?(?:MHz|GHz)\b)")
def norm(s): return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s or "").lower()).strip()
ap = argparse.ArgumentParser(); ap.add_argument("--windows", default="data/kit/big/windows_all.jsonl"); ap.add_argument("--out", default="data/kit/num"); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--gpu-mem", type=float, default=0.90); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max-sent", type=int, default=6); a = ap.parse_args()
rng = random.Random(a.seed); outdir = R / a.out; outdir.mkdir(parents=True, exist_ok=True); outp = outdir / f"rows_s{a.shard}.jsonl"
PROMPT = ("Each line below is a sentence from a telecom document. For EACH sentence write 2 short exam questions whose answer is the exact "
          "numeric value, identifier or parameter mentioned (keep the unit; answer must be copied verbatim from the sentence). Questions must "
          "name the entity/context so they are answerable without the sentence. One JSON object per line, nothing else:\n"
          "{{\"i\": <sentence number>, \"q\": \"...\", \"a\": \"...\"}}\n\nSENTENCES:\n{sents}")
jobs = []
for k, l in enumerate(open(R / a.windows, encoding="utf-8")):
    if k % a.nshards != a.shard: continue
    w = json.loads(l); sents = [s.strip() for s in re.split(r"(?<=[.;])\s+", w["text"]) if 40 < len(s) < 400 and NUM.search(s)]
    if not sents: continue
    rng.shuffle(sents); sents = sents[:a.max_sent]; jobs.append((w["win_id"], sents))
print(f"shard {a.shard}/{a.nshards}: {len(jobs):,} windows with numeric sentences, {sum(len(s) for _, s in jobs):,} sentences", flush=True)
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(GEN); llm = LLM(model=GEN, tensor_parallel_size=1, gpu_memory_utilization=a.gpu_mem, max_model_len=4096, dtype="bfloat16", seed=a.seed)
prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(sents="\n".join(f"{i+1}. {s}" for i, s in enumerate(ss)))}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for _, ss in jobs]
outs = [o.outputs[0].text for o in llm.generate(prompts, SamplingParams(temperature=0.7, top_p=0.95, max_tokens=900))]
st = collections.Counter(); n = 0
with open(outp, "w", encoding="utf-8") as f:
    for (wid, ss), o in zip(jobs, outs):
        for line in o.splitlines():
            line = line.strip()
            if not line.startswith("{"): continue
            try: r = json.loads(line)
            except Exception: st["badjson"] += 1; continue
            i = r.get("i"); q = r.get("q"); ans = r.get("a")
            if not (isinstance(i, int) and 1 <= i <= len(ss) and isinstance(q, str) and isinstance(ans, str) and ans.strip()): st["bad"] += 1; continue
            ok = norm(ans) in norm(ss[i - 1]) and NUM.search(ans) is not None and len(q) > 20
            st[ok] += 1
            if ok: f.write(json.dumps(dict(prompt=q.strip(), completion=ans.strip(), win_id=wid), ensure_ascii=False) + "\n"); n += 1
print(f"numeric rows kept {n:,}; gate {dict(st)}"); print("NUM_DONE")
