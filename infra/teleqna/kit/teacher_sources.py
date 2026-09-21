#!/usr/bin/env python3
"""Teacher-written sources for the functionally uncovered questions (last resort, explicitly allowed by the project owner):
Qwen3.5-122B via the Viettel gateway writes 3 independent reference passages per question from the QUESTION + OPTIONS only
(never the answer key), then — label-free gate — answers the MCQ from each passage alone; a question is kept only when
all 3 passages imply the same option (consensus) and the closed-book teacher answer agrees. Output windows go through the
same 8B functional check as every other source (gold used for measurement only). Needs $GATEWAY_KEY in the environment."""
import json, os, re, sys, time, argparse, collections, concurrent.futures, urllib.request, urllib.error
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
URL = os.environ.get("GATEWAY_URL", "https://stream-netmind.viettel.vn/aigw/ai/v1/chat/completions"); KEY = os.environ.get("GATEWAY_KEY", "")
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="Qwen/Qwen3.5-122B-A10B-FP8"); ap.add_argument("--n", type=int, default=3); ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--questions", default="data/kit/uncovered_functional.jsonl"); ap.add_argument("--out", default="data/kit/teacher"); a = ap.parse_args()
if not KEY: sys.exit("GATEWAY_KEY not set")
outdir = R / a.out; outdir.mkdir(parents=True, exist_ok=True)
test = {r["sample_id"]: r for r in map(json.loads, open(R / "data/eval/otfull10000.jsonl", encoding="utf-8"))}
qs = [json.loads(l)["sample_id"] for l in open(R / a.questions, encoding="utf-8")]
def chat(prompt, temperature, max_tokens=700):
    payload = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature, "max_tokens": max_tokens}
    last = ""
    for attempt in range(1, 6):
        try:
            req = urllib.request.Request(URL, data=json.dumps(payload).encode(), method="POST", headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
            with urllib.request.urlopen(req, timeout=600) as r: raw = json.load(r)
            return raw["choices"][0]["message"].get("content") or ""
        except urllib.error.HTTPError as e: last = f"HTTP {e.code}"
        except Exception as e: last = repr(e)[:100]
        time.sleep(min(60, 3 * attempt))
    return ""
def opts(r): return "\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(r["choices"]))
WRITE = ("You are writing a reference passage for a telecom study guide. Question under study (do NOT answer it, do NOT mention option letters):\n"
         "{q}\nOptions the student will see:\n{o}\n\nWrite a precise, factual passage (150-250 words) in the style of the relevant standard, textbook or paper, "
         "stating the facts needed to settle this question, with exact terms, values and the responsible entities. If you are not confident of a fact, omit it rather than guess. Passage only.")
ASK = ("Using ONLY the passage below, answer the question. Reply with exactly 'ANSWER: <letter>', or 'ANSWER: NONE' if the passage does not settle it.\n\nPASSAGE:\n{p}\n\nQUESTION: {q}\n{o}")
CB = ("Answer the following multiple choice question. Reply with exactly 'ANSWER: <letter>'.\n\n{q}\n{o}")
def letter(t):
    m = re.search(r"ANSWER\s*:\s*([A-E]|NONE)", t or "", re.I); return m.group(1).upper() if m else "?"
def work(q):
    r = test[q]; passages = [chat(WRITE.format(q=r["question"], o=opts(r)), 0.7) for _ in range(a.n)]
    implied = [letter(chat(ASK.format(p=p, q=r["question"], o=opts(r)), 0.0, 20)) for p in passages]
    cb = letter(chat(CB.format(q=r["question"], o=opts(r)), 0.0, 20))
    return dict(sample_id=q, passages=passages, implied=implied, closed_book=cb)
done = {}
prev = outdir / "teacher_raw.jsonl"
if prev.exists():
    for l in open(prev, encoding="utf-8"): x = json.loads(l); done[x["sample_id"]] = x
todo = [q for q in qs if q not in done]; print(f"{len(qs)} questions, {len(done)} done, {len(todo)} to do", flush=True)
with open(prev, "a", encoding="utf-8") as f, concurrent.futures.ThreadPoolExecutor(a.workers) as ex:
    for k, x in enumerate(ex.map(work, todo)):
        f.write(json.dumps(x, ensure_ascii=False) + "\n"); f.flush(); done[x["sample_id"]] = x
        if (k + 1) % 25 == 0: print(k + 1, "done", flush=True)
st = collections.Counter(); nw = 0
with open(outdir / "teacher_windows.jsonl", "w", encoding="utf-8") as fw, open(R / "data/eval/tb_teacher.jsonl", "w", encoding="utf-8") as fe:
    for q, x in done.items():
        imp = [i for i in x["implied"] if i not in ("NONE", "?")]
        if len(imp) < a.n or len(set(imp)) != 1: st["no consensus"] += 1; continue
        if x["closed_book"] != imp[0]: st["closed-book disagrees"] += 1; continue
        st["kept"] += 1; r = test[q]
        for p in x["passages"]: fw.write(json.dumps(dict(win_id=f"tq{nw:06d}", text=p, for_q=[q], src="teacher:" + a.model), ensure_ascii=False) + "\n"); nw += 1
        body = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(x["passages"]))
        fe.write(json.dumps({**r, "question": f"Reference material retrieved from the telecom literature. It may or may not contain the answer.\n\n{body}\n\n---\n\n{r['question']}", "n_ctx": a.n}, ensure_ascii=False) + "\n")
print("gate:", dict(st), "windows", nw); print("TEACHER_DONE")
