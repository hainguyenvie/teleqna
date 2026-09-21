#!/usr/bin/env python3
"""Dense rerank of the deep BM25 candidates for the uncovered questions with Qwen3-Embedding-8B (last-token
pooling, query instruction), top-8 per question -> data/kit/tb_deep_windows.jsonl and an 8B eval set
data/eval/tb_deep8.jsonl (RAG format, 12k budget). Query = question + options (no answer index)."""
import json, torch, collections
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
R = Path.home() / "projects/teleqna/runs/teleqna-8b"; EMB = str(Path.home() / "projects/_shared/models/Qwen3-Embedding-4B")
HEADER = "Reference material retrieved from the telecom literature. It may or may not contain the answer."
test = {r["sample_id"]: r for r in map(json.loads, open(R / "data/eval/otfull10000.jsonl", encoding="utf-8"))}
cands = [json.loads(l) for l in open(R / "data/kit/tb_deep_candidates.jsonl", encoding="utf-8")]
tok = AutoTokenizer.from_pretrained(EMB, local_files_only=True, padding_side="left"); model = AutoModel.from_pretrained(EMB, dtype=torch.bfloat16, local_files_only=True).cuda().eval()
@torch.no_grad()
def embed(texts, bs=32, maxlen=1024):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i+bs], padding=True, truncation=True, max_length=maxlen, return_tensors="pt").to("cuda")
        h = model(**enc).last_hidden_state[:, -1]; out.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu())
    return torch.cat(out)
INSTR = "Instruct: Given a telecom exam question, retrieve the passage that contains the fact needed to answer it\nQuery: "
qtok = AutoTokenizer.from_pretrained(str(Path.home() / "projects/_shared/models/Qwen3-8B"), local_files_only=True)
def trunc(s, n):
    ids = qtok(s, add_special_tokens=False)["input_ids"]; return s if len(ids) <= n else qtok.decode(ids[:n])
nw = 0
with open(R / "data/kit/tb_deep_windows.jsonl", "w", encoding="utf-8") as fw, open(R / "data/eval/tb_deep8.jsonl", "w", encoding="utf-8") as fe:
    for c in cands:
        q = test[c["sample_id"]]; texts = [x["text"] for x in c["cands"]]
        qe = embed([INSTR + q["question"] + " " + " ".join(q["choices"])], bs=1); de = embed(texts)
        top = (de @ qe.T).squeeze(1).topk(min(8, len(texts))).indices.tolist()
        refs = [trunc(texts[i], 12000 // 8) for i in top]
        for i in top: fw.write(json.dumps(dict(win_id=f"td{nw:06d}", text=texts[i], for_q=[c["sample_id"]], src=c["cands"][i]["src"]), ensure_ascii=False) + "\n"); nw += 1
        body = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(refs))
        fe.write(json.dumps({**q, "question": f"{HEADER}\n\n{body}\n\n---\n\n{q['question']}", "n_ctx": len(refs)}, ensure_ascii=False) + "\n")
print(f"reranked {len(cands)} questions -> {nw} windows"); print("DENSE_DEEP_DONE")
