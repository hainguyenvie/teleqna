#!/usr/bin/env bash
# Chạy thử image đã kéo ngược từ Docker Hub, không docker:
#   * entrypoint.sh NGUYÊN VĂN từ layer của image (không sửa) cho mode serve,
#   * env lấy từ config của image (chỉ đổi PORT để khỏi đụng cổng khác),
#   * python3 = venv có đúng vLLM 0.26.0 như base image,
#   * card: chỉ 4–7, chỉ card đang 0 MiB.
# Sau READY: replay nguyên văn 10.000 request body của orchestrator và chấm.
set -u
T=$HOME/image_test; R=$HOME/projects/teleqna/runs/teleqna-8b
export PATH=$T/venv-vllm-0.26.0/bin:$PATH
eval "$(python3 - "$T/config.json" <<'PY'
import json, sys, shlex
env = json.load(open(sys.argv[1]))["config"]["Env"]
for e in env:
    k, v = e.split("=", 1)
    if k in ("PATH",) or k.startswith(("LD_", "NV_", "NVIDIA_", "CUDA_")): continue
    print(f"export {k}={shlex.quote(v)}")
PY
)"
export PORT=8030
CARD=""; for c in 4 5 6 7; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$u" -le 1000 ] && { CARD=$c; break; }; done
[ -n "$CARD" ] || { echo "!! không có card trống trong 4–7"; exit 1; }
export CUDA_VISIBLE_DEVICES=$CARD
echo "card $CARD | SERVED_NAMES=$SERVED_NAMES | PORT=$PORT | vllm $(python3 -c 'import vllm;print(vllm.__version__)')"
MODEL_DIR=$T/root/opt/model bash "$T/root/opt/entrypoint.sh" serve > "$T/serve.log" 2>&1 &
SP=$!
for i in $(seq 1 120); do grep -qE "=== READY ===|NOT READY|process died|không thấy GPU" "$T/serve.log" && break; kill -0 $SP 2>/dev/null || break; sleep 5; done
echo "--- entrypoint output:"; grep -E "===|self-test|model  |served|endpoint|decode|Point|model id|curl" "$T/serve.log" | head -16
grep -q "=== READY ===" "$T/serve.log" || { echo "!! entrypoint không READY"; tail -20 "$T/serve.log"; kill $SP; exit 1; }
python3 - <<'PY'
import json, re, urllib.request, time, os
from concurrent.futures import ThreadPoolExecutor
port = os.environ["PORT"]; base = f"http://127.0.0.1:{port}"
print("--- /v1/models:", [m["id"] for m in json.load(urllib.request.urlopen(base + "/v1/models"))["data"]])
R = os.path.expanduser("~/projects/teleqna/runs/teleqna-8b")
bodies = [json.loads(l) for l in open(f"{R}/gsma_serve/orch_requests.jsonl", encoding="utf-8")]
test = [json.loads(l) for l in open(f"{R}/data/eval/otfull10000.jsonl", encoding="utf-8")]
def key(q, ch): return (q.strip(), tuple(c.strip() for c in ch))
gold = {key(r["question"], r["choices"]): r["answer"] for r in test}
def qkey(content):
    body = content.split("\n\n", 1)[1]; q, opts = body.split("\n\n", 1)
    return key(q, [l[3:] for l in opts.strip().split("\n")])
def ask(b):
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r: return json.load(r)
first = ask(bodies[0])
print("--- một response đầy đủ (request = dòng 1 của file orchestrator, gửi nguyên văn):")
print(json.dumps({k: first[k] for k in ("object", "model", "choices", "usage")}, ensure_ascii=False)[:600])
t0 = time.time()
with ThreadPoolExecutor(max_workers=64) as ex: outs = list(ex.map(ask, bodies))
ok = unp = think = names = 0; fin = {}
for b, d in zip(bodies, outs):
    c = d["choices"][0]; txt = (c["message"]["content"] or "").strip(); fin[c["finish_reason"]] = fin.get(c["finish_reason"], 0) + 1
    names += d["model"] == b["model"]; think += "<think>" in txt
    m = re.fullmatch(r"ANSWER: ([A-E])", txt)
    if not m: unp += 1; continue
    ok += (ord(m.group(1)) - 65) == gold[qkey(b["messages"][0]["content"])]
n = len(bodies)
print(f"--- replay {n} request body của orchestrator trong {time.time()-t0:.0f}s")
print(f"    accuracy {100*ok/n:.2f}%")
print(f"    đúng định dạng 'ANSWER: X' tuyệt đối (fullmatch): {n-unp}/{n}   | có <think>: {think}")
print(f"    response.model == request.model ('Qwen3-8B-Telco'): {names}/{n}   | finish_reason: {fin}")
PY
kill $SP 2>/dev/null; sleep 5; pkill -f "port $PORT" 2>/dev/null; true
echo RUNTEST_DONE
