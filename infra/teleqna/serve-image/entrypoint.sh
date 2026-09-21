#!/bin/bash
# Một tiến trình, một cổng: vLLM phục vụ thẳng thư mục model.
#
#   serve    đứng yên cho orchestrator / satellite chấm. Mặc định.
#   verify   bật cùng stack đó, chạy HARNESS CHÍNH THỨC (gsma-labs/evals trên
#            Inspect AI) trên đủ 10.000 câu TeleQnA, in điểm, thoát. Điểm là thứ
#            máy nhận đo được, không phải thứ image tự khai.
set -u
MODE=${1:-serve}
case "$MODE" in serve|verify) ;; *) echo "usage: [serve|verify]"; exit 2 ;; esac

MODEL=${MODEL_DIR:-/opt/model}
FIRST_NAME=$(echo "$SERVED_NAMES" | awk '{print $1}')

echo "=== teleqna closed-book serving image  (mode: $MODE) ==="
echo "model        : $MODEL"
echo "served names : $SERVED_NAMES"
echo "endpoint     : $HOST:$PORT/v1   (orchestrator gửi model=\"$FIRST_NAME\")"
echo "decode       : greedy do generation_config.json trong model, thinking tắt do chat_template.jinja"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || {
  echo "!! không thấy GPU. Chạy với --gpus all và NVIDIA Container Toolkit."; exit 1; }

python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name $SERVED_NAMES \
  --host "$HOST" --port "$PORT" \
  --dtype bfloat16 \
  --tensor-parallel-size "$VLLM_TP" \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  --gpu-memory-utilization "$VLLM_GPU_MEM_UTIL" \
  --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
  --enable-prefix-caching &
VLLM_PID=$!

shutdown() { echo "shutting down"; kill "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null; exit 0; }
trap shutdown TERM INT

for i in $(seq 1 180); do
  python3 - "http://127.0.0.1:$PORT/health" <<'PY' 2>/dev/null && break
import sys, urllib.request
try: urllib.request.urlopen(sys.argv[1], timeout=3)
except Exception: sys.exit(1)
PY
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "!! engine chết lúc khởi động"; exit 1; }
  sleep 5
done

# Một câu thật, đúng template mà harness dùng, chạy qua đúng đường mà orchestrator
# sẽ đi. READY chỉ được in khi câu này trả về đúng định dạng 'ANSWER: X'.
SELFTEST_OK=0
if python3 - <<'PY'; then SELFTEST_OK=1; fi
import json, os, re, sys, urllib.request
port = os.environ["PORT"]; name = os.environ["SERVED_NAMES"].split()[0]
user = ("Answer the following multiple choice question. The entire content of your response should be of the "
        "following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of A,B,C,D.\n\n"
        "What does RAN stand for in 3GPP?\n\nA) Radio Access Network\nB) Random Access Node\n"
        "C) Rate Adaptive Network\nD) Remote Antenna Node")
try:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({"model": name, "messages": [{"role": "user", "content": user}]}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    d = json.load(urllib.request.urlopen(req, timeout=300))
except Exception as exc:
    print(f"!! self-test lỗi: {exc}"); sys.exit(1)
body = d["choices"][0]["message"]["content"] or ""
if "<think>" in body: print("!! self-test: model còn bật thinking -> chat template sai"); sys.exit(1)
m = re.search(r"(?i)^ANSWER\s*:\s*([A-D])\s*$", body.strip())
if not m: print(f"!! self-test: không ra định dạng ANSWER: X (nhận {body[:80]!r})"); sys.exit(1)
print(f"self-test: {body.strip()!r}  tokens={d['usage']['completion_tokens']}  (đáp án đúng là A)")
sys.exit(0)
PY

if [ "$MODE" = "verify" ]; then
  echo
  echo "=== verify: harness CHÍNH THỨC của GSMA, đủ 10.000 câu TeleQnA ==="
  echo "trên H200 khoảng 2 phút"
  OUT=${VERIFY_OUT:-/tmp/verify_teleqna}
  # inspect eval chỉ nhận đường dẫn TƯƠNG ĐỐI so với cwd (pathlib.glob từ chối
  # pattern tuyệt đối) — nên cd vào package rồi gọi teleqna/teleqna.py.
  cd /opt/eval/evals
  PYTHONPATH=/opt/eval/pylibs:/opt/eval OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1" OPENAI_API_KEY=dummy \
  INSPECT_LOG_DIR="$OUT" \
    python3 /opt/eval/inspect_cli.py eval teleqna/teleqna.py \
      --model "openai/$FIRST_NAME" -T full=true --max-connections "$VERIFY_CONNECTIONS"
  rc=$?
  python3 /opt/eval/verify_summary.py "$OUT" || true
  kill "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null
  exit "$rc"
fi

if [ "$SELFTEST_OK" != "1" ]; then
  cat <<MSG

=== NOT READY ===
Cổng đã mở nhưng một câu thật không trả về đáp án chấm được. Phục vụ tiếp chỉ tạo
ra 10.000 câu sai im lặng, nên thoát. Xem log phía trên.

MSG
  kill "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null; exit 1
fi

cat <<MSG

=== READY ===
Orchestrator / satellite trỏ vào:  http://<host>:$PORT/v1
model id:                          $FIRST_NAME
Kiểm nhanh:                        curl http://<host>:$PORT/v1/models
Chấm lại chính thức:               docker run --gpus all <image> verify

MSG

wait "$VLLM_PID"
