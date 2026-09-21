#!/usr/bin/env bash
# Dựng và push image KHÔNG cần docker daemon, chạy thẳng trên dev pod.
#
# Vì sao không dùng kaniko: pod kaniko không kéo nổi rootfs 10 GB của base image
# từ CDN Docker Hub (connection reset 7/7 lần, luôn ở giây thứ 4), trong khi dev
# pod tải bình thường. crane giải quyết triệt để: base image được **copy chéo repo
# ngay trong registry** (mount blob, 11 giây, không tải byte nào), rồi chỉ các
# layer của mình được upload.
#
#   bash build_crane.sh            dựng + push hainh67/teleqna-serve:wise-o3
set -euo pipefail
R=$HOME/projects/teleqna/runs/teleqna-8b
SRC=$HOME/projects/teleqna/runs/image-build/src
STAGE=${STAGE:-$HOME/projects/teleqna/runs/image-build/stage}
CRANE=$HOME/bin/crane
BASE_SRC=${BASE_SRC:-vllm/vllm-openai:v0.26.0}
REPO=${REPO:-hainh67/teleqna-serve}
BASE_TAG=${BASE_TAG:-$REPO:base-vllm-0.26.0}
DEST=${DEST:-$REPO:wise-o3}
MODEL=${MODEL:-$R/models/kit/wise_o3}

echo "=== 1. base image -> repo đích (mount blob trong registry) ==="
"$CRANE" copy "$BASE_SRC" "$BASE_TAG" --platform linux/amd64

echo "=== 2. dựng cây file đúng đường dẫn trong image ==="
rm -rf "$STAGE"; mkdir -p "$STAGE/opt/eval" "$STAGE/opt/hf" "$STAGE/opt/model"
# hardlink 16 GB trọng số thay vì copy
for f in "$MODEL"/*; do ln "$f" "$STAGE/opt/model/$(basename "$f")" 2>/dev/null || cp "$f" "$STAGE/opt/model/"; done
cp -r "$SRC/eval" "$STAGE/opt/eval/evals"
cp "$SRC/registry_teleqna.py" "$STAGE/opt/eval/evals/_registry.py"
cp "$SRC/verify_summary.py" "$SRC/inspect_cli.py" "$STAGE/opt/eval/"
cp -r "$SRC/hf_cache/hub" "$STAGE/opt/hf/hub"
install -m 0755 "$SRC/entrypoint.sh" "$STAGE/opt/entrypoint.sh"
# Thư viện chấm điểm: pip --target, không venv (venv gắn cứng đường dẫn tuyệt đối).
# Cài sẵn một lần vào $PYLIBS rồi tái dùng: PyPI trên node này hay timeout giữa chừng,
# và một lần hỏng sẽ vứt luôn cây stage vừa dựng.
PYLIBS=${PYLIBS:-$HOME/projects/teleqna/runs/image-build/pylibs}
if [ ! -d "$PYLIBS" ]; then
  for i in 1 2 3 4 5; do
    python3 -m pip install --quiet --no-cache-dir --retries 10 --timeout 120 --target "$PYLIBS" \
      "inspect-ai==0.3.188" "datasets==4.6.1" "pyarrow==22.0.0" "openai==2.26.0" && break
    echo "pip lần $i hỏng, thử lại"; sleep 15
  done
fi
PYTHONPATH="$PYLIBS" python3 -c "import inspect_ai, datasets, pyarrow, openai" \
  || { echo "!! $PYLIBS không dùng được"; exit 1; }
cp -r "$PYLIBS" "$STAGE/opt/eval/pylibs"

echo "--- hợp đồng artefact (kiểm trước khi push, y như RUN trong Dockerfile) ---"
python3 - "$STAGE" <<'PY'
import json, sys, glob, os
s = sys.argv[1]
g = json.load(open(f"{s}/opt/model/generation_config.json"))
assert g.get("do_sample") is False and g.get("temperature") == 0.0 and g.get("top_p") == 1.0, g
t = open(f"{s}/opt/model/chat_template.jinja").read()
assert "enable_thinking is not defined" in t and "set enable_thinking = false" in t
for p in ("opt/model/config.json", "opt/eval/evals/teleqna/teleqna.py", "opt/eval/evals/_registry.py",
          "opt/eval/inspect_cli.py", "opt/eval/verify_summary.py", "opt/entrypoint.sh"):
    assert os.path.exists(f"{s}/{p}"), p
assert glob.glob(f"{s}/opt/hf/hub/datasets--GSMA--ot-full/snapshots/*/teleqna/test-00000-of-00001.parquet")
print("artifact contract ok: greedy defaults + thinking off + task + dataset + entrypoint")
PY

echo "=== 3. đóng layer ==="
cd "$STAGE"
tar --numeric-owner --owner=0 --group=0 -cf /tmp/l_model.tar opt/model
tar --numeric-owner --owner=0 --group=0 -cf /tmp/l_rest.tar opt/eval opt/hf opt/entrypoint.sh
ls -la /tmp/l_model.tar /tmp/l_rest.tar

echo "=== 4. append + push ==="
"$CRANE" append -b "$BASE_TAG" -f /tmp/l_rest.tar -t "$REPO:stage1" 
"$CRANE" append -b "$REPO:stage1" -f /tmp/l_model.tar -t "$DEST"

echo "=== 5. metadata (entrypoint / cmd / env / cổng) ==="
"$CRANE" mutate "$DEST" -t "$DEST" \
  --entrypoint /opt/entrypoint.sh --cmd serve \
  --exposed-ports 8000/tcp \
  --env SERVED_NAMES="Qwen3-8B-Telco teleqna-8b-closedbook wise-o3" \
  --env PORT=8000 --env HOST=0.0.0.0 \
  --env VLLM_TP=1 --env VLLM_MAX_MODEL_LEN=4096 --env VLLM_GPU_MEM_UTIL=0.85 --env VLLM_MAX_NUM_SEQS=64 \
  --env VERIFY_CONNECTIONS=64 \
  --env HF_HOME=/opt/hf --env HF_HUB_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 \
  --env TOKENIZERS_PARALLELISM=false --env OMP_NUM_THREADS=8

rm -f /tmp/l_model.tar /tmp/l_rest.tar
echo "=== xong: $DEST ==="
"$CRANE" config "$DEST" | python3 -c "import json,sys; c=json.load(sys.stdin)['config']; print('Entrypoint', c.get('Entrypoint'), '| Cmd', c.get('Cmd'), '| Ports', c.get('ExposedPorts')); print('Env:'); [print('   ', e) for e in c.get('Env', []) if not e.startswith(('PATH=','LD_','NV_','CUDA_','NCCL_'))]"
echo BUILD_PUSH_DONE
