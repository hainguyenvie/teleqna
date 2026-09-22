#!/usr/bin/env bash
# Mode `verify` của image: dùng entrypoint lấy từ registry, chỉ đổi tiền tố /opt/ -> cây đã giải nén
# (dev pod không có quyền ghi /opt). Chạy harness chính thức offline trên 10.000 câu.
set -u
T=$HOME/image_test
export PATH=$T/venv-vllm-0.26.0/bin:$PATH
eval "$(python3 - "$T/config.json" <<'PY'
import json, sys, shlex
for e in json.load(open(sys.argv[1]))["config"]["Env"]:
    k, v = e.split("=", 1)
    if k in ("PATH",) or k.startswith(("LD_", "NV_", "NVIDIA_", "CUDA_")): continue
    print(f"export {k}={shlex.quote(v)}")
PY
)"
export PORT=8031 HF_HOME=$T/root/opt/hf VERIFY_OUT=$T/verify_out
CARD=""; for c in 4 5 6 7; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$u" -le 1000 ] && { CARD=$c; break; }; done
[ -n "$CARD" ] || { echo "!! không có card trống trong 4–7"; exit 1; }
export CUDA_VISIBLE_DEVICES=$CARD
sed "s#/opt/#$T/root/opt/#g" "$T/root/opt/entrypoint.sh" > "$T/entrypoint_prefixed.sh"
echo "card $CARD | verify offline | HF_HOME=$HF_HOME"
MODEL_DIR=$T/root/opt/model bash "$T/entrypoint_prefixed.sh" verify 2>&1 | grep -vE "INFO|WARNING|it/s|Loading|EngineCore|^\s*$" | tail -25
echo VERIFYTEST_DONE
