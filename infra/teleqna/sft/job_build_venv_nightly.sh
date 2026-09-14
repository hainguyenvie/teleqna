#!/usr/bin/env bash
# Build a second serving venv on vLLM nightly, for Qwen3.5 only.
#
# Our venv has vLLM 0.11.0, whose model registry knows Qwen3MoeForCausalLM and
# Qwen3NextForCausalLM but has no entry for Qwen3.5's `qwen3_5_moe` — the arch
# uses hybrid linear/full attention plus an MTP head, and the model card says
# plainly that vLLM main is required. So this is a new tree, not an upgrade:
# venv keeps scoring the 8B arms at 0.11.0 so that every number in the
# comparison table stays on one stack, and venv-vllm-nightly serves the 122B.
#
# No GPU needed, which is why it runs here on the protected pod while card 0
# trains and card 1 waits for weights.
#
# It is built under this pod's own tree, not under telelogs-base/venvs, because
# telelogs-base is mounted read-only here — the first attempt died on exactly
# that. The tree is still shared: spare4, which owns the runtime door to cards
# 1 and 5, sees this path as telelogs-base/runs/teleqna-sft/venvs/... and a venv
# invoked through its own bin/python resolves site-packages from that directory,
# so one build serves both pods.
set -uo pipefail
ROOT=/workspace/teleqna-sft
VENV="$ROOT/venvs/venv-vllm-nightly"
PY311="$(command -v python3.11 || echo /opt/conda/bin/python3.11)"
mkdir -p "$ROOT/venvs"

echo "python: $PY311 -> $("$PY311" -V 2>&1)"
# curl is not installed in this image, so probe with the interpreter we have.
"$PY311" - <<'PY'
import urllib.request
for url in ("https://pypi.org/simple/", "https://wheels.vllm.ai/nightly"):
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            print(f"  {url} -> {r.status}")
    except Exception as e:
        print(f"  {url} -> {type(e).__name__}: {e}")
PY

if [ ! -x "$VENV/bin/python" ]; then
  "$PY311" -m venv "$VENV" || { echo "ABORT: venv creation failed"; exit 20; }
fi
"$VENV/bin/pip" install -q --upgrade pip setuptools wheel

# --pre plus the nightly index: the nightly wheels are prereleases, and without
# --pre pip silently resolves back to the last stable, which is exactly the
# 0.11.x that cannot load this model.
echo "#### install START $(date -Iseconds)"
"$VENV/bin/pip" install --pre vllm \
  --extra-index-url https://wheels.vllm.ai/nightly \
  2>&1 | tail -25
rc=${PIPESTATUS[0]}
echo "#### install rc=$rc $(date -Iseconds)"
[ "$rc" -eq 0 ] || { echo "ABORT: pip install failed"; exit 21; }

"$VENV/bin/python" - <<'PY'
import vllm, torch, transformers
print("vllm", vllm.__version__, "torch", torch.__version__, "tf", transformers.__version__)
from vllm.model_executor.models.registry import ModelRegistry
names = sorted(n for n in ModelRegistry.get_supported_archs() if "3_5" in n or "3.5" in n or "Qwen3" in n)
print("qwen archs:", names)
# The whole point of the new tree: if this name is absent the 122B cannot load,
# and it is better to learn that here than after 250GB has finished landing.
assert any("Qwen3_5" in n or "Qwen3.5" in n for n in names), \
    "nightly still has no Qwen3.5 arch — check the wheel actually installed"
print("OK: Qwen3.5 architecture is registered")
PY
echo "#### VENV DONE $(date -Iseconds)"
