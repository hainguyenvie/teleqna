set -o pipefail
source venv/bin/activate
export PYTHONPATH=$PWD/src:$PWD/third_party/gsplat/examples
echo "=== imports ==="
for m in diff_gaussian_rasterization_bp einops s2fft jax sklearn scipy plyfile gsplat; do
  python -c "import $m; print(\"  $m OK\", getattr($m,\"__version__\",\"\"))" 2>&1 | tail -1
done
echo "=== pip reachability ==="
timeout 60 pip download --no-deps --no-binary :none: -d /tmp/pipprobe einops 2>&1 | tail -4
echo PROBE_DONE
