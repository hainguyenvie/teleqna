#!/usr/bin/env bash
# Tải layer trọng số của hainh67/teleqna-serve:wise-o3 THẲNG từ registry, có resume (HTTP Range):
# CDN của Docker Hub reset kết nối giữa chừng, crane thì tải lại từ đầu mỗi lần.
set -u
T=$HOME/image_test; REPO=hainh67/teleqna-serve
D=$(python3 -c "import json; m=json.load(open('$T/manifest.json')); print(m['layers'][-1]['digest'])")
SIZE=$(python3 -c "import json; m=json.load(open('$T/manifest.json')); print(m['layers'][-1]['size'])")
F=$T/blobs/${D#sha256:}.tgz; mkdir -p "$T/blobs"
AUTH=$(python3 -c "import json,os; print(json.load(open(os.path.expanduser('~/.docker/config.json')))['auths']['https://index.docker.io/v1/']['auth'])")
echo "layer $D  size $SIZE"
for i in $(seq 1 200); do
  TOKEN=$(curl -s -H "Authorization: Basic $AUTH" "https://auth.docker.io/token?service=registry.docker.io&scope=repository:$REPO:pull" | python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])')
  have=$(stat -c %s "$F" 2>/dev/null || echo 0)
  [ "$have" -ge "$SIZE" ] && break
  curl -sSL -C - --retry 3 -H "Authorization: Bearer $TOKEN" -o "$F" "https://registry-1.docker.io/v2/$REPO/blobs/$D" 2>/dev/null
  now=$(stat -c %s "$F" 2>/dev/null || echo 0)
  echo "attempt $i: $(( now / 1048576 )) / $(( SIZE / 1048576 )) MiB"
  [ "$now" -ge "$SIZE" ] && break
  sleep 3
done
got="sha256:$(sha256sum "$F" | cut -d' ' -f1)"
[ "$got" = "$D" ] || { echo "!! DIGEST MISMATCH $got"; exit 1; }
echo "digest ok $D"
tar xzf "$F" -C "$T/root" && rm -f "$F"
echo "--- extracted:"; ls "$T/root/opt/model"
a=$(sha256sum "$T/root/opt/model/model.safetensors" | cut -d' ' -f1)
b=$(sha256sum "$HOME/projects/teleqna/runs/teleqna-8b/models/kit/wise_o3/model.safetensors" | cut -d' ' -f1)
echo "registry copy $a"; echo "original      $b"
[ "$a" = "$b" ] && echo MODEL_BYTES_IDENTICAL || echo "!! MODEL BYTES DIFFER"
echo PULLBACK_DONE
