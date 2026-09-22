# teleqna-serve-image — đóng gói TeleQnA closed-book thành một image

## Cái này đóng gói cái gì

Một model 8B duy nhất, trả lời một lượt, **không** retrieval, **không** pipeline,
**không** tool. Khác hẳn telelogs: ở đó image mang theo cả scaffold DSPy vì điểm
nằm ở hệ thống; ở đây điểm nằm trong trọng số, nên image chỉ có trọng số cộng hai
file cấu hình quyết định con số đó.

```
orchestrator / satellite ──▶ :8000  vLLM (thư mục /opt/model)
                                      trả "ANSWER: X", 5 token
```

Trọng số: Qwen3-8B → study-kit trên 114k cửa sổ nguồn → vote-distillation →
sibling-contrast → ensemble-distillation → ghép trọng số 3 dòng → on-policy
distillation (giáo viên OTel-31B đọc nguồn) → nội suy. Chi tiết trong
`../METHOD_STUDYKIT.md` và `../PLAN_CLOSED_BOOK_8B.md`.

## Hai file làm nên 0,9 điểm

Phục vụ thư mục model bằng vLLM trần là ra đúng điểm đã đo, vì thư mục tự mang:

* `generation_config.json` — `do_sample=false, temperature=0, top_p=1`. Mặc định
  gốc của Qwen3 là `temperature 0,6 / top_p 0,95 / top_k 20`; để nguyên thì chấm
  chính thức ra **0,818** thay vì **0,823**. Bỏ hẳn `temperature` còn tệ hơn
  (**0,814**) vì vLLM quay về mặc định 1,0.
* `chat_template.jinja` — `enable_thinking` mặc định **false**. Không có dòng này
  model sẽ sinh `<think>…`, vừa tốn token vừa lệch định dạng.

Dockerfile kiểm cả hai lúc **build**, hỏng thì hỏng ở đó chứ không phải ở request
đầu tiên.

## Image đã đẩy (21/09/2026)

```
hainh67/teleqna-serve:wise-o3          <- bản chốt, 34 layer, 20,46 GiB nén
hainh67/teleqna-serve:base-vllm-0.26.0 <- base copy chéo repo, để build lại không phải kéo CDN
hainh67/teleqna-serve:stage1           <- trung gian (base + eval + dataset), giữ để build lại nhanh
```

Repo **private**. Entrypoint `/opt/entrypoint.sh`, cmd `serve`, cổng 8000, env đầy đủ
(kiểm bằng `crane config`). Nội dung đã kiểm từ registry: `/opt/entrypoint.sh`,
`/opt/eval/{inspect_cli.py,verify_summary.py,evals/teleqna/teleqna.py,evals/_registry.py}`,
parquet `GSMA/ot-full` và 19.028 file thư viện chấm điểm (ABI cpython-312, khớp base Ubuntu 24.04).

## Đã kéo ngược về và chạy thật (22/09/2026)

Không có docker trên H200 và cluster không cho tạo pull secret, nên dựng lại container
từ **chính các byte trên Docker Hub**: `pullback_test.sh` tải từng layer của
`wise-o3` thẳng từ registry (có resume, CDN reset giữa chừng), kiểm sha256 từng layer
khớp manifest, giải nén thành cây `/opt` y như trong image. `runtest.sh` rồi chạy
**nguyên văn** `entrypoint.sh` lấy từ layer đó, với env đọc từ config của image, trên
vLLM **0.26.0** (đúng bản của base image), card 4.

| kiểm tra | kết quả |
|---|---|
| trọng số tải về so với bản gốc | **giống từng byte** (sha256 `252d94d0…`) |
| entrypoint | self-test `ANSWER: A` 5 token → `=== READY ===` |
| `/v1/models` | `Qwen3-8B-Telco`, `teleqna-8b-closedbook`, `wise-o3` |
| replay nguyên văn 10.000 request body của orchestrator | **82,25 %**, 27 giây |
| định dạng: `ANSWER: X` khớp tuyệt đối (fullmatch) | **10.000 / 10.000**, 0 câu có `<think>` |
| `response.model == request.model` | 10.000 / 10.000, `finish_reason: stop` 10.000 |
| mode `verify` (harness chính thức, offline, thư viện trong image) | **0,8217**, 10.000 / 10.000 mẫu |

`verifytest.sh` chạy mode verify với entrypoint của image, chỉ đổi tiền tố `/opt/`
sang cây đã giải nén (dev pod không có quyền ghi `/opt`).

## Chạy

```bash
docker run --gpus all -p 127.0.0.1:8000:8000 hainh67/teleqna-serve:wise-o3
# chờ dòng "=== READY ==="
curl http://127.0.0.1:8000/v1/models
```

hoặc `docker compose up -d` với `docker-compose.yml` kèm theo (đã ghim card 6).

## Chấm lại

```bash
docker run --gpus all --rm hainh67/teleqna-serve:wise-o3 verify
```

Harness chính thức (gsma-labs/evals trên Inspect AI), đủ 10.000 câu, dataset
`GSMA/ot-full` đã đóng trong image. Kỳ vọng **0,822–0,823**.

## Repo Docker Hub phải tạo PRIVATE trước

Image mang theo parquet của `GSMA/ot-full` (3,5 MB) để `verify` chạy offline.
Push vào một repository chưa tồn tại sẽ tạo nó **public**. Tạo
`hainh67/teleqna-serve` ở chế độ Private trên hub.docker.com **trước** lần push đầu.

## Build

Xem `BUILD.md`. Tóm tắt: trọng số 16,4 GB đã nằm trên đĩa H200, máy local không có
docker, nên build bằng **kaniko** trong một pod thường (`build_pod.yaml`), giống
hệt cách telelogs đã làm.

## File

| file | việc |
|---|---|
| `Dockerfile` | image; kiểm hợp đồng artefact lúc build |
| `entrypoint.sh` | `serve` (mặc định) và `verify`; self-test một câu thật trước khi in READY |
| `verify_summary.py` | in lại điểm lần verify gần nhất, kèm phân tách theo môn |
| `docker-compose.yml` | bản triển khai có ghim card, healthcheck, network `telco` |
| `API.md` | hợp đồng request/response cho orchestrator |
| `BUILD.md` | dựng context và build bằng kaniko |
| `build_pod.yaml` | pod kaniko |
