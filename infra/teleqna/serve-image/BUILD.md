# Build và push image teleqna-serve

## Kết luận trước: dùng `build_crane.sh`, không dùng kaniko

```bash
ssh H200_Tensara 'cd ~/projects/teleqna/runs/image-build && setsid nohup bash build_crane.sh > build.log 2>&1 &'
ssh H200_Tensara 'tail -f ~/projects/teleqna/runs/image-build/build.log'   # chờ BUILD_PUSH_DONE
```

Không cần docker daemon, không cần pod. Chi tiết vì sao ở dưới.

## Vì sao không build trên máy local

| | |
|---|---|
| `docker` trên máy local | **có**, nhưng trọng số 16 GB nằm trên node |
| kéo 16 GB về rồi đẩy 16 GB lên từ mạng nhà | hàng giờ, vô nghĩa |
| `docker`, `podman`, `nerdctl` trên dev pod | **không có**, không có socket nào |

Nên image phải được dựng **trên server**, mà server không có daemon.

## Vì sao không dùng kaniko (đã thử, đã hỏng)

Telelogs tháng 8 build được bằng kaniko trong pod. Ngày 21/09 thì không:

| lần | kết quả |
|---|---|
| 1 | `failed to get filesystem from image: read tcp …13.35.190.91:443: connection reset by peer` ở giây thứ 4 |
| 2 | như trên, dù đã thêm `--image-download-retry 5` |
| 3 | pod bị **Evicted: node had condition DiskPressure** (node đang 93% đĩa) |
| 4 | vòng lặp 8 lần thử: **7/7 lần đều reset** ở cùng chỗ — kéo rootfs 10 GB của base image |

Dev pod tải cùng blob đó bình thường (~8 MB/s), nên đây là vấn đề của đường mạng
pod kaniko, không phải của node. Đã dọn 207 GB checkpoint để hết DiskPressure,
nhưng lỗi kéo base image thì không sửa được từ phía mình.

## Đường đang dùng: crane (go-containerregistry)

`crane` là một binary tĩnh, không daemon, và quan trọng nhất: **copy base image
chéo repo ngay trong registry** bằng blob mount — 11 giây, không tải byte nào.
Chỉ layer của mình mới phải upload.

```
1. crane copy vllm/vllm-openai:v0.26.0  ->  hainh67/teleqna-serve:base-vllm-0.26.0   (11 giây)
2. dựng cây /opt/{model,eval,hf,entrypoint.sh}   (hardlink 16 GB trọng số, không copy)
3. kiểm hợp đồng artefact (greedy + thinking off + task + dataset)   <- y như RUN trong Dockerfile
4. tar 2 layer  ->  crane append  ->  hainh67/teleqna-serve:wise-o3
5. crane mutate: entrypoint, cmd, env, cổng 8000
```

Thư viện chấm điểm cài bằng `pip install --target` (không venv, vì venv gắn cứng
đường dẫn tuyệt đối) và được giữ lại ở `image-build/pylibs` để dùng lại — PyPI
trên node này hay timeout giữa chừng, một lần hỏng là vứt cả cây stage.

**Chạy detached.** Container của dev pod bị restart vài lần trong ngày; một lần
restart giữa lúc upload 16 GB là mất 20 phút. `setsid nohup … &` rồi theo dõi log.

## Điều kiện tiên quyết

**1. Repo Docker Hub phải là PRIVATE.** Image mang parquet `GSMA/ot-full`.
Đã tạo `hainh67/teleqna-serve` ở chế độ private ngày 21/09 (qua API, xác nhận
`is_private: true`). Push vào repo chưa tồn tại sẽ tạo nó **public**.

**2. Credential.** `~/.docker/config.json` trên server (chmod 600), lấy từ
`~/.docker/config.json` của máy local, tài khoản `hainh67`. crane đọc thẳng file này.

## Kiểm sau khi push

```bash
crane config hainh67/teleqna-serve:wise-o3          # entrypoint, env, cổng
docker run --gpus all --rm hainh67/teleqna-serve:wise-o3 verify
```

Kỳ vọng `accuracy 0.822–0.823` trên 10.000 câu. Ra **0,818** nghĩa là
`generation_config.json` không phải bản greedy; thấy `<think>` nghĩa là
`chat_template.jinja` không phải bản tắt thinking. `build_crane.sh` kiểm cả hai
trước khi push nên hai lỗi này đáng lẽ không tới được đây.

## `Dockerfile` còn dùng làm gì

Giữ làm công thức chuẩn cho ai có docker daemon: `docker build -t … .` với context
gồm `model/`, `eval/`, `hf_cache/`. Nội dung image do `build_crane.sh` tạo ra là
tương đương — cùng base, cùng file, cùng env — chỉ khác cách lắp.
