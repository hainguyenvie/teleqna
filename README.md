# teleqna — cột TeleQnA của GSMA Open Telco leaderboard

Repo này tách khỏi [`telelogs`](../telelogs) ngày 14/09/2026. Trước đó toàn bộ
track nằm trong `infra/teleqna/` của repo đó; **lịch sử git của giai đoạn 1–3
vẫn nằm ở repo telelogs** (các commit `ce760f5` → `efb679f`), repo này bắt đầu
lại từ một commit sạch.

TeleQnA là cột lớn nhất: 10.000 trong 19.588 câu mà `average` của leaderboard
cộng vào — nhiều hơn tổng sáu benchmark còn lại.

## Bắt đầu từ đâu

| Tài liệu | Nội dung |
|---|---|
| [infra/teleqna/README.md](infra/teleqna/README.md) | Nhật ký chính của track: benchmark, contamination, mọi arm đã chạy |
| [PLAN_8B.md](infra/teleqna/PLAN_8B.md) · [PLAN_8B_DATA.md](infra/teleqna/PLAN_8B_DATA.md) | Kế hoạch pivot sang 8B/9B và thiết kế dữ liệu |
| [RESULTS_9B.md](infra/teleqna/RESULTS_9B.md) | Kết quả 9B — mốc cao nhất của track |
| [RESULTS_ENTIGRAPH.md](infra/teleqna/RESULTS_ENTIGRAPH.md) · [RESULTS_DRIFT_REFUTED.md](infra/teleqna/RESULTS_DRIFT_REFUTED.md) · [RESULTS_INTERNALS_8B.md](infra/teleqna/RESULTS_INTERNALS_8B.md) · [RESULTS_KSWEEP.md](infra/teleqna/RESULTS_KSWEEP.md) | Các nhánh đã đóng, kèm bằng chứng đóng |
| [H200_SERVER_GUIDE.md](H200_SERVER_GUIDE.md) | Cách chạy trên server H200 sau khi tách |
| [infra/teleqna/dashboard/](infra/teleqna/dashboard/) | Trang tổng hợp (baseline, journey) |

## Bố cục

```text
infra/teleqna/           mã nguồn + nhật ký track
├── sft/                 SFT/DPO/armE-H: script, job wrapper, dữ liệu sinh
├── synth/               sinh dữ liệu tổng hợp, multiplicity
├── analysis/            phân tích lỗi, coverage, contamination
├── results/             kết quả đã lưu (nguồn của mọi con số trong tài liệu)
├── dashboard/           trang HTML tổng hợp
├── thirdparty/          corpora ngoài (gitignore, nhiều GB — quét trước khi dùng)
└── adaptlkey_nemotron_data/   corpus nhiễm, giữ để tái lập bản quét (gitignore)
data/teleqna/            parquet gốc + bản .jsonl đọc được
artifacts/teleqna-gateway/   kết quả chạy qua gateway (122B FP8, think/no-think)
backups/h200/            snapshot server đã verify sha256 (gitignore)
```

## Trên server

Sau khi tách (xem [H200_SERVER_GUIDE.md](H200_SERVER_GUIDE.md)):

```text
~/projects/teleqna/runs/{teleqna-8b, teleqna-sft, teleqna-spare1, teleqna-spare4}
~/projects/_shared/{models, corpora, hf-cache}     ← dùng chung với telelogs
```

## Kỷ luật dữ liệu

Benchmark này **công khai toàn bộ**, kể cả trường `explanation` của cả 10.000
câu. Vì thế ranh giới "được phép train" không thể dựa vào tên file:

- Mọi corpus ngoài phải quét bằng `infra/teleqna/scan_contamination.py` +
  `verify_contamination.py` **trước khi** đưa vào train. Đã có corpus chứa
  9.986/10.000 dòng teleqna.
- `data/train/eligible/` là nơi duy nhất job train được đọc; `eval/` và
  `restricted/` (answer key, explanation, haystack) không bao giờ là train input.
- Mọi adapter phải có arm base đo cùng model-load, nếu không con số chỉ là
  diagnostic.

Bản thân dữ liệu benchmark được commit ở đây vì repo là private. **Không public
repo khi các file này còn trong lịch sử git.**
