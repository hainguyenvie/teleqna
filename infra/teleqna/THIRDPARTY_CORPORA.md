# Corpora ngoài — đã xoá khỏi đĩa local 14/09/2026, đây là cách lấy lại

`thirdparty/` và `adaptlkey_nemotron_data/` từng chiếm 20 GB trên máy local. Chúng đều tải lại được
từ Hugging Face và **mọi kết quả quét đã lưu trong `results/`** (xem cột "bằng chứng còn lại"), nên
bản trên đĩa chỉ là bản sao tiện tay. Đã xoá để lấy chỗ; file này là bản ghi provenance.

| Thư mục | Nguồn | Dung lượng | Nội dung | Bằng chứng còn lại |
|---|---|---:|---|---|
| `thirdparty/gsma-3gpp` | `GSMA/3GPP` (git clone) | 4,8 GB | 277.726 file, spec 3GPP Rel-8→Rel-20, có `marked/` và `original/` | `results/otel_coverage.json`, `results/errors_to_specs.json` |
| `thirdparty/otel-llm-data` | `farbodtavakkoli/OTel-LLM` | 1,4 GB | `OTel-LLM-Data.jsonl`, 606.237 record SFT, apache-2.0 | `results/otel_llm_data_scan.json`, `results/otel_verify.json`, `results/otel_scan.stdout` |
| `thirdparty/telco-common-corpus` | `GSMA/Telco-Common-Corpus` | 8,9 GB | 306 file, `data/tcc_*.parquet`, ~10 tỷ token | `results/` (coverage), README.md §corpus |
| `adaptlkey_nemotron_data` | `AdaptKey/AdaptKey-Nemotron-30b` | 4,8 GB | `train.jsonl` 1.303.277 record + test/validation | `results/adaptkey_crossbench.json` — **nhiễm nặng**: ~9.986/10.000 dòng teleqna |

## Lấy lại

```bash
cd infra/teleqna
# 3GPP: KHÔNG dùng snapshot_download — 84.220 file làm gãy khâu liệt kê metadata
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://huggingface.co/datasets/GSMA/3GPP thirdparty/gsma-3gpp

# các repo còn lại: snapshot_download, bọc retry vì CDN hay đứt giữa chừng
HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
from huggingface_hub import snapshot_download
for repo, dest in [('farbodtavakkoli/OTel-LLM','thirdparty/otel-llm-data'),
                   ('GSMA/Telco-Common-Corpus','thirdparty/telco-common-corpus'),
                   ('AdaptKey/AdaptKey-Nemotron-30b','adaptlkey_nemotron_data')]:
    for _ in range(5):
        try: snapshot_download(repo, repo_type='dataset', local_dir=dest); break
        except Exception as e: print('retry', repo, e)
"
```

## Trước khi dùng lại bất kỳ corpus nào ở đây

Quét contamination **trước**, không phải sau:

```bash
python3 infra/teleqna/scan_contamination.py --corpus <file> --k 8
python3 infra/teleqna/verify_contamination.py --corpus <file>
```

`scan_contamination.py` đếm n-gram và chỉ cho cận dưới; `verify_contamination.py` mới là cái kết luận.
AdaptKey đã trượt bài kiểm tra này — đừng train trên nó, và đừng giả định các corpus còn lại sạch chỉ vì
lần quét trước nói thế với một bộ câu hỏi khác.
