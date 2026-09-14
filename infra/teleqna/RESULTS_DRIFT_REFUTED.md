# Không phải drift ăn mất kiến thức — kiến thức chưa bao giờ vào

*Đo 2026-08-20 từ per-row artifacts. Script: `/tmp/drift2.py`, `/tmp/ch.py`
(nguồn trong `scratchpad/`). Tài liệu này bác bỏ một phát biểu tôi đã đưa ra ở
lượt trước.*

## Phát biểu bị bác bỏ

> "Mọi thất bại trước là thất bại vì DRIFT, không phải vì kiến thức không vào."

Sai. Tôi tổng hợp nó từ hai dòng memory (`campaign 1: sửa 76 phá 109`,
`arm H: C_hard +13,73`) mà chưa tách được **số hạng học** ra khỏi **số hạng
trôi**. Khi tách ra, số hạng học bằng không ở mọi arm.

## Phép tách

Trên per-row: `fixed` = base SAI → arm ĐÚNG (số hạng học);
`broke` = base ĐÚNG → arm SAI (số hạng trôi).

Null không giả định mà đọc từ các arm **DPO** — repo đã chứng minh chúng không
nạp được kiến thức (fix-rate phẳng qua mọi bin tương đồng cặp), nên chúng cho
đúng mức churn của một run không học được gì.

### 1. CPT campaign 1 — số hạng học nằm trong null

| arm | acc | changed | **đi đúng hướng** | fixed% | broke% |
|---|---:|---:|---:|---:|---:|
| CPT1 step200 | 71,10 | 157 | 40,8% | 24,62 | 12,57 |
| CPT1 step600 | 70,70 | 185 | 41,1% | 29,23 | 14,73 |
| CPT1 final | 71,40 | 180 | **42,8%** | 29,62 | 13,92 |
| NULL dpo1 | 68,50 | 207 | 36,7% | 29,23 | 17,70 |
| NULL dpo2 | 73,60 | 56 | 46,4% | 10,00 | 4,05 |
| NULL dpo3 | 73,80 | 90 | **48,9%** | 16,92 | 6,22 |
| NULL dpo4 | 72,60 | 144 | 45,1% | 25,00 | 10,68 |

192,6M token văn bản nguồn **lái kém hơn** train sở thích chữ cái (42,8% vs
48,9%). Không có arm nào ra ngoài dải null.

### 2. Đúng môn mà corpus được dựng cho — trùng khít null

`select_cpt_corpus.py` chọn doc theo lỗi của base, và toàn bộ 11.296 spec chunk
đi kèm. Standards specifications là môn đích:

| môn | n | base | CPT | **CPT fix%** | **NULL fix%** |
|---|---:|---:|---:|---:|---:|
| Standards specifications | 200 | 62,0 | 58,5 | **27,63** | **27,63** |
| Research publications | 450 | 76,0 | 75,1 | 31,48 | 29,63 |
| Research overview | 200 | 78,5 | 74,0 | 27,91 | 37,21 |

Trùng tới chữ số thập phân thứ hai. Và đường liều **phẳng** — Std-spec fix%:
22,37 → 25,00 → 28,95 → 26,32 → 27,63 → 26,32 → 26,32 → 27,63. Kiến thức tích luỹ
phải cho đường đi lên; đây là nhiễu quanh một hằng số.

### 3. Arm H — "transfer sạch đầu tiên" là artefact của cách chia bucket

`armH_vs_G.py` viết: *"C_hard đi 3,48 → 17,21 với 77 thắng 10 thua, trên câu nó
chưa từng thấy. Đó là transfer thật."* Tách ra:

| bucket | n | base% | base-wrong | base-right | fixed | **fix%** | broke | brk% | đi đúng hướng |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 9.848 | 80,62 | 1.909 | 7.939 | 587 | **30,75** | 926 | 11,66 | 38,8% |
| A | 5.841 | 99,62 | 22 | 5.819 | 15 | 68,18 | 247 | 4,24 | 5,7% |
| B | 2.062 | 82,25 | 366 | 1.696 | 188 | 51,37 | 458 | 27,00 | 29,1% |
| C | 1.457 | 27,93 | 1.050 | 407 | 307 | 29,24 | 211 | 51,84 | 59,3% |
| **C_hard** | **488** | **3,48** | **471** | **17** | 77 | **16,35** | 10 | 58,82 | 88,5% |

Trong C_hard base chỉ đúng **17 dòng** — nên nhiều nhất nó *có thể* phá 17. Tỉ số
"77 thắng 10 thua" là hệ quả số học của thành phần bucket, không phải của học
được gì. Tín hiệu học thật trong bucket đó là **16,35%**, tức **bằng một nửa**
mức 30,75% của chính arm H trên toàn tập. Arm H học được **ít hơn** ở đúng chỗ
người ta khen nó.

### 4. Đã có sẵn trong repo: targeting bằng không

RESULTS_9B §8e chia ot-full theo `src_id`: 7.996 dòng **có** câu train nhắm vào
vs 2.004 dòng **không** → **+0,46 vs +0,55**. Bằng nhau.

## Kết luận

Bốn họ training độc lập — CPT trên văn bản nguồn, SFT answer-key trên corpus
hoàn hảo, arm H trên 40.730 MCQ synthetic, synth14k trên 13.942 MCQ hợp lệ — và
ở mọi nơi có control nhắm được, **số hạng học bằng null**. Cái làm điểm âm không
phải drift nuốt mất kiến thức; là churn ngẫu nhiên có lợi suất thấp hơn thiệt hại
của chính nó.

**Hệ quả cho câu hỏi full-weight CPT: full-weight là biến sai.** Null của số hạng
học không phải triệu chứng thiếu dung lượng — LoRA r=64 trên toàn bộ linear layer
với 192,6M token thừa sức dịch chuyển fact. Triệu chứng là **gradient của LM loss
trên văn bản nguồn không tạo ra kiến thức trích xuất được**. Full weight khuếch
đại cả hai số hạng như nhau; nó không đổi tỉ số.

Đây chính xác là hiện tượng Allen-Zhu & Li mô tả (*Physics of LM 3.1*,
[2309.14316](https://arxiv.org/abs/2309.14316)): fact không được augment thì được
**nhớ** nhưng **0% trích xuất được**, và SFT sau đó không cứu được. Biến duy nhất
chưa từng được thay đổi trong dự án này là **số dạng diễn đạt mỗi fact** — audit
`views.jsonl` cho thấy 97,19% có `spec` rỗng và thực chất chỉ có 3 view cùng một
hình dạng task, nên liều Ovadia (~10/fact) chưa bao giờ được đưa vào.

## Phép đo mà mọi arm trước đây đã thiếu

Mọi arm đều chấm **accuracy trên benchmark**, thứ trộn lẫn học và trôi. Không arm
nào chạy phép thử tách được hai thứ đó. Phép thử đó là:

1. Chọn N fact có thể kiểm chứng, mỗi fact viết K dạng khác nhau.
2. Train. **Giữ lại một tập fact không train làm control.**
3. Đo trích xuất trên câu hỏi held-out về **đúng** những fact đã train, so với
   tập control.

Chênh lệch giữa hai tập là số hạng học, đo trực tiếp, không bị drift làm nhiễu.
Quét K = 1, 3, 10, 30 cho đường liều augmentation — đó là đại lượng chưa ai đo.
Nếu chênh lệch đó bằng không ở K=30 thì kênh weights đóng dứt điểm và không cần
chạy chiến dịch nào nữa.
