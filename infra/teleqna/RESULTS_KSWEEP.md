# Số hạng học không phải null — nó bằng null vì liều chưa bao giờ được đưa vào

*Chạy 2026-08-20 trên hgx046. Code: `runs/teleqna-8b/code/kswp/`
(`gen_facts.py`, `gen_views.py`, `probe.py`, `train_k.py`, `lenctl.py`, `ci.py`).
Tài liệu này sửa kết luận của chính [`RESULTS_DRIFT_REFUTED.md`](RESULTS_DRIFT_REFUTED.md).*

## 1. Thiết kế

Mọi arm trước đây trong dự án chấm **accuracy trên benchmark**, thứ **cộng** số
hạng học với số hạng trôi. Thí nghiệm này tách chúng bằng
difference-in-differences:

```
số hạng học = (train_sau − train_trước) − (control_sau − control_trước)
```

Drift đánh cả hai nửa như nhau nên nó tự triệt tiêu.

**Dữ liệu** — 6.000 chunk 3GPP/IEEE (corpus công khai, hợp lệ) → OTel-31B viết
`(fact, answer span, 2 câu hỏi probe)`. Sáu cổng **chuỗi**, không có LLM judge:
answer phải là substring nguyên văn của chunk gốc (fact bịa **trượt cơ học** ở
đây — 701 dòng bị loại đúng vì lý do này), phải nằm trong statement, **không**
được lộ trong câu probe, dài 3–60 ký tự. Còn **4.483/6.000 (74,7%)**.

**Lọc trước khi chia:** base Qwen3-8B trích xuất được 12,6% → bỏ, giữ **3.920**
fact base không biết. Rồi sinh 30 dạng viết lại mỗi fact, mỗi dạng phải chứa
answer span nguyên văn và không trùng lặp → **3.725 fact** đạt ≥30 dạng
(median 36). Chia **1.862 train / 1.863 control**.

**PRE = 0,00% ở cả hai nửa**, 0 generation rỗng. Nên ước lượng rút gọn thành
`DiD = train_sau − control_sau`.

## 2. Kết quả

| arm | DiD thô | 95% CI | DiD (40 ký tự đầu) | cổng format |
|---|---:|:---:|---:|---|
| K=1 | +0,51 | [−0,08, +1,13] | +0,46 | ok |
| K=3 | −0,67 | [−1,61, +0,27] | −0,80 | ok |
| K=10 | **+2,39** | [+1,00, +3,79] | +2,15 | ok |
| **K=30 @3e-5** | **+4,78** | **[+3,36, +6,15]** | **+3,28** | dài dòng |
| K=1×30 @3e-5 | +1,45 | [+0,51, +2,42] | +1,34 | ok |
| K=30 **full-weight** @1e-5 | +1,02 | [+0,11, +1,93] | +1,00 | ok |
| K=30 @1e-4 | +0,27 | — | +0,08 | **SẬP** |
| K=1×30 @1e-4 | +0,78 | — | +0,46 | **SẬP** |

### Phép so sánh quyết định

Cùng token budget, cùng số step, cùng LR — khác **duy nhất** số dạng mỗi fact:

```
K=30 dạng khác nhau   +4,78pp
K=1 lặp 30 lần        +1,45pp
--------------------------------
MULTIPLICITY EFFECT   +3,33pp   95% CI [+1,67, +5,05]   CÓ Ý NGHĨA
  (chấm 40 ký tự đầu) +1,93pp   95% CI [+0,35, +3,52]   CÓ Ý NGHĨA
```

**Số dạng mỗi fact là đòn bẩy thật, không phải số token.** Đây đúng là biến
`PLAN_8B.md` §4 ghi là biến thể duy nhất chưa bị bác bỏ, và nó chưa bao giờ được
đưa vào: audit `views.jsonl` cho thấy 97,19% có `spec` rỗng và thực chất chỉ 3
dạng cùng một hình dạng task.

### Các control đã qua

- **PRE = 0,00%** ở cả hai nửa, theo đúng construction.
- **Foreign-answer:** tỉ lệ một generation chứa answer của **fact khác** ≤ **0,13%**
  ở mọi arm. Nên hit của arm K=30 dài dòng **không** phải may nhờ độ dài.
- **Chấm lại theo 40 ký tự đầu** (cân bằng độ dài giữa arm 196 ký tự và arm 11
  ký tự): hiệu ứng vẫn còn và vẫn có ý nghĩa.
- **Cổng format** bắt được hai arm @1e-4 đã sập.

## 3. Lỗi thiết bị đo phải ghi lại

Cổng format ban đầu của tôi là "số generation rỗng = 0". **Sai.** Model mất hợp
đồng trả lời không sinh chuỗi rỗng — nó sinh **văn xuôi dài lặp lại**. Arm K=30
@1e-4 có 0 dòng rỗng nhưng mọi generation dài đúng 209 ký tự (chạm trần token)
và đọc lại cùng một câu spec bất kể câu hỏi:

```
'The UE shall not expect to detect a DCI format with a BWP indicator field...'   ← 6/6 mẫu giống hệt
```

Arm K=1×30 @1e-4 còn tệ hơn: loss 0,36, 94,2% generation >100 ký tự, chỉ **17,6%**
phân biệt nhau. Nó thuộc lòng 1.862 câu train và đọc lại chúng.

Cổng đúng là **phân bố độ dài + tỉ lệ generation phân biệt**, đã vá vào
`probe.py`. Nếu không vá, cặp so sánh quyết định đã chạy trọn trong vùng hỏng và
cho `multiplicity = −0,51pp` — ngược dấu với sự thật.

Đây là lần thứ ba độc lập hiện tượng "SFT thô trên văn xuôi phá hợp đồng trả lời"
xuất hiện, sau arm answer-key (`unparsed 10 → 96`) và CPT campaign 1.

## 4. Trả lời thẳng câu hỏi full-weight CPT

| | DiD |
|---|---:|
| LoRA r64, K=30, lr 3e-5 | **+4,78** |
| Full weight, K=30, lr 1e-5 | **+1,02** |

Hai arm **không** khớp LR nên đây không phải phép so dung lượng sạch. Nhưng nó đủ
để nói: **full weight không mua được gì**, và biến dịch chuyển số hạng học là
**K, không phải dung lượng**. Kết luận ở lượt trước giữ nguyên — full-weight là
biến sai.

## 5. Đọc con số cho đúng độ lớn

Đừng đọc +4,78 như một chiến thắng. Ở liều tốt nhất, trên **chính những fact được
train trực tiếp 30 lần bằng 30 cách viết**:

```
trích xuất được  10,31%      (fact đã train)
                  5,53%      (fact chưa train)
--------------------------------------------
vẫn không trích xuất được:   89,7% số fact đã dạy
```

So với phép đo cũ của repo: cùng loại fact **đặt trong context** đáng **+22,2**.
Kênh weights ở liều cao nhất từng chạy cho ~4,8pp trên chính fact nó vừa học.

Thêm nữa, arm K=30 **dài dòng** (p50 196 ký tự). Đưa nguyên nó vào TeleQnA sẽ phá
hợp đồng `ANSWER: X` — đúng cơ chế đã giết arm answer-key. Muốn chuyển thành điểm
benchmark thì bắt buộc có format anchor, và repo đã đo sẵn cái tốt nhất
(self-replay, +9,7 trong contrast có kiểm soát).

## 6. Kết luận

Kênh weights **mở nhưng hẹp**. Nó bằng null trong bốn chiến dịch trước không phải
vì cơ chế không tồn tại, mà vì **liều chưa bao giờ được đưa vào** — và không arm
nào chạy phép đo tách được học khỏi trôi để phát hiện điều đó.

Cái đã chứng minh được: ≥10 dạng mỗi fact tạo ra kiến thức trích xuất được, và
30 dạng tốt hơn 10. Cái **chưa** chứng minh được, và là câu hỏi tiếp theo: một
mức trích xuất ~5pp trên fact đã dạy có chuyển thành điểm TeleQnA hay không —
thí nghiệm này cố tình không đo benchmark, để giữ ước lượng sạch.
