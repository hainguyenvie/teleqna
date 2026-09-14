# Bên trong Qwen3-8B: đọc thẳng residual stream, và cái trần nó dựng lên

*Đo ngày 2026-08-20 trên `tensara-dev-0`. Toàn bộ closed-book, không RAG, không
train một dòng nào của ot-full. Code: `code/lens.py`, `code/lens_an.py`,
`code/qk.py`, `code/qk_an.py`, `code/probe.py`, `code/probe2.py` trong
`runs/teleqna-8b`.*

Tài liệu này không dùng lại chẩn đoán nào có sẵn. Mọi con số đo lại từ đầu.

## 0. Cổng kiểm chứng

Prompt là template của harness, chat-template `enable_thinking=False`, rồi
teacher-force đúng chuỗi `ANSWER:` để vị trí đọc chính là vị trí phát ra chữ cái.
Logit của layer cuối trên tập letter token cho **71,80** trên 10.000 dòng, so với
71,84 (vLLM) và 71,95 (transformers) đã ghi. Lệch dưới sàn nhiễu A/A 0,50pp →
phép đọc là trung thực, mọi số dưới đây đọc được.

## 1. Đáp án sinh ra ở đúng một layer

Logit lens qua cả 37 layer:

```
L00..L24   ~22%  (đúng bằng mức đoán)
L25        69,58        <- toàn bộ đáp án xuất hiện trong MỘT layer
L26 68,13 · L27 65,30 · L28 68,23 · L29 61,44 · L30 70,19 · L31 67,70
L32 70,34 · L33 71,54 · L34 71,76 · L35 71,94 · L36 71,80
```

24 layer đầu không mang thông tin nào về chữ cái. Từ L25 trở đi có 12 layer
"sống", và chúng dao động chứ không đơn điệu — L29 tụt 8 điểm rồi hồi lại.

**Trần cộng gộp qua layer: 83,33%** (gold là argmax ở ít nhất một layer sống).
Kiểm chứng đoán mò, theo đúng luật của repo: cùng phép tính với một đáp án SAI cố
định chỉ cho **19,00%**, nên 11,53 điểm chênh kia là tín hiệu thật, không phải may.

Nhưng không một quy tắc label-free nào chạm tới nó:

| cách gộp | điểm |
|---|---:|
| chỉ layer cuối | **71,80** |
| trung bình log-prob 12 layer sống | 72,04 |
| bỏ phiếu đa số qua các layer | 71,51 |
| trung bình log-prob 8 layer cuối | 71,64 |
| contrast kiểu DoLa, tốt nhất (final − L28) | 66,90 |

Và chỗ quyết định: trên 488 dòng layer cuối bất đồng với đa số các layer, **layer
cuối đúng 38,93% còn đa số chỉ 32,99%**. Layer cuối đã là phép đọc tốt nhất.

## 2. Selection bias không phải là bệnh — chẩn đoán cũ đọc sai vì điều kiện trên tập lỗi

`PLAN_8B.md` §1 kết luận model "đoán B/C quá nhiều và bỏ sót E", dẫn ICLR'24
token bias, và đề xuất một lớp PriDe debias. Histogram đó tính **trên tập lỗi**.
Tính trên cả 10.000 dòng:

```
gold   A 22,1  B 21,8  C 21,7  D 21,5  E 12,9
model  A 22,2  B 22,3  C 23,1  D 22,2  E 10,2
```

Gần như đã hiệu chỉnh. Lệch lớn nhất là E, 2,7 điểm phân bố — tương đương ~1 điểm
accuracy nếu sửa được hoàn hảo. **PriDe gần như không còn gì để lấy.** Điều kiện
trên tập lỗi luôn tạo ra hình dạng bias kể cả khi model không lệch.

Bias thật thì nằm ở các layer giữa và **chính model tự sửa**: L29 dự A 37,4% / E
2,4%; L31 lật sang E 23,2% / A 12,4%; các layer cuối kéo về đúng biên gold. Đó là
việc mà 11 layer cuối đang làm.

## 3. QK-score: head "select-and-copy" có thật, nhưng không thắng ở đây

Cài đặt theo [arXiv:2410.02343], có sửa cho Qwen3: GQA 32 q-head / 8 kv-head, áp
q_norm/k_norm, **không** áp RoPE. Token đại diện mỗi lựa chọn là token cuối của
dòng lựa chọn đó (Qwen gộp `\n` với dấu câu đứng trước, `.\n` / `)\n` — kiểm
100% 10.000 dòng).

Chọn head trên **synth5k** (câu sinh từ corpus, hợp lệ), chấm trên ot-full:

| | synth | ot-full |
|---|---:|---:|
| head tốt nhất chọn trên synth = **L25H28** | 75,60 | **71,01** |
| top-3 head | 75,82 | 71,12 |
| top-5 head | 76,02 | 70,79 |
| top-32 head | 75,22 | 70,12 |
| logit của chính model | 76,30 | **71,80** |

Head phổ quát là thật: tương quan thứ hạng head giữa hai tập là **r = 0,967**, và
L25H28 đứng đầu ở cả hai. Nhưng kênh này **thấp hơn logit 0,79 điểm**. Trộn vào
logit tốt nhất được +0,04 (w=0,1) — dưới sàn nhiễu.

Lý do bài báo được +7..16pp trên LLaMA2-7B mà ở đây không: phần lớn lợi ích của
họ là "model biết nhưng không phát ra được chữ cái". Qwen3-8B có **1 dòng
unparsed trên 10.000**. Phần đó ở đây vốn đã bằng không.

Kênh này *độc lập* — 304 dòng chỉ nó đúng, union 74,84 — nhưng trên 942 dòng bất
đồng, logit đúng 42,99% còn QK chỉ 32,27%. Lại đúng hình dạng cũ.

## 4. Phép thử quyết định: có phép đọc tuyến tính nào tốt hơn lm_head không?

Đây là điều kiện cần của mọi can thiệp bên trong (KAPPA, steering, gắn head mới):
nếu một ánh xạ được fit tự do trên cùng hidden state không thắng nổi lm_head đóng
băng, thì không có gì để căn chỉnh vào.

Probe **ORACLE** — fit bằng chính đáp án chuẩn của ot-full, cross-validate 5 fold.
Không nộp được, chỉ để đo trần:

| phép đọc | ot-full |
|---|---:|
| lm_head đóng băng (chính model) | **71,80** |
| tuyến tính, một layer, tốt nhất (L35) | 71,10 |
| tuyến tính, ghép hidden L25..L36 (49k chiều) | 69,05 |
| MLP-512 trên hidden ghép | 71,44 |
| tuyến tính trên letter-logit từng layer (60 chiều) | 71,90 |
| tuyến tính trên hidden + letter-logit | 69,22 |
| **MLP-256 trên letter-logit từng layer** | **72,41** |

Trần của **mọi** phép đọc nội bộ, kể cả khi được cầm đáp án, là **72,41 = +0,61**.
Dưới ngưỡng ý nghĩa của repo, và không hợp lệ.

Probe **HỢP LỆ** (fit trên synth6k rồi áp sang ot-full) chỉ đạt 58–61 ở mọi layer
— probe không chuyển được giữa hai tập, đúng như Orgad et al. (ICLR'25) mô tả.

Ghi chú trung thực: probe một-layer tốt nhất ở L25 cho **70,53**, trùng khít con
số mà campaign probe trước đã báo. Tôi vào việc với giả thuyết rằng số đó là sản
phẩm của thiết kế hỏng (prompt bị viết lại, đọc last-token, chỉ 3 layer). Tôi
dựng feature đúng vị trí sinh đáp án, đủ 37 layer, và **ra đúng cùng một số**.
Giả thuyết của tôi sai; kết luận cũ đúng, chỉ là chưa từng được chứng minh bằng
trần.

## 5. Kết luận

**83,33% nằm rải trong các layer là độ phủ, không phải tín hiệu nhận dạng được.**
Tôi chứng minh điều đó bằng cách đưa đáp án chuẩn cho một probe và nó vẫn không
lấy được: trần 72,41.

Điều này đóng lane "can thiệp bên trong" bằng một lập luận về trần, chứ không
phải bằng một chuỗi thí nghiệm thất bại:

- **KAPPA** ([arXiv:2509.23782]) cần knowledge-probe > output của model. Ở đây
  bất đẳng thức **ngược chiều** (71,10 < 71,80), nên không có subspace nào để căn.
  Ngoài ra nó cần nhãn của chính benchmark, và tác giả ghi rõ transfer chéo tập
  là yếu — bản hợp lệ ở đây đo được 58–61.
- **DoLa / layer contrast**: âm nặng, −4,9 ở cấu hình tốt nhất.
- **QK-score / select-and-copy head**: −0,79, trộn vào được +0,04.
- **PriDe / debias chữ cái**: biên đã gần đúng, dư địa lý thuyết ~1 điểm.

Và đây là lần thứ **năm** cùng một hình dạng xuất hiện trong dự án, giờ ở tầng
biểu diễn chứ không còn ở tầng lấy mẫu: *thêm một kênh độc lập thì oracle tăng,
điểm không tăng.* Generator diversity +3,04 coverage / +0,47 điểm. Option-blind
union +6,40 / +0,00. QK-score union +3,04 / −0,79. Any-layer 83,33 / +0,24.

Nút thắt không nằm trong Qwen3-8B, và không đọc ra được từ Qwen3-8B. Thứ duy nhất
từng dịch chuyển nó trong toàn bộ lịch sử dự án là **một model thứ hai độc lập**
(31B phá thế hoà trên 31,7% dòng sát nút: +3,50).
