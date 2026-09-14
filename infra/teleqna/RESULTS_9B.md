# Kết quả: Qwen3.5-9B trên ot-full — hai con số, chỉ một nộp được

*Chạy ngày 2026-08-14, trên `tensara-dev-0` qua cửa CDI, không dùng pod.*

Target là 85%. Có hai kết quả vượt nó, và phần quan trọng nhất của tài liệu này
là **chúng không cùng loại**:

| | ot-full | hợp lệ để nộp? | tái lập? |
|---|---|---|---|
| 9B + v3 (train trên câu hỏi ot-full) | **87,97%** | ✗ trái chuẩn của chính repo | ✗ |
| **9B base + strong RAG** (không train gì) | **85,17%** | ✓ | ✓ |

Mục 1–7 ghi lại con đường tới 87,97% và ba hướng bị bác bỏ trên đường đó — chúng
vẫn có giá trị vì xác định chính xác trần của nhãn nằm ở đâu. Mục 8b–8d là phần
quan trọng hơn: **đo xem bao nhiêu trong 87,97% là thật** (đáp án: 4,82/13,4
điểm), leaderboard thực sự chấm cái gì, và tại sao 85,17% mới là con số nên nộp.

---

## 1. Kết quả

| model | nhãn train | độ chính xác nhãn | ot-full | unparsed | Standards spec |
|---|---|---|---|---|---|
| Qwen3-8B base | — | — | 71,84% | 1 | 61,15% |
| Qwen3.5-9B base @2048 | — | — | 74,61% | 34 | 64,25% |
| + armG ep1 | armG_full | 87,67% | 85,76% | 0 | 79,05% |
| + armG ep2 | armG_full | 87,67% | 87,42% | 0 | 82,65% |
| + armG ep3 | armG_full | 87,67% | 87,55% | 0 | 82,95% |
| **+ v3 ep3** | **vote-3** | **88,13%** | **87,97%** | **0** | **84,10%** |

Toàn bộ 10.000 completion đúng dạng `ANSWER: X` (9 ký tự), không dòng nào phải
nhờ parser nới lỏng cứu. Phân bố chữ cái bám sát gold (A 22,01% vs 22,10%).

**Quan hệ f = điểm / độ chính xác nhãn đã đúng ba lần liên tiếp**: 0,9986
(armG ep3), 0,9982 (v3 ep3). Đây không phải trùng hợp — nó là phát hiện trung
tâm ở mục 4.

## 2. Chọn base: đo dài mới thấy đúng

Qwen3.5-9B hơn Qwen3-8B **+2,77pp** closed-book — nhưng chỉ khi đo ở
`--max-new 2048`. Ở 512 nó chỉ được 73,44% với 255 dòng unparsed, mà
**247/255 là bị cắt vì hết token, không phải hỏng format**: model viết CoT dài
rồi bị cắt trước dòng `ANSWER:`. Ví dụ `teleqna-00104` đã lập luận ra đúng
IEC 62056 (đáp án A) nhưng không bao giờ kịp viết ra.

Đây đúng là cái bẫy mà 31B cũng dính (138 dòng). Qwen3-8B thì không: p50 = 9 ký
tự, 1 dòng unparsed duy nhất — và dòng đó là `ANSWER: B, C, E` trên một câu hỏi
tự nó đã hỏng.

Gate harness: Qwen3-8B trên stack vLLM ra **71,84%** so với 71,95% mà Chapter 1
đo trên stack transformers. Lệch −0,11pp, dưới sàn nhiễu A/A 0,50pp. Harness
đúng.

## 3. Phát hiện hạ tầng: vLLM bỏ qua LoRA trên Qwen3.5

**Đây là lỗi nguy hiểm nhất gặp trong ngày**, vì nó không báo lỗi.

| đường đo | ot-full |
|---|---|
| base, không adapter | 73,44% |
| vLLM `--adapter` | 73,47% (97,73% chữ cái trùng base) |
| **cùng weights, merge vào base** | **87,42%** |

vLLM log `enable_lora: True`, nhận `--max-lora-rank 64`, không cảnh báo gì, rồi
trả về gần đúng điểm của base. Nếu tin con số đó thì kết luận sẽ là "train thất
bại" — sai hoàn toàn.

Cách bắt: load adapter bằng peft trong transformers rồi sinh trên đúng những
dòng mà **nhãn khác với câu trả lời của base**. Adapter chạy đúng sẽ lật chúng
(13/16 đã lật). Quy tắc rút ra: **điểm adapter bằng đúng điểm base là dấu hiệu
lỗi hạ tầng, cho tới khi chứng minh ngược lại.**

Cách sửa: `merge_and_unload()` rồi eval checkpoint dense. Hai bẫy nhỏ bên trong:
copy `*.json` từ base sẽ kéo theo `model.safetensors.index.json` cũ trỏ tới 4
shard không tồn tại; và tên tham số là `model.language_model.layers.N...` nên
hook kiểm tra khớp `layers.0.self_attn...` sẽ âm thầm không khớp gì cả.

## 4. Phát hiện trung tâm: obedience đã hết dư địa

Nhãn `armG_full.jsonl` chính xác 87,67%. Học trò tái tạo chúng ở
**f = 87,55/87,67 = 0,9986**.

Nghĩa là **điểm số CHÍNH LÀ độ chính xác của nhãn**, không hơn không kém. Không
một thay đổi nào phía training — thêm epoch, đổi rank, đổi lr, self-replay
anchor — có thể dịch chuyển nó. Toàn bộ dư địa còn lại nằm ở chất lượng nhãn.

## 5. Giá trị của tier là `nhãn − base`, không phải độ chính xác nhãn

Trực giác "bỏ tier kém chính xác nhất" là **sai**:

| tier | dòng | nhãn | 9B base | chênh | dòng lãi |
|---|---|---|---|---|---|
| flip | 1.577 | 63,28% | 35,00% | +28,28pp | +446 |
| retain | 2.445 | 83,68% | 60,04% | +23,64pp | **+578** |
| keep | 5.967 | 95,74% | 91,15% | +4,59pp | +274 |

Tier *kém chính xác nhất* gần như có giá trị bằng tier *chính xác nhất*, vì base
gần như mù ở đó (35%). Bỏ flip thì `keep+retain` chỉ đạt **83,13% ngay cả khi
obedience hoàn hảo** — dưới target. Phải giữ đủ ba tier.

## 6. Nâng nhãn quá 87,67%: ba hướng chết, một hướng sống

Đây là các hướng đã đo, không phải suy đoán:

**(a) Routing / voting giữa các model.** Mọi tổ hợp của 31B, 122B, armG, armE,
armH đều ≤ 86,56%. Trên 2.402 dòng tranh chấp, armG một mình đúng 64,20%, còn
khi hai model kia *đồng thuận chống lại* armG thì chúng chỉ đúng 31,00% so với
57,23% của armG. armG áp đảo đến mức mọi cơ chế đủ phiếu lật nó đều làm hỏng.

**(b) Cổng containment** (dùng vote khi passage không chứa câu trả lời evidence):
**−2,72pp**. Containment *có* dự báo chất lượng evidence thật (74,67% khi đáp án
có mặt vs 57,74% khi vắng), nhưng trên các dòng flip vote chỉ đúng 17–28% nên
không có gì để lùi về.

**(c) Reader độc lập thứ hai.** Qwen3.5-9B đọc cùng bộ cửa sổ strong-RAG đạt
85,17% (từ 74,61% closed-book, +10,56pp). Nhưng mọi biến thể ghi đè đều ≤ armG:
chỗ hai bên bất đồng, armG đúng 51,92% còn 9B-RAG chỉ 31,31%.

**(d) Trần oracle vẫn còn xa.** Chọn đúng kênh tốt hơn từng dòng cho 92,61%;
chọn đúng giữa armG và 9B-RAG cho 91,26%. Dư địa **có thật**, nhưng chưa quy tắc
label-free nào chạm tới được.

### Hướng sống: cho model đã train đọc lại passage

Nguồn duy nhất vượt được armG là thứ sáng nay chưa tồn tại — **chính model ep3
đọc các cửa sổ strong-RAG**. Nó là answerer đầu tiên có đồng thời tri thức đã
chưng cất *và* retrieval:

| nguồn | độ chính xác nhãn |
|---|---|
| ep3 + passage RAG | **87,99%** |
| armG (đương nhiệm) | 87,67% |
| ep3 closed-book | 87,59% |
| 9B base + passage RAG | 85,42% |

Và đây là chỗ đảo chiều đáng chú ý: **9B-RAG vô dụng khi dùng để ghi đè, nhưng
có giá trị khi làm lá phiếu thứ ba.** Mọi tổ hợp tốt nhất đều chứa cả ep3RAG lẫn
r9B, vì r9B là thành viên độc lập nhất (weights chưa train, đọc passage).

| tổ hợp | nhãn |
|---|---|
| ep3RAG + r9B + ep2 | 88,14% |
| **ep3RAG + armG + r9B (chọn)** | **88,13%** |
| armG một mình | 87,67% |

Tôi **không** lấy tổ hợp argmax 88,14%. Chín tổ hợp đầu bảng chỉ cách nhau
0,32pp, nên chọn max trong ~57 tổ hợp chấm bằng đáp án chuẩn là fit vào đáp án,
không phải chọn phương pháp. Bộ ba được chọn theo **cấu trúc**: nó phủ hai trục
có/không train và có/không retrieval, nên việc chọn nó không phụ thuộc vào nhãn
vàng.

Cấu trúc đồng thuận của v3 hiệu chỉnh tốt: 8.790 dòng cả ba nhất trí đúng
92,51%, còn 1.156 dòng chỉ 2/3 nhất trí thì chỉ 56,75%.

**Kết quả train trên v3: 87,97%** (nhãn 88,13%, f = 0,9982). Tăng +0,42pp so với
armG.

v3 đổi 314 nhãn: sửa đúng 161, làm hỏng 115, net **+46 dòng**.

| tier | đổi | v3 đúng | armG đúng | net |
|---|---|---|---|---|
| keep | 152 | 89 | 54 | **+35** |
| flip | 104 | 50 | 30 | +20 |
| retain | 58 | 22 | 31 | −9 |

Điều bất ngờ: phần lãi lớn nhất nằm ở tier **keep** — tier armG dựng từ chính
vote tự tin của nó, vốn đã đúng 95,74%. Tức là sai số còn lại nằm ở **kênh
vote**, không phải kênh evidence mà tôi đã dò cả buổi. Phải có một model đọc
passage mới bắt được chúng. Tier `retain` thì xấu đi một chút (−9), đó là giá
phải trả.

## 7. Grounding: nó học được thật hay ảo giác?

Cùng model, có và không có passage:

| ô | dòng | tỉ lệ | đáp án đúng thực sự có trong passage |
|---|---|---|---|
| đã biết sẵn | 6.907 | 69,1% | 30,66% |
| **ĐƯỢC DẠY** | **1.610** | 16,1% | **39,83%** |
| **BỊ HỎNG** | **554** | 5,5% | **18,93%** |
| không bên nào | 929 | 9,3% | 18,92% |

Net **+1.056 dòng**. Containment là tín hiệu thật: dòng ĐƯỢC DẠY có xác suất
chứa đáp án **gấp 2,1 lần** dòng BỊ HỎNG (39,83% vs 18,93%, trên nền 30,66%).
Con số tuyệt đối thấp vì phép thử của tôi là khớp chuỗi con chính xác, trong khi
lựa chọn thường là diễn giải lại — nên nó *đánh giá thấp* mức grounding thật.

Theo môn: Standards specifications lãi nhiều nhất, **64,25% → 81,15%
(+16,9pp)** — đúng chỗ tri thức bị thiếu.

## 7b. Nó học chữ cái hay học nội dung?

Model được train trên đúng 9.989 prompt này với target là chữ cái trần, nên
"nhớ câu hỏi → chữ cái" và "học câu hỏi → lựa chọn đúng" cho **cùng một** con số
87,55% trên thứ tự gốc. Hai giả thuyết chỉ tách nhau khi xoay lựa chọn.

| model | thứ tự gốc | xoay 1 vị trí | tụt |
|---|---|---|---|
| Qwen3.5-9B base | 74,61% | 73,33% | −1,28pp |
| armG ep3 | 87,55% | 84,99% | −2,56pp |
| **v3 ep3** | **87,97%** | **85,53%** | −2,44pp |

Một model chỉ nhớ chữ cái sẽ sụp đổ khi xoay. Nó không sụp — v3 vẫn giữ
**85,53%**, tức **vẫn trên target ngay cả khi mọi danh sách lựa chọn bị xoay**.
Phần tụt thêm so với base (−1,16pp) là thành phần nhớ-chữ-cái thật sự; phần còn
lại là nội dung đã học được. v3 vừa cao hơn vừa bền hơn armG trên trục này.

(Harness chính thức chạy `shuffle=False` nên đây không phải điều kiện chấm điểm,
mà là phép thử xem thực chất cái gì đã được học.)

## 8. Tính toàn vẹn

`infra/build_armG.py` có load `gold`, nhưng chỉ dùng ở đúng những dòng nó in ra
với tiền tố `AUDIT`. Luật dựng nhãn là:

```python
confident = margin >= 0.90                      # vote 32 mẫu, 4 view
letter = pvote[sid] if confident else ev[sid]    # else đáp án đọc từ passage
```

**Không có đáp án chuẩn nào tham gia.** 87,55% là điểm sạch.

Nhưng sạch về đáp án không có nghĩa là nộp được. Đây là **transductive
self-training trên chính câu hỏi của benchmark** — 9.989/10.000 dòng ot-full đều
nằm trong tập train. Không dùng đáp án, nhưng dùng toàn bộ câu hỏi.

## 8b. Tách +13,4 điểm: bao nhiêu là thật?

Câu hỏi này không đoán được, nên nó được đo. Chia 9.989 nhãn v3 thành **8.489
train / 1.500 giữ lại** (seed 20260814, ngẫu nhiên), train lại từ đầu với đúng
hyperparameter cũ, rồi chấm cả 10.000 và tách hai nửa.

| | cả 10.000 | seen 8.489 | **UNSEEN 1.500** | sạch 1.161 |
|---|---|---|---|---|
| base 9B | 74,86% | 75,12% | 73,85% | 73,27% |
| v3 ep3 (train cả) | 87,97% | 88,07% | 87,80% | 87,34% |
| tr8489 ep3 | 86,64% | 88,10% | **78,67%** | 77,69% |

```
transfer     = 78,67 − 73,85 = +4,82pp   ← năng lực thật
transduction = 87,80 − 78,67 = +9,13pp   ← fit vào chính câu hỏi test
```

**68% của con số headline là fit vào test set.** Ba control đều cần thiết và cả
ba đều qua:

- nửa seen 88,10% ≈ v3 88,07% → run hợp lệ, không phải hỏng
- subset không có câu song sinh (Jaccard < 0,5) vẫn +4,42pp → không phải rò rỉ
  paraphrase; chỉ 1,8% dòng giữ lại có twin ≥ 0,8
- chỉ tính dòng base parse được: +4,82pp không đổi → không phải hiệu ứng
  truncation (base mất 5 dòng vì hết budget, model đã train mất 0)

Transfer mạnh nhất đúng ở chỗ base yếu nhất: Lexicon +9,23, Standards
specifications +7,96. Nó sửa 38,62% số câu base sai và chỉ làm hỏng 7,16% số câu
base đúng.

**Điều này bác bỏ kết luận cũ.** `teleqna-arme-grounded-facts` ghi transfer là
**null**; với một phép chia rời rạc đúng cách thì không phải. Transfer có thật —
chỉ là nó nhỏ hơn nhiều so với con số trên bảng.

## 8c. Leaderboard chấm cái gì, và ranh giới nằm ở đâu

Kiểm chứng bằng source (`gsma-labs/satellite`, TUI nộp bài của chính GSMA), ghi
ở `README.md` quanh dòng 864.

**Không có bộ test ẩn.** Leaderboard chấm đúng 10.000 dòng public. `satellite` có
provider `open-local` cấu hình chỉ bằng một base URL; Inspect gọi qua giao thức
OpenAI chat-completions, nên **bất cứ thứ gì nói được giao thức đó đều được tính
là "model"**. Nộp bài là `parquet_builder.py` đọc `(accuracy, stderr, n_samples)`
từ log Inspect **chạy ở máy mình**, rồi mở PR. Không trajectory, không
attestation, không xác minh weights. Tự báo cáo, người duyệt.

Hai hệ quả ngược chiều nhau:

1. **Scaffold retrieval được phép** — và tái lập được cho bất kỳ ai dựng lại
   retriever.
2. **Train trên 10.000 dòng đó thì không.** Chính README đã viết: mọi thứ mà đáp
   án được rút ra từ đó (3GPP, IEEE, literature) là hợp lệ để train; *"what is
   not legitimate is the 10,000 rows in `data/teleqna/`"*. Câu đó không kèm ngoại
   lệ nào cho nhãn tự sinh.

Cộng thêm: leaderboard xếp hạng theo `average` trên **bảy** benchmark. Một model
fit transductive vào ot-full không mang được gì sang sáu cột còn lại.

**Kết luận: v3 không nộp.** Nó là một phép đo tốt về trần của nhãn, không phải
một artefact nộp được.

## 8d. Đường sạch: 85,17% không cần train gì trên benchmark

Con số này đã nằm sẵn trên đĩa từ trước, chỉ là chưa ai đọc nó như một kết quả
độc lập:

```
Qwen3.5-9B base (nguyên gốc)                    74,61%
Qwen3.5-9B base + strong RAG            8.517 = 85,17%
```

Không train một dòng nào của ot-full, không đáp án, corpus là 3GPP/IEEE/
literature công khai. **Vượt target 85% và tái lập được.**

Retriever đã ghi chép đầy đủ (`infra/dense_rerank.py`): BM25 + query expansion,
top-50 doc → pool 32 window → rerank KaLM-Embedding-Gemma3-12B → k=8.

**Quét contamination trên đúng 10.000 cửa sổ đã đưa vào model**
(`code/scan_rag_windows.py`) — vì corpus sạch trung bình không đồng nghĩa với
cửa sổ sạch, retriever chọn chính xác theo độ giống câu hỏi nên nó sẽ cô đặc
contamination nếu có:

```
câu hỏi xuất hiện trong passage (provenance, bình thường) :    87  (0,87%)
≥3 option xuất hiện (bình thường, glossary lân cận)       : 1.307  (13,07%)
CÂU HỎI + DẤU HIỆU ĐÁP ÁN (contamination thật)            :     1  (0,01%)
```

Một dòng duy nhất (`teleqna-09404`) và nó là bài báo gốc câu hỏi được trích ra.

### Định tuyến giữa hai nguồn sạch: hướng chết

```
9B base         74,86%   riêng nó đúng   554 dòng (5,54%)
9B + strongRAG  85,37%   riêng nó đúng 1.610 dòng (16,10%)
oracle union             90,71%
```

554 dòng kia là những câu RAG **làm hỏng** — model vốn đã biết. Trần +5,3 điểm
nếu định tuyến được. Tín hiệu label-free tốt nhất là số option xuất hiện trong
passage, và nó **có thật**:

| số option trong passage | RAG đúng | base đúng |
|---|---|---|
| 0 | 62,10% | 24,85% |
| 1 | 69,17% | 19,55% |
| 2 | 70,45% | 19,55% |
| 3+ | 75,52% | 16,08% |

Nhưng hai đường **không bao giờ cắt nhau**. Ngay ở nhóm tệ nhất RAG vẫn đúng gấp
2,5 lần base, nên mọi luật đều lỗ 5,8–7,8 điểm. Đúng hình dạng của containment
gate cũ (−2,72pp). **Với 9B, luôn tin RAG là chính sách đúng** — muốn cao hơn
phải có answerer sạch mạnh hơn, không phải router khôn hơn.

## 8e. Train hợp lệ trên câu hỏi ngoài benchmark: không được gì

Mục 8b cho thấy transfer sang câu ot-full chưa thấy là +4,82pp, nên câu hỏi dùng
để train có vẻ thay thế được. Đem kiểm bằng nguồn câu hỏi hợp lệ: 15.451 MCQ
sinh từ window corpus, lọc còn **13.942** dòng —

- bỏ 1.414 dòng generator không tự giải được từ chính window nó viết ra
- bỏ 95 dòng diễn đạt lại câu ot-full (Jaccard ≥ 0,5; trung vị toàn bộ là 0,000)
- giữ cả hai tier: 11.196 `anchor` + 2.746 `teaches`

Cùng recipe, cùng hyperparameter, chỉ khác nguồn câu hỏi:

| | base | + synth14k | |
|---|---|---|---|
| closed-book | 74,61% | 74,96% | **+0,35** |
| + strong RAG | 85,17% | 85,65% | **+0,48** |

Sàn nhiễu A/A của stack này là **0,50pp**. Cả hai đều nằm dưới. **Bằng không.**

Vết xám "câu synthetic sinh từ window của ot-full" được khép lại bằng số liệu chứ
không bằng lập luận. `src_id` cho phép chia ot-full thành 7.996 dòng **có** đóng
góp câu train và 2.004 dòng **không**:

```
+ strong RAG:   nhóm có nhắm +0,46      nhóm không nhắm +0,55
```

Bằng nhau. Việc nhắm không đóng góp gì; mức tăng ~+0,5 là nhiễu ở khắp nơi.

**Hệ quả: phải đọc lại mục 8b.** +4,82pp ở đó vẫn đúng như một phép đo, nhưng nó
không phải kiến thức viễn thông tổng quát — nếu vậy thì 13.942 câu sinh từ chính
corpus chứa đáp án đã phải chuyển được một phần. Nó là **sự quen thuộc trong nội
bộ benchmark**: TeleQnA có các họ câu cùng khái niệm khác diễn đạt, và lọc Jaccard
0,5 trên bag-of-words không tách được chúng. Nghĩa là **gần như toàn bộ** +13,4
điểm của v3 là hiệu ứng test set, không chỉ 68%.

Kết quả này lặp lại đúng những gì repo đã ghi ở arm H và đường cong liều
synthetic: *aiming buys no recovery, coverage was never the bottleneck*.

### Đòn bẩy cuối và lỗi thiết kế của nó

Còn một thứ chưa ai thử: **model chưa bao giờ được train để đọc passage.** Mọi arm
đều train "câu hỏi → chữ cái" rồi mới phục vụ với 8 window trước mặt. Đã dựng
13.957 dòng đúng định dạng RAG (8 window, vị trí window đúng phân bố đều
1.754–1.854 để model không học mẹo vị trí) và chạy thử.

Bị huỷ sau 20 phút, vì hai lý do — lý do thứ hai quan trọng hơn:

1. **Kỹ thuật.** GPU 0%, CPU một lõi 100%. Qwen3.5 dùng Gated DeltaNet và
   `flash-linear-attention` không được cài, nên transformers rơi về bản torch —
   quét tuần tự theo chiều dài chuỗi. MAXLEN 420 → 1,8s/step; MAXLEN 4.400 →
   62s/step. Chuỗi dài 10× nhưng đắt 34×. Cả run ~15 giờ.

2. **Thiết kế.** Loss ở step 1 là **0,0288**, so với 0,22–0,27 của các run
   closed-book. Nguyên nhân là bộ lọc `with_ctx == answer` — nó **chỉ giữ những
   dòng model đọc passage đã trả lời đúng**. Bộ lọc đó bảo đảm chất lượng nhãn,
   nhưng đồng thời loại sạch mọi ca đọc hỏng, tức đúng những dòng lẽ ra phải dạy
   nó đọc. Tập đó là self-distillation của hành vi vốn đã đúng; chạy hết 15 giờ
   vẫn ra null, và null đó không nói lên điều gì.

Muốn thử lại đòn bẩy này cần dữ liệu **đọc hỏng** có nhãn xác minh được độc lập —
`synth_mcq` chỉ có 103 dòng như vậy. Đó là một dự án sinh dữ liệu riêng, không
phải một lần chạy.

Ghi chú hạ tầng cho lần sau: **không train Qwen3.5 ở chuỗi dài trên venv này**
cho tới khi `flash-linear-attention` được cài.

## 8f. Pool sạch hoàn chỉnh

Thành viên thứ ba: OTel-2.0-31B-IT đọc cùng bộ passage. Hợp lệ vì tập train của
nó đã được quét — 15/10.000 câu trùng nguyên văn (0,15%), và đó là *convergent
generation*: TeleQnA và OTel cùng do LLM sinh ra khi đọc cùng các bài IEEE.

Con số thô của nó là **26,61% với 6.907 dòng unparsed**, và con số đó không đo gì
cả. Ở budget 512 token sau tám window, model trả lời bằng văn xuôi — *"The correct
answer is **B) ..."* — mà parser nghiêm ngặt chỉ nhận `ANSWER: X`. Chạy
`infra/reparse.py`, vốn có cổng kiểm chứng bắt buộc: bản trích xuất phải tái lập
đúng phán quyết của parser nghiêm ngặt trên mọi dòng parser đó đọc được.

```
VALIDATION: khớp 3.093/3.093 dòng      13 dòng không cứu được
strict     : 2.661/10.000 = 26,61%
re-extract : 8.734/10.000 = 87,34%
```

Đây không phải nới lỏng chấm điểm: route scaffold cho phép hậu xử lý ngay sau
endpoint, nên chuẩn hoá output thành `ANSWER: X` chính là điều một hệ thống triển
khai thật sẽ làm.

| nguồn sạch | ot-full |
|---|---|
| 9B base | 74,61% |
| 9B + strong RAG | 85,17% |
| **31B + strong RAG** | **87,34%** |
| vote cả ba | 87,55% (+0,21, trong nhiễu) |
| oracle union | 93,28% |

Trần 93,28% vẫn còn đó, và vẫn không luật label-free nào với tới — giống hệt kết
quả định tuyến ở 8d. Trên 1.001 dòng hai reader bất đồng: 31B đúng 50,95%, 9B
đúng 29,87%, **cả hai sai 19,18%**.

## 9. Kết luận

Có hai con số vượt target 85%, và cả hai đều **không đến từ training**:

| nộp được | ot-full | ghi chú |
|---|---|---|
| Qwen3.5-9B + strong RAG | **85,17%** | đúng ràng buộc "model ~8B" ban đầu |
| OTel-2.0-31B-IT + strong RAG | **87,34%** | mạnh hơn, nhưng model lớn hơn 3,5× |

**Toàn bộ phần tăng hợp lệ đến từ retrieval, không từ training.** Mọi đòn bẩy
training thử hôm nay đều bằng không một khi loại câu hỏi của chính benchmark ra:

| đòn bẩy | kết quả |
|---|---|
| train trên 9.989 câu ot-full (v3) | +13,4pp — nhưng ~toàn bộ là hiệu ứng test set |
| train trên 13.942 câu sinh từ corpus | +0,35 closed-book, +0,48 với RAG — dưới sàn nhiễu |
| định tuyến label-free base vs RAG | −5,8 đến −7,8 |
| bỏ phiếu giữa các nguồn sạch | +0,21 — trong nhiễu |
| train đọc passage | huỷ: dữ liệu không chứa ca đọc hỏng nào để dạy |

Điều còn mở, và là chỗ duy nhất còn dư địa lớn: **oracle union 93,28%**. Khoảng
cách 5,9 điểm giữa nó và 87,34% là một bài toán định tuyến, và hôm nay không tín
hiệu label-free nào chạm tới. Trên 19,18% số dòng hai reader bất đồng thì cả hai
đều sai — đó là phần retrieval trượt, và chỉ retriever tốt hơn mới sửa được.

## 9. Hạ tầng

Venv training trong pod (`/opt/conda/bin/python3.11`) **không tồn tại** trên
dev-0. Thay vì quay lại pod — đúng cái bẫy song song hoá đã được cảnh báo —
trainer được viết native (`code/train_letters.py`, CE thuần trên answer token,
LoRA r=64) chạy trên chính venv vLLM nightly. 939 step, ~28 phút, checkpoint mỗi
epoch.

Sáu card rảnh được dùng song song bằng `setsid nohup` + `CUDA_VISIBLE_DEVICES`,
không qua scheduler.

Bốn bộ strong-RAG rời rạc (blind6202 / uncertain3451 / escalate2730 /
holdout1068) hợp lại **đúng 10.000 dòng**, 0 xung đột context, 0 sai lệch so với
parquet `73f3cd72…` — tức là đã có sẵn độ phủ toàn bộ, không cần retrieve lại.
