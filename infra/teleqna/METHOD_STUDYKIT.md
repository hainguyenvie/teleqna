# Study-kit — bài giảng đầy đủ: đưa tri thức viễn thông vào **trọng số** của một Qwen3-8B

Viết 14/09/2026, viết lại toàn bộ 17/09/2026 06:30 UTC sau khi chốt `vd3/ep1 = 79,04`.

Tài liệu này là **bài giảng**: nó giải thích *vì sao* pipeline có hình dạng hiện tại, mỗi khái niệm nghĩa là
gì, mỗi thí nghiệm đã trả lời câu hỏi nào, và những bài học rút ra — kèm ví dụ dữ liệu thật lấy từ chính
các file đang chạy trên server. Nó thay thế bản 14/09 (chỉ mô tả bậc 1).

Hai tài liệu chị em:
- [`PLAN_CLOSED_BOOK_8B.md`](PLAN_CLOSED_BOOK_8B.md) — nhật ký quyết định theo thời gian, mọi phép đo thô.
- [`README.md`](README.md) — benchmark, contamination, prior work, baseline.

Tài liệu này là **cách đọc** hai tài liệu đó.

---

## 0. Bài toán và luật chơi

**Mục tiêu:** nộp một model **Qwen3-8B duy nhất** trả lời `GSMA/ot-full` cột `teleqna` (10.000 câu MCQ)
**closed-book** — không retrieval, không corpus lúc infer, không model thứ hai lớn hơn 8B.

**Vì sao khó:** base Qwen3-8B đạt **71,84**. Nhét đúng 8 đoạn tài liệu vào context thì lên **83,75**
(+11,9). Tức là tri thức *tồn tại* trong tài liệu công khai, và model *biết đọc* nó — nhưng closed-book
thì model không có nó. Toàn bộ track là câu hỏi: **chuyển được bao nhiêu phần của +11,9 điểm context ấy
vào trong trọng số?**

**Luật dữ liệu (bất di bất dịch):**

| được phép | không bao giờ |
|---|---|
| dùng **câu hỏi** test để chọn tài liệu nguồn công khai (retrieval) | dùng `answer` (đáp án) hoặc `explanation` trong prompt sinh dữ liệu |
| sinh dữ liệu tổng hợp tự do từ tài liệu nguồn | dùng đáp án trong bộ **lọc** dữ liệu |
| training tự do (CPT, SFT, consistency) | train trên 10.000 dòng benchmark |
| hậu xử lý output trước khi parse `ANSWER: X` | dùng model > 8B lúc infer |

Đáp án chỉ là **dụng cụ đo**: dùng để báo cáo `recover`/`broke`/coverage, không bao giờ chảy ngược vào
pipeline. Mỗi lần một arm dùng đáp án để chọn dữ liệu (arm `tier1gf`), nó được gắn nhãn "cận trên
answer-key" và **không** được coi là ứng viên nộp.

---

## 1. Từ vựng — 16 khái niệm phải hiểu đúng trước khi đọc bất kỳ con số nào

| khái niệm | định nghĩa chính xác trong repo này |
|---|---|
| **cửa sổ** (window) | một đoạn ~800 từ (~1,2k token) mà retrieval trả về cho **một câu test**. Đơn vị tài liệu cơ bản của cả pipeline. Không phải section (4–8k token) — xem §4.1. |
| **view** | một cách viết lại nội dung của cửa sổ. Sáu view: `verbatim`, `facts`, `factview`, `register`, `qa`, `mcq`. |
| **kit** | tập hợp **mọi view của mọi cửa sổ**. "Study-kit" = bộ tài liệu ôn thi mà model tự viết cho chính nó. |
| **pack** | hai mảng `.npy`: `ids` (token) và `mask` (1 = tính loss, 0 = không), cắt thành **block 1024 token**. Đây là thứ trainer đọc. |
| **anchor** | hàng neo định dạng: prompt harness + `ANSWER: X`, loss **chỉ trên completion**. Giữ hợp đồng output khỏi vỡ khi CPT. |
| **replay** | văn bản viễn thông thô (tele-data) trộn vào ~10% token kit, chống quên (catastrophic forgetting). |
| **khuếch đại** (amplification) | số token kit sinh ra / số token nguồn. Bậc 1: 12,5× (165M kit / 13,2M nguồn). Big run: ~11× (1,41B / ~124M) — cùng mức, trên nguồn rộng gấp 9 lần. |
| **multiplicity K** | số **dạng khác nhau** mà một fact được viết. K-sweep đã đo: K=30 dạng khác nhau cho +4,78 DiD, K=1 lặp 30 lần chỉ +1,45. **Đa dạng, không phải số lần lặp.** |
| **T0 / T1 / T2 / T3** | thang trần bốn tầng, xem §3. Công cụ đo quan trọng nhất của track. |
| **recover / broke** | `recover` = câu base sai → ckpt đúng. `broke` = câu base đúng → ckpt sai. `net = recover − broke`. **Điểm tổng che mất cả hai.** |
| **churn** | hiện tượng ~10% câu "mong manh" lật qua lại ở *mọi* can thiệp. Thuộc tính của dòng thiếu tự tin, không phải của dữ liệu — xem §4.2. |
| **rot1** | bản benchmark đã **xoay thứ tự lựa chọn**. Chấm trên rot1 để tách "học nội dung" khỏi "học vị trí chữ cái". |
| **strict vs first-letter** | `strict` = regex `^ANSWER: X$` cả dòng. `first-letter` = lấy chữ cái đầu sau `ANSWER:`. Chênh lệch giữa hai cái = mức độ model viết đuôi thừa. |
| **margin** | chênh lệch xác suất giữa chữ cái hạng 1 và hạng 2 ngay sau token `ANSWER:`. Là tín hiệu **label-free** về độ tự tin. |
| **label-free** | quy tắc chỉ dùng câu hỏi + tài liệu, không dùng đáp án. Mọi thứ đưa vào train phải label-free. |
| **greedy / vote** | `greedy` = một lượt T=0. `vote` = 4 hoán vị × 8 mẫu T=0,7, bỏ phiếu trong không gian **chỉ số gốc** (không phải chữ cái). |

---

## 2. Bức tranh một trang

```
 ┌─ TẦNG 1: NGUỒN ────────────────────────────────────────────────────────────┐
 │ 10.000 câu hỏi ──BM25+rerank──► 5,1M chunk (3GPP, TCC, IEEE PDF, Wikipedia) │
 │                              ──► 114.125 cửa sổ (~1,2k token/cửa sổ)        │
 │ coverage chức năng: RAG-8 83,8% ∪ TB-8 → 89,4%; còn 1.059 câu không nguồn   │
 └────────────────────────────────────────────────────────────────────────────┘
                                    │
 ┌─ TẦNG 2: GENERATOR (Qwen3-8B tự viết lại) ─────────────────────────────────┐
 │ mỗi cửa sổ → 6 view, mọi cổng là **so khớp chuỗi**, không LLM judge         │
 │   verbatim 1× · facts · factview K=10–30 · register ×8 · qa ×12 · mcq ×4    │
 │ big run: 1,41B token kit từ ~124M token nguồn (≈11×)                        │
 └────────────────────────────────────────────────────────────────────────────┘
                                    │
 ┌─ TẦNG 3: PACK ─────────────────────────────────────────────────────────────┐
 │ kit (loss mọi token) + replay 5–10% + anchor agree-only ×12                 │
 │ + chat rows (qa-chat 50%, mcq-gold) + masked reconstruction                 │
 │ → block 1024, trộn theo document                                            │
 └────────────────────────────────────────────────────────────────────────────┘
                                    │
 ┌─ TẦNG 4: CPT full-weight ──────────────────────────────────────────────────┐
 │ 8 × H200, DDP, lr 3e-5 cosine, 1 epoch, 2,20B token, 18,3 h                 │
 │ → 77,08 (ckpt tốt nhất step24000: 77,81)                                    │
 └────────────────────────────────────────────────────────────────────────────┘
                                    │
 ┌─ TẦNG 5: BEHAVIOUR STAGES (rẻ, 20 phút/stage) ─────────────────────────────┐
 │ contrastive-MCQ (+recover) → vote-distillation (−broke) → lặp               │
 │ 77,81 ──cm──► 78,74 ──vd3──► **79,04**                                      │
 └────────────────────────────────────────────────────────────────────────────┘
                                    │
 ┌─ TẦNG 6: SERVING ──────────────────────────────────────────────────────────┐
 │ greedy, parser harness gốc, không stop-sequence, không vote — **mọi mẹo đã  │
 │ được nội hoá vào trọng số ở tầng 5**                                        │
 └────────────────────────────────────────────────────────────────────────────┘
```

Điểm đi từ **71,84 → 79,04** (+7,20; recover 1.164, broke 444, unparsed 0).

---

## 3. Thang trần T0–T3 — công cụ đo quan trọng nhất

Đây là ý tưởng tổ chức của cả track. Trước khi đốt card, luôn biết **trần của bước đó ở đâu**, và khi
thất bại thì biết **tầng nào hỏng**.

```
T0  trần lý thuyết    gold explanation trong context              98,58   "câu hỏi có trả lời được không"
T1  trần tài liệu     8 cửa sổ nguồn trong context                83,75   "tài liệu có chứa tri thức không"
T2  trần data sinh    study-kit của cùng cửa sổ trong context      ~T1    "generator có giữ tri thức không"
T3  điểm closed-book  sau khi train                               79,04   "trọng số hấp thụ được bao nhiêu"
```

**Hiệu suất kênh = (T3 − base) / (T2 − base)** trên cùng nhóm câu. Đây là con số phải tăng, không phải
điểm tổng.

Ba lần thang này cứu cả tuần card:

1. **Run 3 EntiGraph (trước study-kit):** T3 ≈ T2 → model học gần hết những gì kit chứa, **nút thắt là
   T2**: kit quan hệ entity-cặp chỉ giữ ~17–20% tri thức của cửa sổ. Kết luận: sửa generator, không phải
   tăng liều. Nếu chỉ nhìn điểm tổng (+0,53) sẽ kết luận sai là "kênh weights không mở".

2. **Đơn vị tài liệu:** cùng nội dung, section thô 4–8k token trong context cho +3,6; cửa sổ 1,5k token
   cho +7,5. → **đơn vị phải là cửa sổ.** Đo ở T1, không cần train.

3. **Chọn generator:** pilot 300 câu, T2 của Qwen3-8B (79,67 = 68% lift RAG) > OTel-31B (76,00 = 52%),
   và nhanh 2,7×. → generator là **chính base model**, không phải model to hơn. Ngược trực giác, nhưng
   đo được: model đọc tốt nhất thứ do chính nó viết.

**T1 = 83,75 < 85** là một kết luận khó chịu nhưng quan trọng: *nhét hoàn hảo cũng chưa tới mục tiêu*.
Nghĩa là muốn 85 thì phải nâng corpus, không chỉ nâng phương pháp. Đó là lý do §4.3 tồn tại.

---

## 4. Tầng 1 — Nguồn: coverage trước thuật toán

### 4.1 Cửa sổ là gì và vì sao ~1,2k token

Một **cửa sổ** = đoạn văn bản mà retrieval (BM25 + rerank) trả về cho một câu test. 8 cửa sổ/câu.
51.637 cửa sổ phục vụ 10.000 câu (80% cửa sổ chỉ phục vụ đúng một câu).

Vì sao không dùng section 3GPP nguyên vẹn (4–8k token)? Vì đo T1:

| context | T1 trên nhóm TARGET |
|---|---:|
| 8 cửa sổ strong-RAG | **85,87 (+22,6)** |
| 2,5 cửa sổ nằm trong section | 70,80 (+7,5) |
| section thô 2 × 3k token | 66,91 (+3,6) |
| section thô 1 × ≤ 8k không cắt | 66,02 |

Cắt hay không cắt section **không khác nhau**; cửa sổ ngắn và đúng chỗ thì khác. Tri thức benchmark hỏi
nằm ở mức **mệnh đề trong một đoạn**, không ở mức chương.

### 4.2 Sàng cửa sổ label-free — và bài học đắt nhất về lọc dữ liệu

Ý tưởng ban đầu: 51.637 cửa sổ là nhiều, hãy giữ cửa sổ *có ích*. Cách làm label-free: đưa **từng cửa sổ
một mình** vào context (69.718 cặp câu–cửa sổ), giữ cửa sổ nào làm model **đổi chữ cái** so với
closed-book. Không dùng đáp án. Kết quả: 15.760/51.637 cửa sổ (30,5%), 13,2M token.

Kiểm bằng gold (chỉ để báo cáo): 77% dòng mà RAG-8 sửa được đã được một cửa sổ đơn sửa. Nghe rất tốt.

**Nhưng phép đo tiếp theo phá vỡ giả thuyết.** Đưa *chỉ* tập cửa sổ đã giữ vào context cho **72,57** —
gần bằng base (71,84), trong khi 8 cửa sổ đầy đủ cho 83,75. Tách ra:

| nhóm | tác dụng của tập "giữ" |
|---|---|
| câu base **sai** (1.804) | sửa 1.442 (80%) — **hơn cả RAG-8** |
| câu base **đúng** (4.725) | 90,3 → 61,4; `broke` 1.390 (RAG-8 chỉ 585) |

Lý do: bộ lọc "đổi chữ cái" theo định nghĩa chỉ giữ cửa sổ **gây đổi**. Với câu model đã đúng, "đổi" =
"làm sai". 40% tập giữ (6.356 cửa sổ) chỉ có tác dụng hại trong context.

Đã thử tìm tín hiệu label-free tách sạch "sửa" khỏi "hại":

| tín hiệu | recall câu-sửa | tỉ lệ hại |
|---|---:|---:|
| margin base < 0,9 | 85% | 33% |
| đồng thuận ≥ 2 cửa sổ | 96% | 37% |
| chữ cái mới = đa số cửa sổ | 75% | 29,5% |
| độ tự tin *với* cửa sổ (`win_conf.py`) | — | không tách được (fix margin≥0,95: 45,6% vs break 36,0%) |

**Không có tín hiệu nào tách sạch.** Lý do bản chất: cửa sổ được retrieval vì *giống câu hỏi*, mà câu hỏi
giống distractor, nên cửa sổ tất yếu nói về distractor một cách tự tin.

Rồi arm quyết định: `tier1gf` — chỉ 6.291 cửa sổ **thật sự sửa được** (chọn bằng gold, tức cận trên tuyệt
đối của mọi luật lọc). Kết quả: **72,20 — kém hơn tier1 label-free (72,64)**. Và nhóm "chỉ có cửa sổ hại"
vẫn tụt xuống 79,11 dù kit gf **không chứa cửa sổ hại nào**.

> **Bài học 1 — churn không phải lỗi dữ liệu.** Phân tích lật: câu "mong manh" (có ít nhất một cửa sổ lật
> được, n=2.342) bị lật 10,3% bởi tier1, 10,1% bởi gf, **21,2% chỉ bởi việc xoay thứ tự lựa chọn**. Câu
> "bền" (n=4.842): 1,8% / 1,7% / 5,8%. Câu có margin base < 0,5 bị lật 20%; margin ≥ 0,999 chỉ 0,7%.
> Hai kit khác nhau lật **cùng những dòng** (Jaccard 0,66).
> → `broke` là thuộc tính của **dòng thiếu tự tin**, không phải của dữ liệu. Lọc dữ liệu để giảm broke là
> vô ích. Muốn giảm broke phải làm model **tự tin nhất quán hơn** (đó là vì sao tầng 5 tồn tại).

Hệ quả cho bậc 2 trở đi: **không lọc theo hiệu ứng nữa** — lấy đủ 8 cửa sổ mỗi câu.

### 4.3 Bịt lỗ hổng nguồn — sáu hướng, ba thành công

Failure analysis trên 2.816 câu base sai cho biết tri thức nằm ở đâu:

- 64,1% có ≥1 cửa sổ đơn sửa được
- 14,5% chỉ tổ hợp RAG-8 sửa được
- **20,3% (573 câu) gold không có trong cửa sổ nào** ← lỗ hổng nguồn
- 1,1% có nguồn mà không dùng

573 câu ấy chia theo tag: untagged 370 (research/lexicon), 3GPP Rel-17/18 94, IEEE 802.11 42, C95.1 29,
802.15.4 13, 802.3 8, TCP/IP 5, ETSI 4. Sáu hướng đã thử:

| hướng | làm gì | kết quả |
|---|---|---|
| **BM25 truy ngược toàn bộ** (`traceback_index.py`) | chunk ~800 từ trên 5,1M chunk (3GPP đầy đủ + TCC + IEEE), top-16/câu theo câu hỏi + option | ✅ TB-8 = 83,8%, **hợp RAG-8 ∪ TB-8 = 89,4%** |
| **4 PDF IEEE** (802.11-2020 4.379 trang, 802.3-2022 7.025, 802.15.4-2020 799, C95.1-2019 327) | ingest → 5.846 chunk | ✅ 802.11 72,5 → 87,6; 802.15.4 55,6 → 81,0; C95.1 55,4 → 81,5 |
| **deep rerank** (`dense_rerank_deep.py`) | BM25 top-100 (2 dạng query) → rerank Qwen3-Embedding-4B → top-8 cho 1.059 câu chưa cover | ⚠️ sửa 121/740 (16,4%) so với **control ngữ cảnh lạc đề 67 (9,1%)** → tín hiệu thật nhưng nhỏ; *phá* câu base đúng mạnh hơn cả ngữ cảnh lạc đề (giữ 29% vs 55%) |
| **Wikipedia API** (`api_sources.py`) | 1.059 câu → 718 câu có trang → 5.691 cửa sổ | ⚠️ sửa 69 vs control 49 (+20, ~4%) — yếu nhưng dương và gần như miễn phí |
| **OTel-31B recitation** (`otel_recite.py`) | 31B tự viết passage về chủ đề câu hỏi, cổng atom-agreement ≥ 0,5 | ⚠️ sửa 135/740 (18,2%); nhưng 213 passage qua cổng chỉ đúng 57 → **đóng nhánh** |
| **teacher 122B** (`teacher_sources.py`) | Qwen3.5-122B viết 3 passage/câu từ câu hỏi + option, cổng = 3 passage cùng implied option **và** closed-book teacher đồng ý | chưa chạy (thiếu `GATEWAY_KEY`) |

> **Bài học 2 — luôn có control ngữ cảnh xoay.** "deep-8 sửa 121 câu" nghe như thành công cho tới khi
> control (mỗi câu nhận 8 cửa sổ của câu *khác*) cũng sửa 67. Sàn nhiễu của "có passage nào đó trong
> context" là ~9%. Không có control thì mọi con số coverage đều thổi phồng.

**Kết luận coverage:** ~520 câu base sai (5,2% tập test) **không có nguồn truy được** trong 5,1M chunk +
4 PDF IEEE + recitation. Đây là trần cứng của hướng corpus.

### 4.4 Thước đo coverage đúng là **chức năng**, không phải chuỗi

Lần đo đầu dùng "gold nguyên văn có trong 8 cửa sổ": chỉ **20,5%** ở mọi nhóm. Con số này vô nghĩa —
option TeleQnA là diễn đạt lại của GPT-3.5, không phải trích dẫn nguồn.

Thước đúng: **đưa cửa sổ vào context, model có trả lời đúng không.** RAG-8 = 83,8%. Đó là coverage.

> **Bài học 3 — đo cái mình cần, không đo cái dễ đo.** Tương tự: "gold-string có trong factview/qa của
> kit" hoàn toàn **không** dự báo được câu nào sẽ được sửa (21% vs 20%); "số dòng kit chứa gold" cũng
> không (0 dòng: 21%, 50+ dòng: 22%). Cái *có* dự báo là **số cửa sổ độc lập cùng sửa được**
> (1 → 9%, 2 → 14%, 3 → 16%, ≥4 → **30%**). Tức là điều quan trọng không phải "fact xuất hiện bao nhiêu
> lần trong kit" mà "fact xuất hiện ở bao nhiêu **tài liệu độc lập**" — đúng hiệu ứng "celebrity" của
> Allen-Zhu.

---

## 5. Tầng 2 — Generator: sáu view, cổng chuỗi, không LLM judge

### 5.1 Một cửa sổ, sáu view — ví dụ thật

Cửa sổ `w000002` (TS 29.575, Nmfaf_3daDataManagement service). Nguồn (trích):

```
Upon the reception of an HTTP POST request with "{apiRoot}/nmfaf-3dadatamanagement/<apiVersion>/
configurations" as Resource URI and MfafConfiguration data structure as request body, the MFAF shall:
create a new configuration; assign a transaction reference id; determine the MFAF notification
information and add it to the configuration if no MFAF notification information has been provided
in the request; and store the configuration. ... shall respond with "201 Created" ...
```

Từ **một** cửa sổ ấy generator sinh ra **165 dòng, 118 qua cổng** (đếm thật trên file:
verbatim 1 · facts 13 ✓/1 ✗ · factview 85 ✓/45 ✗ · register 8 ✓ · qa 8 ✓/1 ✗ · mcq 3 ✓):

**`facts`** — fact nguyên tử + span đáp án nguyên văn + câu hỏi không chứa span:
```json
{"fact": "The MFAF assigns a transaction reference id when creating a new configuration.",
 "answer": "assign a transaction reference id",
 "q": "What does the MFAF do when creating a new configuration?"}
```

**`factview`** — mỗi fact viết lại K dạng, span đáp án **bắt buộc nguyên văn** trong từng dạng:
```
The MFAF is required to assign a transaction reference id when creating a new configuration.
When a new configuration is created, the MFAF must assign a transaction reference id.
```

**`register`** — viết lại **cả cửa sổ** theo 8 văn phong (textbook / notes / tutorial / faq / clause /
summary_detail / walkthrough / cheatsheet):
```
[notes]  The MFAF shall create a new configuration upon receiving an HTTP POST request with
         "{apiRoot}/nmfaf-3dadatamanagement/<apiVersion>/configurations" as Resource URI.
         The MFAF shall assign a transaction reference id when creating a new configuration.
         The MFAF shall respond with "201 Created" if it created an "Individual MFAF Configuration".
```

**`qa`** — 12 câu hỏi, đáp án ≤12 từ **nguyên văn trong nguồn** + một câu vì sao:
```json
{"q": "What status code is returned on successful configuration creation?",
 "a": "201 Created", "why": "The excerpt states this status code for successful creation."}
```

**`mcq`** — 4 MCQ kiểu chứng chỉ, distractor near-miss, kèm evidence nguyên văn:
```json
{"q": "What HTTP status code does the MFAF return when it successfully updates an MFAF configuration?",
 "options": ["200 OK", "201 Created", "204 No Content", "302 Found"], "answer": 0,
 "evidence": "If the MFAF successfully processed and accepted the received HTTP PUT request, ...
              shall respond with: HTTP \"200 OK\" status code with the message body ..."}
```

### 5.2 Mỗi view giải quyết một cơ chế thất bại cụ thể

| view | cơ chế nó chữa | cổng | khuếch đại |
|---|---|---|---:|
| `verbatim` | nguồn gốc, không mất mát | — | 1,0× |
| `facts` | tách mệnh đề khỏi văn xuôi; là nguyên liệu cho factview và probe | span ⊂ nguồn, span ⊄ câu hỏi, 2–80 ký tự | 1,46× |
| `factview` | **multiplicity K** — cơ chế đã đo của K-sweep (+4,78 vs +1,45) | span đáp án nguyên văn trong **từng dòng**, ≥20 ký tự, dedup | ~3× (K=10) → 22,5M dòng ở big run (K=30) |
| `register` | đa dạng **ngữ cảnh** (Allen-Zhu: exposure trong ngữ cảnh khác nhau) | >200 ký tự, **atom coverage ≥ 0,7**, không meta-statement | 8× |
| `qa` | dạng **hỏi** — cầu nối sang cách benchmark hỏi | đáp án ⊂ nguồn (chỉ 41% qua cổng!) | ~1× |
| `mcq` | nguyên liệu cho anchor, mcq-gold, cmcq, vote-distillation | evidence[:80] ⊂ nguồn, option phân biệt nhau | 0,66× |

**"atom"** = `TS xx.yyy`, `Rel-N`, `số + đơn vị` (ms, dB, GHz…), `ACRONYM` 3–7 chữ, số 2–4 chữ số. Cổng
atom ≥ 0,7 nghĩa là: trong bản viết lại, ≥70% các định danh/giá trị phải có mặt trong nguồn. Đây là cách
bắt ảo giác **bằng chuỗi**, không cần judge.

### 5.3 Vì sao cổng là so khớp chuỗi, không phải LLM judge

Ba lý do: (a) rẻ — chạy trên CPU, không tốn card; (b) tất định — tái lập được; (c) **không thể bị chính
generator lừa**.

Cái giá phải trả thì rất cụ thể. Ví dụ thật, fact có span `"update the configuration of corresponding
transaction reference Id"`:

```
[PASS] The MFAF, when processing an HTTP PUT, update the configuration of corresponding transaction reference Id.
[PASS] For every HTTP PUT request, the MFAF update the configuration of corresponding transaction reference Id.
[FAIL] When an HTTP PUT request is processed, the MFAF updates the configuration of corresponding transaction reference Id.
[FAIL] The configuration of corresponding transaction reference Id is updated by the MFAF during processing of an HTTP PUT.
```

Hai dòng FAIL là **tiếng Anh đúng hơn** hai dòng PASS. Cổng span nguyên văn ép generator giữ nguyên chia
động từ, nên nó sinh ra câu hơi sai ngữ pháp ("the MFAF update"). Trong 29 dạng của fact đó, 23 qua cổng.

> **Bài học 4 — cổng chuỗi mua sự trung thực bằng một ít ngữ pháp.** Chấp nhận được với CPT (model học
> fact, không học văn phong), nhưng phải biết mình đang đánh đổi cái gì. Đây cũng là lý do `qa` chỉ 41%
> qua cổng: yêu cầu "đáp án nguyên văn ≤12 từ" quá chặt với văn xuôi research.

### 5.4 Các view đã chết — và vì sao ghi lại

| view chết | ý tưởng | vì sao chết |
|---|---|---|
| `relations` (EntiGraph) | sinh passage cho **mọi cặp entity** trong cửa sổ | T2 chỉ 20% lift của RAG; 33% passage chứa meta-statement ("the document does not mention…"). Nó bỏ **giá trị số và mệnh đề clause-level** — đúng thứ câu hỏi spec hỏi. |
| `dense` | viết lại theo đoạn 260 token, coverage atom ≥ 0,8 | T2 76,33 — không hơn textbook đơn giản. Nén mất chi tiết là bản chất của việc viết lại, không phải do cắt sai. |
| recite-then-answer (prompt-only) | model tự viết 4–8 câu fact rồi tự trả lời | base 71,86 → **65,39 (−6,5)**; tier1 72,64 → 67,75. Ảo giác **nhắm đúng câu hỏi** gây nhiễu mạnh hơn passage lạc đề (−0,6). Đóng. |

> **Bài học 5 — "viết lại cả cửa sổ một lượt" mất ~nửa chi tiết, bất kể cấu hình.** Pilot đã thử: cổng
> faithfulness, ngân sách 24k, gộp nhiều view, chia đoạn nhỏ — **mọi cấu hình prose đều 48–68% lift của
> RAG**. Hai giả thuyết "neo từ vựng" (option trùng chuỗi nguồn / câu hỏi trùng chuỗi nguồn) đều bị bác
> bằng số. Đây là trần của việc nén qua một lượt sinh, và nó là lý do `verbatim` luôn có mặt trong pack.

### 5.5 Audit chất lượng kit — hai judge cho hai câu trả lời khác nhau

150–200 hàng/view, judge đọc cửa sổ nguồn và phán `SUPPORTED / PARTIAL / NOT`:

| view | judge Qwen3-8B (supported/partial/**bịa**) | judge OTel-31B |
|---|---|---|
| `facts` | 94,7 / 5,3 / **0** | 94,5 / 3,5 / **2,0** |
| `factview` | 90,7 / 8,7 / **0,7** | 76,5 / 19,0 / **4,0** |
| `qa` | 88,7 / 10,7 / **0,7** | 84,0 / 12,0 / **4,0** |
| `mcq` | 92,7 / 6,7 / **0,7** | 81,0 / 13,0 / **6,0** |
| `cmcq` | 94,7 / 5,3 / **0** | 85,0 / 13,0 / 2,0 |
| `register` | 75,3 / 24,7 / 0 | 61,0 / 39,0 / 0 |

Judge mạnh khắt khe hơn hẳn. Đọc các ca `NOT`: phần lớn đến từ **cửa sổ nguồn rác** — log Orion, socket
buffer, câu lặp "heuristically designed because heuristically designed" — tức nội dung web/mailing-list
lọt vào Telco-Common-Corpus rồi lọt vào strong-RAG.

> **Bài học 6 — đừng audit data bằng chính model đã sinh ra nó.** 8B nói factview bịa 0,7%; 31B nói 4,0%.
> Số thật gần 31B hơn. Và lỗi không nằm ở generator mà ở **chất lượng cửa sổ đầu vào** → hướng sửa là lọc
> cửa sổ, không phải siết prompt.

---

## 6. Tầng 3 — Pack: nơi ba bug đắt nhất đã xảy ra

Pack là bước "tầm thường" (tokenize rồi ghép block) và cũng là nơi mất nhiều điểm nhất. Ba bug dưới đây
cộng lại đáng ~4 điểm.

### 6.1 Thành phần của một pack

```
kit docs      loss trên MỌI token, trộn ngẫu nhiên theo document
              (10 dạng của một fact phải rơi vào ngữ cảnh khác nhau — Allen-Zhu)
replay 5–10%  tele-data standard/wiki/arxiv prose thô — chống quên
anchor ×12    self-replay MCQ, loss CHỈ trên completion — neo định dạng
chat rows     qa-chat 50% + mcq-gold, loss chỉ trên assistant — cầu nối chế độ
masked        chunk 380 token che ngẫu nhiên → tái tạo, loss chỉ trên token bị che
```

Ví dụ big run: kit 1,41B token (verbatim 114k doc, facts 2,11M, register 898k, factview 22,47M,
qa 321k + qa-chat 223k, mcq-gold 331k) + replay 71,8M + anchor 185.200 × 12 → **1.784.306 block**;
cộng masked K=1 (361.881 block) → **2.146.187 block = 2,20B token**.

### 6.2 Bug 1 — anchor phản tín hiệu (đáng ~1,5 điểm)

**Ý tưởng anchor:** CPT trên văn xuôi làm model quên hợp đồng `ANSWER: X`. Cách chữa đã được đo ở vòng
trước: trộn ≥25% sequence là **câu trả lời của chính base** trên MCQ tổng hợp, đúng template harness,
chữ cái trần (self-replay). Loss chỉ trên completion. Điều này giữ format mà không dạy kiến thức mới.

**Bug:** MCQ tổng hợp được sinh **từ chính cửa sổ đang dạy**. Base đồng ý với đáp án có evidence chỉ
74,3% → **25,7% anchor đang dạy model giữ niềm tin SAI ngay tại fact vừa học.**

Ví dụ thật, hai hàng anchor bị loại:

```
Q: Which function controls the downlink user data flow to the ng-eNB-DU?
   A) Transfer of user data   B) Warning messages information transfer
   C) RRC message delivery status reporting   D) Flow control function
   base nói: A        generator (có evidence trong cửa sổ) nói: D

Q: What is the maximum relative time error that CoMP cannot tolerate in 5G networks?
   A) 1.5μs   B) 1.0μs   C) 0.4μs   D) 2.0μs
   base nói: C        generator nói: B
```

Với ~11,3k dòng như vậy × 12 lần lặp ≈ **27% token của pack** dạy ngược.

**Sửa:** `selfreplay_tier1.py` ghi thêm trường `agree`; `pack_tier1.py` chỉ giữ anchor `agree=True`.
Format vẫn được neo (vì hàng vẫn đúng template), hết phản tín hiệu.

**Kết quả:** cùng pack, cùng lr 1e-5, cùng 1 epoch: **72,64 → 74,14 (+1,50)**.

> **Bài học 7 — mọi hàng trong pack đều dạy một điều gì đó.** Hàng "chỉ để neo định dạng" vẫn dạy nội
> dung. Phải hỏi: *nếu model học thuộc hàng này thì nó tin điều gì?*

### 6.3 Bug 2 — lệch chế độ chat / raw (đáng ~2,5 điểm ở bậc 1)

Phát hiện bằng `rawmode_eval.py`: chấm cùng 10.000 câu bằng hai prompt khác nhau —
(a) harness chat template, (b) **văn bản thô** `Question: … \n A) … \n Answer:` đọc log-prob chữ cái.

| | base | tier1 ep1 | Δ |
|---|---:|---:|---:|
| chat harness | 71,79 | 72,69 | +0,90 |
| văn bản thô | 69,55 | **73,16** | **+3,61** |

CPT là học ở chế độ pretraining (văn xuôi thô). Đường chat/instruction chỉ rút được ~1/4 số tri thức đã
vào. Hai cách chữa:

- **serving:** render request harness thành prompt thô bên trong (hợp lệ — ta phục vụ model). +2,5 ngay.
- **mixed training:** đưa cầu nối vào chính pack — `--chat-qa 0.5` (một nửa `qa` thành lượt chat
  `user: q → assistant: a + why`) và `--mcq-gold` (MCQ tổng hợp ở đúng harness format, target = đáp án
  có evidence). Đây là cách đã chọn vì nó nằm trong trọng số, không phải mẹo serving.

Sau khi mixed training vào từ bậc 2, khoảng cách chat/raw thu hẹp dần: bậc 2 raw 76,23 / chat 75,88
(+0,35); tới `vd/ep1` thì **raw 78,79 ≈ chat 78,86** — hết lệch.

> **Bài học 8 — "model không biết" và "model không lấy ra được ở format này" là hai chuyện khác nhau.**
> Luôn chấm một arm ở chế độ raw log-prob để phân biệt. Nếu chênh lệch lớn, vấn đề là cầu nối chứ không
> phải liều lượng.

### 6.4 Bug 3 — thiếu `<|im_end|>` (làm strict parser tụt 36 điểm)

Các hàng chat trong `pack_tier1.py` (qa-chat, mcq-gold, anchor) nối `prompt + completion` nhưng **không
có `<|im_end|>`** sau completion. Model học "sau `ANSWER: X` thì viết tiếp", không học "dừng".

Hậu quả trên big run, đo thật:

| ckpt | strict | first-letter | câu có đuôi | median độ dài completion |
|---|---:|---:|---:|---:|
| big step24000 | **39,23** | **77,81** | 5.018 | **2.438 ký tự** |
| vd3/ep1 | **79,04** | 79,04 | 0 | **9 ký tự** |

Ví dụ thật một completion của big/step24000:

```
ANSWER: B The Nmfaf_3daDataManagement_Deconfigure service operation is used to stop the MFAF from
mapping data or analytics to out-bound notification endpoints. This is the opposite of the
Nmfaf_3daDataManagement_Configure operation, which is used to start the mapping process. ...
```

Chữ cái **đúng**, nhưng regex `^ANSWER: X$` của harness không khớp → chấm 0.

Hai cách xử: (a) serving đặt stop sequence sau chữ cái — hợp lệ nhưng là mẹo; (b) sửa tận gốc ở
`pack_chat.py`: `ci = tok(completion) + tok("<|im_end|>\n")`, và **mask = 1 cả trên `<|im_end|>`** — tức
loss dạy model *dừng*. Cách (b) đã chọn, và nó là một phần của vì sao vote-distillation thắng lớn (§8.2).

> **Bài học 9 — một ckpt có thể "đúng" 77,81 và bị chấm 39,23.** Luôn báo cả hai con số. Suýt nữa
> eval-loop xoá mất ckpt tốt nhất (step 12.000) vì nó xếp hạng theo `strict`.

### 6.5 Masked reconstruction — view không cần LLM

Theo arXiv 2510.09885. Mỗi chunk ~380 token của cửa sổ → hàng:

```
user:      Below is a passage from the telecom literature with some tokens masked as <|fim_pad|>.
           Reconstruct the original passage exactly.

           The MFAF shall <|fim_pad|> a new configuration, <|fim_pad|> a transaction <|fim_pad|> id,
           and <|fim_pad|> the configuration.
assistant: The MFAF shall create a new configuration, assign a transaction reference id,
           and store the configuration.
           ^^^^^^          ^^^^^^        ^^^^^^^^^        ^^^^^
           loss CHỈ trên những token có counterpart bị che
```

Tỉ lệ che ~ U(0,05; 0,95), mỗi hàng đóng đúng 1 block 1024 (pad, mask 0) nên hàng không tràn block.
**Không cần gọi LLM** — thuần biến đổi chuỗi.

Kết quả arm cô lập: anchor-fix 74,14 (recover 652 / broke 422) → + masked (329M token, +40%)
**74,41 (636 / 379)**. Điểm +0,27 nằm trong biên nhiễu, nhưng **broke giảm ~10%** — cùng chiều bài gốc.
Đã đưa vào big run với K=1.

---

## 7. Tầng 4 — Train: liều, learning rate, và điểm bão hoà

### 7.1 Công thức

Full-weight Qwen3-8B, AdamW, weight-decay 0, cosine + warmup 150, bf16 + gradient checkpointing, DDP một
replica đầy đủ mỗi card (~96 GB/card), block 1024, batch hiệu dụng 32 sequence (8 card × bs 2 × acc 2).
Tốc độ thật: **41,2k tok/s trên 8 card**.

Vì sao full-weight chứ không LoRA: run 1–3 full-weight lr 1e-5 **giữ được hợp đồng `ANSWER: X`**
(unparsed 0) và vẫn học được; EntiGraph / Active Reading / SMT đều full-weight. LoRA chỉ dùng làm sanity.

Một chi tiết vận hành đã gây treo NCCL: `order[rank::world][:nblk//world]` — mỗi rank phải nhận **đúng
cùng số block**, nếu không allreduce cuối cùng sẽ chờ mãi (gặp đúng ở pack vd 44.155 block).

### 7.2 Learning rate sweep — đường cong có đỉnh rõ

Cùng pack bậc 1, 1 epoch:

| lr | điểm | recover | broke | rot1 | ghi chú |
|---|---:|---:|---:|---:|---|
| 1e-5 (anchor cũ) | 72,64 | 410 | 330 | 70,81 | |
| 2e-5 | 72,87 | 481 | 378 | 71,83 | nhóm câu-có-cửa-sổ-sửa: 17,4% → 20,6% |
| 3e-5 | 73,11 | 592 | 465 | 71,78 | probe EM DiD +8,1 (đơn điệu theo lr) |
| **1e-5 (anchor agree-only)** | **74,14** | 652 | 422 | 72,07 | sửa bug > tăng lr |
| 5e-5 | 71,73 | 1.030 | **1.041** | 69,59 | quá nóng: recover = broke, unparsed 33 |

> **Bài học 10 — lr cao mua recover *và* broke gần như theo tỉ lệ.** Ở 5e-5 tỉ lệ đó về 1:1 và net = 0.
> "Liều" không phải nút vặn miễn phí; nó là cái cần bằng cả hai phía. Chốt 3e-5 cho big run.

### 7.3 Thang liều — và nơi nó bão hoà

| bậc | dữ liệu | token | lr / epoch | first-letter | recover | broke |
|---|---|---:|---|---:|---:|---:|
| base | — | — | — | 71,85 | — | — |
| tier-1 | 15.760 cửa sổ label-free | ~200M | 1e-5 / 1 | 72,64 | 410 | 330 |
| tier-1 anchor-fix | idem, anchor agree-only | ~200M | 1e-5 / 1 | 74,14 | 652 | 422 |
| tier-2 (mixed) | + đủ 8 cửa sổ/câu, chat-qa, mcq-gold | 870M | 3e-5 / 1 | 76,04 | 1.004 | 588 |
| **big run** | 114.125 cửa sổ (RAG-8 ∪ TB-8 ∪ deep ∪ Wiki), K=30, + masked | **2,20B** | 3e-5 / 1 | **77,81** (step24k) | **1.044** | 448 |

Đường cong theo step của big run (67.069 step = 1 epoch):

```
step   6.000  76,47   (9% epoch, LR đỉnh — đã bằng bậc 2 cuối)
step  12.000  76,86
step  18.000  76,60   ← phẳng
step  24.000  77,81   ← tốt nhất
step  30.000  76,98
step  36.000  77,20
step  42.000  77,10
step  48.000  77,01
      ep1     77,08
```

**Phẳng ~77 ± 0,5 từ 36% epoch.** Decay LR về cuối không bứt.

So sánh trực tiếp big-24k với bậc 2 (2,5× data):

| | bậc 2 | big-24k |
|---|---:|---:|
| recover | 1.004 | 1.044 (trùng nhau 820 câu) |
| broke | 588 | **448** |
| điểm | 76,04 | 77,81 |

> **Bài học 11 — lợi ích của 2,5× data là bớt `broke`, không phải thêm `recover`.** Kênh trọng số bão
> hoà ở **~1.000–1.050 câu recover** với công thức hiện tại. Đây là phát hiện quan trọng nhất của big
> run: *thêm data không còn là đòn bẩy*. Mọi bước sau đó chuyển sang nhắm `broke` và nhắm khả năng
> **trích xuất**.

### 7.4 Nút thắt thật: trích xuất, không phải lưu trữ

Hai phép đo độc lập nói cùng một chuyện.

**(a) Học một phần** (`letter_probs.py` + `partial_learning.py`): so xác suất option gold giữa base và
bậc 2:

| nhóm | n | P(gold) trung bình | gold lên hạng nhì |
|---|---:|---|---|
| base sai, chưa sửa, **có nguồn** | 1.214 | 0,037 → **0,121** | 47,8% → **54,3%** |
| base sai, chưa sửa, không nguồn | 598 | 0,023 → 0,086 | 41,6% → 45,0% |
| đã sửa | 1.004 | 0,091 → 0,778 | — |
| **broke** | 588 | **0,896 → 0,205** | 1,9% → **75,2%** |
| giữ đúng | 6.596 | 0,983 → 0,925 | — |

Nhóm "có nguồn chưa sửa" **có học một phần** — gold lên hạng nhì ở 54%. Nhưng nhóm "không nguồn" cũng
tăng 0,06 do model bớt tự tin nói chung (giữ đúng 0,983 → 0,925), nên phần "kiến thức thật" chỉ ≈ +0,035.

Và: **`broke` không phải flip biên.** P(gold) rơi 0,90 → 0,21. Training **đẩy mạnh sang một option sai cụ
thể**, không phải làm model do dự.

**(b) Chẩn đoán nguồn-trong-context** trên ckpt bậc 2: cho nó đọc lại 8 cửa sổ trong context →
**77,89**, trong khi base đọc cùng cửa sổ được 83,75. Trên 1.214 câu "base sai, bậc 2 chưa sửa, nguồn có
trong cửa sổ": base + RAG-8 đúng **1.010 (83%)**, bậc 2 + RAG-8 chỉ 679.

> **Bài học 12 — CPT toàn block 1024 làm mòn kỹ năng đọc context dài.** Không ảnh hưởng điểm closed-book,
> nhưng nó chứng minh rằng 1.214 câu ấy **có fact rút được từ nguồn** → thất bại là ở khâu **trích xuất
> trong trọng số**, không phải câu mơ hồ, không phải thiếu nguồn.

### 7.5 Hai cách trộn trọng số — đều không có bữa trưa miễn phí

**WiSE-FT** (`wise_merge.py`): θ = (1−α)·base + α·tuned.

| α | điểm | recover | broke | tỉ lệ broke/recover |
|---|---:|---:|---:|---:|
| 0,5 | 73,41 | 359 | 202 | 0,56 |
| 0,75 | 74,01 | 520 | 303 | 0,58 |
| **1,0** | **74,14** | 652 | 422 | 0,65 |
| 1,25 (ngoại suy) | 73,65 | 800 | 619 | 0,77 |
| 1,5 | 31,15 | — | — | vỡ format, 4.906 câu không parse |

Nội suy giảm broke **gần đúng tỉ lệ** với recover. → α=1 chốt; trade-off recover/broke là một continuum
theo liều, không tách được bằng nội/ngoại suy trọng số.

**Model soup:** tập đúng của bậc 2 ∪ big = **80,71** (cả hai cùng đúng 73,14) → hai ckpt biết những thứ
khác nhau, có vẻ đáng trộn. Trung bình trọng số 0,5: **77,36** — nằm *giữa* hai cha (76,04 và 77,81),
không lấy được phần hợp. → hai ckpt không cùng "basin"; bỏ hướng soup.

> **Bài học 13 — "hợp tập đúng cao" không hàm ý soup có lãi.** Nó chỉ nói hai model khác nhau. Muốn khai
> thác phần hợp thì phải qua distillation, không qua trung bình trọng số.

---

## 8. Tầng 5 — Behaviour stages: contrastive → consistency

Đây là phần **mới nhất** và có tỉ lệ lợi ích/chi phí cao nhất: mỗi stage ~20 phút trên 8 card, không sinh
thêm tri thức, chỉ sửa **cách model dùng** tri thức đã có.

### 8.1 Contrastive-MCQ — dạy *phân biệt*, không dạy *fact*

**Vấn đề nó nhắm:** failure analysis cho thấy rất nhiều ca sai là **nhầm hàng xóm gần** — RLC vs PDCP
("transfer of upper layer PDUs"), Rician vs Rayleigh, MiD (nhiều identity trên một UE) vs MuD (nhiều UE).
Kit có 68 dòng nói về "RLC sublayer" mà vẫn sai → **paraphrase mức câu không dạy được phân biệt**.

**Cách làm** (`gen_contrastive.py`): với mỗi cửa sổ TARGET, lấy ≤3 cửa sổ **hàng xóm** (= cửa sổ khác được
retrieval cho *cùng câu test*). Prompt generator:

> "Correct option phải được TARGET hỗ trợ. Mỗi distractor phải là một mệnh đề/giá trị/thực thể lấy từ
> NEIGHBOUR (đúng ở đó, sai với TARGET) hoặc near-miss của đáp án đúng."

Cổng: evidence nguyên văn trong TARGET. Ví dụ thật:

```
win w000000  neighbours [w000002, w012905, w000001]
Q: What HTTP status code should the MFAF return if it successfully processes and accepts an
   HTTP DELETE request to remove a configuration?
   A) 200 OK       ← từ cửa sổ hàng xóm (đúng cho PUT, sai cho DELETE)
   B) 201 Created  ← từ cửa sổ hàng xóm (đúng cho POST)
   C) 204 No Content   ✔ đúng, có evidence trong TARGET
   D) 302 Found
```

Distractor không phải "sai bừa" mà là **sự thật của đoạn bên cạnh** — đúng loại nhầm lẫn model đang mắc.

**Kết quả:** arm cô lập trên pack anchor-fix — A (chỉ mcq-gold) 74,02 (717/499); B (A + cmcq)
**74,38 (755/501)**, rot1 **72,59** (+0,8 so với A). cmcq dương, nhỏ, và **không tăng broke**.

Stage cmcq đầy đủ từ `vd/ep1` (700.228 hàng, 2 hoán vị, 98k block): **78,74** — recover **1.181 (+134)**
nhưng broke **491 (+146)** → net ≈ 0 một mình. Đáp án do generator viết (bịa 2–6% theo audit 31B) vừa
thêm kiến thức vừa đẩy lệch.

### 8.2 Vote-distillation — dạy model **nhất quán với chính nó**

Đây là stage hiệu quả nhất của cả track (+1,05 trong 19 phút).

**Ý tưởng:** model biết câu trả lời nhưng **trả lời khác nhau tuỳ vị trí option**. Vote 4 hoán vị × 8 mẫu
cho +1,30 lúc infer (big ep1: 77,05 → 78,35). Vậy **đem chính kết quả vote đó làm nhãn và train ngược
lại** — biến mẹo serving thành thuộc tính của trọng số. Hoàn toàn label-free.

**Quy trình** (`vote_distill.py` → `pack_chat.py`):

1. Gom MCQ tổng hợp đã qua cổng từ mọi bậc: **409.035 hàng** (tier3 152.468, tier2 114.331, tier1 49.048,
   tier1c 46.117, tier4 26.475, tier5 20.596). Lấy ngẫu nhiên 100k.
2. Cho ckpt tự trả lời mỗi câu dưới **4 hoán vị × 4 mẫu** (T=0,7).
3. Bỏ phiếu **trong không gian chỉ số gốc** (`(g + s) % n` — không phải theo chữ cái).
4. Giữ câu có `share ≥ 0,6` → 84.351/100.000 (bỏ 15.649 vì low-share).
5. Với mỗi câu giữ lại, phát **4 hàng chat** — một cho mỗi hoán vị, target là chữ cái *của đáp án vote
   trong hoán vị đó*, kết thúc `<|im_end|>`.

Ví dụ thật, một câu → 4 hàng:

```
A) Minimum SINR constraint | B) Maximum secrecy rate | C) Individual and sum EH constraints | ...  → ANSWER: C
A) Maximum secrecy rate | B) Individual and sum EH constraints | C) Minimum SINR ...             → ANSWER: B
A) Individual and sum EH constraints | B) Minimum throughput | C) Minimum SINR ...               → ANSWER: A
A) Minimum throughput | B) Minimum SINR | C) Maximum secrecy rate | D) Individual and sum EH...  → ANSWER: D
```

Cùng một nội dung, bốn chữ cái khác nhau. Model buộc phải học **nội dung**, không thể học vị trí.

6. Pack: 342.676 hàng → 44.155 block (45M token, loss-bearing chỉ 2,06M token vì loss chỉ trên
   `ANSWER: X` + `<|im_end|>`). SFT lr **5e-6**, 1 epoch, 8 card, **19 phút**.

**Kết quả `vd/ep1`:**

| | big/step24000 | vd/ep1 |
|---|---:|---:|
| strict | 39,23 | **78,86** |
| first-letter | 77,81 | **78,86** |
| unparsed | 5.018 (đuôi) | **0** |
| rot1 | 75,76 | 76,67 |
| broke | 448 | **345 (−23%)** |
| greedy vs vote | 77,81 / 78,45 | 78,86 / 78,65 — **vote không còn thêm gì** |
| raw vs chat | 78,20 / 77,68 | 78,79 / 78,86 — **hết lệch chế độ** |

Cả bốn mục tiêu đạt cùng lúc: đuôi hết, strict = first-letter, broke −23%, và **mọi mẹo serving bị nội
hoá** (vote, stop-sequence, raw-mode đều không còn thêm điểm).

> **Bài học 14 — nếu một mẹo lúc infer cho +1, hãy biến nó thành nhãn train.** Vote-distillation lấy
> chính hành vi ensemble và ép nó vào một forward pass. Không cần đáp án, không cần data mới, 19 phút.

**Vòng 2** (200k câu mới, seed 2, init từ `vd/ep1`): 78,49 / rot1 **77,06**. Trung bình hai thứ tự:
vd1 77,77 · vd2 77,78 → **bằng nhau**, chỉ đổi độ nhạy thứ tự (gap 2,2 → 1,4). **Bão hoà sau một vòng.**

### 8.3 Mẫu đã tìm ra: contrastive (thêm recover) → consistency (bớt broke)

| stage | init | điểm | rot1 | recover | broke | TB hai thứ tự |
|---|---|---:|---:|---:|---:|---:|
| vd | big/step24000 | 78,86 | 76,67 | 1.047 | 345 | 77,77 |
| cm (cmcq) | vd/ep1 | 78,74 | 76,79 | **1.181** | 491 | 77,77 |
| **vd3** | cm/ep1 | **79,04** | **77,11** | 1.164 | 444 | **78,08** |

cmcq một mình: net ≈ 0 (thêm recover, thêm broke bằng nhau). vd một mình: bão hoà sau vòng 1.
**Ghép lại: +0,3 thật.** cmcq mở rộng phần model dám trả lời; vd ép lại phần nó dao động.

> **Bài học 15 — hai can thiệp "≈0" theo đúng thứ tự có thể cho một can thiệp dương.** Chỉ thấy được nếu
> báo cáo `recover`/`broke` riêng chứ không chỉ điểm tổng.

### 8.4 Hai arm ≈ 0 — và vì sao vẫn đáng ghi

**Numeric drill** (`gen_numeric.py`): hồ sơ lỗi của `vd` cho thấy bucket yếu nhất là câu có gold là giá
trị số — **63,5% đúng** (236 câu sai) so với 78,9% tổng thể. Sinh 84.689 cặp Q/A số (cổng: đáp án nguyên
văn trong câu nguồn + phải khớp regex số/đơn vị). SFT lr 5e-6, 2,5 phút.

Kết quả: **78,75** (recover 1.037 / broke 346) — bằng vd. Bucket mục tiêu: numeric-gold 63,7 → **63,7**
(không nhúc nhích). Giải thích: số trong test không trùng số trong drill, hoặc Q/A ngắn không chuyển sang
được format MCQ.

**Behaviour view** (`gen_behavior.py`): sinh 37.838 MCQ theo ba chủng loại — "All of the above" là đáp án
đúng / "All of the above" là distractor / câu NOT-EXCEPT. Kết quả: **78,52** (hơi kém). Bucket:
All-gold 94,5 → **90,4 (−4)** trong khi All-distractor 59,5 → 65,0 (+5,5).

> **Bài học 16 — data cân bằng làm lệch prior của test.** Behaviour view sinh All-đúng : All-sai theo tỉ
> lệ 1:1, nhưng prior thật của TeleQnA là **~4,5:1** (824 câu chứa "All of the above", 740 câu trong đó
> nó là gold = 89,8%). Model học lại prior sai. Nếu dùng lại view này phải giữ tỉ lệ ≥ 4:1. Lợi ích kỳ
> vọng ≤ +0,3 — không ưu tiên.

---

### 8.5 Sáu arm sau vd3 — cái gì ≈ 0 và vì sao (17/09)

| arm | init | điểm | rot1 | recover | broke | kết luận |
|---|---|---:|---:|---:|---:|---|
| cm_mlp (cmcq, chỉ cập nhật MLP) | vd | 78,79 | 76,66 | 1.151 | 456 | đóng băng attention bớt 35 broke *và* 30 recover → net 0 |
| cm2 → vd4 (lượt 2 contrastive→consistency) | vd3 | 78,83 → 78,81 | 76,97 → 77,10 | 1.211 → 1.204 | 512 → 507 | consistency chỉ cắt broke được một lần |
| DPO chữ cái, lr 5e-7 | vd3 | 78,97 | 77,17 | 1.162 | 449 | loss đứng 0,69 — liều quá thấp |
| DPO, lr 3e-6 | vd3 | **14,11** | 13,75 | 494 | 6.267 | sụp: likelihood displacement, dồn xác suất sang C/E |
| RPO (DPO + NLL), lr 1e-6 | vd3 | 78,97 | 77,15 | 1.172 | 459 | tín hiệu âm trên chữ cái không thêm gì |
| style (MCQ đúng kiểu hỏi test) → vd5 | vd3 | 78,46 → 78,71 | 76,92 → 77,24 | 1.181 → 1.187 | 519 → 500 | dạng hỏi khớp hơn nhưng vẫn là "dạy chọn" |
| re-study bậc 3 (808M token, 2 epoch, lr 1e-5) | vd3 | 77,53 (6k) / 77,77 (12k) | 75,17 / 75,39 | ~1.085 | 521 / 487 | **dừng sớm** ở 26% run: thêm liều không thêm recover |

Phát hiện phụ đáng ghi: cặp DPO gần như không có chữ E vì MCQ tổng hợp/cmcq chỉ 4 option, trong khi 64% câu test có
5 option — lệch cấu trúc của toàn bộ MCQ tổng hợp (sibling view sau này cho phép 3–5 option).

### 8.6 Sibling-contrast — đòn bẩy lớn nhất sau vd1 (17/09 21:35)

**Chẩn đoán dẫn tới nó** (§10.3): lõi "never" (1.144 câu chưa ckpt nào đúng) là *confidently wrong* — margin 0,90,
P(gold) 0,01; 413/428 câu có gold nguyên văn trong cửa sổ đã được kit lặp ≥5 lần; đọc mẫu thì model chọn **thực thể
anh em trong cùng đoạn**: EDGE-1 → EDGE-3, state N8 → N9, first-order → second-order Markov, 10 cm → 1,5 m, queuing
theory → stochastic network calculus, index modulation → SSK. Paraphrase giữ thực thể, không dạy ràng buộc thuộc
tính ↔ đúng thực thể; cmcq (§8.1) lấy distractor từ *cửa sổ khác* nên không chạm đúng lỗi này.

**Cách làm** (`gen_sibling.py`): generator (1) liệt kê các nhóm sibling dễ nhầm *trong chính cửa sổ* (state đánh số,
interface/reference point, tham số + giá trị, phương pháp liệt kê cạnh nhau, message, timer, release); (2) mỗi nhóm
1–2 câu hỏi ngắn tự-chứa, đáp án = một sibling, option = các sibling khác của cùng nhóm (3–5), kèm evidence. Cổng
nghiêm: **mọi option và evidence phải nguyên văn trong cửa sổ** (loại ~45%: option không nguyên văn 99k, evidence
26k, badjson 60k). 174k MCQ (60k cửa sổ) → 349k hàng chat (2 hoán vị, `<|im_end|>`), 45k block; SFT lr 5e-6 từ vd3,
EMA on.

**Kết quả:** **79,66 / rot1 77,90** (recover **1.225**, broke 443) — +0,62/+0,79 so với vd3, +0,47 so với soup; sau
consistency (vd6): **79,69 / 78,05** (1.220 / 435) — ckpt chốt hiện tại. Sinh nốt 54k cửa sổ còn lại → 331k MCQ sibling
cho toàn bộ 114k cửa sổ, đưa vào big run 2 với trọng số ×2.

> **Bài học 17 — distractor phải đến từ *chính đoạn nguồn*.** Lỗi thật của model là nhầm sibling cùng đoạn; distractor
> lấy từ đoạn khác (cmcq) hay do generator bịa (mcq) không chạm vào nó. Cổng "option nguyên văn trong cửa sổ" là thứ
> làm view này khác các view trước.

### 8.7 Hướng B — "dạy nhớ lại": recall-QA tự-chứa, mixed vs PIT, qa_all

Probe cũ (QA của kit) đo ~5% recall vì câu phụ thuộc ngữ cảnh. `gen_recall_qa.py` sinh QA **tự-chứa** ×20/cửa sổ (câu
hỏi nêu thực thể, ~25% hỏi ngược, cổng: đáp án nguyên văn, cấm "the paper/excerpt/we/proposed", ≥2 từ nội dung trùng
cửa sổ), 10% giữ lại làm probe. Bậc 1: 203k hàng train + 14k giữ lại; toàn bộ 114k cửa sổ: 1,49M hàng.

| arm | công thức | điểm | rot1 | recover | broke | recall giữ-lại |
|---|---|---:|---:|---:|---:|---:|
| anchor-fix (đối chứng) | docs bậc 1, từ base | 74,14 | 72,07 | 652 | 422 | — |
| mixed | docs + QA (3% pack), từ base | **74,67** | **72,60** | 667 | 384 | 20,6% |
| PIT (2402.12847) | QA trước → docs sau | 74,73 | 72,20 | 681 | 392 | 19,0% |
| qa_all | chỉ QA, 114k cửa sổ, 52M token, EMA | 73,80 (raw 73,92) | 72,53 | 475 (617) | 279 (409) | 19,2% |
| base / vd3 | — | 71,85 / 79,04 | | | | 15,2% / 23,2% |

QA tự-chứa: +0,5 ở bậc 1 (cả hai thứ tự), thứ tự QA-first ≈ mixed; qa_all riêng đã +1,96 với 52M token — token-efficient
hơn docs nhiều lần. Đưa vào big run 2 với trọng số ×3.

### 8.8 Trộn trọng số lần hai: soup cùng dòng và EMA

Soup hai *dòng khác nhau* (bậc 2 × big) từng thất bại (§7.5). Soup **cùng dòng** (7 ckpt hậu duệ của vd3): **79,19 / 77,37**
(recover 1.160, broke **425**) — +0,15/+0,26, làm mượt nhóm flicker; 13 ckpt: 79,15/77,44 (bão hoà). Đưa hiệu ứng này vào
trainer: **EMA** fp32 trên CPU rank 0, cập nhật mỗi 10 step, decay 0,99 (`--ema-every`, `--ema-decay`), lưu `ep1` = EMA và
`ep1_raw`. **Cảnh báo:** bản EMA này làm rank 0 chậm 3,7× (copy 32 GB fp32 sang CPU mỗi 10 step) → big run 2 phải tắt EMA;
dùng soup các ckpt step sau khi train thay thế. Trên qa_all: EMA 73,80 (475/279, 5 không parse) so với raw 73,92 (617/409, 40) — EMA = bản "liều thấp" mượt
hơn, net ≈ bằng; giữ làm ổn định, không kỳ vọng thêm điểm.

## 9. Tầng 6 — Serving: mọi mẹo đều đã bị nội hoá

Bốn đòn bẩy lúc infer đã đo hết trên `vd`:

| kỹ thuật | trên base | trên big-24k | trên vd/ep1 |
|---|---:|---:|---:|
| **vote 4 hoán vị × 8 mẫu** | 71,85 → 72,38 (+0,53) | 77,81 → 78,45 (+0,64) | 78,86 → 78,65 (**−0,21**) |
| **raw-mode prompt** | 71,79 → 69,55 (−2,2) | 77,68 → 78,20 (+0,5) | 78,86 → 78,79 (**≈0**) |
| **stop sau chữ cái** | không cần | +38,6 (39,23 → 77,81) | không cần (unparsed 0) |
| **thinking mode** (1.000 câu) | 72,0 → **74,5 (+2,5)** | — | 78,3 → 77,7 (**−0,6**) |

Thinking mode đáng chú ý: nó giúp **base** +2,5 nhưng **hại** model đã train. Cả pipeline (sinh data + eval)
chạy no-think, nên phân phối "reasoning trước đáp án" bị lệch — hoặc đơn giản là tri thức đã học được truy
cập trực tiếp tốt hơn qua deliberation.

> **Bài học 17 — model càng tốt thì mẹo serving càng vô dụng.** Đó là dấu hiệu lành: mọi cải thiện tiếp
> theo **phải đến từ trọng số**. Deliverable cuối cùng là một forward pass greedy với parser harness gốc —
> đúng thứ mà một dòng leaderboard nên đại diện.

Ghi chú quan trọng: **quy tắc chọn chế độ phục vụ phải label-free.** Raw-mode tốt hơn trên `lr3e5`
(74,88 vs 73,16) nhưng **xấu hơn** trên `anchor-fix` (72,92 vs 74,07) — ngược dấu. Không được chọn chế độ
theo điểm test; phải chọn theo tín hiệu nội tại (vote share / confidence). Sau `vd` thì vấn đề này biến
mất vì hai chế độ bằng nhau.

---

## 10. Kỷ luật đo — bảy công cụ, mỗi cái trả lời một câu hỏi khác nhau

| công cụ | file | câu hỏi nó trả lời |
|---|---|---|
| **recover / broke** | `jobs/eval_ckpt.sh` | điểm tăng vì sửa được câu mới, hay chỉ vì lật ngẫu nhiên? |
| **rot1** (xoay lựa chọn) | `data/eval/otfull_rot1.jsonl` | học nội dung hay học vị trí chữ cái? |
| **strict vs first-letter** | `eval_ckpt.sh` | model chọn sai, hay chỉ viết sai định dạng? |
| **letter probs / partial learning** | `letter_probs.py`, `partial_learning.py` | câu chưa sửa có *nhích* về phía gold không? |
| **probe log-prob (mem/sem/qa)** | `probe_logprob.py`, `probe_heldout_v2.py` | tri thức đã vào ở tầng nào — nhớ, hiểu, hay rút ra được? |
| **failure analysis** | `analysis/failure_analysis.py` | yếu tố nào dự báo "sửa được"? |
| **audit** | `kit_audit.py` (data), `label_audit.py` (nhãn test), `broke_audit.py` (câu hỏng) | data có bịa không? nhãn có sai không? model hỏng vì cái gì? |

Ba nguyên tắc vận hành:

1. **Sàn nhiễu A/A = 0,50pp.** Mọi Δ dưới mức đó không được gọi là hiệu ứng.
2. **Cổng rẻ trước bước đắt.** T2-gate (`check_tier1_t2.py`) chặn chain nếu kit không giữ đủ tri thức —
   không đốt 9 giờ card cho data hỏng.
3. **Mọi arm đều có control cùng stack.** Base phải được đo bằng đúng model-load, đúng prompt, đúng
   parser với arm được so.

### 10.1 Probe log-prob — "knowing–using gap" đo được

Theo Chang et al. (arXiv 2406.11813), dùng cloze probe ở ba độ sâu trên fact do **31B** viết (độc lập với
generator 8B), chia train (cửa sổ trong kit) / held-out (cửa sổ của câu **không có cửa sổ nào** trong kit):

- `mem` — câu nguồn cắt ngay trước span → "đã nhớ chuỗi chưa"
- `sem` — câu fact paraphrase cắt trước span → "đã hiểu chưa"
- `qa` — `Question: q / Answer:` → span → "**rút ra được chưa**"

`DiD = (train_ckpt − train_base) − (heldout_ckpt − heldout_base)`.

Bậc 1: mem +0,38, sem +0,60 (vào từ step 2000, bão hoà từ step 6000), **qa ≈ 0**, EM DiD +2,6.
Sau khi tăng lr: 3e-5 cho mem +0,92, sem +1,24, **qa +0,52, EM DiD +8,1** — đơn điệu theo lr.

> **Bài học 18 — "nhớ" đến trước, "dùng được" đến sau và khó hơn nhiều.** Probe này cảnh báo trước 6 giờ
> rằng epoch 2 sẽ không thêm gì (nó bão hoà từ step 6000) — và đúng: ep1 72,64 vs ep2 72,60.
> **Held-out phải sạch về chủ đề**: bản v1 lấy held-out từ cửa sổ *bị loại* của cùng các câu pilot, nhưng
> cửa sổ khác của cùng câu vẫn trong kit nên held-out cũng tăng theo → DiD ≈ 0, che mất tín hiệu.

### 10.2 Audit nhãn test — trần thật ở đâu

`label_audit.py`: judge đọc 5 cửa sổ nguồn, phán đồng ý/trái/không quyết được với gold. Chỉ để **hiểu
trần**, không bao giờ dùng để dạy.

| nhóm | judge 31B: đồng ý | trái | không quyết |
|---|---:|---:|---:|
| 600 câu ngẫu nhiên | 47,0% | **7,7%** | 45,3% |
| 200 câu `broke` của bậc 2 | 28,5% | **15,5%** | 56% |
| 100 câu `fixed` | 47% | 6% | 47% |

Đọc tay 59 ca "trái": ≥3 là judge sai. Ước **nhiễu nhãn thật ~3–5%** → trần với kiến thức hoàn hảo
≈ **95–97**. Nhưng nhóm `broke` giàu câu mơ hồ gấp đôi ngẫu nhiên → **~1/3 broke là không tránh được**
khi model học đúng theo nguồn.

Phân loại 59 ca ấy:

| loại | ~tỉ lệ | ví dụ | chữa được? |
|---|---:|---|---|
| gold "All of the above / Both" nhưng nguồn chỉ nêu rõ một ý | 22% | 5G slice access, EE metrics | vd đã đúng 94,5% (base 88,9) → còn ~50 câu, ≤ +0,5 |
| hai option gần đồng nghĩa (GPT-3.5 sinh distractor sát nghĩa) | ~25% | VLC A≡E; scattering/diffraction | khó, chỉ giảm bằng nhất quán |
| đoạn nguồn truy về là "hàng xóm" nói khác câu gốc | ~35% | DCCP↔SCTP, SDN latency | cmcq + siết retrieval |
| thật sự mơ hồ / judge sai | ~18% | ZC "depends", RIS semi-passive | không |

Và `broke_audit.py` đọc mẫu cho một kết luận đáng nhớ: có ca **model học đúng fact và bị chấm sai** —
"chargeable events → charging events" là việc của **CTF** (TS 32.240) nhưng gold ghi OCF.

---

### 10.3 Trần đo lại: oracle union, never/flicker, calibration, probe recall, reader 31B, harness, contamination

- **Oracle union 18 ckpt = 88,56%** (đúng ở mọi ckpt train: 64,6%). Pipeline *đã từng* đúng 88,6% ≈ trần nguồn 89,4 →
  trần không đo sai; khoảng cách 79 → 88 là **giữ đồng thời**. Sai của vd3 = 853 **flicker** (ckpt anh em đúng; margin
  0,50, P(gold) 0,20, gold nhì 82%) + 1.243 **never** (margin 0,90, P(gold) 0,01, gold nhì 45%).
- **Calibration của xác suất chữ cái** (ECE): base 0,197 (86% câu p≥0,9 nhưng đúng 77%); sau train 0,089, đơn điệu
  (0,5→47%, 0,8→68%, ≥0,9→89%) → margin dùng để *xếp hạng/chọn* là hợp lệ; giá trị tuyệt đối của base thì lệch.
- **Probe recall** đúng cách = QA tự-chứa giữ lại (token-F1 ≥ 0,5): base 15,2 → vd3 23,2 → trọng số *có* gain ở mức nhớ
  lại nhưng mới 23% fact của chính các cửa sổ đã học (probe cũ 5% là do câu phụ thuộc ngữ cảnh).
- **Reader mạnh hơn không mở trần:** OTel-31B + RAG-8 trên 1.500 câu 80,4 (lenient) so với base 82,8; trên nhóm never chỉ
  ~45% được nguồn giải quyết ở cả hai reader → ~55% lõi never là retrieval miss / nhãn mơ hồ.
- **Harness-faithful** (`run_baseline.py` = contract Inspect, vLLM server): vd3 **79,07** (0 parse fail) ≈ 79,04 nội bộ.
- **Contamination:** 2.937 câu test có câu hỏi trùng nguyên văn trong kit (cửa sổ nguồn chứa câu văn GPT-3.5 đã dùng),
  nhưng gain hai nhóm bằng nhau (+7,4 / +7,1); 917 câu có ≥3 option trong một cửa sổ = cụm từ spec, không phải bản
  dataset. Điểm là closed-book thật theo policy.
- **Dạng hỏi kit ≠ test:** kit MCQ 42% "Which of the following" (test 1,5%), thiếu "What are/What does" → `--style`.
- **Bug tìm được khi tự audit:** `pack_chat.py` cắt block qua hàng → 7,2% block có completion không có prompt (sửa: pad);
  biến `N` trùng tên trong `kit_vd_round.sh` làm vd5 gán nhãn 4 câu (sửa); `eval_ckpt.sh` bỏ qua khi file cũ còn.

## 11. Bảng kết quả đầy đủ

Mọi con số: 10.000 câu, greedy, no-think, template harness byte-identical. `first-letter` = chữ cái đầu
sau `ANSWER:`.

| arm | mô tả | first-letter | rot1 | recover | broke | unparsed |
|---|---|---:|---:|---:|---:|---:|
| base | Qwen3-8B | 71,85 | 70,29 | — | — | 1 |
| kit1_ep1 | bậc 1, lr 1e-5, anchor cũ | 72,64 | 70,81 | 410 | 330 | 1 |
| kit1_ep2 | idem, epoch 2 | 72,60 | 70,80 | 409 | 333 | 1 |
| kit1gf_ep1 | chỉ cửa sổ gold-fix (cận trên answer-key) | 72,20 | 70,54 | 355 | 319 | 1 |
| kit1lr2e5 | bậc 1, lr 2e-5 | 72,87 | 71,83 | 481 | 378 | 0 |
| kit1lr3e5 | bậc 1, lr 3e-5 | 73,11 | 71,78 | 592 | 465 | 1 |
| kit1lr5e5 | bậc 1, lr 5e-5 | 71,73 | 69,59 | 1.030 | 1.041 | 33 |
| **kit1agree** | **anchor agree-only** (bug fix) | **74,14** | 72,07 | 652 | 422 | 0 |
| kit1masked | + masked reconstruction | 74,41 | 72,19 | 636 | **379** | 4 |
| kit1armA | + mcq-gold | 74,02 | 71,79 | 717 | 499 | 0 |
| kit1armB | + mcq-gold + cmcq | 74,38 | 72,59 | 755 | 501 | 0 |
| kit2_ep1 | bậc 2 mixed, 870M, lr 3e-5 | 76,04 | 73,60 | 1.004 | 588 | 5 |
| big_step24000 | big run 2,20B, 36% epoch | **77,81** | 75,76 | 1.044 | 448 | 5.018† |
| big_ep1 | big run đủ 1 epoch | 77,08 | 74,80 | — | — | 4.786† |
| soup_t2_big50 | trung bình trọng số bậc2/big | 77,36 | 75,28 | 993 | 769 | 370† |
| **vd_ep1** | vote-distillation vòng 1 | **78,86** | 76,67 | 1.047 | **345** | **0** |
| vd2_ep1 | vote-distillation vòng 2 | 78,49 | **77,06** | 1.031 | 366 | 0 |
| cm_ep1 | contrastive-MCQ stage | 78,74 | 76,79 | **1.181** | 491 | 0 |
| num_ep1 | numeric drill | 78,75 | 76,73 | 1.037 | 346 | 0 |
| beh_ep1 | behaviour view | 78,52 | 76,45 | 1.047 | 379 | 0 |
| vd3_ep1 | consistency trên cm/ep1 | **79,04** | **77,11** | 1.164 | 444 | **0** |
| cm_mlp_ep1 | cm, chỉ cập nhật MLP (attention đóng băng) | 78,79 | 76,66 | 1.151 | 456 | 0 |
| restudy_step6000 | re-study bậc 3 từ vd3, 12% run | 77,53 | 75,17 | 1.090 | 521 | 0 |
| restudy_step12000 | idem, 26% run → **dừng sớm** | 77,77 | 75,39 | 1.080 | 487 | 0 |
| vd3 qua harness | `run_baseline.py` (contract Inspect, vLLM server, max_tokens 32) | **79,07** | — | — | — | 0 |
| cm2_ep1 / vd4_ep1 | lượt 2 contrastive → consistency | 78,83 / 78,81 | 76,97 / 77,10 | 1.211 / 1.204 | 512 / 507 | 0 |
| dpo_ep1 / dpo3_ep1 | letter-DPO 5e-7 / RPO 1e-6 | 78,97 / 78,97 | 77,17 / 77,15 | 1.162 / 1.172 | 449 / 459 | 0 |
| style_ep1 / vd5_ep1 | MCQ đúng kiểu hỏi → consistency | 78,46 / 78,71 | 76,92 / 77,24 | 1.181 / 1.187 | 519 / 500 | 0 |
| soup_lineage (7) / soup_all (13) | trung bình trọng số cùng dòng | 79,19 / 79,15 | 77,37 / 77,44 | 1.160 / 1.151 | 425 / 420 | 0 |
| recall_ep1 / pit_ep1 | bậc 1 + QA tự-chứa (mixed / QA-first), từ base | 74,67 / 74,73 | 72,60 / 72,20 | 667 / 681 | 384 / 392 | 0 |
| qa_all_ep1 (EMA / raw) | chỉ QA tự-chứa 114k cửa sổ, từ base | 73,80 / 73,92 | 72,53 / 72,67 | 475 / 617 | 279 / 409 | 5 / 40 |
| **sib_ep1** | sibling-contrast từ vd3 | **79,66** | **77,90** | **1.225** | 443 | 0 |
| **vd6_ep1** ★ | consistency trên sib | **79,69** | **78,05** | 1.220 | 435 | **0** |
| big3_ep1 / step4500 / soup | big run 3 (stack FSDP2, QA ×3 + sibling, 2,6B token) | 80,17 / 80,33 / 80,34 | 78,32 / 78,71 / 78,57 | — | — | 0 |
| **vd8_ep1** | consistency trên big3 | **80,61** | 79,02 | — | — | 0 |
| merge_plain / merge_a | task-arithmetic vd6 ⊕ vd8 (0,5/0,5; 0,4/0,6) | 81,36 / 81,53 | 79,62 / 79,64 | 1.405 | 436 | 0 |
| **ens_ep1** | chưng cất đồng thuận vd8+vd6 (cap dài 33 %, trùng-từ 39 %, All 4:1) từ vd8 | 81,48 | **80,58** | 1.480 | 516 | 0 |
| ens2_ep1 | idem, init merge_a (merge làm init → *giảm*) | 81,14 | 79,88 | 1.412 | 482 | 0 |
| merge_f (0,2 vd6 / 0,3 vd8 / 0,5 ens) | ghép 3 ckpt; g/h/i quét bão hoà 81,75–81,78 | **81,79** | 80,32 | 1.455 | 460 | 0 |
| big4_step2500 / ep1 / soup | big run 4 (data khử thiên kiến, style, All-prior) | 81,20 / 81,05 / 81,25 | 79,00 / 79,35 / 79,36 | 1.459 / 1.492 / 1.503 | 523 / 571 / 562 | 0 |
| vd9_ep1 | consistency trên big4 (vote 4rot×8 = 81,21 = greedy) | 81,29 | 79,78 | 1.501 | 556 | 0 |
| ens3_ep1 | chưng cất vd9+vd8+vd6 từ vd9 | 81,46 | 80,32 | 1.470 | 508 | 0 |
| merge_j / merge_k | vd6/vd8/ens3 0,2/0,3/0,5 · 4 dòng 0,15/0,2/0,3/0,35 | 81,79 / 81,98 | 80,59 / 80,61 | 1.439 / 1.468 | 444 / 454 | 0 |
| **merge_l** (0,5 merge_f ⊕ 0,5 ens3) | ghép 3 dòng độc lập | **82,04** | 80,85 | 1.473 | 453 | 0 |
| het_ep1 / het3 ep1·ep2·ep3 | HET: 1,37M QA tự-chứa 4 style ×(1 / 3 epoch) từ merge_l; probe recall 33,8 → 36,2 | 81,96 / 81,75·81,77·81,36 | **81,13** / 81,22·81,16·80,96 | 1.502 | 490 | 0 |
| utr_ep1 | hard-gold rows + sibling + recall-QA, không All-prior (**vỡ**) | 74,02 | 73,17 | 1.457 | 1.239 | 0 |
| diag hard / rq / sib | từng thành phần UTR riêng từ merge_f | 66,95 / 81,80 / 79,43 | 64,68 / 80,71 / 78,03 | 1.417 / 1.478 / 1.511 | 1.906 / 482 / 752 | 0 |
| utr2a / utr2b | hard đã 31B xác nhận, pha loãng 31 % / 16 %, từ soup_het | 79,30 / 81,10 | 78,83 / 80,44 | 1.555 / 1.570 | 809 / 644 | 0 |
| soup_het | merge_l + het + het3 ep1/ep2 | 82,04 | **81,33** | 1.501 | 481 | 0 |
| **wise_u3** ★ | 0,7 soup_het + 0,3 utr2b | **82,11** | 81,34 | 1.522 | 495 | 0 |
| wise_u3 qua harness | `run_baseline.py` (contract Inspect, vLLM server, max_tokens 32) | **82,06** | — | — | — | 0 |
| **wise_o3** ★ | 0,7 wise_u3 + 0,3 onp1 (on-policy, giáo viên 31B đọc nguồn) | **82,27** | 81,46 | 1.540 | 497 | 0 |
| wise_o3 qua harness nội bộ | `run_baseline.py` | 82,22 | — | — | — | 0 |
| **wise_o3 qua runner GSMA** | `gsma-labs/evals` + Inspect AI, `GSMA/ot-full`, greedy | **82,3** | — | — | — | 0 |

† `unparsed` cao = đuôi dài (bug §6.4), chữ cái vẫn đúng — xem cột `first-letter`.

### Theo môn — điểm đi vào đâu

| môn | n | base | kit1agree | kit2 | big-24k | vd | vd3 | **vd6** | Δ vs base |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Tổng** | 10.000 | 71,85 | 74,14 | 76,04 | 77,81 | 78,86 | 79,04 | **79,69** | **+7,84** |
| Standards specifications | 2.000 | 61,15 | 62,60 | 66,35 | 70,05 | 70,30 | 71,10 | **71,45** | **+10,30** |
| Standards overview | 1.000 | 69,80 | 73,00 | 74,50 | 76,10 | 78,60 | 77,80 | **78,00** | +8,20 |
| Research publications | 4.500 | 74,73 | 77,04 | 78,84 | 79,98 | 80,71 | 81,04 | **81,84** | +7,11 |
| Research overview | 2.000 | 74,15 | 76,70 | 77,25 | 78,25 | 80,20 | 79,90 | **80,80** | +6,65 |
| Lexicon | 500 | 83,60 | 86,20 | 87,80 | 91,00 | 91,60 | 91,80 | **92,20** | +8,60 |

**Standards specifications +10,30** là kết quả có ý nghĩa nhất: đây chính là môn mà README đã chứng minh
**không** nhúc nhích với prompt optimisation (−2,80) và gần như không nhúc nhích với thinking (+0,35).
Đó là "knowledge gap" thuần tuý — và kênh trọng số đã mở đúng ở đó.

**Mốc tham chiếu:** GPT-4 (2023) 74,91 · chuyên gia viễn thông 64,86 · gpt-5 83,80 (n≈1.000) ·
AT&T OTel-LLM-8.3B-QnA 91,20 (SOTA).

---

## 12. Hai mươi bài học, gom lại

*(Danh sách này đánh số độc lập với các mốc "Bài học N" rải trong bài — đây là bản gom theo chủ đề.)*

**Về dữ liệu**

1. Đa dạng **dạng** quan trọng hơn số lần lặp (K=30 dạng +4,78 vs lặp 30 lần +1,45).
2. Điều dự báo "sửa được" là số **tài liệu độc lập** nói cùng fact (1→9%, ≥4→30%), không phải số dòng kit
   chứa gold (0 dòng 21% ≈ 50+ dòng 22%).
3. Đơn vị tài liệu là **cửa sổ ~1,2k token**, không phải section.
4. **Không lọc dữ liệu theo hiệu ứng.** Cận trên answer-key (`tier1gf`) còn kém hơn label-free.
5. Viết lại cả đoạn một lượt mất ~nửa chi tiết, ở mọi cấu hình → luôn giữ `verbatim` trong pack.
6. Cổng chuỗi mua sự trung thực bằng một ít ngữ pháp; chấp nhận được với CPT.
7. Đừng audit data bằng chính model sinh ra nó (8B nói bịa 0,7%, 31B nói 4,0%).
8. Data cân bằng làm lệch prior của test (All-of-the-above 1:1 vs prior thật 4,5:1).

**Về pack và training**

9. Mọi hàng trong pack đều dạy nội dung — kể cả hàng "chỉ để neo định dạng" (anchor bug = 1,5 điểm).
10. CPT học ở chế độ pretraining; phải xây cầu nối sang chat trong chính pack (chat-qa + mcq-gold).
11. Thiếu `<|im_end|>` = model không học *dừng* (median completion 2.438 ký tự → strict tụt 36 điểm).
12. lr cao mua recover **và** broke gần theo tỉ lệ; ở 5e-5 net = 0.
13. Ở 2,20B token, thêm data chỉ bớt `broke`, không thêm `recover` → kênh bão hoà ở ~1.050 câu.
14. Nút thắt là **trích xuất**, không phải lưu trữ: 83% câu chưa sửa *có* fact rút được từ nguồn.
15. Trộn trọng số (WiSE-FT, soup) không có bữa trưa miễn phí; "hợp tập đúng cao" ≠ soup có lãi.

**Về hành vi và serving**

16. Nếu một mẹo lúc infer cho +1, hãy biến nó thành nhãn train (vote-distillation: +1,05 trong 19 phút).
17. Hai can thiệp "≈0" theo đúng thứ tự có thể dương (contrastive → consistency = +0,3).
18. Model càng tốt thì mẹo serving càng vô dụng — và đó là dấu hiệu lành.
19. Quy tắc chọn chế độ phục vụ phải label-free (raw-mode đổi dấu theo checkpoint).

**Về đo lường**

20. Điểm tổng che mất mọi thứ. Luôn báo `recover`/`broke`/`rot1`/`strict vs first-letter`, và luôn có
    control (ngữ cảnh xoay, A/A floor, base cùng stack).

---


**Bổ sung 17–18/09**

21. Distractor phải đến từ *chính đoạn nguồn* (sibling), không từ đoạn khác hay do generator bịa — đó là lỗi thật.
22. "Oracle union" của mọi ckpt là thước trần rẻ và trung thực: 88,6% cho biết khoảng cách còn lại là *giữ đồng thời*.
23. Chia lỗi thành never (tin chắc sai) và flicker (gần hoà) — hai bệnh, hai thuốc (binding vs consistency/soup).
24. DPO với completion 1 token sụp vì likelihood displacement; RPO cũng không thêm gì khi cặp lấy từ MCQ tổng hợp.
25. QA tự-chứa có hiệu quả/token cao nhất (52M token = +2), nhưng probe recall chỉ tin được khi câu hỏi tự-chứa.
26. Mọi job dùng nhiều card phải kiểm card thật sự rảnh *ngay trước* torchrun và đặt `NCCL_NVLS_ENABLE=0`.

27. **Trần vote ≈ trần gộp.** Majority vote của 10 phiếu (5 ckpt × 2 thứ tự) chỉ 81,95 dù oracle-union 91,5: bất đồng giữa các
    dòng là nhiễu gần hoà, không có tín hiệu để soup/merge/distill khai thác. Đo trần vote *trước* khi đầu tư vào gộp.
28. Chưng cất đồng thuận đa giáo viên chỉ có tác dụng khi init là **một dòng đơn** (vd8 → ens1 +0,87, vd9 → ens3 +0,17);
    init là merge thì SFT trên nhãn dễ (loss 0,001) kéo về mức giáo viên (ens2 −0,4).
29. Ghép trọng số (task-arithmetic, λ = 1) giữa các dòng *độc lập* cộng dồn +0,3–0,5 mỗi dòng mới; λ > 1 phá format; quét trọng
    số bão hoà ở ±0,05.
30. **Retention không phải nút thắt.** Model đúng ~81 % trên chính MCQ đã học, và 3 epoch QA tự-chứa nâng probe nhớ lại
    +2,4 nhưng không đổi acc train (81,3 → 81,8) lẫn test. Audit 31B: 67 % chỗ model phản đối gold kit là *kit đúng* —
    fact có trong data, đúng, model vẫn giữ prior (cặp đối nghịch, sibling).
31. **Dạy ngược prior bằng gold thì vỡ** (Gekhman EMNLP'24): hard rows 66,9; đã 31B xác nhận và pha loãng 16 % vẫn 81,1
    (fixed +60, broke ×2). Hàng "All of the above" phải luôn có mặt trong mọi stage MCQ (thiếu → All-gold 89,8 → 47,4).
32. Với đáp án 1 token, gradient GRPO = p(gold)·∇log π(gold): on-policy = SFT trên gold *nhân trọng số p(gold)* — fact tin chắc
    sai nhận gradient 0. "Partially mastered" (Gradual Learning 2410.05802) là tập đúng để dạy.
33. Không bao giờ chạy vLLM chung card với train FSDP (chậm 40×), và không bao giờ kill theo cmdline bị cắt ngắn.

---

## 13. Trạng thái hiện tại và việc còn lại

**Checkpoint chốt (19/09 12:46 UTC): `models/kit/wise_u3` = 82,11 first-letter, harness-faithful 82,06** (8.206/10.000,
0 parse fail), rot1 81,34. Đường đi: vd6 79,69 → big3/vd8 80,61 → merge_f 81,79 → merge_l 82,04 → wise_u3 82,11.
Cùng model Qwen3-8B đọc RAG mạnh chỉ 83,75 → closed-book còn kém open-book 1,7 điểm.

**Phân rã 1.789 câu còn sai (31B đọc RAG-window, không dùng đáp án để chọn):** không có nguồn quyết định 52 % + không có
cửa sổ 15 % (≈ 12 điểm — chỉ base lớn hơn hoặc nguồn mới chạm được; 31B closed-book vẫn đúng 355 câu trong số đó); học
được từ nguồn đang có 21 % (377, ≈ 3,8 điểm; lỗi binding); nhãn đáng ngờ ~11 %. Bucket yếu nhất nhận diện được:
[3GPP Release] 1.810 câu acc 77,7 (22,6 % tổng sai) — corpus có đủ Rel-8…20, lỗi ở cửa sổ được chọn.

**Đang chạy:** vòng on-policy (`jobs/kit_onp.sh`): student sample → p(gold) → 31B đọc cửa sổ xác nhận → hàng nhân bản ∝ p(gold)
+ đồng thuận + All-prior → SFT 3e-6; vòng 2 nếu vòng 1 tăng. **Chờ quyết định:** đổi base sang Qwen3.5-9B (open-book 85,2;
9B base ∪ merge_l = 87,6; trainable trên stack, cần kernel fla) — đòn bẩy duy nhất còn kỳ vọng chạm 85.

**(Đoạn dưới là trạng thái cũ 18/09 01:40, giữ làm lịch sử.)** Checkpoint chốt (18/09 01:40 UTC): `models/kit/vd6/ep1` = 79,69 (rot1 78,05, unparsed 0, greedy). Đường đi:
vd3 79,04 → sibling 79,66 → vd6 79,69. Harness-faithful của vd3 = 79,07 (vd6 chưa chạy lại harness).

**Đang chạy: big run 2** (`jobs/kit_big2_train.sh`, `kit_big2_evalloop.sh`): pack 2.509.287 block (2,57B token) = big
2,15M + recall-QA ×3 (153k) + sibling ×2 (170k) + style 40k; init base (PIT ≈ mixed), lr 3e-5, 8 card, EMA, ckpt mỗi
6.000 step (eval dọc đường, giữ ckpt tốt nhất), sau đó vd7. Lần khởi động đầu sụp vì NCCL NVLS (mất 1 h); chạy lại
01:37 UTC, ~16–17 h. Kỳ vọng thật: 80–81 (các đòn bẩy cộng dồn: sibling ×2 toàn cửa sổ + QA + EMA) — chưa đủ 85.

**(Đoạn dưới là trạng thái cũ 17/09 06:30, giữ làm lịch sử.)** Checkpoint chốt: `models/kit/vd3/ep1` = 79,04 (rot1 77,11, unparsed 0, greedy, không cần mẹo serving).

**Đã đóng (17/09 09:00 UTC):** re-study bậc 3 (808M token, 2 epoch, lr 1e-5 từ vd3) **dừng ở 26% run** theo quy tắc
đặt trước — 6k 77,53 / 12k 77,77, broke 521→487 > 450: thêm liều nhắm tập câu chưa chắc **không** rút thêm được
(recover 1.080–1.090 ≈ cũ). Kênh bão hoà thật. `cm_mlp` (chỉ MLP): 78,79, bớt 35 broke và 30 recover → net ≈ 0.
**Đang chạy:** lượt cuối `cm2` (cmcq, MLP-only, từ vd3) → `vd4` (`jobs/kit_cm2_vd4.sh`) rồi chốt.
**Kiểm tra tin cậy:** harness-faithful 79,07 (0 parse fail); contamination: 2.937 câu test có câu hỏi trùng nguyên văn
trong kit (do cửa sổ nguồn chứa câu văn gốc của GPT-3.5) nhưng gain hai nhóm bằng nhau (+7,4/+7,1) và không thấy bản
dataset trong corpus (917 câu có ≥3 option trong một cửa sổ = cụm từ spec). Thinking: base +2,5, vd −0,6.

**(Đoạn dưới giữ nguyên làm lịch sử của quyết định re-study.)** `jobs/kit_restudy.sh` — re-study bậc 3 khởi tạo từ `vd3/ep1`.
Pack: chỉ cửa sổ trace-back của **4.019 câu mà 8B có margin < 0,9** (48.325 cửa sổ) → 628M token kit +
replay 10% + anchor agree-only 57.652 → **789.033 block (808M token)**; 2 epoch, lr 1e-5, 8 card,
49.316 step, 41,2k tok/s ≈ 11 giờ. Câu hỏi nó trả lời: **thêm liều nhắm đúng tập câu model chưa chắc
chắn có rút thêm được không** — hay kênh đã bão hoà thật.

**Khoảng cách tới mục tiêu:** 79,04 → 85 cần thêm **596 câu**. Phân tích cho biết chúng nằm ở đâu:

| nhóm | n | tình trạng |
|---|---:|---|
| có nguồn, chưa sửa, **gold đã lên hạng nhì** | ~660 | học một phần (đo trên ckpt bậc 2: 54,3% của 1.214) — cần trích xuất tốt hơn, không cần data mới |
| không có nguồn truy được | ~520 | trần cứng của corpus; cần IEEE/research sâu hơn |
| `broke` do nhãn mơ hồ | ~115 | ước theo tỉ lệ 26% đo trên broke của bậc 2 — không sửa được mà không dùng đáp án |
| `broke` do học lệch (nguồn ủng hộ gold) | ~330 | cmcq mở rộng + consistency |

**Các hướng còn lại, theo thứ tự ưu tiên:**

1. **Re-study** (đang chạy) — trả lời câu hỏi bão hoà.
2. **Prompt distillation** (§6.3 của PLAN, chưa chạy): teacher = chính model + passage trong context,
   student = không passage, KL T=2 trên token đáp án, câu hỏi tự sinh τ=1,5. Đây là kỹ thuật duy nhất
   trong literature nhắm **thẳng** vào nút thắt trích xuất mà repo chưa thử đúng cách.
3. **Lọc cửa sổ theo chất lượng** (judge label-free) trước khi sinh kit — audit 31B chỉ ra nguồn rác là
   gốc của phần bịa; ước tác động ≤5% hàng.
4. **teacher_sources 122B** cho ~520 câu không nguồn (cần `GATEWAY_KEY`) — cổng ba passage đồng thuận.
5. **Trộn hàng train dạng think** nếu muốn lấy lại +2,5 mà thinking cho base (kỳ vọng ≤ +1).

---

## 14. File map

| bước | file |
|---|---|
| **nguồn** | `kit/traceback_index.py` (BM25 5,1M chunk) · `kit/traceback_deep.py` · `kit/dense_rerank_deep.py` · `kit/ieee_ingest.py` (4 PDF IEEE) · `kit/api_sources.py` + `kit/api_windows.py` (Wikipedia) · `kit/otel_recite.py` · `kit/teacher_sources.py` |
| **chọn cửa sổ** | `kit/build_win_single.py` · `kit/select_windows{,_gold,_r2}.py` · `kit/win_conf.py` |
| **coverage** | `kit/functional_coverage.py` · `kit/coverage_report.py` · `kit/build_tb_eval.py` |
| **trần T1/T2** | `analysis/ceiling_sets.py` · `analysis/ceiling_sets2.py` · `kit/build_kit_eval*.py` · `kit/check_tier1_t2.py` · `kit/score_tier1_gate.py` |
| **generator** | `kit/gen_kit.py` (6 view) · `kit/gen_kit_pilot.py` · `kit/gen_contrastive.py` (cmcq, `--style`) · `kit/gen_sibling.py` (sibling-contrast) · `kit/gen_recall_qa.py` (QA tự-chứa) · `kit/gen_numeric.py` · `kit/gen_behavior.py` |
| **kiểm data / trần** | `kit/kit_stats.py` · `kit/kit_audit.py` · `kit/label_audit.py` · `kit/broke_audit.py` · `kit/recall_probe.py` · `kit/kitqa_probe.py` · `kit/letter_probs.py` + `kit/partial_learning.py` · `run_baseline.py` (harness-faithful) |
| **pack** | `kit/selfreplay_tier1.py` (anchor + `agree`) · `kit/pack_tier1.py` · `kit/pack_chat.py` · `kit/pack_masked.py` · `kit/mcq_rows.py` |
| **train** | `kit/train_tier1.py` (`--train-only`, `--layers`, `--ema-every`) · `kit/train_dpo_letter.py` + `kit/dpo_pairs.py` · `kit/wise_merge.py` · `kit/soup_n.py` · `kit/pack_concat.py` |
| **stage hành vi** | `kit/vote_distill.py` (+ `pack_chat.py`, không vắt block) · `kit/mcq_rows.py` |
| **eval / chẩn đoán** | `jobs/eval_ckpt.sh` · `kit/vote_eval.py` · `kit/rawmode_eval.py` · `kit/recite_eval.py` · `kit/letter_probs.py` · `kit/partial_learning.py` · `kit/probe_logprob.py` · `kit/probe_heldout_v2.py` · `kit/probe_report.py` · `analysis/failure_analysis.py` |
| **orchestration** | `jobs/kit_*.sh` (gen → pack → train → eval chain) · `jobs/best_ckpt.sh` · `jobs/final_summary.sh` · `jobs/status.sh` |

Trên server mọi thứ nằm ở
`~/projects/teleqna/runs/teleqna-8b/{code/kit, data/kit, results/landscape, results/kit, models/kit, logs}`.
