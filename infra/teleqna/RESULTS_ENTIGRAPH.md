# EntiGraph-inspired synthetic CPT, full weight: +0,65pp — và nó không phải kiến thức

*Chạy 2026-08-21 trên hgx046, 8×H200. Code: `runs/teleqna-8b/code/eg/`.
Không dòng benchmark nào vào train. Không RAG lúc serving.*

## 1. Đã làm gì

Theo [arXiv:2409.07431](https://arxiv.org/abs/2409.07431) (ICLR'25 oral), có sửa
và **phải gọi đúng tên là EntiGraph-inspired**, không phải reproduction:

| | paper | run này |
|---|---:|---:|
| token nguồn | 1,3M | 24,1M |
| token synthetic | 455M | **58,9M** |
| khuếch đại | 350× | **2,44×** |
| pair mỗi document | **tất cả** | 6 ngẫu nhiên + 2 triple |
| phủ cạnh đồ thị entity | 100% | **26,0%** |
| token synthetic / câu hỏi test | 98.700 | **5.889** |
| replay | RedPajama 10% (tổng quát) | self-replay MCQ (đặc thù task) |

Pipeline: 44.816 chunk sạch (dedup 38.370, loại 2 chunk chứa nguyên văn câu hỏi
benchmark) → 148.592 entity → 313.712 lượt relation-analysis → scan sau sinh
loại 68 passage (8-gram câu hỏi + đúng gold option) → 313.644 passage / 57,4M
token + anchor 97.594 dòng → full-weight Qwen3-8B, DDP 8 GPU, lr 1e-5 cosine,
2 epoch, 4.308 step, 41k tok/s.

## 2. Kết quả

**ot-full, 10.000 dòng, closed-book:**

```
base   71,87%
ep2    72,52%      delta +0,65pp
fixed 414  broke 349  net +65  discordant 763
McNemar p = 0,020        bootstrap 95% CI [+0,10, +1,20]
unparsed: base 0, ep2 0     gen-len p50 = 9 ký tự
```

**Đây là lần đầu tiên trong dự án một arm training closed-book vượt base một
cách có ý nghĩa thống kê.** Và full-weight CPT ở lr 1e-5 **không phá hợp đồng
trả lời** — bác bỏ dứt điểm cách giải thích "CPT thất bại vì drift format".

Theo môn:

| môn | n | base | ep2 | delta |
|---|---:|---:|---:|---:|
| Lexicon | 500 | 83,20 | 86,40 | **+3,20** |
| Standards specifications | 2000 | 60,85 | 61,65 | +0,80 |
| Research publications | 4500 | 74,84 | 75,38 | +0,53 |
| Research overview | 2000 | 74,15 | 74,60 | +0,45 |
| Standards overview | 1000 | 70,30 | 70,30 | 0,00 |

(dev-1000 cho Standards spec **−0,50**; trên 10.000 dòng nó là **+0,80**. Tín
hiệu theo môn ở n=1000 là nhiễu — đừng đọc bảng môn dưới vài nghìn dòng.)

## 3. Phép thử giết chết giả thuyết kiến thức

Nếu CPT dạy được fact từ corpus, phần tăng **bắt buộc** phải tập trung ở những
dòng mà corpus **có chứa** đáp án. Chia theo đúng tiêu chí đó:

| | n | base | ep2 | **delta** |
|---|---:|---:|---:|---:|
| **gold text CÓ trong corpus** | 3.483 | 72,93 | 72,95 | **+0,03** |
| gold text KHÔNG có trong corpus | 6.517 | 71,31 | 72,29 | **+0,98** |

**Toàn bộ phần tăng nằm ở những dòng corpus không chứa đáp án.** Ở đúng chỗ ta
biết chắc kiến thức có mặt, lợi ích là **không**.

Tập "có trong corpus" được xác định bằng khớp chuỗi nguyên văn nên **độ chính
xác cao** (ít dương tính giả). Nó cho **+0,03**. Không có cách đọc nào biến con
số đó thành "kiến thức đã vào nhưng còn ít" — một hiệu ứng bị thu nhỏ vì thiếu
liều vẫn phải **cùng dấu và tập trung đúng chỗ**, chứ không bằng không đúng chỗ
và dương ở chỗ khác.

Vậy **+0,65pp là thật nhưng không phải kiến thức.** Nó là thích nghi văn phong /
thuật ngữ / hiệu chỉnh — và bằng chứng phụ khớp với cách đọc đó: môn lãi nhất là
**Lexicon +3,20**, tức thuật ngữ và viết tắt, chứ không phải Standards spec.

## 4. Corpus không giàu hơn ở chỗ model sai

| | n | gold có trong corpus |
|---|---:|---:|
| base ĐÚNG | 7.180 | 35,14% |
| base SAI | 2.820 | **34,04%** |

Bằng nhau. Corpus **không** mang nhiều thông tin hơn về những câu model trượt.

```
base sai VÀ gold có trong corpus :   960 dòng =  9,60pp   <- trần tuyệt đối
base sai VÀ gold KHÔNG có        : 1.860 dòng = 18,60pp   <- không phương pháp
                                                             corpus-only nào với tới
```

Lưu ý trung thực: khớp nguyên văn là **cận dưới** của độ phủ (lựa chọn thường là
diễn giải lại), nên 34,83% đánh giá thấp độ phủ thật. Nhưng **so sánh tương đối**
giữa hai nhóm không bị lệch bởi thiên lệch đó, và nó phẳng.

## 5. Kết luận

- Full-weight CPT **giữ được format** ở lr 1e-5. Giả thuyết drift-format: bác bỏ.
- Synthetic CPT cho **+0,65pp có ý nghĩa** — đòn bẩy training hợp lệ đầu tiên
  của dự án vượt được nhiễu.
- Nhưng nó **không phải kênh kiến thức**: lợi ích bằng 0 ở đúng nơi kiến thức có
  mặt, và corpus không giàu hơn ở nơi model sai.
- Scale lên 230M token sẽ khuếch đại **cùng cơ chế đó** — thích nghi văn phong —
  chứ không mở kênh kiến thức. Cận trên hợp lý là vài điểm, không phải +16,73 như
  paper, vì điều kiện tiên quyết của paper (test query hỏi về chính nội dung
  corpus) chỉ đúng với ~35% benchmark này, và ở 35% đó hiệu ứng đang bằng 0.

---

# 6. Phép thử liều (2026-08-21, chiều): liều tăng 3,8× đúng chỗ — kênh vẫn đóng

Cách đọc duy nhất còn sống sau mục 5 là *"chưa đủ liều"*. Nên liều được đổ vào
**đúng và chỉ** nơi kiến thức có thể đo được.

**Thiết kế.** 3.483 dòng có gold nằm trong corpus → tìm chunk **thật sự chứa**
gold đó (cap 6 chunk/dòng, ưu tiên chunk hiếm để một thuật ngữ phổ biến không
kéo cả corpus vào) → **9.055 chunk mục tiêu = 20,2% corpus**. Sinh **toàn bộ**
cạnh entity còn thiếu cho riêng chúng: **6 → ~23 pair/chunk (×3,8)**, thêm
155.345 passage / 30,8M token. Train lại **từ base**, cùng hyperparameter,
grad-accum 2→4 trên 4 card để **giữ nguyên effective batch 32** của run-1. Anchor
tự co giãn giữ 24,6% sequence (run-1: 23,7%). Khác biệt duy nhất giữa hai run là
liều trên nhóm mục tiêu.

| | run-1 | run-2 (liều ×3,8) |
|---|---:|---:|
| token synthetic | 57,4M | **88,2M** |
| pair/chunk mục tiêu | 6 | **~23** |
| ot-full toàn bộ | 72,52 (+0,65) | **72,50 (+0,63)** |
| McNemar | p=0,020 | p=0,027 |

**Trên nhóm mục tiêu — nơi liều tăng gấp gần 4 lần:**

| | run-1 | run-2 | |
|---|---:|---:|---|
| delta (gold CÓ trong corpus) | +0,03 | **−0,34** | |
| **đi đúng hướng** | **50,2%** | **47,9%** | tiêu chí đặt trước: **>55%** |
| fixed / broke | 127 / 126 | 137 / 149 | |

| | run-1 | run-2 |
|---|---:|---:|
| delta (gold KHÔNG trong corpus) | +0,98 | **+1,15** |
| đi đúng hướng | 56,3% | 57,5% |

**Trượt tiêu chí, và trượt theo hướng ngược.** Tăng liều 3,8× đúng vào các tài
liệu chứa đáp án làm nhóm đó **tệ đi** (50,2 → 47,9, tức dưới mức tung đồng xu),
trong khi toàn bộ phần lãi vẫn nằm ở nhóm corpus **không** chứa đáp án và còn
nhích lên (+0,98 → +1,15).

Thêm 30,8M token nhắm đích đổi điểm tổng đi **−0,02pp**. Bằng không.

## 7. Kết luận cuối của lane này

- **Kênh kiến thức corpus → weights: đóng.** Không phải vì thiếu liều — liều đã
  được tăng 3,8× đúng chỗ và kết quả đi xuống. Không phải vì drift format —
  unparsed 0, gen-len p50 = 9 ở cả hai run. Không phải vì corpus bẩn — 0 dòng
  chứa câu hỏi nguyên văn, 179 dòng nghi ngờ đã bị loại.
- **Cái có thật là +0,65pp thích nghi văn phong/thuật ngữ**, và nó **bão hoà
  ngay**: 57M token cho +0,65, 88M token cho +0,63.
- Trần vật lý đã đo: **1.860 dòng (18,6%)** là base sai **và** đáp án không tồn
  tại trong corpus — không phương pháp corpus-only nào chạm tới.

**Điểm hợp lệ tốt nhất của Qwen3-8B closed-book, không RAG, không train trên
benchmark: 72,52.**

---

## 8. Run 3 — pre-registration, written before the eval existed

Timestamp: 2026-08-22 05:0x UTC. `results/eg/otfull_eg3.jsonl` does not exist yet;
training is at step ~550/5502. Everything below is fixed now so it cannot be
tuned to the result.

### 8.1 What run 3 changes, and what it does not

| | run 1 | run 2 (dose) | **run 3** |
|---|---:|---:|---:|
| source document unit | chunk, 540 tok | chunk, 540 tok | **3GPP section, 4,769 tok** |
| entities per document | 7 | 7 | **40** |
| relation calls per document | 6 | 23 | **~734** |
| amplification per document | 3.7x | 8.6x | **35.6x** |
| synthetic tokens after gates | 57.4M | 85.7M | **77.0M** |
| anchor | x7, 23.7% of sequences | x11, 24.6% | x7, 23.9% |

Run 3 carries **fewer** tokens than run 2 and only 1.34x run 1. It is therefore
not a dose experiment — run 2 is the dose-matched control that already went
down. The only variable moved is how many distinct rewritings each fact gets,
which is the variable the K-sweep found a real learning term on (+3.33pp at
>=10 forms per fact, matched tokens).

### 8.2 Corpus gates actually applied

- faithfulness: passage dropped if <70% of its capitalised atoms appear in the
  source section — 6,686 dropped (2.0%)
- `no_relationship` passages dropped: 10,579 (3.4%) — zero-information, and they
  teach the model to answer "no relationship is stated"
- style cleanup: assistant preamble stripped (19.1% of passages carried one),
  markdown stripped (67.3% carried it). A CPT corpus that teaches those tokens
  spends its gradient on style, which is the drift channel prior runs lost to.
  After cleanup: residual preamble 0.27%, residual markdown 0.
- benchmark gate: passage carrying a test question 8-gram **and** its gold option
  verbatim — 208 dropped
- exact duplicates: 0
- kept: 310,832 passages / 77.0M tokens / 88,012 blocks of 1024

### 8.3 The estimator, and why the original one was not usable

The pre-registered read-out was raw `TARGET - OTHER > +1.0pp`. Running it on
runs 1 and 2 — which had no multiplicity and so should read zero — gives
**+0.93pp and +0.60pp**. The threshold sat inside its own null band.

The confound is difficulty: TARGET is at base 63.13%, OTHER at 74.09%, so any
global gain lands harder on TARGET. Standardising on the binary base outcome
alone (v2) over-corrects to -1.11 / -1.50pp, because a base-correct row drawn
from a hard pool is a more marginal correct and breaks more readily on its own
(TARGET mean max-p 0.9251 vs OTHER 0.9610).

**v3**, `code/eg3/groups_eval3.py`: strata = base_correct x margin quintile,
stratum weights from TARGET, flip rates from OTHER, bootstrap on the difference
with both groups resampled jointly. On the two null runs it reads:

| | raw TARGET-OTHER | v2 | **v3** |
|---|---:|---:|---:|
| run 1 | +0.93pp | -1.11pp | **+0.19pp, p=0.842** |
| run 2 | +0.60pp | -1.50pp | **-0.21pp, p=0.779** |

v3 is flat on both, with 95% CI half-width ~1.5pp.

### 8.4 The criterion for run 3

**KNOWLEDGE v3 > +1.5pp with a 95% CI excluding 0.** Anything smaller is inside
the band that runs 1 and 2 already occupy without any multiplicity.

Secondary, reported either way:
- HOLDOUT delta — transfer to sections never synthesised over
- OTHER delta — global/style term
- overall ot-full delta and `unparsed`, so a format collapse cannot be read as a score
- per-stratum fix/break tables, so a floor effect cannot hide inside the summary

Three readings fixed in advance:
1. v3 > +1.5 and CI excludes 0 -> multiplicity opens the weights channel; scale the section count.
2. v3 inside the band with OTHER > 0 -> third confirmation that the gain is style, not knowledge; the corpus-only weights route is closed and the remaining lever is retrieval or an independent verifier.
3. HOLDOUT >> 0 while v3 is flat -> general transfer, not fact-specific knowledge.

### 8.5 Group integrity, checked before the verdict

Verified against `data/eg2/windows.jsonl`, `data/eg2/qsplit.json` and
`data/eg3/sections.jsonl` (23,170 sections, 447 selected):

| group | has an eg2 source window | inside a selected section | role |
|---|---|---|---|
| TARGET 1,904 | 1,904 / 1,904 | all | where the effect has to land |
| OTHER 6,596 | 6,542 / 6,596 | none | style control |
| HOLDOUT 1,500 | **0 / 1,500** | none | transfer control |

HOLDOUT is not merely unselected: no window of any holdout question exists in
any of the 23,170 sections, so nothing about them could have been synthesised.
Leakage of a holdout question into a selected section: **0**.

The greedy selection also ran to exhaustion, not to its 700-section budget:
1,904 seen questions are reachable by any section at all, and the 447 chosen
sections cover all 1,904. There is no unselected section that could have added
a target question.
