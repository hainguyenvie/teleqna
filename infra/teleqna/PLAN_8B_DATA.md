# Chiến lược dữ liệu: chứng minh tri thức có thật trước khi train

Viết 2026-08-14, nối tiếp [`PLAN_8B.md`](PLAN_8B.md). File kia trả lời "đi đâu";
file này trả lời "lấy dữ liệu ở đâu và làm sao biết nó tốt". Số liệu teacher lấy
từ [`analysis/teacher_map.py`](analysis/teacher_map.py).

**Luật nền, kế thừa từ Chapter 2:** mọi tiêu chí phải là **hành vi đo được**,
không phải phán xét của một LLM judge. Confusability và resolvability đã được
làm theo luật này; grounding cũng phải vậy.

---

## 1. Evidence ledger — một bảng 10.000 dòng, mọi thứ khác đọc từ đó

Không dựng thêm bộ dữ liệu rời rạc nữa. Một artifact duy nhất, khoá theo
`sample_id`, ba khối cột:

**Khối PRIOR — model tự nó có gì (không context)**

| cột | cách lấy |
|---|---|
| `p_prior` | tỉ lệ đúng trên 8 mẫu closed-book, prompt gốc của harness |
| `margin` | khoảng cách phiếu giữa lựa chọn 1 và 2 trong 32-vote |
| `stable_4view` | đúng ở bao nhiêu trong 4 cách trình bày (đã có cho Qwen3-8B: 5.994 / 2.391 / 1.615) |
| `prior_class` | `known-solid` (≥6/8 và ổn định) · `shaky` · `unknown` |

**Khối EVIDENCE — corpus cho được gì**

| cột | cách lấy |
|---|---|
| `passages` | top-k của cấu hình retrieval **mạnh** (89,5% answer-in-context, không phải cấu hình 43,0% đã deploy) |
| `co_presence` | số lựa chọn (kể cả distractor) xuất hiện trong cùng một cửa sổ — tín hiệu chất lượng retrieval không cần nhãn, đã đo: 0 distractor → +0,42 · 1 → +4,48 · ≥2 → +11,20 |
| `p_ctx` | tỉ lệ đúng trên 8 mẫu khi có passage |
| `span` | câu model trích ra làm bằng chứng |
| `span_verified` | `span` có phải substring thật của passage không (so khớp chuỗi đã chuẩn hoá) |
| `cf_follows` | model có đổi đáp án theo passage đã bị sửa ngược không |
| `foreign_ok` | model có còn đúng khi đưa passage lạc đề cùng độ dài/văn phong không |
| `verdict` | xem §2 |

**Khối LABEL — kết luận**: `label`, `tier`, `label_source`, `excluded_reason`.

Ledger này là thứ được commit và review, không phải các file `*.jsonl` rời.

---

## 2. Bốn phép thử grounding — trả lời "nó giải được thật hay ảo giác"

Đúng câu hỏi bạn đặt: base + RAG trả lời đúng **không chứng minh** tài liệu chứa
tri thức. Bốn phép thử, chạy trên cùng một (câu hỏi, passage):

1. **Lift** — `p_ctx − p_prior > 0`. Điều kiện cần, quá yếu nếu đứng một mình.
2. **Span-citation, kiểm bằng chuỗi** — bắt model trích nguyên văn câu trong
   passage đỡ cho đáp án; ta so khớp substring. Trích dẫn bịa **trượt cơ học**,
   không cần ai phán xét. Đây là phép thử chống ảo giác rẻ nhất và không gian lận được.
3. **Counterfactual swap — phép thử quyết định.** Sửa passage một lượng tối
   thiểu để nó đỡ cho một *distractor* thay vì gold (đổi con số, đổi tên bản tin,
   đổi release). Hỏi lại. Nếu đáp án **đi theo passage đã sửa** → model đang thực
   sự đọc. Nếu không đổi → cái đúng lúc nãy là prior/định kiến chứ không phải
   grounding. Đây chính là bucket "conviction" (93 dòng, 19,5% phần dư RAG) mà
   `ctx_failure_anatomy.py` đã tìm ra nhưng chưa dùng làm cổng lọc.
4. **Foreign-passage control** — passage lạc đề, cùng độ dài và văn phong. Vẫn
   đúng → passage không phải nguyên nhân. Control này đã được hiệu chuẩn ở phép
   đo trần: explanation đúng +22,3 điểm, explanation của dòng khác −0,6 điểm.

**Ma trận phán quyết**

| verdict | điều kiện | dùng làm gì |
|---|---|---|
| `GROUNDED` | lift>0 ∧ span_verified ∧ cf_follows ∧ ¬foreign_ok | **tài liệu chứa tri thức, đã chứng minh** → tier TEACH |
| `PRIOR` | p_prior cao, lift≈0 | tier ANCHOR (giữ, không dạy lại) |
| `CONVICTION` | đúng có context nhưng ¬cf_follows | không tính là grounded; nhãn phải lấy từ nguồn khác |
| `LUCKY` | đúng nhưng ¬span_verified hoặc foreign_ok | loại — đây chính là ảo giác/ăn may bị bắt |
| `CONTRADICTS` | passage grounded nhưng đỡ cho đáp án ≠ gold công bố | **ứng viên nhãn hỏng** — không train, ghi vào ledger |
| `NOT-COVERED` | không passage nào đạt | corpus/retrieval không với tới → không bịa nhãn |

`CONTRADICTS` là sản phẩm phụ có giá trị: ta đã biết 142 dòng (1,42%) không ai
giải được kể cả khi có gold explanation. Train trên một nhãn sai là dạy một sự
thật sai — phải tách ra chứ không để lẫn.

---

## 3. "Tri thức của tập test đã được cover chưa" — trả lời bằng ba con số khác nhau

Ba khái niệm đang bị gộp làm một; phải tách:

| tầng | định nghĩa | số hiện có |
|---|---|---|
| **có tài liệu nguồn** | tồn tại document công khai chứa câu trả lời | 93,5% (9.348/10.000); phần thiếu là 652 dòng IEEE + Bluetooth |
| **retrieval lấy được** | passage lấy về có chứa đáp án (đo lỏng) | 89,5% với cấu hình mạnh, 43,0% với cấu hình đã deploy |
| **grounded đã kiểm chứng** | qua đủ 4 phép thử §2 | **chưa đo — đây là con số cần đo, và nó sẽ thấp hơn 89,5%** |

Deliverable của bước này là bảng coverage theo subject × tag, kèm danh sách dòng
`NOT-COVERED`. Nếu tỉ lệ grounded chỉ đạt ~70%, thì trần của mọi phương pháp
"dạy bằng tài liệu" là ~70% + phần model tự biết — và ta biết điều đó **trước**
khi đốt card, chứ không phải sau ba arm thất bại.

---

## 4. Nội tại vs được dạy — kế toán "học cái mới, quên cái cũ"

Đây là ràng buộc bạn nêu, và dự án đã có đủ số liệu để định lượng nó:

| arm | dòng sửa được | **dòng bị phá** |
|---|---:|---:|
| armG_full (nhãn thật, có tier keep làm neo) | 1.021 | **357** |
| armJ (trộn 30% synthetic) | 961 | 372 |
| merge G+H s=0,6 | 966 | 407 |
| merge G+H s=1,0 | 878 | 595 |
| armH (100% synthetic) | 384 | **926** |

Quy luật đã hiện rõ: **tỉ lệ dòng lạ trong tập train tăng thì số dòng bị phá
tăng theo, gần như tuyến tính.** Cái giữ được phân phối là tier neo — 5.967 dòng
mà chính model trả lời đúng và tự tin, dùng làm target chính câu trả lời của nó
(self-replay anchor: +9,7 điểm trong một contrast có kiểm soát, damage
17,3% → 3,9%).

**Ba quy tắc bắt buộc, đưa vào script chứ không để trong đầu:**

1. **Bảng damage per-bucket phải chạy TRƯỚC khi train.** Với mỗi bucket
   (`prior_class` × `label_source`): n, độ chính xác của base, độ chính xác của
   nhãn, Δ kỳ vọng. **Từ chối bất kỳ tier nào có nhãn kém hơn base trên bucket
   đó.** Đây đúng là cái đã bắt được bẫy "dùng evidence khắp nơi" (headline
   86,83 nhưng lỗ 0,80 vì bucket A: base đúng 99,0%, nhãn chỉ 97,4%).
2. **Tier ANCHOR không phải tuỳ chọn** và phải chiếm ≥50% số dòng train.
3. **`broken` là chỉ số hạng nhất**, báo cáo cùng `fixed` mọi lúc. Điểm là
   `net`. Đặt ngưỡng dừng: `broken > 450` thì arm bị loại bất kể headline.

---

## 5. Teacher: chưng cất OTel-2.0-31B-IT vào 8B

Teacher là **OTel-2.0-31B-IT** — 79,54 closed-book, 87,24 khi mang adapter arm G.
Số liệu từ [`analysis/distill_31b_to_8b.py`](analysis/distill_31b_to_8b.py).
(Bản OTel-LLM-8B-IT đã release là một ứng viên khác và đã bị loại — xem §5b.)

### Teacher mạnh ở đâu, tính theo dòng chứ không theo %

| slice | n | 8B | 31B | 31B+G | 8B sai → 31B+G đúng | 8B đúng → 31B+G sai | net |
|---|---:|---:|---:|---:|---:|---:|---:|
| Research publications | 4.500 | 74,84 | 82,13 | 87,84 | 751 | 166 | **+585** |
| Standards specifications | 2.000 | 61,40 | 70,45 | 82,30 | 521 | 103 | **+418** |
| Research overview | 2.000 | 74,25 | 80,35 | 86,95 | 333 | 79 | +254 |
| Standards overview | 1.000 | 69,70 | 79,20 | 91,50 | 249 | 31 | +218 |
| Lexicon | 500 | 83,40 | 90,00 | 94,20 | 60 | 6 | +54 |
| — tag 3GPP | 1.810 | 60,61 | 70,66 | 85,91 | 527 | 69 | **+458** |
| — tag IEEE | 647 | 67,85 | 76,04 | 78,21 | 118 | 51 | +67 |

Teacher thắng ở **mọi** slice, nhưng giá trị rất lệch: 3GPP là chỗ nó vượt xa
nhất (+25,3 điểm, tỉ lệ đổi 527:69), IEEE là chỗ nó gần như không giúp gì
(+10,4 điểm, 118:51) — khớp với lỗ hổng corpus IEEE đã biết. Nếu phải gate theo
slice thì IEEE là slice duy nhất đáng cân nhắc bỏ.

### Sampling teacher trực tiếp ≈ dùng file nhãn của nó

| nguồn | sửa được bao nhiêu trong 2.805 lỗi của 8B |
|---|---:|
| 31B base | 1.370 (48,8%) |
| 31B + armG | 1.914 (68,2%) |
| **file nhãn armG_full** | **1.955 (69,7%)** |

Chênh lệch hai chiều chỉ 19 và 60 dòng. **Chạy teacher nhiều lượt hơn không phải
chỗ còn headroom** — file nhãn đã mang gần hết tri thức của teacher trên đúng
những dòng cần. Headroom nằm ở tier `flip` (63,28%) và ở cổng lọc dưới đây.

### Cổng lọc: giữ câu trả lời của học trò ở đâu

| phương án | label acc | fix | **broken** | net |
|---|---:|---:|---:|---:|
| A. dùng nhãn teacher khắp nơi (file armG_full nguyên bản) | 87,67% | 1.955 | 388 | **+15,67pt** |
| C. dự đoán của 31B+armG khắp nơi | 87,24% | 1.914 | 385 | +15,29pt |
| F. chỉ nhận tier `keep`/`retain`, còn lại giữ 8B | 83,23% | 1.262 | **134** | +11,28pt |
| G. chỉ nhận khi hai lượt 31B đồng ý, còn lại giữ 8B | 82,43% | 1.196 | 148 | +10,48pt |
| B. dự đoán 31B base khắp nơi | 80,65% | 1.370 | 538 | +8,32pt |
| E. giữ 8B khi no-think == thinking, còn lại teacher | 80,11% | 927 | 112 | +8,15pt |

Và một **trần oracle**, không phải công thức dùng được:

| | label acc | fix | broken | net |
|---|---:|---:|---:|---:|
| D. giữ 8B ở dòng nó đúng cả 4 view, còn lại teacher | **89,65%** | 1.955 | **190** | **+17,65pt** |

D dùng `correct`, tức là **có dùng answer key** — nó không phải recipe. Giá trị
của nó là định giá cái cổng: nếu tín hiệu tự tin của học trò tốt bằng chính độ
đúng của nó, ta được **+2 điểm label accuracy và giảm một nửa số dòng bị phá**.
Xấp xỉ label-free của cổng đó là margin 32-vote của chính 8B trải trên 4 view —
trên 31B, margin ≥0,90 cho tier `keep` chính xác 95,80%. **Đây là lý do bước 0
phải chạy sweep 8 mẫu × 4 view trước khi dựng nhãn**: nó là thứ biến D từ oracle
thành recipe. Proxy rẻ tiền "no-think trùng thinking" chỉ đạt 80,11% (phương án
E), tức là không đủ.

Chọn A hay F là núm vặn điểm ↔ an toàn: A hơn 4,4 điểm nhưng phá gấp 2,9 lần.
Đề xuất: dựng nhãn theo A, rồi áp cổng margin của học trò lên tier `flip` —
đó là nơi 254 trong 388 dòng bị phá đang nằm.

## 5b. Ứng viên bị loại: OTel-LLM-8B-IT (bản 8B đã release)

Ghi lại vì nó từng được nêu như một nguồn tri thức khả dĩ. Trên cùng 10.000
dòng nó đạt 72,20 / 64,00 / 63,64 / 59,10 / 49,55 theo năm subject — **thấp hơn
Qwen3-8B ở mọi slice**.

**OTel-LLM-8B-IT bị loại khỏi pipeline nhãn.** Nó thấp hơn Qwen3-8B ở *mọi*
slice, không có túi năng lực nào để khai thác. Đúng, nó sửa được 895 dòng mà
Qwen3-8B sai — nhưng không có tín hiệu nào nhận ra chúng trước:

```
nhãn armG_full nguyên bản                          labelacc 87,67%  fix 1955  broken 388  net +15,67pt
nhãn armG_full, chỉ giữ khi OTel-8B xác nhận       labelacc 77,68%  fix  688  broken 119  net  +5,69pt
khi OTel-8B đồng ý với OTel-31B, trên dòng 8B sai   đáp án chung đúng 47,37%
```

Cho nó quyền phủ quyết làm mất 10 điểm label accuracy. Ý tưởng "chưng cất chỗ
OTel làm tốt" là hợp lý về nguyên tắc nhưng **bản 8B đã release không có chỗ nào
làm tốt** — bản 8.3B-QnA đạt 91,20 chưa bao giờ được công bố. Teacher thật là
**OTel-2.0-31B-IT**, và 122B chỉ dùng làm phiếu thứ hai ở tier `flip` (nơi hai
model đồng thuận: 82,39%).

Ngược lại, một cổng bảo thủ hữu ích: chỉ nhận nhãn khi hai lượt 31B đồng ý →
`broken` giảm từ 388 xuống **148** (label acc 82,43%, net +10,48). Đó là núm
vặn giữa điểm và độ an toàn, không phải quyết định đúng/sai.

---

## 6. Bộ train cuối: bốn tier, và ý tưởng mới duy nhất

| tier | chọn dòng nào | target | vai trò |
|---|---|---|---|
| **ANCHOR** | `prior_class = known-solid` | chính câu trả lời của model, chữ cái trần | giữ phân phối, chặn quên |
| **TEACH** | `verdict = GROUNDED` ∧ prior yếu | đáp án passage đỡ, chữ cái trần | dạy tri thức đã chứng minh có thật |
| **CONSISTENCY** | `stable_4view` ∈ 1..3, đa số ổn định | đáp án đa số, chữ cái trần | 2.391 dòng dao động |
| **EXCLUDE** | `NOT-COVERED`, `CONTRADICTS`, `LUCKY` | — | không bịa nhãn |

**Ý tưởng mới, và là thứ chưa arm nào làm: multi-view transductive.** Mỗi dòng
được train **dưới nhiều cách trình bày** với **cùng một nhãn nội dung**. Bốn bộ
prompt đã tồn tại trên đĩa, không phải sinh mới — nhưng phải gọi đúng tên chúng,
vì dashboard đang mô tả sai:

| file | instruction | thứ tự lựa chọn | câu hỏi |
|---|---|---|---|
| `otfull_passk` | template harness | gốc | nguyên văn |
| `otfull_p1` | template harness | **xoay 1** | nguyên văn |
| `otfull_p2` = `otfull_rot2` (trùng byte) | template harness | **xoay 2** | nguyên văn |
| `otfull_cot` | "reason briefly ≤80 words" | gốc | nguyên văn |

Không câu nào được diễn đạt lại; dashboard gọi p1/p2 là "hai cách diễn đạt lại
trung tính" là sai. Bản thân phép đo thì làm đúng — nó gọi các mode là
`plain/cot/rot1/rot2` và đã map chữ cái về chỉ số gốc trước khi vote. Nên
"presentation fragility" ở đây nghĩa là **bất biến với thứ tự lựa chọn và với
instruction**, chưa từng đo bất biến với cách diễn đạt. Muốn cái sau thì phải
sinh paraphrase, hiện chưa có.

**Mốc phải tái lập trên 8B.** Trên 31B, vote chéo bốn mode ở agreement ≥ 0,70
cho **7.533 câu ở label accuracy 91,65%** — cao hơn file nhãn armG_full (87,67%)
ở 75% coverage, và trên 2.123 câu "fragile" thì 92,46%. Đó chính là cổng oracle
ở §5 dưới dạng label-free. Bước 0 đo đúng đại lượng này cho học trò.

Ba lý do nó là bước đi đúng, mỗi lý do gắn với một kết quả đã đo:

- **Allen-Zhu & Li:** tri thức chỉ trích xuất được nếu được augment nhiều dạng.
  Probe của ta nói kiến thức đang lưu *theo từng cách diễn đạt* (xoay lựa chọn
  thì giữ 79,0→80,2; đổi cách hỏi thì sập còn 3,4). Multi-view chữa đúng chỗ đó.
- **Ovadia:** số dạng mỗi fact mới là biến quan trọng, bão hoà quanh 10. Các
  campaign trước viết mỗi fact đúng một lần. Đây là biến duy nhất chưa từng được
  thay đổi — thí nghiệm merge/mix đã chứng minh *tỉ lệ trộn* không phải đòn bẩy.
- **Nó tự chặn đường tắt học vẹt chữ cái.** Bản xoay có chữ cái khác cho cùng
  nội dung, nên model **không thể** buộc câu hỏi vào một chữ cái. Arm G có 1,4
  trong 7,02 điểm dính vào chữ cái; multi-view làm phần đó bằng không theo thiết
  kế, và trả lời luôn phần "transductive nghĩa là gì" một cách trung thực hơn.

Và nó không mở lại nhánh đã đóng: mọi prompt vẫn là câu hỏi của benchmark, nên
không có drift phân phối kiểu arm H.

---

## 7. Eval: toàn bộ ot-full, và bộ control thay cho holdout

Theo yêu cầu, bỏ holdout — mọi arm chấm trên đủ 10.000 dòng. Đổi lại, phải bù
bằng control, nếu không sẽ không phân biệt được học nội dung với học chữ cái:

| control | đọc ra cái gì | mốc đã có |
|---|---|---|
| `rot2` toàn bộ 10.000 | phần điểm bám nội dung vs bám vị trí | arm G 82,41 vs base xoay 76,69 → 80% bám nội dung |
| reframe (`probe_viewtransfer`) | tri thức có tách khỏi cách diễn đạt không | trained 3,4 vs base 2,6 — chỗ này đang gần như bằng không |
| fixed/broken theo bucket | kế toán quên | §4 |
| label-accuracy accounting | điểm vượt độ chính xác nhãn = có gì đó khác đang xảy ra | nhãn 87,67% |
| memorisation probe | contamination mới do train tạo ra | luật cũ của repo |

`rot2` và reframe **thay thế** vai trò của holdout: chúng đo cùng một thứ
(khái quát hoá) mà không phải hy sinh dòng nào khỏi tập train.

---

## 8. Thứ tự chạy, và tài nguyên

**hgx046 lúc 14/08: 6/8 card H200 trống hoàn toàn** (card 0, 2, 3, 4, 6, 7 ở
0 MiB). Card 1 và 5 do `fd-train-a/b` giữ, và đó cũng là đúng 2 allocation
`nvidia.com/gpu` của device plugin. Sáu card kia chỉ với tới qua cửa CDI
(`runtimeClassName: nvidia`, `NVIDIA_VISIBLE_DEVICES=all`, pin bằng
`CUDA_VISIBLE_DEVICES` theo index **host**) — spec sẵn ở `sft/pod_spare4.yaml`.
Hai bẫy bắt buộc nhớ: `limits.cpu: "32"` + `OMP_NUM_THREADS=8` (nếu không torch
mở ~768 thread trong quota 8 core và bò), và `timeout --foreground` trong mọi
job (nếu không job runner giết trượt, python giữ card).

| bước | nội dung | card | ghi chú |
|---|---|---:|---|
| 0 | baseline Qwen3.5-9B + Qwen3-8B cùng stack, sweep 4-view × 8 mẫu | 2 | cổng quyết định base |
| 1 | retrieval mạnh cho cả 10.000 dòng, dựng khối EVIDENCE | 2 | BM25 + dense rerank |
| 2 | bốn phép thử grounding §2 → ledger + bảng coverage §3 | 3 | ~4 lượt sinh/dòng |
| 3 | bảng damage per-bucket, dựng bộ train 4 tier | 0 | CPU, không cần card |
| 4 | train multi-view + eval ot-full + rot2 + reframe | 2 | LoRA r64 |

Bước 0–3 **không train gì cả** và đó là chủ ý: dự án này đã đốt mười lăm arm vì
train trước khi biết dữ liệu có gì. Chi phí của bước 0–3 là khoảng một ngày trên
6 card đang rảnh, và nó cho ra con số coverage đã kiểm chứng — thứ quyết định
85% có với tới được bằng tài liệu hay không.

**Rủi ro tài nguyên:** cửa CDI không có bảo vệ của scheduler; lần trước tenant
khác lấy lại card 5 và 7 trong vài giờ. An toàn cho eval ngắn, rủi ro cho train
dài — checkpoint dày và ưu tiên xếp việc dài vào các card đang trống nhất.
