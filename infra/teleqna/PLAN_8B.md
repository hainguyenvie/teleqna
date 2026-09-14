# Quay về base ~8B: chẩn đoán, tài liệu tham chiếu, và đường đi tới 85%

Viết 2026-08-14. Mọi con số trong file này đo lại từ per-row artifacts bằng
[`analysis/error_structure_8b.py`](analysis/error_structure_8b.py) và
[`analysis/pivot_readiness_8b.py`](analysis/pivot_readiness_8b.py), không lấy lại
từ báo cáo cũ. Đích: **85,00% trên `GSMA/ot-full` teleqna, 10.000 dòng**, với
base thuộc họ Qwen cỡ 8–9B.

Giả định đang dùng, nêu rõ để bác bỏ được: (1) base phải là Qwen ~8–9B, Qwen3
hoặc Qwen3.5 đều được; (2) train transductive trên câu hỏi của benchmark mà
không dùng answer key là chấp nhận được — đây là điều repo đã chấp nhận khi
chọn arm G-full 87,24 làm ứng viên submit; (3) vẫn giữ luật closed-book lúc
serving, retrieval chỉ dùng ở phía teacher.

---

## 1. Điểm xuất phát thật của Qwen3-8B

Bốn arm đã có per-row (no-think, thinking, và bản xoay lựa chọn của cả hai),
join theo `sample_id` trên đúng 10.000 dòng:

| arm | acc |
|---|---:|
| no-think | 71,95 |
| thinking | 73,97 |
| no-think + rotate | 70,36 |
| thinking + rotate | 73,58 |

Phân tầng theo số arm trả lời đúng — đây là bảng quan trọng nhất của file này:

| đúng ở | dòng | tỉ lệ |
|---|---:|---:|
| 4/4 arm | 5.994 | 59,94% |
| 3/4 | 841 | 8,41% |
| 2/4 | 937 | 9,37% |
| 1/4 | 613 | 6,13% |
| **0/4 (hard-core sai)** | **1.615** | **16,15%** |

**23,91% benchmark được quyết định bởi cách trình bày, không phải kiến thức.**
Và hệ quả sắc hơn: nếu chỉ sửa tính nhất quán mà không thêm gì khác, trần là
**83,85%** — *không đủ để chạm 85*. Đây là lý do chiến lược dưới đây phải có
hai tầng chứ không một.

### Lỗi nằm ở đâu (2.805 lỗi no-think)

| chiều | bucket | dòng | acc | lỗi | % lỗi | hard-core |
|---|---|---:|---:|---:|---:|---:|
| subject | Research publications | 4.500 | 74,84 | 1.132 | 40,4% | 656 |
| | Standards specifications | 2.000 | 61,40 | 772 | 27,5% | 460 |
| | Research overview | 2.000 | 74,25 | 515 | 18,4% | 304 |
| | Standards overview | 1.000 | 69,70 | 303 | 10,8% | 161 |
| | Lexicon | 500 | 83,40 | 83 | 3,0% | 34 |
| số lựa chọn | 5-choice | 6.441 | 70,97 | 1.870 | **66,7%** | 1.093 |
| | 4-choice | 3.456 | 73,81 | 905 | 32,3% | 509 |
| shape cue | plain (không có "All of the above"…) | 8.631 | 70,72 | 2.527 | **90,1%** | 1.445 |
| | shape-cued | 1.369 | 79,69 | 278 | 9,9% | 170 |
| provenance | không có tag | 7.382 | 74,82 | 1.859 | 66,3% | 1.076 |
| | 3GPP Rel-18 | 780 | 59,49 | 316 | 11,3% | 183 |
| | 3GPP Rel-17 | 733 | 61,12 | 285 | 10,2% | 181 |
| | IEEE | 647 | 67,85 | 208 | 7,4% | 110 |

Histogram chữ cái **trên tập lỗi** — dấu vết selection bias, không phải kiến thức:

```
dự đoán   A 571  B 734  C 716  D 546  E 237
gold      A 721  B 633  C 517  D 539  E 395
```

Model đoán B/C quá nhiều và bỏ sót E: 395 dòng có gold là E nhưng nó chỉ chọn E
237 lần trên toàn bộ tập lỗi. Đây chính xác là token-bias mà ICLR'24 mô tả.

### Ai đã giải được các lỗi đó

| model khác | 8B sai → nó đúng | 8B đúng → nó sai | union |
|---|---:|---:|---:|
| OTel-2.0-31B-IT closed-book | 1.370 | 611 | 85,65% |
| 31B + adapter arm G-full | **1.914** | 385 | **91,09%** |
| 31B + gold explanation (trần) | 2.716 | 53 | 99,11% |
| OTel-LLM-8B-IT (AT&T bản đã release) | 895 | 2.003 | 80,90% |
| chính 8B bật thinking | 756 | 554 | 79,51% |

- **142 dòng (1,42%) không ai giải được kể cả khi có gold explanation** — nhãn
  hỏng, trần thực tế của mọi phương pháp là 98,58%.
- **530 dòng sai ở mọi model chúng ta có** (31B, 31B+armG, 8B-think, OTel-8B).
  Đó là biên giới tri thức thật: 5,3% benchmark. Phân bố: Research pub 217,
  Std spec 161, Research ovw 104, Std ovw 33, Lexicon 15.

Đọc thẳng: trong 2.805 lỗi của Qwen3-8B, chỉ khoảng **530 là thiếu kiến thức
cứng**. 2.275 dòng còn lại đã nằm trong tầm với của một nguồn nhãn tốt hơn hoặc
của chính model nếu nó trả lời nhất quán.

---

## 2. Vì sao mọi lần train trên 8B đã thất bại — và lý do không phải cái repo đang ghi

Chapter 2 kết luận rằng mọi campaign kiến thức chết vì **target form**: file
train viết target dạng prose, model không tái tạo dạng đó lúc eval. Kết luận đó
đúng cho các file của Chapter 2. **Nó không giải thích thất bại của Qwen3-8B ở
Chapter 1.** Quét lại target form của toàn bộ file train từng dựng:

| file | key | bare `ANSWER: X` | ghi chú |
|---|---|---|---|
| `train_v3.jsonl` (36,6MB — SFT của 8B) | completion | **400/400** | đã đúng dạng |
| `dpo_pairs{,_v2,_v3}.jsonl` | chosen | **400/400** | đã đúng dạng |
| `cpt_mcq_anchors.jsonl` | completion | 400/400 | đã đúng dạng |
| `anchor_mcq.jsonl` | completion | 0/400 | prose 72 từ |
| `answerkey_mcq_all.jsonl` | completion | 0/400 | prose 23 từ |
| `answerkey_fact_*.jsonl` | completion | 0/400 | prose, không có chữ cái |
| `m1.jsonl` | completion | 0/400 | prose 54 từ |
| `armF/armG/armG_full/armJ` | completion | 400/400 | dạng đã sửa |

Vậy nguyên nhân thật của Chapter 1 trên 8B là cái khác, và nó đã được định
lượng ở Chapter 2 dưới tên **arm H**: `train_v3` gồm 40.730 **câu hỏi synthetic**,
không một câu nào là câu của benchmark. Arm H trên 31B là đúng thí nghiệm đó và
cho net −2,54 với 926 dòng bị phá, phân bố câu trả lời trôi về phía "câu hỏi
sinh từ passage". Chapter 1 gọi hiện tượng này là "SFT phá 3 dòng để dạy được 2";
đó là cùng một bệnh, ở cùng một liều.

> **Kết luận có sức nặng nhất của tài liệu này: công thức arm G — câu hỏi thật
> của benchmark, nhãn label-free ba tầng, target chữ cái trần — chưa bao giờ
> được chạy trên Qwen3-8B.** Mọi lần train 8B đều dùng câu hỏi synthetic. Lane
> có bằng chứng mạnh nhất của dự án chưa từng được thử ở cỡ 8B.

### Số học của lane đó, tính trên bộ nhãn đã có sẵn trên đĩa

`data/train/eligible/armG_full.jsonl`, 9.989 dòng, dựng không dùng answer key:

| tier | n | label acc | 8B hiện đúng | 8B đã đồng ý sẵn với nhãn |
|---|---:|---:|---:|---:|
| keep | 5.967 | 95,74% | 88,29% | 90,92% |
| retain | 2.445 | 83,68% | 55,75% | 60,86% |
| flip | 1.577 | 63,28% | 35,45% | 28,85% |
| **tổng** | **9.989** | **87,67%** | 71,98% | — |

Đổi chiều từng dòng nếu 8B tuân nhãn tuyệt đối:

```
8B sai  & nhãn đúng : +1.955   (retain 779, flip 693, keep 483)
8B đúng & nhãn sai  :   −388   (flip 254, retain 96, keep 38)
net                 : +1.567 dòng = +15,69 điểm  ->  87,67%
```

Điểm dự kiến theo mức tuân nhãn (obedience), phần còn lại giữ hành vi cũ:

| obedience | điểm |
|---:|---:|
| 100% | 87,67 |
| 98% | 87,35 |
| 95% | 86,88 |
| 90% | 86,10 |
| **80%** | **84,53** |

**85% đạt được ngay khi obedience ≥ 82%.** Arm V trên 31B fit nhãn tới 99,91%;
arm F đạt obedience 92,4% ở tier khó nhất. Biên an toàn là thật, không phải hy vọng.

Và bộ nhãn này đúng ở **969/1.615 (60,0%)** dòng hard-core của 8B — tức là nó
tấn công đúng phần mà tính nhất quán không với tới.

---

## 3. Tài liệu tham chiếu, và cái mỗi bài nói về đúng lỗi của chúng ta

| bài | nói gì | áp vào đây |
|---|---|---|
| Allen-Zhu & Li, *Physics of LM 3.1* ([2309.14316](https://arxiv.org/abs/2309.14316), ICML'24) | kiến thức chỉ *trích xuất được* nếu được augment (paraphrase, đảo câu, dịch) lúc pretrain; không augment thì model nhớ nhưng 0% trích xuất được, và SFT sau đó không cứu được | giải thích chính xác probe `probe_viewtransfer`: xoay lựa chọn thì model giữ (79,0→80,2), *đổi cách hỏi* thì sập còn 3,4. Kiến thức của ta lưu theo từng cách diễn đạt. Đây là lý do CPT và arm H thất bại — chúng viết mỗi fact một lần |
| Ovadia et al., *Fine-Tuning or Retrieval?* ([2312.05934](https://arxiv.org/abs/2312.05934), EMNLP'24) | FT thua RAG cho kiến thức mới; số paraphrase mỗi fact có tác dụng nhưng bão hoà quanh **10** | chẩn đoán liều: các campaign fact của ta chưa bao giờ đạt ~10 dạng/fact. Kết luận "kênh kiến thức đóng" được rút ra ở liều dưới ngưỡng |
| *TTRL: Test-Time RL* ([2504.16084](https://arxiv.org/abs/2504.16084), NeurIPS'25) | RL trên chính test set không nhãn, reward = majority vote của model | arm G là bản supervised của đúng họ này. Hợp thức hoá phương pháp luận, và chỉ ra biến thể RL nếu tầng 1 bão hoà |
| *Flip-Flop Consistency* ([2510.14242](https://arxiv.org/abs/2510.14242)) | train **không nhãn** để output bất biến qua các perturbation của prompt; cải thiện cả consistency lẫn accuracy | đúng công cụ cho 2.391 dòng unstable của 8B (23,91%) |
| Ren et al., *Consistency Alignment* ([2403.14221](https://arxiv.org/abs/2403.14221)) | 2 giai đoạn: instruction-augmented SFT rồi consistency alignment; +2,5% trung bình | cho biết thứ tự: augment trước, align sau — không trộn |
| Zheng et al., *LLMs Are Not Robust MCQ Selectors* ([2309.03882](https://arxiv.org/abs/2309.03882), ICLR'24) + PriDe | selection bias là **token bias** trên A/B/C/D; PriDe ước lượng prior bằng hoán vị trên một mẫu nhỏ rồi debias phần còn lại, **không cần nhãn** | khớp histogram lỗi ở §1 (E dự 237 / gold 395). Một lớp debias inference-time gần như miễn phí |
| *Permutation-Aware GRPO* ([2603.21016](https://arxiv.org/html/2603.21016v1), 2026) | cross-permutation advantage + reward phạt lệch giữa các hoán vị; lợi ích chủ yếu ở **consistency** | bản có nhãn; thay nhãn bằng majority vote thì thành label-free và ghép thẳng vào tầng 3 |
| *ORANSight-2.0* ([2503.05200](https://arxiv.org/abs/2503.05200)) | RANSTRUCT: teacher RAG sinh instruction, QLoRA fine-tune 18 model từ 5 họ | nguồn dữ liệu instruction O-RAN; base của họ là Qwen2.5, không dùng lại |
| *Tele-LLMs* ([2409.05314](https://arxiv.org/abs/2409.05314)) | CPT 1B–8B trên Tele-Data, 5.000 giờ A6000 | không có bản Qwen; đánh giá trên Tele-Eval chứ không phải TeleQnA |

Hai bài đầu cộng lại cho một dự đoán kiểm chứng được: **nếu viết lại mỗi fact
thành ≥10 dạng khác nhau rồi mới train, kênh kiến thức có thể mở** — đó là biến
thể duy nhất của arm H còn chưa bị bác bỏ bởi thí nghiệm merge/mix.

---

## 4. Base ~8–9B: có sẵn cái nào đã học tri thức viễn thông không?

Đã tra thực tế trên HuggingFace và trong các paper, không suy đoán:

| ứng viên | tình trạng | phán quyết |
|---|---|---|
| **Qwen3.5-9B / -9B-Base** (Qwen, 02–03/2026) | MMLU-Pro 82,5 · GPQA-D 81,7 · 262k ctx · có cả bản Base · kiến trúc Gated DeltaNet + sparse MoE. Không phải model telecom | **ứng viên số 1.** Không có CPT telecom, nhưng **TelecomGPT-R1 — #1 GSMA leaderboard, average 89,6% — dựng trên Qwen3.5-27B**, tức cùng họ đã được chứng minh ở đúng bảng xếp hạng này. `venv-vllm-nightly` phục vụ họ Qwen3.5 đã dựng sẵn trên server (làm cho 122B) |
| **Qwen3-8B** | baseline 71,95 đã đo, mọi per-row artifact đã có, so sánh được với toàn bộ lịch sử dự án | **giữ làm arm đối chứng.** Trọng số đã bị xoá khỏi `shared/hf-cache`, cần tải lại ~16GB |
| **OTel-LLM-8B-IT** (AT&T, đã release) | 60,87 trên ot-full. Phân tích lỗi mới: **3.793/3.913 lỗi không phát ra dòng `ANSWER:`** (nó trả `A) NOMA`), 282 parse-fail, 320 truncated, letter bias nặng (B 3.247 vs D 1.181). Vẫn fix được 895 dòng mà Qwen3-8B sai | **không dùng làm base** — hành vi trả lời đã bị RAG/abstention training phá. Dùng được như *phiếu bầu phụ* trong bộ sinh nhãn. Bản **8.3B-QnA đạt 91,20 chưa bao giờ được release** (kiểm tra lại 14/08: `farbodtavakkoli` chỉ có 31B-IT + embedding + reranker) |
| **Tele-LLMs** (Yale) | TinyLlama/Phi/Gemma/Llama-3.2/Llama-3-8B-Tele. **Không có bản Qwen**; lớn nhất là `AliMaatouk/LLama-3-8B-Tele` | loại — sai họ base, base cũ, đánh giá trên Tele-Eval |
| **ORANSight-2.0 Qwen** (NextGLab) | có `ORANSight_Qwen_7B_Instruct` (và 1.5B/3B/14B/32B), nền **Qwen2.5**, QLoRA trên dữ liệu O-RAN | loại làm base (Qwen2.5 yếu hơn Qwen3-8B, và TeleQnA gần như không có nội dung O-RAN). Dữ liệu RANSTRUCT thì đáng xem |
| **TelecomGPT-R1** (KU-DFI) | Qwen3.5-**27B**, 3 giai đoạn post-training, corpus 158.915 ví dụ cân theo 4 trục (protocol 22,7% / knowledge QA 15,3% / modeling 43,5% / fault 18,5%), average 89,6% | quá lớn để làm base 8B, nhưng là **bằng chứng công thức**: base Qwen3.5 + SFT domain + RL có verifiable signal thắng ở chính bảng này. Cũng là ứng viên teacher |
| linh tinh trên HF (`phi-2-telecoms`, `falcon-7b-qlora-telecom`, `raoulbia/Qwen2.5-7B-3GPP-NR`, `kopioO/3gpp-*-llama3.2-3b`) | quy mô đồ chơi, không có eval công bố | loại. Nếu vẫn muốn dùng: bắt buộc chạy `scan_contamination.py` + `verify_contamination.py` trước — luật đã bắt được AdaptKey |

**Kết luận:** không tồn tại một Qwen ~8B đã CPT telecom đáng để kế thừa. Lợi thế
khả dĩ đến từ **base mới hơn** (Qwen3.5-9B), không phải từ base telecom.

---

## 5. Chiến lược tới 85%

85% = 8.500/10.000. Từ Qwen3-8B no-think cần **+1.305 dòng**, tức sửa 46,5% số lỗi.

### Tầng 0 — đo, trước khi train bất cứ thứ gì (≈1 ngày, 1 card)

1. `Qwen3.5-9B` closed-book trên cả 10.000 dòng, stack vLLM nightly, no-think,
   512 token — cùng harness với mọi arm Chapter 2.
2. `Qwen3-8B` chạy lại trên **đúng stack đó** để có A/A offset (Chapter 1 dùng
   stack khác; offset đã đo là +0,2pp no-think nhưng phải xác nhận lại).
3. Trên base được chọn: 8 mẫu × 4 cách trình bày (như sweep 320k của 31B) để có
   pass@1/pass@8 và tập unstable của chính nó.

Cổng quyết định: nếu Qwen3.5-9B ≥ 78 closed-book thì mọi tầng sau rẻ đi rõ rệt
và 85% gần như chỉ còn là bài toán kỹ thuật.

### Tầng 1 — transductive label distillation (đòn bẩy chính, dữ liệu đã có)

Train base đã chọn trên `armG_full.jsonl` như hiện trạng: prompt là prompt của
harness, target là `ANSWER: X` trần, LoRA r64, cùng trainer/LR/epoch với arm G.

- Dự báo: **84,5 – 87,7** tuỳ obedience (bảng ở §2). Chạm 85 ở obedience ≥82%.
- Bắt buộc kèm: rotation control (`rot2`), holdout, và memorisation probe chạy
  lại trên checkpoint — luật của repo, không bỏ.
- Bẫy phải phòng: 388 dòng mà base đúng còn nhãn sai (254 nằm ở tier flip). Trên
  31B, "dùng evidence khắp nơi" có headline cao hơn nhưng lỗ 0,80 điểm vì đúng
  cơ chế này. Dựng lại tier `keep` bằng **vote của chính base mới** thay vì kế
  thừa margin của 31B, rồi chạy bảng damage per-bucket *trước khi train*.

### Tầng 2 — nâng chất lượng nhãn (mỗi điểm label accuracy là một điểm score)

Tier `flip` đang ở 63,28% trên 1.577 dòng và là mắt xích yếu nhất. Ba nguồn đã
đo sẵn, chưa được ghép:

- retrieval mạnh: **89,5%** answer-in-context so với 43,0% của cấu hình đã deploy;
- 122B làm phiếu thứ hai: nơi hai model đồng ý, độ chính xác **82,39%**;
- vote 32 mẫu của chính base mới, cộng với co-presence của lựa chọn trong một
  cửa sổ retrieval (tín hiệu chất lượng retrieval không cần nhãn, §"corpus is
  provenance").

Số học: đưa flip từ 63,28 → 78 là +233 dòng đúng = **+2,3 điểm** trên bộ nhãn,
kéo label accuracy tổng lên ~90 và trần của tầng 1 lên tương ứng.

### Tầng 3 — nhất quán, không cần nhãn (đánh vào 23,91% unstable)

Chạy như một arm độc lập rồi mới cộng, để đo được tính cộng tính:

- Flip-Flop Consistency trên các perturbation đã có sẵn (`otfull_p1/p2/cot/rot2`);
- hoặc PA-GRPO phiên bản label-free: nhóm hoán vị + advantage chéo + reward phạt
  lệch, pseudo-label bằng majority vote (TTRL);
- cộng một lớp PriDe debias lúc inference — không cần train, không cần nhãn.

Trần khi đứng một mình là 83,85%, nên tầng 3 **không thay thế** tầng 1; nó nhắm
vào phần "8B đúng ở một cách hỏi nhưng không ở cách khác" mà nhãn không chạm tới.

### Tầng 4 — dự phòng, chỉ mở nếu tầng 1+2 dừng dưới 85

Biến thể duy nhất của arm H chưa bị bác bỏ: viết mỗi fact thành **≥10 dạng khác
nhau** (Ovadia) và trộn ở tỉ lệ synthetic thấp cùng anchor thật (Allen-Zhu).
Lưu ý thí nghiệm merge/mix đã chứng minh *tỉ lệ* không phải đòn bẩy — chỉ đổi
**số dạng mỗi fact** mới là cơ chế mới.

### Không làm (đã có bằng chứng đóng)

CPT ở mọi liều · prompt/GEPA · GRPO thuần trên decoding · train chỉ bằng câu hỏi
synthetic (`train_v3`, arm H) · answer-key · corpus AdaptKey · tăng rank/epoch ·
tinh chỉnh tỉ lệ trộn hay thứ tự curriculum của arm H.

### Kỷ luật đo, giữ nguyên từ Chapter 2

Mọi arm phải có arm base cùng stack · noise floor A/A 0,50pp · `reparse.py`
trước khi tin bất kỳ điểm long-context nào · `analyze_arm.py` để tách dòng
trained/never-trained · holdout không bao giờ dùng để chọn · quét contamination
mọi corpus/checkpoint bên thứ ba trước khi dùng.

---

## 6. Rủi ro đã biết

1. **Transductive.** Điểm số gắn với đúng 10.000 câu hỏi này. Nếu ban tổ chức
   đổi tập test, phần lớn lợi ích mất. Rotation control trên 31B nói 80% lợi ích
   bám nội dung chứ không bám vị trí chữ cái, nhưng đó là câu trả lời cho một câu
   hỏi hẹp hơn. Cần nêu rõ khi báo cáo, như arm G đã làm.
2. **Obedience của 8B có thể thấp hơn 31B**, vì 8B phải đổi ~26% câu trả lời của
   mình (agreement với nhãn: keep 90,92 / retain 60,86 / flip 28,85) trong khi
   31B đổi ít hơn. Đây là biến số chưa đo và là rủi ro chính của tầng 1.
3. **Nhãn kế thừa từ 31B không khớp margin của 8B** — tier `keep` phải dựng lại
   từ vote của base mới, nếu không sẽ dẫm vào bẫy bucket A theo chiều ngược lại.
4. **Trần cứng 98,58%** vì 142 dòng nhãn hỏng, và 530 dòng không model nào của
   chúng ta giải được. Đừng lập kế hoạch vượt qua chúng.
