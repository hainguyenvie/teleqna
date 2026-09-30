# TeleQnA API — hướng dẫn sử dụng

Endpoint trả lời trắc nghiệm viễn thông **closed-book**: không truy xuất tài liệu,
không model phụ, không tool. Nói giao thức OpenAI chat-completions nên orchestrator
gọi nó y hệt gọi một model — và lần này bên trong **đúng là** một model.

## Địa chỉ

```
POST  http://127.0.0.1:20501/v1/chat/completions
GET   http://127.0.0.1:20501/v1/models
GET   http://127.0.0.1:20501/health
```

## Gửi cái gì

Đúng định dạng trong `data/teleqna.jsonl` của bộ request mẫu:

```json
{
  "model": "Qwen3-8B-Telco",
  "messages": [
    {"role": "user", "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of A,B,C,D.\n\nWhat is the diversity gain ...\n\nA) 0\nB) 4\nC) 2\nD) 1"}
  ]
}
```

Một message `user` duy nhất, chứa nguyên văn prompt do harness dựng (template
`multiple_choice(cot=False)` của Inspect AI). Service nhận cả ba tên model:
`Qwen3-8B-Telco`, `teleqna-8b-closedbook`, `wise-o3`.

**Không cần gửi tham số decode.** Thư mục model đã mang sẵn:

| file trong model | tác dụng | mất bao nhiêu điểm nếu thiếu |
|---|---|---|
| `generation_config.json` (`do_sample=false, temperature=0, top_p=1`) | greedy | **0,5** (mặc định Qwen3 `temp 0,6/top_p 0,95` cho 81,8 thay vì 82,3) |
| `chat_template.jinja` (`enable_thinking` mặc định **false**) | trả lời thẳng `ANSWER: X`, không `<think>` | **~0,4** cộng nguy cơ tràn token |

Nếu người gọi **tự** gửi `temperature`/`top_p`, vLLM sẽ nghe theo và điểm đổi.
Muốn con số đã đo thì đừng gửi — bộ request mẫu không gửi, đã kiểm cả 10.000 dòng.

`stream: true` không dùng trong bài chấm; vLLM hỗ trợ nhưng chưa đo.

## Nhận lại cái gì

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "Qwen3-8B-Telco",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "ANSWER: A"}, "finish_reason": "stop"}],
  "usage": {"completion_tokens": 5}
}
```

Luôn đúng một dòng `ANSWER: <chữ cái>`, 5 token. Trong 10.000 câu của bài chấm
chính thức **không có câu nào** parser không đọc được (`unparsed = 0`).

## Điểm

| đo bằng | kết quả |
|---|---|
| harness chính thức `gsma-labs/evals` + Inspect AI, `GSMA/ot-full` (teleqna, test, 10.000 câu), greedy | **0,822–0,823** (stderr 0,004) |
| bản tự viết theo hợp đồng Inspect (`run_baseline.py`) | 82,22 |
| Qwen3-8B gốc, cùng đường chấm | 71,8 |

Dao động ±0,1 giữa các lần chạy là do continuous batching của vLLM, không phải do model.

## Chấm lại ngay trong container

```bash
docker run --gpus all --rm hainh67/teleqna-serve:wise-o3 verify
```

Bật đúng stack đó, chạy task chính thức trên đủ 10.000 câu (dataset đã đóng sẵn
trong image nên không cần mạng), in accuracy kèm phân tách theo môn, rồi thoát.
Trên H200 khoảng 2 phút.
