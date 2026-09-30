# TeleQnA closed-book service: một model 8B duy nhất, sau một URL OpenAI
# /v1/chat/completions mà orchestrator (hoặc gsma-labs/satellite) trỏ thẳng vào.
#
# Khác telelogs ở một điểm quan trọng: KHÔNG có pipeline, không DSPy, không tool.
# Điểm 82,2 % là của chính trọng số trả lời một lượt. Mọi thứ quyết định con số đó
# nằm trong thư mục model chứ không nằm trong cờ dòng lệnh:
#   * generation_config.json  -> do_sample=false, temperature=0, top_p=1  (greedy)
#   * chat_template.jinja     -> enable_thinking mặc định FALSE
# Phục vụ thư mục này bằng vLLM trần, không cờ nào thêm, là ra đúng điểm đã đo.
# Thiếu một trong hai thứ trên mất 0,5–0,9 điểm (đã đo, xem API.md).
#
# Base pin v0.26.0: điểm được đo trên vLLM 0.26.1rc1.dev542 (nightly). v0.26.0 là
# bản phát hành gần nhất; continuous batching của vLLM tạo dao động ±0,1 điểm giữa
# các lần chạy, đổi minor version sẽ đổi dải đó một lượng không biết trước.
FROM vllm/vllm-openai:v0.26.0

# Thư viện cho harness chấm lại (mode `verify`) cài bằng pip --target vào /opt/eval/pylibs, KHÔNG venv:
# entrypoint gọi `PYTHONPATH=/opt/eval/pylibs python3 /opt/eval/inspect_cli.py`, và image thật (build_crane.sh)
# có đúng bố cục này. inspect-ai kéo openai/pydantic riêng nên không cài chung vào site-packages của vLLM.
RUN python3 -m pip install --no-cache-dir --target /opt/eval/pylibs \
      "inspect-ai==0.3.188" "datasets==4.6.1" "pyarrow==22.0.0" "openai==2.26.0"
COPY inspect_cli.py /opt/eval/inspect_cli.py

# Task chính thức của GSMA (gsma-labs/evals, MIT) — nguyên văn, không sửa một dòng:
# hf_dataset("GSMA/ot-full", name="teleqna", split="test") + multiple_choice(cot=False) + choice().
COPY eval/ /opt/eval/evals/
# _registry.py rút gọn: chỉ teleqna, vì image không mang 7 benchmark còn lại.
COPY registry_teleqna.py /opt/eval/evals/_registry.py
# Cache HF của GSMA/ot-full (3,5 MB) để `verify` chạy được khi máy không có mạng.
COPY hf_cache/ /opt/hf/
# Trọng số bf16 đã gộp: Qwen3-8B → study-kit → vote/ens/merge → on-policy → wise_o3.
COPY model/ /opt/model/
COPY entrypoint.sh /opt/entrypoint.sh
COPY verify_summary.py /opt/eval/verify_summary.py

# Hỏng thì hỏng lúc BUILD, không phải lúc request đầu tiên.
RUN chmod +x /opt/entrypoint.sh \
 && test -f /opt/model/config.json \
 && test -f /opt/model/generation_config.json \
 && test -f /opt/model/chat_template.jinja \
 && test -f /opt/eval/evals/teleqna/teleqna.py \
 && test -f /opt/eval/evals/_registry.py \
 && test -f /opt/hf/hub/datasets--GSMA--ot-full/snapshots/*/teleqna/test-00000-of-00001.parquet \
 && python3 -c "\
import json; g=json.load(open('/opt/model/generation_config.json')); \
assert g.get('do_sample') is False and g.get('temperature')==0.0 and g.get('top_p')==1.0, g; \
t=open('/opt/model/chat_template.jinja').read(); \
assert 'enable_thinking is not defined' in t and 'set enable_thinking = false' in t; \
print('artifact contract ok: greedy defaults + thinking off')" \
 && PYTHONPATH=/opt/eval/pylibs python3 -c "import inspect_ai, datasets; print('inspect_ai', inspect_ai.__version__)"

# Orchestrator gọi với "model": "Qwen3-8B-Telco". Cổng 20501 (8000 bị cấm trên server bench).
EXPOSE 20501

ENV SERVED_NAMES="Qwen3-8B-Telco teleqna-8b-closedbook wise-o3" \
    PORT=20501 \
    HOST=0.0.0.0 \
    VLLM_TP=1 \
    VLLM_MAX_MODEL_LEN=4096 \
    VLLM_GPU_MEM_UTIL=0.85 \
    VLLM_MAX_NUM_SEQS=64 \
    VERIFY_CONNECTIONS=64 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=8

ENTRYPOINT ["/opt/entrypoint.sh"]
CMD ["serve"]
