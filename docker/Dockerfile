# Bài nộp vòng Docker. BTC mount private test vào /data, chạy entrypoint, lấy
# result/submission.jsonl. Container chạy HOÀN TOÀN offline.
#
# Build context là gốc repo (xem docker/README.md); mọi thứ do ta viết nằm trong docker/.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    CUBLAS_WORKSPACE_CONFIG=:4096:8

# libsndfile cho soundfile (đọc .wav), ffmpeg cho các định dạng khác. Không cài gì thừa:
# image đã 14 GB vì trọng số model.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Ghim đúng phiên bản đã chạy được trên máy dev, không dùng khoảng >=: một bản minor mới có
# thể đổi signature TrainingArguments/generate và làm chết lượt chạy duy nhất ta được chấm.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 \
 && pip install --no-cache-dir \
        transformers==5.13.0 \
        peft==0.20.0 \
        accelerate==1.15.0 \
        bitsandbytes==0.50.2 \
        numpy==2.5.4 \
        soundfile==0.14.0 \
        psutil==7.2.2 \
        huggingface-hub==0.36.0

# Trọng số nhúng sẵn: container không có mạng nên không tải được gì lúc chạy. Hai model này
# là toàn bộ hệ thống -- 4.0B (LLM) + 1.5B (ASR), dưới trần 6B của luật.
COPY models/hf-cache/hub/models--Qwen--Qwen3-4B-Instruct-2507 /opt/hf/hub/models--Qwen--Qwen3-4B-Instruct-2507
COPY models/hf-cache/hub/models--vinai--PhoWhisper-large      /opt/hf/hub/models--vinai--PhoWhisper-large
COPY finetune/lora_V4                                          /opt/adapter

# Mã nguồn. BTC yêu cầu gói nộp gồm cả code huấn luyện, nên scripts/ vào trọn.
COPY src     /app/src
COPY scripts /app/scripts
COPY docker/entrypoint.py /app/entrypoint.py

WORKDIR /app
# Luật: "collect the result at result/submission.jsonl" -- đường dẫn tương đối, nên nó là
# /app/result/submission.jsonl. RESULT_DIR cho phép đổi nếu BTC chờ ở chỗ khác.
ENV RESULT_DIR=/app/result \
    DATA_DIR=/data \
    MAX_SECONDS=30 \
    USE_ADAPTER=1

ENTRYPOINT ["python3", "/app/entrypoint.py"]
