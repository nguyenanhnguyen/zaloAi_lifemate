# ZAC 2026 — LifeMate Agent

> Giải pháp cho cuộc thi [Zalo AI Challenge 2026 — LifeMate](https://challenge.zalo.ai)  
> Biến yêu cầu giọng nói tiếng Việt thành chuỗi tool calls thực thi chính xác.

---

## Kiến trúc Pipeline

| Thành phần | Mô hình | Tham số | VRAM (RTX 2050) | VRAM (A30 Docker) |
|---|---|---|---|---|
| **ASR** | `openai/whisper-large-v3-turbo` | 0.81B | ~1.0 GB (int8_float16) | ~1.6 GB (FP16) |
| **LLM Agent** | `Qwen/Qwen3-4B-Instruct` (4-bit BnB) | 4.0B | ~2.5 GB | ~8.0 GB (BF16/vLLM) |
| **Tổng** | | **4.81B** | **~3.5 GB / 4 GB** | **~10 GB / 24 GB** |

---

## Cấu trúc thư mục

```
a_zaloAI/
├── data/
│   └── train+public_test/
│       └── train+public_test/
│           ├── train/
│           │   ├── audio/<id>/<id>.wav
│           │   ├── train.jsonl
│           │   ├── train_tools_catalog.json
│           │   ├── train_environment/<id>/state.json
│           │   └── zaic_tools/            ← execution environment
│           └── public_test/
│               ├── audio/<id>/<id>.wav
│               ├── public_test.jsonl
│               ├── public_test_tools_catalog.json
│               ├── public_test_environment/<id>/state.json
│               ├── sample_submission.jsonl
│               └── zaic_tools/
├── models/
│   ├── whisper/                           ← openai/whisper-large-v3-turbo
│   └── qwen3-4b/                          ← Qwen/Qwen3-4B-Instruct
├── checkpoints/
│   └── whisper-finetuned/                 ← LoRA checkpoint sau fine-tune
├── src/
│   ├── __init__.py
│   ├── asr.py                             ← Whisper transcription module
│   └── pipeline.py                        ← Pipeline chính (ASR → LLM → env)
├── scripts/
│   ├── infer.py                           ← Inference chính → submission.jsonl
│   ├── train_asr.py                       ← Fine-tune Whisper (Giai đoạn 2)
│   ├── evaluate.py                        ← Evaluate local trên train split
│   └── download_models.py                 ← Download models về offline
├── docs/
│   └── context.txt                        ← Mô tả bài toán & pipeline
├── requirements.txt
└── README.md
```

---

## Cài đặt

### 1. Tạo virtual environment

```bash
python -m venv env
# Windows
env\Scripts\activate
# Linux/macOS
source env/bin/activate
```

### 2. Cài PyTorch với CUDA 12.4

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

> **Note:** Driver 621.01 trên RTX 2050 hỗ trợ CUDA 12.x. CUDA 12.4 wheel là lựa chọn ổn định nhất.

### 3. Cài các package còn lại

```bash
pip install -r requirements.txt
```

---

## Download Models

```bash
# Download cả hai model (cần ~6GB ổ cứng)
python scripts/download_models.py --models all

# Chỉ download Whisper (~1.5GB)
python scripts/download_models.py --models whisper

# Chỉ download Qwen (~4.5GB)
python scripts/download_models.py --models qwen
```

Models lưu tại `models/whisper/` và `models/qwen3-4b/`.

---

## Chạy

### Inference (tạo submission)

```bash
# Chạy toàn bộ public_test → submission.jsonl
python scripts/infer.py --split public_test

# Debug với 10 sample đầu
python scripts/infer.py --split public_test --max-samples 10 --output debug_sub.jsonl
```

### Evaluate local (train split, có label)

```bash
# Chạy pipeline rồi evaluate
python scripts/evaluate.py --split train --max-samples 50

# Evaluate file submission có sẵn
python scripts/evaluate.py --submission submission.jsonl
```

### Fine-tune Whisper

```bash
# Fine-tune 3 epoch trên train split (Giai đoạn 2)
python scripts/train_asr.py --split train --epochs 3

# Dùng model local đã download
python scripts/train_asr.py --local-model models/whisper --epochs 3
```

---

## Pipeline chi tiết (4 bước)

```
Audio (WAV 16kHz)
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│ Bước 1 — Context-Biased ASR                             │
│  • faster-whisper int8_float16 (greedy, beam_size=1)    │
│  • initial_prompt = tên app + từ khoá từ history        │
│  • Kết quả: transcript tiếng Việt                       │
└─────────────────────┬───────────────────────────────────┘
                      │ transcript
                      ▼
┌─────────────────────────────────────────────────────────┐
│ Bước 2 — Dynamic Schema Pruning                         │
│  • Lọc tools_catalog theo installed_apps của sample     │
│  • Prompt: ~1.200 tokens (thay vì ~6.000 tokens toàn bộ)│
└─────────────────────┬───────────────────────────────────┘
                      │ pruned schema
                      ▼
┌─────────────────────────────────────────────────────────┐
│ Bước 3 — Vòng lặp ReAct (LLM + env.validate/call)      │
│  • Luật 1: Out-of-scope → tool_calls: [] (điểm 1.0)    │
│  • Luật 2: x-ref → gọi read tool trước                 │
│  • Luật 3: Thiếu info → sys.ask_missing_info            │
│  • Luật 4: env.validate() trước env.call() (miễn phí)  │
└─────────────────────┬───────────────────────────────────┘
                      │ tool_calls
                      ▼
┌─────────────────────────────────────────────────────────┐
│ Bước 4 — Chuẩn hóa fuzzy args                          │
│  • Unicode NFC, cắt từ đệm cảm thán                    │
│  • token-F1 > 0.6 với label                             │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
           submission.jsonl
    {"id": ..., "tool_calls": [...], "state_hash": "..."}
```

---

## Evaluation Metrics

- **State match**: `state_hash` dự đoán == `state_hash` gold (tất cả trạng thái app đúng).
- **Tool-call F1**: `F1 = 2PR/(P+R)` với P = M/|pred|, R = M/|gold|.
  - Fuzzy args (x-match): token-F1 > 0.6.
  - Out-of-scope: điểm 1.0 khi cả hai đều rỗng.

---

## Reproducibility

Tất cả seed cố định = **42** (Python, NumPy, PyTorch).  
ASR: `beam_size=1` (greedy). LLM: `temperature=0`, deterministic decoding.

---

## Yêu cầu kỹ thuật

| Ràng buộc | Giá trị |
|---|---|
| Tổng tham số | < 6B (pipeline: ~4.81B ✅) |
| LLM params | < 5B (Qwen3-4B: 4.0B ✅) |
| Context length LLM | ≤ 16K tokens (dùng ~4K ✅) |
| Thời gian / sample | < 30 giây |
| Offline | Không dùng API ngoài ✅ |
