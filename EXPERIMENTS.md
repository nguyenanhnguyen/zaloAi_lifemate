# Lịch sử thử nghiệm — ZAC 2026 LifeMate

Ghi lại từng lần chạy: tên run, mô hình, điểm chính thức và nhận xét. Chỉ thêm vào, không xoá.

---

## v1–v4 (local, baseline)
- **Pipeline**: Whisper + Qwen2.5-3B llama.cpp
- **Điểm**: chưa ghi nhận / thử nghiệm ban đầu

---

## local – PhoASR-small + Qwen3.5-4B Q8 (llama.cpp)
- **Script**: `python scripts/infer.py --split public_test --output submission.jsonl`
- **ASR**: `Qualcomm-AI-Research/PhoASR-whisper-small` (rev `da2d9ae`)
- **LLM**: `unsloth/Qwen3.5-4B-GGUF` → `Qwen3.5-4B-Q8_0.gguf` (rev `e87f176`) qua llama-server
- **Điểm chính thức**: **22.52000**
- **Thời gian/mẫu**: 50–471 s (trung bình ~167 s) — *vượt giới hạn 30 s; chỉ để đo chất lượng*
- **VRAM dùng**: 787 MB (llama.cpp chạy phần lớn trên CPU, swap RAM)
- **RAM process tree**: 6.6–8.2 GB
- **stop_reason phổ biến**: `done` (37/50), `step_limit` (3), `asked_missing` (5), `empty_response` (4), `repeated_invalid` (1)
- **Ghi chú**: Điểm cải thiện đáng kể so với local trước. LLM rất chậm vì chạy CPU llama.cpp.

---

## kaggle – `qwen_asr06_qwen4_v7` (chưa hoàn chỉnh)
- **Notebook**: `lifemate_kaggle.ipynb`
- **ASR**: `Qwen/Qwen3-ASR-0.6B-hf` (rev `7f1569a`) — bfloat16 GPU
- **LLM**: `Qwen/Qwen3-4B-Instruct-2507` (rev `cdbee75`) — bfloat16 GPU
- **Điểm chính thức**: *Chưa có* (notebook không chạy hoàn chỉnh lần nào)
- **Lỗi phát hiện**:
  1. `torch.use_deterministic_algorithms(True, warn_only=False)` → ném RuntimeError khi warm-up vì nhiều CUDA op không deterministic
  2. `processor.decode(generated_2D_tensor, ...)` → decode toàn bộ batch như một chuỗi, không trả về list → `zip(paths, texts)` lặp qua ký tự thay vì string
  3. `assert counts == {'asr': 782426112, 'llm': 4022468096}` cứng — fail nếu revision thay đổi nhỏ

---

## kaggle – `qwen_asr06_qwen4_v8` (bản mới, đang chạy)
- **Notebook**: `lifemate_kaggle.ipynb` (cập nhật 2026-10-31)
- **ASR / LLM**: giữ nguyên Qwen3-ASR-0.6B + Qwen3-4B-Instruct-2507
- **Sửa so với v7**:
  - `warn_only=True` → không crash khi warm-up
  - Decode ASR theo từng hàng: `[processor.decode(generated[k], ...) for k in range(batch)]`
  - `assert counts ==` → `logger.warning` (soft check, giới hạn 5B/6B vẫn còn)
  - `'code_hash'` tĩnh → `'pipeline_version': 'v8'` (tránh fingerprint lỗi khi notebook thay đổi)
- **Điểm**: *Chờ kết quả*
- **Kỳ vọng**: Qwen3-4B-Instruct-2507 full precision bfloat16 tốt hơn Q8 quantized; ASR 0.6B mới hơn PhoASR-small

---

## Bài học rút ra

| Vấn đề | Nguyên nhân | Cách xử lý |
|--------|-------------|------------|
| `use_deterministic_algorithms` crash | Nhiều kernel CUDA không có bản deterministic | `warn_only=True` hoặc bỏ |
| `processor.decode(2D_tensor)` | API nhận 1D, batch cần lặp | Loop `decode(row)` hoặc `batch_decode` |
| Assert cứng trên số tham số | Revision model có thể thay đổi nhẹ | Chuyển thành warning, chỉ giữ kiểm tra giới hạn tổng |
| llama.cpp CPU quá chậm | Llama.cpp không offload VRAM đủ trên Windows | Dùng transformers full GPU (Kaggle A30/T4) |
| `code_hash` tĩnh trong fingerprint | Hash tính tay, lỗi thời khi sửa notebook | Dùng `pipeline_version` string |
