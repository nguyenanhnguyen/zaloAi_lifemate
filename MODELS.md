# Model cache — ở đâu, dùng cho gì

Ghi lại vì cache Hugging Face đã từng làm đầy ổ C (chỉ còn 6.7GB, 2026-10-31). Đã dọn sang
`D:\z_DaiHoc\a_zaloAI\models\hf-cache\hub\`. Không set `HF_HOME` toàn máy (tránh quên chỉnh lại
sau này) — mỗi lệnh tự trỏ đúng nơi, xem phần "Lệnh" dưới đây.

## Dùng bởi pipeline LifeMate (project này)

| Model | Vai trò | Nơi lưu |
|---|---|---|
| `vinai/PhoWhisper-large` | ASR (`src/asr.py`) — đổi từ Qwen3-ASR-0.6B vì WER tiếng Việt thấp hơn hẳn, xem `EXPERIMENTS.md` | `D:\z_DaiHoc\a_zaloAI\models\hf-cache\hub\models--vinai--PhoWhisper-large` |
| `Qwen/Qwen3-4B-Instruct-2507` | LLM tool-calling (`src/llm_agent.py`) | `D:\z_DaiHoc\a_zaloAI\models\hf-cache\hub\models--Qwen--Qwen3-4B-Instruct-2507` |

### ASR đã thử và **đã bị loại** — giữ lại để đối chiếu, không dùng để nộp

Cả hai đã đo trên split `train` có nhãn và **thua PhoWhisper-large ~5 điểm WER** (xem `EXPERIMENTS.md`
mục local v4). Vẫn nằm trong `ASR_MODELS` của `src/asr.py` để chạy lại bằng `--asr-model`, và để thử
hướng ghép nhiều ASR còn đang tồn đọng.

| Model | Dung lượng | Nơi lưu |
|---|---|---|
| `erax-ai/EraX-WoW-Turbo-V1.1` | 1.6 GB | `...\hf-cache\hub\models--erax-ai--EraX-WoW-Turbo-V1.1` |
| `openai/whisper-large-v3-turbo` | 1.6 GB | `...\hf-cache\hub\models--openai--whisper-large-v3-turbo` |

Xoá nếu cần chỗ (tải lại được bằng `--prepare-only --asr-model <repo>`):

```powershell
Remove-Item -Recurse -Force D:\z_DaiHoc\a_zaloAI\models\hf-cache\hub\models--erax-ai--EraX-WoW-Turbo-V1.1
Remove-Item -Recurse -Force D:\z_DaiHoc\a_zaloAI\models\hf-cache\hub\models--openai--whisper-large-v3-turbo
```

## Chạy full bf16 trên GPU 4GB (RTX 2050) — ĐÃ XÁC NHẬN chạy được, không cần đổi driver

Pipeline **không** tự hạ precision xuống 4-bit hay đẩy model sang CPU để "vừa" VRAM nhỏ — chạy đúng
model/độ chính xác như trên A30 (bf16 đầy đủ), chấp nhận chậm hơn.

Tổng 2 model bf16 ≈ 3.1GB (ASR) + 8.0GB (LLM) = **~11GB**, vượt xa 4GB VRAM của RTX 2050. Phần vượt
tràn sang RAM hệ thống (32GB máy test) — **Windows/driver NVIDIA đã tự làm việc này, không cần bật
gì trong NVIDIA Control Panel** (giả thuyết ban đầu về "CUDA Sysmem Fallback Policy" sai — xem
`EXPERIMENTS.md` phần "v2 bf16 full" để biết chi tiết điều tra).

**OOM thật gặp phải và cách đã sửa**: context 8192 token mặc định ban đầu khiến 1 lượt attention cần
cấp tức thời 5.93GB, vượt ngưỡng cấp phát khả dụng → `torch.OutOfMemoryError`. Đã đổi default
`--context-size` 8192→**4096**, `--max-new-tokens` 384→**192** (khớp notebook Kaggle T4) trong
`scripts/infer.py` — **không cần truyền tay**, chạy lệnh bình thường là đủ:

```bash
python scripts/infer.py --split public_test --max-samples 2 --max-seconds 36000 \
  --output debug.jsonl --metrics-output debug.metrics.jsonl
```

Kết quả thật (2026-10-31): 2/2 mẫu `public_test` chạy xong, không fail, ~12-31 phút/mẫu (xem
`EXPERIMENTS.md`). `PYTORCH_ALLOC_CONF=expandable_segments:True` đã thử thêm trong code nhưng
**không có tác dụng trên Windows** (log báo "not supported on this platform") — giữ lại vô hại,
không phải fix chính.

**Nếu vẫn OOM với context mặc định 4096** (ví dụ mẫu có lịch sử hội thoại rất dài): giảm thêm qua
`--context-size 3072` hoặc đóng các app khác đang dùng GPU/RAM (trình duyệt, game) để có thêm RAM
trống trước khi chạy.

`src/models.py::model_path()` tự tìm theo thứ tự: `$HF_HOME` (nếu đặt) → `~/.cache/huggingface/hub`
(C:) → `models/hf-cache/hub` của project (D:). Vì 2 model trên đã nằm ở D:, chạy
`python scripts/infer.py ...` bình thường sẽ tự thấy, **không cần set gì thêm**.

## Không dùng bởi project này — đã dời từ C: sang D: để trả chỗ ổ C

Các model này thuộc công việc/project khác trên máy, không liên quan LifeMate. Đã **chuyển vật lý**
(không xoá) sang `D:\z_DaiHoc\a_zaloAI\models\hf-cache\hub\` để giải phóng ổ C. Nếu project khác cần
chúng và bị lỗi "model not found", trỏ tạm `HF_HOME` về đây cho riêng lệnh đó (không set vĩnh viễn):

```powershell
# PowerShell — chỉ áp dụng cho lệnh này, không ảnh hưởng session khác
$env:HF_HOME = "D:\z_DaiHoc\a_zaloAI\models\hf-cache"; python your_script.py
```

```bash
# bash
HF_HOME="D:/z_DaiHoc/a_zaloAI/models/hf-cache" python your_script.py
```

| Model | Kích thước |
|---|---|
| `BAAI/bge-m3` | 4.3 GB |
| `BAAI/bge-reranker-v2-m3` | 2.2 GB |
| `CS221DoAn/vietnamese_food_order_extraction` | 515 MB |
| `Qwen/Qwen3-Reranker-0.6B` | 1.2 GB |
| `intfloat/multilingual-e5-base` | 1.1 GB |

## Model cũ đã xoá hẳn (không còn pipeline nào trong repo này dùng)

Xoá thẳng khỏi C: (không chuyển, vì chắc chắn không còn cần — xem `EXPERIMENTS.md` phần lý do đổi
pipeline sang GPU transformers):

| Model | Lý do xoá |
|---|---|
| `Qualcomm-AI-Research/PhoASR-whisper-small` | ASR cũ, thay bằng Qwen3-ASR-0.6B-hf |
| `Qwen/Qwen3-4B` (bản base, không phải `-Instruct-2507`) | Không được code nào tham chiếu |
| `unsloth/Qwen3.5-4B-GGUF` | LLM cũ chạy qua llama.cpp CPU, đã bỏ hoàn toàn |
| `vinai/PhoWhisper-large` | ASR cũ không còn dùng |

Nếu cần lại, `python scripts/infer.py --prepare-only` sẽ tải đúng 2 model project này cần; các model
khác tải lại bằng `huggingface_hub.snapshot_download("<repo>")` bình thường khi cần.

## Kiểm tra nhanh dung lượng

```bash
df -h /c/ /d/
du -sh ~/.cache/huggingface/hub/*/ 2>/dev/null          # còn sót gì ở C không
du -sh "/d/z_DaiHoc/a_zaloAI/models/hf-cache/hub"/*/     # kiểm kê D
```
