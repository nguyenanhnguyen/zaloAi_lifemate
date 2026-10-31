# Model cache — ở đâu, dùng cho gì

Ghi lại vì cache Hugging Face đã từng làm đầy ổ C (chỉ còn 6.7GB, 2026-10-31). Đã dọn sang
`D:\z_DaiHoc\a_zaloAI\models\hf-cache\hub\`. Không set `HF_HOME` toàn máy (tránh quên chỉnh lại
sau này) — mỗi lệnh tự trỏ đúng nơi, xem phần "Lệnh" dưới đây.

## Dùng bởi pipeline LifeMate (project này)

| Model | Vai trò | Nơi lưu |
|---|---|---|
| `vinai/PhoWhisper-large` | ASR (`src/asr.py`) — đổi từ Qwen3-ASR-0.6B vì WER tiếng Việt thấp hơn hẳn, xem `EXPERIMENTS.md` | `D:\z_DaiHoc\a_zaloAI\models\hf-cache\hub\models--vinai--PhoWhisper-large` |
| `Qwen/Qwen3-4B-Instruct-2507` | LLM tool-calling (`src/llm_agent.py`) | `D:\z_DaiHoc\a_zaloAI\models\hf-cache\hub\models--Qwen--Qwen3-4B-Instruct-2507` |

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
