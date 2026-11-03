# baseline_2252 — bản chạy đạt 22.52, truy lại từ git

Đây là **bản phục dựng** của lần chạy ghi 22.52 trên leaderboard (`results/local_v1`), do
Codex làm trước khi dự án chuyển sang Qwen3-4B. Folder này tồn tại để bản đó không bị mất,
và để so sánh được với mọi bản sau.

## Truy vết: vì sao chắc chắn là commit này

`pipeline/` là cây làm việc tại **`7002744`** (31/10/2026, "update gitignore"), lấy ra bằng
`git archive`. Bằng chứng, chứ không phải suy đoán:

| | |
|---|---|
| `submission.jsonl` tại `7002744` | md5 `8e65b65a4b7c0a17bab102a1b3ad8ba1` |
| `results/local_v1/submission.jsonl` | md5 `8e65b65a4b7c0a17bab102a1b3ad8ba1` |
| `submission.metrics.jsonl` tại `7002744` | md5 `b1dff0b2a5932d9993866fad6a11ab86` |
| `results/local_v1/submission.metrics.jsonl` | md5 `b1dff0b2a5932d9993866fad6a11ab86` |

Khớp byte-đúng cả hai file. Và `git diff f71bc40 7002744 -- src/` **rỗng**, nên code suy luận
không đổi giữa lần chạy và commit — không có khoảng mờ nào để nghi ngờ.

Hai commit sau (`0498c10`) có cùng md5 nhưng chỉ khác `.gitignore`; `7002744` là commit sớm
nhất mang đủ bằng chứng.

## Không có model đã train

Cần nói rõ, vì đây là điều hay bị nhớ sai: **lần chạy 22.52 không fine-tune gì cả.** Nó là
Qwen3.5-4B Q8 nguyên bản qua llama.cpp, zero-shot, có bật thinking. Không tồn tại checkpoint
hay adapter nào để truy lại — không phải vì bị xoá, mà vì chưa từng có.

Việc fine-tune trong dự án này bắt đầu ở `319423a` (01/11), trên một model **khác**
(Qwen3-4B-Instruct-2507), và chưa ra điểm.

## Trọng số: đã bị xoá khỏi đĩa, nhưng tải lại được chính xác

Tìm toàn máy không còn file `.gguf` nào, và cache HuggingFace không còn
`models--unsloth--Qwen3.5-4B-GGUF` lẫn `models--Qualcomm-AI-Research--PhoASR-whisper-small`.
Không sao: code pin cả repo lẫn revision, nên tải lại cho ra đúng bit cũ.

| | Repo | Revision | File |
|---|---|---|---|
| LLM | `unsloth/Qwen3.5-4B-GGUF` | `e87f176479d0855a907a41277aca2f8ee7a09523` | `Qwen3.5-4B-Q8_0.gguf` (~4.2 GB) |
| ASR | `Qualcomm-AI-Research/PhoASR-whisper-small` | `da2d9ae0ac0a6a6faba20add61c2e980ccd85037` | snapshot |

Chạy `python restore.py` trong folder này để tải cả hai về `models/`.

Ngoài ra cần **`llama-server`** của llama.cpp (không có trong git, không tải tự động được
bằng script này). Code tìm nó theo thứ tự: biến môi trường `LLAMA_SERVER` → đệ quy trong
`models/llama.cpp/` → `PATH`.

## Cấu hình đã chạy

| | |
|---|---|
| LLM | Qwen3.5-4B **Q8_0** GGUF, llama.cpp server, `--flash-attn on`, `--jinja` |
| Context | 12288 |
| `max_new_tokens` | 1024 |
| Thinking | **bật**, `--reasoning-budget 512`, `--reasoning-format deepseek` |
| ASR | PhoASR-whisper-small |
| `transformers` | **4.57.6** (repo hiện tại dùng 5.13.0 — phải pin lại để chạy bản này) |

## Cảnh báo quan trọng: 22.52 không phải điểm hợp lệ

Từ `submission.metrics.jsonl` của chính lần chạy đó:

| | |
|---|---|
| Thời gian/mẫu, trung vị | **149.3 giây** |
| p90 | 294.1 giây |
| Tối đa | 529.7 giây |
| Mẫu vượt 30 giây | **634/634 (100%)** |

Luật là 30 giây/mẫu trên 1×A30 24GB, timeout tính 0 điểm. Mốc đó **chỉ được áp ở vòng
Docker**, nên ở vòng public test bản này nộp được và ghi 22.52 thật. Nhưng trong container
nộp ngày 16/11 thì cả 634 mẫu đều timeout, và điểm là **0**.

Nên con số 22.52 dùng được để làm gì, và không dùng được để làm gì:

- **Dùng được:** nó chứng minh pipeline này (parser, định tuyến, vòng gọi tool) *đúng về
  logic* — 22.52 so với 13.41 của `kaggle_v1` là chênh thật về chất lượng quyết định.
- **Không dùng được:** làm mục tiêu cần vượt. Điểm hợp lệ tốt nhất ta có là **13.41**.
  Không có cách nào ép 149 giây xuống 30 giây bằng cách tắt thinking hay đổi quant — đó là
  gấp 5 lần.

## Hai câu còn phải kiểm trước khi dùng lại Qwen3.5

1. **Giấy phép.** Luật loại thẳng mọi licence NonCommercial. `unsloth/Qwen3.5-4B-GGUF` là
   bản quantize lại của người khác; phải đọc licence của cả repo gốc và repo quantize.
2. **Ngày phát hành.** Chỉ model công khai trước **26/10/2026** được dùng. Chưa ai trong dự
   án kiểm ngày của Qwen3.5-4B.

Nếu một trong hai câu trả lời là không, thì folder này thuần là tư liệu, và mọi so sánh với
22.52 chỉ còn giá trị chẩn đoán.

## Nội dung folder

```
baseline_2252/
  README.md     file này
  restore.py    tải lại hai model theo đúng revision đã pin
  pipeline/     cây code tại 7002744, nguyên vẹn, kèm submission + metrics của lần chạy
  models/       chỗ restore.py đặt trọng số (gitignore)
```
