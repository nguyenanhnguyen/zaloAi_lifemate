# local_v1 — PhoASR-small + Qwen3.5-4B Q8 qua llama.cpp CPU

**Điểm leaderboard: 22.52** (`public_test`). Đây là **lần chạy duy nhất của dự án có điểm chính
thức** — mọi phiên bản sau đều phải so với con số này.

## Cấu hình

| | |
|---|---|
| ASR | `Qualcomm-AI-Research/PhoASR-whisper-small` |
| LLM | `Qwen3.5-4B-Q8_0.gguf` qua **llama.cpp, chạy CPU** (subprocess) |
| Context | **12288 token** |
| Prompt tool | **nguyên catalogue — không nén, không route** (phiên bản này không có `fit_tools`) |
| `max_new_tokens` | **1024** |
| Thinking | **bật**, `--reasoning-budget 512 --reasoning-format deepseek` |
| Tập | `public_test`, 634/634 mẫu |
| Máy | laptop RTX 2050 4GB + 32GB RAM (LLM không dùng GPU) |

Năm dòng cấu hình trên là **công thức duy nhất của dự án có điểm leaderboard tốt**. Đã kiểm chứng
trực tiếp từ `git show 7002744:src/llm_agent.py`, không phải suy đoán. `kaggle_v1` đổi cả năm thứ đó
và tụt xuống 13.41 — xem [`../kaggle_v1/README.md`](../kaggle_v1/README.md).

Lưu ý ASR: bản này dùng PhoASR-whisper-**small**, tức ASR *nhỏ hơn* `PhoWhisper-large` của các bản
sau, mà vẫn hơn 9 điểm. Chất lượng ASR không phải chỗ quyết định điểm; prompt và LLM mới là.

## Số đo

| | |
|---|---|
| Thời gian/mẫu | **trung bình 166.9s**, trung vị 148.8s, min 34.8s, **max 529.7s** |
| Phân bổ | ASR 19.4s (12%) · LLM 147.4s (88%) |
| Vượt 30s | **634/634 — toàn bộ** |
| Thất bại | 1 mẫu |
| RAM đỉnh | 8.2 GB |
| Lượt LLM/mẫu | 4 lượt (135 mẫu), 3 (132), 2 (120), 5 (81), 8 (67) |
| Tool call nộp | 2024 call; **14 mẫu nộp 0 call** |
| `stop_reason` | done 487, empty_response 58, step_limit 43, asked_missing 33, repeated_invalid_call 12, error 1 |

## Bài học

**Điểm 22.52 đạt được bằng một cấu hình không thể nộp Docker.** Luật BTC là 30 giây/mẫu trên A30;
bản này mất trung bình 167 giây và **không một mẫu nào** dưới 30s. Nó chứng minh pipeline đúng về
logic, nhưng toàn bộ backend buộc phải chuyển sang GPU — đó là lý do có v2 trở đi.

**58 mẫu `empty_response`** là con số lớn nhất đáng chú ý ngoài tốc độ: LLM trả lời rỗng, không
gọi tool mà cũng không kết thúc. Cộng với 14 mẫu nộp 0 call, đây là phần mất điểm có thể sửa được
mà không cần đổi model.

## File

| File | Nội dung |
|---|---|
| `submission.jsonl` | bài nộp 634 mẫu đã đạt 22.52 |
| `submission.metrics.jsonl` | số đo từng mẫu: thời gian, transcript, `stop_reason` |

Khôi phục từ git commit `7002744` (hai file này từng bị xoá khỏi repo ở commit `4799657`). Git còn
một bản chạy dở 242 mẫu của cùng cấu hình ở commit `114f997` (trung bình 163.2s) — không lưu ở đây
vì chỉ là tiền tố của chính lần chạy này.
