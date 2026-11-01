# kaggle_v1 — PhoWhisper-large + Qwen3-4B bf16 trên 2x Tesla T4

**Bài nộp hợp lệ đầu tiên chạy trên GPU.** 634/634 mẫu, schema đúng. Trước lần này notebook chưa
bao giờ chạy hết tập.

`RUN_NAME = 'phowhisper_qwen4_v9'`.

## Cấu hình

| | |
|---|---|
| ASR | `vinai/PhoWhisper-large` (rev `b9136a44`) — trên `cuda:1` |
| LLM | `Qwen/Qwen3-4B-Instruct-2507` (rev `cdbee75f`), **bf16 đầy đủ** — trên `cuda:0` |
| Tập | `public_test`, 634/634 mẫu |
| Máy | Kaggle **2x Tesla T4** (15 GiB mỗi card) |
| Context / output | 4096 / 192 token |
| Prompt | **không cắt** — gửi nguyên schema tool |

Hai GPU nên ASR và LLM mỗi bên một card, không phải tranh VRAM như laptop. VRAM đỉnh:
`cuda:0` 12.7 GB (LLM), `cuda:1` 1.6 GB (ASR).

## Số đo

| | |
|---|---|
| Thời gian/mẫu | **trung bình 34.9s**, p95 68.9s, max 90.2s |
| Phân bổ | ASR 2.4s (7%) · LLM 32.5s (93%) |
| Vượt 30s | **332/634 (52%)** |
| Thất bại | 33 mẫu |
| Lượt LLM/mẫu | 2 lượt (235 mẫu), 3 (154), 4 (105), 1 (47), 5 (37), 6 (23) |
| Tool call nộp | 1378 call; **42 mẫu nộp 0 call**; 38 call bị từ chối vì schema |
| `stop_reason` | done 492, asked_missing 109, step_limit 16, time_limit 8, error 6, repeated_invalid 3 |

## Nhận xét

**52% mẫu vượt 30s trên T4.** T4 chậm hơn A30 đáng kể nên con số này không chứng minh được bản
Docker có đạt mốc BTC hay không — nhưng p95 68.9s thì khoảng cách phải bù là lớn, không thể coi là
chắc chắn đạt. `docs/context.txt` ước tính A30 ~2.5s/mẫu cho cấu hình 4.92B; bản này 5.57B và gửi
prompt không cắt, nên cần đo lại trên A30 thật.

**42 mẫu nộp 0 tool call.** Theo luật, mỗi mẫu như vậy chỉ được điểm nếu gold cũng rỗng, không thì 0.
Bản notebook này **vẫn mang bug parser** đã sửa ở local v3: tool call xuất ra dạng văn xuôi bị nhận
là câu trả lời cuối. Chưa port bản vá sang notebook.

**33 mẫu chắc chắn mất điểm** (step_limit 16 + time_limit 8 + error 6 + repeated_invalid 3).

**`local_train_score: null`** — preflight chạy trên `train` nhưng không chấm điểm, nên lần chạy này
không tự biết chất lượng của mình. `scripts/score.py` (thêm ở local v4) sẽ lấp chỗ này.

## File

| File | Nội dung |
|---|---|
| `submission.jsonl` | bài nộp 634 mẫu |
| `metrics.jsonl` | số đo từng mẫu, kèm transcript và VRAM đỉnh theo từng card |
| `summary.json` | tổng hợp của notebook |
| `run.log` | log đầy đủ |

Tải chọn lọc từ Kaggle (569 KB thay vì ~7 GB): tab **Output** → `lifemate/phowhisper_qwen4_v9/`.
Từ lần chạy sau không cần lọc nữa — notebook đã được sửa để đặt model ở `/kaggle/temp`, nằm ngoài
output.

## Điểm leaderboard: **13.41**

**Thấp hơn `local_v1` (22.52) tới 9 điểm.** Chuyển sang GPU, ASR tốt hơn và LLM mới hơn mà **mất
điểm**. Đây là kết quả âm quan trọng nhất của dự án.

### Phần mất điểm đo được

| | Số mẫu | Điểm mất tối đa |
|---|---|---|
| Nộp **0 tool call** | 42 | 6.6 |
| `failed` | 33 | 5.2 |
| *(hai nhóm không trùng mẫu nào)* | **75** | **11.8** |
| `asked_missing` | 109 | 17.2 |

`13.41 + 11.8 = 25.2` — **riêng hai nhóm đầu đã đủ giải thích toàn bộ khoảng cách 9 điểm**, chưa cần
tính tới 109 mẫu `asked_missing`.

**Cả 42 mẫu nộp 0 call đều có `stop_reason: done`.** Đó đúng là dấu vết bug parser đã sửa ở
`local_v3` nhưng chưa port: model xuất tool call dạng văn xuôi, parser đọc thành câu trả lời cuối,
nộp danh sách rỗng và **không báo lỗi gì**. Không mẫu nào trong 42 mẫu đó bị tính là `failed`, nên
không có cảnh báo nào trong summary.

### So với cấu hình đạt 22.52

`local_v1` (xem [`../local_v1/README.md`](../local_v1/README.md)) khác bản này ở **bốn** điểm, không
phải một:

| | local_v1 = **22.52** | kaggle_v1 = **13.41** |
|---|---|---|
| Context | **12288** | 4096 |
| Prompt tool | **nguyên catalogue, không route** | nén + route theo từng tool |
| LLM | **Qwen3.5-4B** Q8 (llama.cpp) | Qwen3-4B-Instruct-2507 bf16 |
| Thinking | **bật, 512 token** | tắt |
| `max_new_tokens` | 1024 | 192 |

Trước đó tôi ghi rằng bản này "không cắt prompt" — **sai**. `MAX_CONTEXT 4096` trừ 192 token output
còn 3904, trong khi prompt đã nén cần trung vị 4553, nên nó route ở phần lớn mẫu.

`kaggle_v2` sửa context + route + parser và nâng `max_new_tokens`. Hai khác biệt còn lại (Qwen3.5 và
thinking) **chưa chạm tới**: `ToolModel` khẳng định `model_type == 'qwen3'` và phần tái dùng KV-cache
chỉ được kiểm chứng cho full attention, còn Qwen3.5 dùng linear attention (DeltaNet) — đổi model là
sửa kiến trúc, không phải đổi config.
