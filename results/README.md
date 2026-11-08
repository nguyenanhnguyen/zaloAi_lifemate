# results/ — lưu trữ các lần chạy đã thành công

Quy ước: **mỗi phiên bản chạy xong và ra kết quả thì có một thư mục riêng ở đây**, chứa file kết quả
phát sinh và một `.md` ghi lại cấu hình, số đo, điểm đạt được và bài học. Thư mục này chỉ thêm, không
sửa, không ghi đè — lần chạy mới luôn vào thư mục mới.

Đọc ở đây **trước khi** bắt đầu một thử nghiệm, để không chạy lại thứ đã có kết luận.

## Đặt tên

- `local_v<N>/` — chạy trên laptop RTX 2050 4GB
- `kaggle_v<N>/` — chạy trên Kaggle (T4/P100)

Hai dãy số độc lập nhau. Số chỉ tăng; một cấu hình đã đổi thì là phiên bản mới, không dùng lại số cũ.

## Đang có gì

| Thư mục | Model | Tập | Điểm | Ghi chú |
|---|---|---|---|---|
| [local_v1](local_v1/) | PhoASR-whisper-small + Qwen3.5-4B Q8 (llama.cpp CPU) | `public_test` 634/634 | **22.52** | Mốc tham chiếu duy nhất có điểm leaderboard |
| local_v2 | PhoWhisper-large + Qwen3-4B bf16 | — | — | Không có file kết quả: 1027s/mẫu, chưa bao giờ chạy hết tập. Chỉ còn ghi chú trong `EXPERIMENTS.md` |
| [local_v3](local_v3/) | PhoWhisper-large + Qwen3-4B 4-bit, prompt 1800 | `public_test` 5 và 10 mẫu | — | Hai lần dò để đo tốc độ, không phải bài nộp |
| [local_v4](local_v4/) | như v3 + ASR tách pha trên GPU | `train` 100 mẫu | **5.50** | Lần đầu đo F1 thật. **Tìm ra router prompt bỏ mất tool cần dùng ở 76% mẫu** — đọc trước khi cắt prompt lần nữa |
| [kaggle_v1](kaggle_v1/) | PhoWhisper-large + Qwen3-4B | `public_test` 634/634 | **13.41** | **Thấp hơn local_v1 9 điểm.** 42 mẫu nộp 0 call + 33 failed = 11.8 điểm mất trắng, đủ giải thích cả khoảng cách |
| [kaggle_v2](kaggle_v2/) | PhoWhisper-large + Qwen3-4B **fp16**, adapter **gộp**, đủ 147 tool | A/B: `train` 50 held-out · C: `public_test` **634/634** | **A 5.50 / B 0.44** · C *chờ portal* | **Phép A/B công bằng đầu tiên về adapter** (khác đúng một cờ): adapter thắng mọi cột — điểm **12.4×**, nhanh 1.40×, quá-30-giây 18% vs 52%, 0 thất bại vs 3. Cơ chế: cùng số call, ít token mỗi bước. **Lượt C là bài nộp**: 634 dòng hợp lệ, `done` 83.6%, mẫu 0-call **62 → 7** so với `local_v10`, 1684 call. Rò còn lại: 17 mẫu chạm trần `MAX_STEPS=8`. **Và: `done` KHÔNG nghĩa là đúng** — `done` 46/50 mà tổng F1 chỉ 2.75 |
| [local_v5](local_v5/) | như v4, bỏ ASR khỏi đường đi | `train` 50 mẫu | **3.90** | Toàn bộ VRAM cho LLM; GPU đỉnh 4096 MiB — bão hoà, tràn shared memory |
| [local_v6](local_v6/) | như v5 + màng chắn `x-ref` | `train` 50 mẫu | **3.90** | A/B đúng một biến. Màng chắn bắn trúng 10 mẫu nhưng tốn lượt LLM: 2 chạm `time_limit`, 2 chạm `step_limit` |
| [finetune_v1](finetune_v1/) | — (chỉ dựng dữ liệu) | — | — | Dataset SFT từ gold trajectory: 1233 train / 243 eval, chỉ **1.5%** token được tính loss |
| [local_v7](local_v7/) | như v6 + **adapter `lora_V4`** | `train` 50 mẫu | 2.57 → **4.00** | **Adapter đầu tiên của dự án được đo.** Bucket `KHÔNG ghi` **không** dịch (thoát 6 / rơi 7 = nhiễu), nhưng **quá 30 giây: 14/50 → 0/50**. Ngân sách prompt bị VRAM ép xuống 2000 token nên là phép đo **dưới mức** |

| [local_v8](local_v8/) | như v7, **không** adapter, `--ask-retries 1` | `train` 50 mẫu | 2.50 | **Tìm ra chỗ rò lớn nhất: `sys.ask_missing_info` là cửa một chiều, và base hỏi lại SAI ở 91% mẫu (21/50).** Mở cửa ra thì thoát 3 / rơi 0 — nhưng dưới trần 30 giây có **23/50 mẫu chạm `time_limit`**, nên chưa dùng được. Để `--ask-retries 0` cho bài nộp, đo lại trên A30 |

| [local_v9](local_v9/) | như v8, đẩy lại **chỉ** lần hỏi khi chưa đọc gì | `train` 50 mẫu | 2.57 / 4.00 | **`--ask-retries 1` nhắm đúng: miễn phí (giây/mẫu 15.8 → 16.4) và dịch một chiều (lora thoát `KHÔNG ghi` 5, rơi 0).** Chưa ăn điểm vì cả 5 vào `ghi sai` — ngân sách 2000 token nên model hành động bằng tool sai. **Bật khi chạy Kaggle.** Và: adapter dạy model hỏi khi chưa đọc ở 24/31 mẫu (base 7/23) |

| [local_v10](local_v10/) | như v9 + adapter, **cả tập** | `public_test` **634/634** | *chờ portal* | **Bài nộp đầy đủ đầu tiên có adapter.** 634 dòng hợp lệ, `done` 73.8%, 30 OOM, 24 quá 30 giây, 62 mẫu 0 call. Local **không chấm được** `public_test` (không có gold) — mốc so là `kaggle_v1` = 13.41. Tìm ra: `--prompt-token-budget` **không** chặn tổng prompt, và sàn 2335 token là một app nên hạ budget vô tác dụng |

> **`done` không phải thước đo chất lượng.** `kaggle_v2` cho thấy `done` 46/50 đi cùng điểm
> 5.50 và 0.44: mô hình tự tin kết thúc rồi gọi sai tool hoặc sai tham số, và `state_hash`
> chấm all-or-nothing nên lệch một tham số là mất trắng cả mẫu. Đọc `stop_reasons` như số đo
> vận hành, không như điểm.
>
> Hai điều cần biết trước khi đọc bảng này. **`local_v1` = 22.52 là đường đi khác** (llama.cpp
> CPU + Qwen3.5-4B Q8 + reasoning), không phải pipeline HF hiện tại — pipeline hiện tại xuất
> phát từ `kaggle_v1` = **13.41**. Và điểm của `local_v5`–`local_v7` đo trên 50 mẫu `train`
> với prompt bị cắt, **không so được** với điểm `public_test` 634 mẫu ở hai dòng đầu.

Lịch sử suy luận đầy đủ, gồm cả những hướng đã thử và bị loại, nằm trong
[`../EXPERIMENTS.md`](../EXPERIMENTS.md). Thư mục này giữ **số đo và file**, `EXPERIMENTS.md` giữ
**lý do**.
