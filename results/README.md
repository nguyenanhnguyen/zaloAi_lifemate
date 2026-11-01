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
| [kaggle_v2](kaggle_v2/) | PhoWhisper-large + Qwen3-4B bf16, **không route prompt** | *chờ chạy* | — | Notebook đã sẵn sàng: context 12288, LLM chia 2 GPU, vá parser. Đọc `.md` trước khi chạy |

Lịch sử suy luận đầy đủ, gồm cả những hướng đã thử và bị loại, nằm trong
[`../EXPERIMENTS.md`](../EXPERIMENTS.md). Thư mục này giữ **số đo và file**, `EXPERIMENTS.md` giữ
**lý do**.
