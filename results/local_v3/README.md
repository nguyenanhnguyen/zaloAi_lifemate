# local_v3 — cắt prompt về 1800 token, LLM 4-bit

Không phải bài nộp. Đây là **hai lần dò trên `public_test`** để đo xem laptop có thể chạy hết tập
trong dưới 24 giờ hay không. Kết quả: có.

## Cấu hình

| | |
|---|---|
| ASR | `vinai/PhoWhisper-large` (rev `b9136a44`), chạy **CPU** |
| LLM | `Qwen/Qwen3-4B-Instruct-2507` (rev `cdbee75f`), **4-bit NF4** |
| Prompt | cắt về **1800 token** bằng định tuyến theo app (`--prompt-token-budget`) |
| Context / output | 4096 / 256 token |
| Ngân sách | `--max-seconds 3600` (nghiên cứu, không phải mốc 30s của BTC) |

## Số đo

| | 5 mẫu | 10 mẫu |
|---|---|---|
| Trung bình | 50.7s | 63.9s |
| ASR | 22.6s (45%) | 24.4s (38%) |
| Thất bại | 0 | 1 (`public_0009`) |

Suy ra **~8.4 giờ cho 600 mẫu** — lần đầu laptop đạt mức chạy được cả tập trong một đêm, từ 171 giờ
của v2.

## Bài học

**Nghẽn là độ dài prompt, không phải VRAM.** Benchmark riêng cho thấy prefill 1k token mất 5.5s còn
3.9k token mất **252.5s**; decode chỉ ~88ms/token. Prompt 3.9k đó gần như toàn bộ là schema của 34
tool, nên cắt nó đi là đòn tăng tốc lớn nhất. Giả thuyết ban đầu của tôi — trọng số bf16 tràn VRAM —
**sai**: chuyển 4-bit chỉ nhanh hơn 2x, không phải 100x.

**Định tuyến phải dùng tiếng Việt.** Bản cắt prompt đầu tiên xếp hạng tool bằng cách so transcript
tiếng Việt với tên tool tiếng Anh (`alarm.update`) → chọn gần như ngẫu nhiên và cắt mất đúng tool cần
(`public_0002` tụt từ `ride_hailing.book` xuống `ask_missing_info`). Sửa bằng `name_vi` trong catalog
và char-3gram đã bỏ dấu.

**Một bug mất điểm trong im lặng.** Model đôi khi xuất tool call dạng văn xuôi
(`sys.ask_missing_info({...})`); parser nhận đó là câu trả lời cuối → **nộp 0 tool call, không báo
lỗi gì**. Phát hiện ở `public_0004`. Notebook kaggle_v1 vẫn mang bug này.

**Còn sai ở đây, đã sửa ở v4:** ASR chiếm 40-45% thời gian mà vẫn bị để trên CPU — comment trong code
lúc đó vẫn ghi "ASR is only ~1% of a sample's time", đúng ở thời v2 nhưng đã lạc hậu.

## File

| File | Nội dung |
|---|---|
| `probe5_public_test.jsonl` / `.metrics.jsonl` / `.log` | lần dò 5 mẫu |
| `probe10_public_test.jsonl` / `.metrics.jsonl` | lần dò 10 mẫu |
