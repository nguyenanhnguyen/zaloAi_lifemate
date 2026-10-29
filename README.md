# ZAC 2026 LifeMate

Một pipeline suy luận: audio tiếng Việt → ASR → LLM chọn tool → `ToolEnv` → `submission.jsonl`.

## Cấu trúc

- `scripts/infer.py`: lệnh chạy duy nhất, đọc split và ghi bài nộp.
- `src/asr.py`: nhận dạng giọng nói bằng Whisper Small qua `transformers`.
- `src/llm_agent.py`: Qwen2.5-3B-Instruct 4-bit, giải mã xác định và đọc quyết định JSON.
- `src/pipeline.py`: lọc tool được phép, gọi tool từng bước, lấy call log và state hash thật.
- `docs/context.txt`, `docs/evaluation.png`: bản ghi quy định và công thức chấm điểm.

## Chạy lần đầu

1. Tạo môi trường Python và cài PyTorch CUDA phù hợp với máy, sau đó `pip install -r requirements.txt`. Kiểm tra `torch.cuda.is_available()` trả về `True` trước khi chạy.
2. Đặt gói dữ liệu BTC tại `data/train+public_test/train+public_test/` như hiện tại.
3. Chạy thử một mẫu:

```bash
python scripts/infer.py --split public_test --max-samples 1 --output debug.jsonl
```

4. Chạy toàn bộ:

```bash
python scripts/infer.py --split public_test --output submission.jsonl
```

Đo thời gian và bộ nhớ khi cho Windows dùng shared GPU memory:

```bash
python scripts/infer.py --split public_test --max-samples 1 --allow-shared-memory --metrics-output debug.metrics.jsonl --output debug.jsonl
```

Khi đã kiểm tra tốc độ, bỏ `--max-samples` và dùng `--output submission.jsonl` để chạy đủ tập. File metrics tách riêng, không nộp. Cột `cuda_peak_allocated_mb` là bộ nhớ PyTorch cấp phát, không gồm shared GPU memory do Windows quản lý; xem Task Manager để đo phần shared.

Mặc định đã dùng cấu hình nhẹ. Cờ `--low-memory` cũ được giữ để các lệnh trước vẫn chạy.

Giới hạn: tối đa 3 lượt LLM/mẫu, 160 token đầu ra/lượt và tổng context 4096 token. Ngân sách mặc định 28 giây/mẫu gồm ASR và LLM, không gồm tải/nạp model. Các bộ giải mã kiểm tra thời hạn giữa các bước sinh token; thời gian một phép tính đang chạy có thể khiến tổng thời gian vượt ngân sách. Khi hết ngân sách, ghi các tool calls thực sự đã gọi và state hash; điều này bảo đảm định dạng, không bảo đảm yêu cầu đã hoàn tất hay đạt điểm. Dùng `--max-steps` (1–3), `--max-seconds` (tối đa 30) để giảm giới hạn.

Mặc định: `openai/whisper-small` và `Qwen/Qwen2.5-3B-Instruct`. Lần chạy đầu sẽ cần tải model nếu cache chưa có. Để chạy offline, chuẩn bị sẵn model và truyền thư mục local:

```bash
python scripts/infer.py --whisper-model models/whisper --llm-model models/qwen2.5-3b
```

Có thể đổi vị trí dữ liệu bằng `--data-root`. Tệp kết quả có đúng một dòng JSON cho mỗi mẫu được chạy, với `id`, `tool_calls` từ `env.get_tool_calls()` và `state_hash` từ `env.get_state_hash()`.

## Quy định cần giữ

- Mỗi mẫu tạo đúng một `ToolEnv`; không đọc trực tiếp bất kỳ tệp nào trong `<split>_environment/`.
- Chỉ gọi tool của các ứng dụng trong `installed_apps` và `sys.ask_missing_info`.
- Khi nộp Docker, toàn bộ model phải có sẵn và inference chạy offline; giới hạn là dưới 30 giây mỗi mẫu, dưới 6B tham số toàn hệ thống và dưới 5B cho LLM.

Repo hiện chỉ chứa **pipeline suy luận**. Chưa có bước huấn luyện ASR bằng dữ liệu ngoài hay fine-tune LLM. Các lệnh train/evaluate/download cũ đã được bỏ vì không thuộc luồng chạy duy nhất và chưa tích hợp với pipeline hiện tại. Chi tiết quy định nằm trong [docs/context.txt](docs/context.txt).
