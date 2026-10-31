# ZAC 2026 LifeMate

Một pipeline suy luận: audio tiếng Việt → ASR → LLM chọn tool → `ToolEnv` → `submission.jsonl`.

## Cấu trúc

- `scripts/infer.py`: lệnh chạy duy nhất, đọc split và ghi bài nộp.
- `src/asr.py`: nhận dạng giọng nói bằng Qwen3-ASR-0.6B-hf qua `transformers`, GPU, batch cả history trong 1 lượt.
- `src/llm_agent.py`: Qwen3-4B-Instruct-2507 bf16 qua `transformers`, GPU, tái dùng KV-cache giữa các lượt tool trong 1 mẫu.
- `src/models.py`: tiện ích tải/định vị checkpoint Hugging Face dùng chung cho ASR và LLM.
- `src/pipeline.py`: lọc tool được phép, gọi tool từng bước, lấy call log và state hash thật.
- `docs/context.txt`, `docs/evaluation.png`: bản ghi quy định và công thức chấm điểm.
- `EXPERIMENTS.md`: lịch sử các lần chạy, model dùng và điểm đạt được — đọc trước khi đổi pipeline.
- `lifemate_kaggle.ipynb`: notebook tương đương để chạy demo trên Kaggle GPU (T4/A30).

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

Đo thời gian/VRAM trên máy yếu (vd. 4 GB): không cần cấu hình gì thêm — chỉ `.to("cuda")`,
driver NVIDIA tự swap sang shared GPU memory (RAM) khi thiếu VRAM. Chạy chậm hơn nhưng không cần
chỉnh layer CPU/GPU thủ công như bản llama.cpp cũ (xem [EXPERIMENTS.md](EXPERIMENTS.md)).

Khi đã kiểm tra tốc độ, bỏ `--max-samples` và dùng `--output submission.jsonl` để chạy đủ tập. File
metrics tách riêng (`*.metrics.jsonl`), không nộp.

Mặc định dùng Qwen3-ASR-0.6B-hf (ASR) + Qwen3-4B-Instruct-2507 (LLM), cả hai bf16 trên GPU —
tổng 4.8B tham số, đúng luật (<6B toàn hệ thống, LLM <5B). Ngân sách mặc định 30 giây/mẫu khớp
luật A30 của BTC; dùng `--max-seconds 180` chỉ để nghiên cứu cục bộ, không phải mốc nộp bài.

Giới hạn: tối đa 8 lượt LLM/mẫu, 384 token đầu ra/lượt LLM, 256 token/lượt ASR, tổng context 8192
token (≤16384 theo luật). Các bộ giải mã kiểm tra thời hạn giữa các bước sinh token; thời gian một
phép tính đang chạy có thể khiến tổng thời gian vượt ngân sách. Khi hết ngân sách, ghi các tool
calls thực sự đã gọi và state hash.

Lần chạy đầu sẽ cần tải model nếu cache chưa có (`python scripts/infer.py --prepare-only`). Để chạy
offline, chuẩn bị sẵn model và truyền thư mục local:

Cache Hugging Face mặc định của dự án nằm tại `models/hf-cache/` trên ổ D, tránh làm đầy ổ C. Có thể
đặt `HF_HOME` trước khi chạy nếu muốn dùng thư mục khác.

```bash
python scripts/infer.py --asr-model models/qwen3-asr-0.6b --llm-model models/qwen3-4b-instruct
```

Có thể đổi vị trí dữ liệu bằng `--data-root`. Tệp kết quả có đúng một dòng JSON cho mỗi mẫu được chạy, với `id`, `tool_calls` từ `env.get_tool_calls()` và `state_hash` từ `env.get_state_hash()`.

## Quy định cần giữ

- Mỗi mẫu tạo đúng một `ToolEnv`; không đọc trực tiếp bất kỳ tệp nào trong `<split>_environment/`.
- Chỉ gọi tool của các ứng dụng trong `installed_apps` và `sys.ask_missing_info`.
- Khi nộp Docker, toàn bộ model phải có sẵn và inference chạy offline; giới hạn là dưới 30 giây mỗi mẫu, dưới 6B tham số toàn hệ thống và dưới 5B cho LLM.

Repo hiện chỉ chứa **pipeline suy luận**. Chưa có bước huấn luyện ASR bằng dữ liệu ngoài hay fine-tune LLM. Các lệnh train/evaluate/download cũ đã được bỏ vì không thuộc luồng chạy duy nhất và chưa tích hợp với pipeline hiện tại. Chi tiết quy định nằm trong [docs/context.txt](docs/context.txt).
