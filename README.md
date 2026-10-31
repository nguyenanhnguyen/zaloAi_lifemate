# ZAC 2026 LifeMate

Một pipeline suy luận: audio tiếng Việt → ASR → LLM chọn tool → `ToolEnv` → `submission.jsonl`.

## Cấu trúc

- `scripts/infer.py`: lệnh chạy duy nhất, đọc split và ghi bài nộp.
- `src/asr.py`: nhận dạng giọng nói bằng `vinai/PhoWhisper-large` qua `transformers`, GPU, batch cả
  history của 1 mẫu trong 1 lượt.
- `src/llm_agent.py`: `Qwen/Qwen3-4B-Instruct-2507` bf16 qua `transformers`, GPU, tái dùng KV-cache
  giữa các lượt tool trong 1 mẫu, tự nén/định tuyến schema tool để vừa context.
- `src/models.py`: tiện ích tải/định vị checkpoint Hugging Face dùng chung cho ASR và LLM.
- `src/pipeline.py`: lọc tool được phép, gọi tool từng bước, lấy call log và state hash thật.
- `docs/context.txt`, `docs/evaluation.png`: bản ghi quy định và công thức chấm điểm của BTC.
- `lifemate_kaggle.ipynb`: pipeline tương đương để chạy trên Kaggle GPU (T4/A30) — nhanh hơn laptop
  nhiều lần, dùng khi cần chạy đủ `public_test`/`private_test`.
- `EXPERIMENTS.md`: lịch sử từng lần chạy, model dùng, điểm đạt được và lý do đổi hướng — đọc trước
  khi đổi model/pipeline để không lặp lại thử nghiệm đã có kết luận.
- `MODELS.md`: model nào lưu ở đâu, dọn cache, chi tiết kỹ thuật việc chạy bf16 trên GPU nhỏ.

## Chạy lần đầu

1. Tạo môi trường Python, cài PyTorch CUDA phù hợp máy, rồi `pip install -r requirements.txt`.
   Kiểm tra `torch.cuda.is_available()` trả `True` trước khi chạy.
2. Đặt gói dữ liệu BTC tại `data/train+public_test/train+public_test/` như hiện tại.
3. Tải model lần đầu (nếu cache chưa có):

```bash
python scripts/infer.py --prepare-only
```

4. Chạy thử 1 mẫu, ngân sách rộng để không bị cắt giữa chừng khi debug cục bộ:

```bash
python scripts/infer.py --split public_test --max-samples 1 --max-seconds 3600 \
  --output debug.jsonl --metrics-output debug.metrics.jsonl
```

5. Chạy đủ tập (xem mục "Ước tính thời gian" dưới — trên máy yếu có thể mất nhiều ngày, nên cân
   nhắc chạy trên Kaggle thay vì ở đây):

```bash
python scripts/infer.py --split public_test --output submission.jsonl
```

Mặc định `--max-seconds 30` (đúng luật A30 của BTC). Dùng `--max-seconds` lớn hơn (vd. `3600`) chỉ
để nghiên cứu cục bộ — thời gian vượt 30s/mẫu không đạt giới hạn cuộc thi, nhưng cần ngân sách rộng
trên máy yếu để pipeline hoàn tất tự nhiên (DONE/asked_missing) thay vì bị cắt lúc đang suy luận.

## Cấu hình mô hình & giới hạn (đúng luật BTC)

- ASR: `vinai/PhoWhisper-large` (1.543B) · LLM: `Qwen/Qwen3-4B-Instruct-2507` (4.022B) · cả 2 **bf16
  đầy đủ**, không quantize — tổng **5.57B < 6B**, LLM **4.02B < 5B**.
- Mặc định: tối đa 8 lượt LLM/mẫu, **256 token đầu ra/lượt LLM**, 256 token/lượt ASR, tổng
  **context 4096 token** (≤16384 theo luật), và **prompt cắt về ~1800 token** qua
  `--prompt-token-budget` — đây là đòn tăng tốc lớn nhất trên GPU nhỏ (prompt 3.9k token khiến
  prefill mất 252s, prompt 1k chỉ 5.5s; xem `EXPERIMENTS.md`).
- Trên GPU dưới 10 GiB VRAM, mặc định `auto` chọn **LLM 4-bit + ASR trên CPU** để trọng số LLM nằm
  trọn trong VRAM (bf16 8.0 GB tràn ra RAM và chậm gấp ~20 lần). Cấu hình này được **log rõ và ghi
  vào metrics**; dùng `--llm-precision bf16 --asr-device cuda` để chạy đúng như A30.

## Ước tính thời gian (đo thật, không phải lý thuyết)

Trên **laptop RTX 2050 4GB + 32GB RAM**, cấu hình mặc định hiện tại (LLM 4-bit resident, ASR trên
CPU, prompt cắt về ~1800 token): **~51s/mẫu** (n=5, dao động 30-96s) → **~8.4 giờ cho 600 mẫu**
(xấu nhất ~16h). Đủ để chạy hết `public_test` trong một đêm.

Trước khi tối ưu, cấu hình bf16 + prompt 3.9k token mất **~1027s/mẫu ≈ 171 giờ** — nghẽn là prefill
vượt VRAM, không phải GPU yếu. Chi tiết benchmark và từng thay đổi nằm trong `EXPERIMENTS.md`.

Lưu ý: cấu hình nhanh này (4-bit, prompt cắt) **khác** cấu hình nộp bài trên A30 (bf16), nên điểm đo
cục bộ không dự đoán chính xác điểm leaderboard. Dùng `--llm-precision bf16` khi cần đối chiếu đúng.

## Kiểm soát khi chạy trên laptop yếu VRAM

- **Chỉ chạy 1 tiến trình GPU cùng lúc** — GPU 4GB không chịu được 2 tiến trình PyTorch tranh VRAM
  (sẽ OOM cho cả hai). Kiểm tra trước khi chạy thêm lệnh:
  ```bash
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv
  ```
- Dừng một lần chạy đang treo (tìm PID qua lệnh trên hoặc Task Manager), xác nhận đúng tiến trình
  trước khi kill:
  ```powershell
  Get-CimInstance Win32_Process -Filter "ProcessId=<PID>" | Select-Object CommandLine
  Stop-Process -Id <PID>
  ```
- Resume một lần chạy bị ngắt giữa chừng (tiếp tục từ mẫu đã xong, không chạy lại từ đầu):
  ```bash
  python scripts/infer.py --split public_test --resume --output submission.jsonl
  ```
- Chạy vài mẫu chỉ định theo ID (debug một mẫu cụ thể bị lỗi):
  ```bash
  python scripts/infer.py --split public_test --sample-ids public_0002 public_0003 \
    --max-seconds 3600 --output debug.jsonl --metrics-output debug.metrics.jsonl
  ```
- Dọn cache model không dùng / xem model nào ở đâu: xem `MODELS.md`.

## Quy định cần giữ (BTC)

- Mỗi mẫu tạo đúng một `ToolEnv`; không đọc trực tiếp bất kỳ tệp nào trong `<split>_environment/`.
- Chỉ gọi tool của các ứng dụng trong `installed_apps` và `sys.ask_missing_info`.
- Khi nộp Docker: toàn bộ model có sẵn, inference chạy offline hoàn toàn; giới hạn 30 giây/mẫu,
  dưới 6B tham số toàn hệ thống, dưới 5B cho LLM, context ≤16384 token.
- Decode tất định (greedy, temperature=0), cố định seed — đã áp dụng trong `src/llm_agent.py` và
  `src/asr.py`.

Repo hiện chỉ chứa **pipeline suy luận**. Chưa có bước huấn luyện ASR bằng dữ liệu ngoài hay
fine-tune LLM. Chi tiết quy định đầy đủ nằm trong [docs/context.txt](docs/context.txt).
