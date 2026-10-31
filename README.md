# ZAC 2026 LifeMate

Một pipeline suy luận: audio tiếng Việt → ASR → LLM chọn tool → `ToolEnv` → `submission.jsonl`.

## Cấu trúc

- `scripts/infer.py`: lệnh chạy duy nhất, đọc split và ghi bài nộp.
- `src/asr.py`: nhận dạng giọng nói qua `transformers`, batch cả history của 1 mẫu trong 1 lượt.
  `ASR_MODELS` giữ các checkpoint đã kiểm tra luật (repo → revision pin); mặc định
  `vinai/PhoWhisper-large`.
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

5. **Chạy đủ tập để có file nộp** (laptop RTX 2050: đo được ~5.7 giờ cho 600 mẫu):

```bash
python scripts/infer.py --split public_test --max-seconds 3600 --output submission.jsonl
```

`--max-seconds 3600` là **bắt buộc trên laptop này**: mặc định là 30 (đúng luật A30 của BTC), nhưng
laptop chạy ~34s/mẫu nên ngân sách 30s sẽ cắt ngang lúc đang suy luận và làm hỏng phần lớn mẫu. Thời
gian vượt 30s/mẫu không đạt giới hạn cuộc thi — nhưng `tool_calls` và `state_hash` thu được vẫn đúng
để đối chiếu chất lượng, còn tốc độ thật thì đo trên A30/Kaggle.

Nếu bị ngắt giữa chừng, chạy lại đúng lệnh trên kèm `--resume` — vừa giữ các mẫu đã xong, vừa dùng
lại `submission.transcripts.jsonl` nên không phải chạy ASR lại.

6. Kiểm tra file nộp trước khi nộp:

```bash
python -c "import json;rows=[json.loads(l) for l in open('submission.jsonl',encoding='utf-8') if l.strip()];print(len(rows),'mẫu');print('schema ok',all(set(r)=={'id','tool_calls','state_hash'} for r in rows))"
```

## Đo điểm thật mà không tốn lượt nộp

Chỉ split `train` có nhãn. Chạy rồi chấm bằng đúng công thức BTC:

```bash
python scripts/infer.py --split train --max-samples 100 --max-seconds 3600 --output train_check.jsonl
python scripts/score.py train_check.jsonl --split train
```

`scripts/score.py` dựng lại `state_hash` chuẩn bằng cách replay `tool_calls` trong nhãn qua một
`ToolEnv` mới (nhãn không chứa hash sẵn), rồi tính F1 tool-call với các tham số `x-match: "fuzzy"`
chấm bằng token-F1 > 0.6 — giống cách BTC tái lập kết quả của ta. Nó in thêm phần tách nguyên nhân
mất điểm (lệch `state_hash`, gold rỗng, số mẫu 0 điểm) và danh sách mẫu điểm thấp nhất để truy lỗi.

## Cấu hình mô hình & giới hạn (đúng luật BTC)

- ASR: `vinai/PhoWhisper-large` (1.543B) · LLM: `Qwen/Qwen3-4B-Instruct-2507` (4.022B) · cả 2 **bf16
  đầy đủ**, không quantize — tổng **5.57B < 6B**, LLM **4.02B < 5B**.
- Mặc định: tối đa 8 lượt LLM/mẫu, **256 token đầu ra/lượt LLM**, 256 token/lượt ASR, tổng
  **context 4096 token** (≤16384 theo luật), và **prompt cắt về ~1800 token** qua
  `--prompt-token-budget` — đây là đòn tăng tốc lớn nhất trên GPU nhỏ (prompt 3.9k token khiến
  prefill mất 252s, prompt 1k chỉ 5.5s; xem `EXPERIMENTS.md`).
- Trên GPU dưới 10 GiB VRAM, mặc định `auto` chọn **LLM 4-bit + ASR chạy thành pha riêng trên GPU**:
  trọng số LLM nằm trọn trong VRAM (bf16 8.0 GB tràn ra RAM và chậm gấp ~20 lần), còn ASR được dùng
  cả card một mình ở pha trước (3.1s/file thay vì 24.5s trên CPU, transcript y hệt). Cấu hình này
  được **log rõ và ghi vào metrics**; dùng `--llm-precision bf16 --asr-phase inline` để chạy đúng A30.

## Ước tính thời gian (đo thật, không phải lý thuyết)

Trên **laptop RTX 2050 4GB + 32GB RAM**, cấu hình mặc định hiện tại: **~34s/mẫu** (n=5, dao động
16-56s) → **~5.7 giờ cho 600 mẫu**. Đủ chạy hết `public_test` trong một đêm.

| Phiên bản | s/mẫu | 600 mẫu | Thay đổi |
|---|---|---|---|
| v2 (bf16, prompt 3.9k) | 1027 | 171 h | nghẽn prefill tràn VRAM |
| v3 (4-bit, prompt 1800, ASR trên CPU) | 51 | 8.4 h | cắt prompt |
| **v4 (ASR thành pha riêng trên GPU)** | **34** | **5.7 h** | ASR nhanh 4.7x, `tool_calls` không đổi |

Lưu ý: cấu hình nhanh này (4-bit, prompt cắt, ASR tách pha) **khác** cấu hình nộp bài trên A30
(bf16, inline), nên điểm đo cục bộ không dự đoán chính xác điểm leaderboard. Dùng
`--llm-precision bf16 --asr-phase inline` khi cần đối chiếu đúng.

## So sánh model ASR (đã đo, đừng đo lại)

```bash
# Đo WER thật trên split train (có nhãn audio_text), không tốn lượt nộp
python scripts/infer.py --split train --max-samples 30 --asr-model erax-ai/EraX-WoW-Turbo-V1.1 \
  --max-seconds 3600 --output train_erax.jsonl
```

| ASR | Params | WER (n=12, train) | GPU s/file |
|---|---|---|---|
| **`vinai/PhoWhisper-large`** (đang dùng) | 1.543B | **23.59%** | 3.1 |
| `erax-ai/EraX-WoW-Turbo-V1.1` | 809M | 28.17% | 0.9 |
| `openai/whisper-large-v3-turbo` | 809M | 28.87% | 1.0 |

Kết luận: **giữ PhoWhisper-large.** Hai model turbo nhanh hơn 3x nhưng WER tệ hơn ~5 điểm, và model
đối chứng cho thấy fine-tune tiếng Việt của EraX chỉ hơn base 0.7 điểm — thứ quyết định là độ sâu
decoder (32 lớp vs 4), không phải dữ liệu huấn luyện. Chi tiết và danh sách ứng viên bị loại vì luật
(tham số/giấy phép/ngôn ngữ) nằm trong `EXPERIMENTS.md`.

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
