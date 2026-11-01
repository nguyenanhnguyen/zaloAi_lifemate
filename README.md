# ZAC 2026 LifeMate

Một pipeline suy luận: audio tiếng Việt → ASR → LLM chọn tool → `ToolEnv` → `submission.jsonl`.

👉 **Việc cần làm bây giờ, theo thứ tự: [RUNBOOK.md](RUNBOOK.md).** File này mô tả hệ thống; RUNBOOK
mô tả các bước phải bấm tay trên Kaggle và đọc số nào để ra quyết định.

## Cấu trúc

- `scripts/infer.py`: lệnh chạy duy nhất, đọc split và ghi bài nộp.
- `src/asr.py`: nhận dạng giọng nói qua `transformers`, batch cả history của 1 mẫu trong 1 lượt.
  `ASR_MODELS` giữ các checkpoint đã kiểm tra luật (repo → revision pin); mặc định
  `vinai/PhoWhisper-large`.
- `src/llm_agent.py`: `Qwen/Qwen3-4B-Instruct-2507` bf16 qua `transformers`, GPU, tái dùng KV-cache
  giữa các lượt tool trong 1 mẫu, tự nén/định tuyến schema tool để vừa context.
- `src/models.py`: tiện ích tải/định vị checkpoint Hugging Face dùng chung cho ASR và LLM.
- `src/text_norm.py`: đổi số nói thành chữ số trong transcript ("sáu mươi phút" → "60 phút"). Cố ý
  bảo thủ: chỉ đổi khi parse sạch, từ chối mọi chỗ nhập nhằng. Bật sẵn, tắt bằng `--raw-numbers`.
- `scripts/build_sft.py` + `scripts/train_sft.py`: dựng dataset SFT từ nhãn `train` và fine-tune
  QLoRA. **Phải nằm trong repo** vì BTC yêu cầu nộp code huấn luyện kèm Docker.
- `scripts/asr_bench.py`: đo WER thật trên `train` theo nhiều chiến lược decode, không tốn lượt nộp.
- `src/pipeline.py`: lọc tool được phép, gọi tool từng bước, lấy call log và state hash thật.
- `docs/context.txt`, `docs/evaluation.png`: bản ghi quy định và công thức chấm điểm của BTC.
- `notebook/lifemate_kaggle_v2.ipynb`: pipeline tương đương để chạy trên Kaggle GPU (2x T4) — bản
  **đang dùng**, gửi cả catalogue tool với `MAX_CONTEXT 12288` và chia LLM lên 2 GPU. Dùng khi cần
  chạy đủ `public_test`/`private_test`. `lifemate_kaggle_v1.ipynb` giữ lại để đối chiếu, **không
  chạy nữa**: nó route prompt và mất điểm nặng (xem `results/kaggle_v2/README.md`).
- `EXPERIMENTS.md`: lịch sử suy luận — vì sao đổi model/pipeline, hướng nào đã thử và bị loại. Đọc
  trước khi đổi model/pipeline để không lặp lại thử nghiệm đã có kết luận.
- `results/`: **lưu trữ các lần chạy đã thành công**, mỗi phiên bản một thư mục
  (`local_v<N>/`, `kaggle_v<N>/`) gồm file kết quả và một `.md` ghi cấu hình, số đo, điểm, bài học.
  Chỉ thêm, không ghi đè. Xem [results/README.md](results/README.md).
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

## Ghép laptop với Kaggle: chạy ASR ở laptop, LLM ở Kaggle

Pipeline tách được làm hai chặng, và **transcript cache là đường nối**. Laptop chạy ASR một mình
vừa 4GB VRAM (3.1s/file), còn Kaggle dành toàn bộ GPU cho LLM — phần thật sự nặng.

**Chặng 1 — laptop sinh transcript** (chỉ ASR, không nạp LLM):

```bash
python scripts/infer.py --split public_test --asr-only \
  --transcript-cache public_test.transcripts.jsonl
```

`--asr-only` dừng ngay sau chặng ASR, **không nạp LLM** — nên card 4GB không bao giờ phải giữ cả
hai model. Bị ngắt giữa chừng thì chạy lại đúng lệnh đó, nó đọc tiếp từ cache.

File ra có khoá **portable**, là đường dẫn tương đối trong split, đúng như `<split>.jsonl` ghi:

```json
{"audio": "audio/public_test_0001/public_test_0001.wav", "text": "..."}
```

Trước v5 khoá là đường dẫn tuyệt đối của Windows (`D:\...\train_0001.wav`) nên Kaggle không dùng
được. Cache cũ vẫn đọc được (nhận cả khoá `path`), nhưng cache mới mới ghép được hai máy.

**Chặng 2 — Kaggle đọc cache**: upload file đó thành một Kaggle Dataset, rồi trong notebook đặt

```python
TRANSCRIPT_CACHE = Path('/kaggle/input/lifemate-transcripts/public_test.transcripts.jsonl')
```

Notebook sẽ bỏ qua ASR cho mọi mẫu có trong cache, và **vẫn tự chạy ASR cho mẫu nào thiếu** — nên
cache thiếu hay chạy dở cũng không sao. `metrics.jsonl` ghi `asr_cached` từng mẫu để biết đã dùng lại
bao nhiêu, và `manifest.json` ghi đường dẫn + số dòng của cache nên lần chạy vẫn tái lập được.

⚠️ **Chỉ dùng cho vòng public test**, nơi ta tự chạy và tự nộp JSONL. Bản Docker phải tự chạy ASR
bên trong container (BTC mount private test rồi chạy, không có cache nào). Đặt `TRANSCRIPT_CACHE =
None` trước khi đóng gói.

## Fine-tune (QLoRA) — laptop chuẩn bị dữ liệu, Kaggle huấn luyện

Nhãn `train` **đã là trajectory agent**: `label.tool_calls` khớp 1:1 với `label.tool_results` ở cả
1434 mẫu có nhãn, nên mỗi mẫu replay thành một hội thoại nhiều lượt mà mỗi lượt assistant là một
gold call, lượt cuối là `DONE`. Không bịa gì cả — đây là lý do fine-tune khả thi ngay.

**Bước 1 — laptop: ASR toàn bộ train rồi dựng dataset.**

```bash
python scripts/infer.py --split train --asr-only --transcript-cache train_all.transcripts.jsonl
python scripts/build_sft.py --transcripts train_all.transcripts.jsonl --text-source asr \
  --out sft --count-tokens
python scripts/train_sft.py --train sft.train.jsonl --eval sft.eval.jsonl --cache sft.tok --tokenize-only
```

Huấn luyện trên **transcript ASR của chính ta**, không phải `audio_text` sạch: lúc chạy thật model
chỉ bao giờ thấy đầu ra ASR ở ~24% WER, học trên văn bản sạch là dạy nó dựa vào những từ nó sẽ không
bao giờ nhận được. Chi phí gần như bằng 0: trung vị chỉ dài hơn 25 token.

| | `audio_text` (nhãn sạch) | **ASR thật** |
|---|---|---|
| Trung vị token | 5281 | **5306** |
| p95 / max | — / 9503 | 8011 / **9554** |
| Vừa seq 8192 | 96.4% | **96.0%** |

Kết quả: **1284 train / 261 eval**, 0 mẫu bị loại, không id nào nằm cả hai bên (chia theo sha256 của
id nên build lại không rò rỉ eval).

**Bước 2 — Kaggle: QLoRA.**

```bash
pip install -q peft bitsandbytes
python scripts/train_sft.py --train sft.train.jsonl --eval sft.eval.jsonl --cache sft.tok --out lora_v1
```

**Bước 3 — chạy inference với adapter:**

```bash
python scripts/infer.py --split train --max-samples 100 --adapter lora_v1 --output ft_check.jsonl
python scripts/score.py ft_check.jsonl --split train
```

bf16 thì adapter được **merge vào trọng số** → còn lại một model Qwen3 thường: giữ nguyên số tham số
(luật <6B không bị ảnh hưởng) và phần tái dùng KV-cache prefix vẫn đúng. 4-bit thì giữ adapter rời,
vì merge vào trọng số 4-bit sẽ lượng tử hoá cả phần delta.

### Loss chỉ tính trên lượt assistant — điểm sống còn

```
sft.train.jsonl: 1233 hội thoại, 51 bị bỏ vì >8192 token; giám sát 96168/6547604 token (1.5%)
sft.eval.jsonl:   243 hội thoại, 18 bị bỏ vì >8192 token; giám sát 18632/1327915 token (1.4%)
```

Prompt trung vị 5306 token, các lượt assistant chỉ vài trăm. **Không mask thì 98.5% gradient đổ vào
học thuộc lại kết quả tool** — thứ model không bao giờ phải sinh ra, vì lúc chạy thật nó đến từ
`ToolEnv` của BTC. `encode()` khôi phục đúng các đoạn assistant bằng cách render prefix tăng dần rồi
diff, và **dừng hẳn nếu template không prefix-stable** thay vì âm thầm train sai.

Ngân sách tín hiệu thật vì thế chỉ **96k token giám sát** (2 epoch ≈ 192k). Trông đợi ở lần này là
**chỉnh hành vi** — gọi đúng tool, đúng dạng đối số, biết dừng — không phải dạy kiến thức mới.

Kiểm tra mask bất cứ lúc nào, không cần GPU, không cần `peft`:

```bash
python scripts/train_sft.py --train sft.train.jsonl --inspect 2
```

Mẫu dài hơn `--max-seq-len` bị **bỏ, không cắt**: cắt là mất phần đuôi, mà đuôi chính là các call
cuối và `DONE` — phần giám sát giá trị nhất của mẫu.

### Vì sao base là Instruct, không phải Thinking

Trajectory của ta không có khối `<think>` nào. Template của `Qwen3-4B-Thinking-2507` mở khối reasoning
**vô điều kiện** và checkpoint được post-train để luôn suy luận, nên SFT trên target không-reasoning
là dạy nó dập đúng hành vi nó được huấn luyện. Template Instruct khớp dữ liệu chính xác. Nó cũng giữ
Kaggle V2 (Thinking, zero-shot) và bản fine-tune là **hai thí nghiệm khác nhau thật sự**.

Đúng luật BTC: chỉ đọc `train` (split duy nhất có nhãn) — gán nhãn `public_test`/`private_test` để
train hay tune là bị cấm; và code huấn luyện nằm trong repo vì BTC yêu cầu nộp kèm trong Docker.

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
  `--prompt-token-budget`.
- ⚠️ **Mặc định này được tối ưu cho tốc độ và nó phá điểm.** Đo trên 100 mẫu `train`: cắt prompt
  về 1800 token khiến router chỉ giữ đủ app cần thiết ở **24% mẫu** (43% mẫu không có app nào đúng)
  → điểm **5.50** so với **22.52** của `local_v1` chạy không cắt prompt. Prompt đầy đủ tốn trung vị
  5882 token (nén còn 4354), nên `context 4096` **bắt buộc** phải cắt. Cấu hình đúng về chất lượng là
  **`--context-size 8192 --prompt-token-budget 6000`** (giữ 100% app). Chi tiết số đo:
  [results/local_v4/README.md](results/local_v4/README.md).
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

**Nhưng bảng trên là bảng tốc độ, không phải bảng chất lượng.** v3 và v4 nhanh nhờ cắt prompt, và
chính việc cắt prompt kéo điểm từ 22.52 xuống 5.50 (xem mục cấu hình ở trên). Tốc độ đo được mà không
chấm điểm là tốc độ của một kết quả sai — **luôn chạy `scripts/score.py` trước khi tin một tối ưu
hoá nào.**

Lưu ý: cấu hình nhanh này (4-bit, prompt cắt, ASR tách pha) **khác** cấu hình nộp bài trên A30
(bf16, inline), nên điểm đo cục bộ không dự đoán chính xác điểm leaderboard. Dùng
`--llm-precision bf16 --asr-phase inline` khi cần đối chiếu đúng.

## So sánh model ASR (đã đo, đừng đo lại)

```bash
# Đo WER thật trên split train (có nhãn audio_text), nhiều chiến lược decode, không tốn lượt nộp
python scripts/asr_bench.py --samples 15 --configs greedy beam5 beam5_norepeat
```

| Chiến lược decode | WER (n=15) | s/file |
|---|---|---|
| **`greedy`** (đang dùng) | 25.23% | **1.9** |
| `beam5` | 24.61% | 22.4 |
| `beam5` + `no_repeat_ngram_size=4` | 26.17% | 21.9 |

Beam search mua 0.62 điểm WER bằng **12 lần thời gian** → loại. Chặn n-gram lặp làm tệ hơn.

**Chuẩn hoá số nói → chữ số** (`src/text_norm.py`, bật mặc định, tắt bằng `--raw-numbers`): không
cải thiện WER (+0.31pp) nhưng đưa **gold numeric argument có mặt trong transcript từ 0% lên 22%** —
và BTC chấm đối số, không chấm từ. Đo end-to-end trên 22 mẫu được route đúng thì **điểm không đổi**
(11.36 cả hai), dù 8/22 mẫu ra `tool_calls` khác: 82% mẫu ở đó mất điểm vì lệch `state_hash` do
nguyên nhân khác, nên phép đo chưa đủ lực để kết luận. Xem `EXPERIMENTS.md`.

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
