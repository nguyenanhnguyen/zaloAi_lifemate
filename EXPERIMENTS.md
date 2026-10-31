# Lịch sử thử nghiệm — ZAC 2026 LifeMate

Ghi lại từng lần chạy: tên run, mô hình, điểm chính thức và nhận xét. Chỉ thêm vào, không xoá.

---

## local v4 – thử ASR khác hẳn + sửa time leak thật của ASR (2026-10-31)

Mục tiêu người dùng đặt ra: **laptop và Kaggle không được chạy trùng nhau**. Notebook v9 đang chạy
`PhoWhisper-large`, nên lần này laptop phải thử một ASR **khác hẳn** để biết chênh lệch thật, chứ
không lặp lại chính model đã biết.

### Luật trước tiên — loại ngay những ứng viên không đủ điều kiện

Ngân sách tham số: hệ thống <6B, LLM đã chiếm 4.022B → **ASR phải <1.978B**. Mọi checkpoint phải
công khai trước 26/10/2026 và có giấy phép cho phép dùng.

| Ứng viên | Kết luận |
|---|---|
| `Qwen/Qwen3-ASR-1.7B` | **Loại** — tên ghi 1.7B nhưng thực tế **2.349B** → tổng 6.37B > 6B |
| `hynt/Zipformer-30M-RNNT-6000h` | **Loại** — `cc-by-nc-nd-4.0` (NonCommercial + NoDerivatives) |
| `nguyenvulebinh/wav2vec2-large-vi-vlsp2020` | **Loại** — `cc-by-nc-4.0` |
| `nvidia/parakeet-tdt-0.6b-v3` | **Loại** — 25 thứ tiếng châu Âu, **không có tiếng Việt** |
| `mistralai/Voxtral-Mini-4B` | **Loại** — 4B, cộng LLM là 8B |
| `erax-ai/EraX-WoW-Turbo-V1.1` | **Nhận** — MIT, 809M, 2025-03-31, fine-tune tiếng Việt từ whisper-large-v3-turbo |
| `openai/whisper-large-v3-turbo` | **Nhận làm đối chứng** — MIT, 809M, cùng kiến trúc nhưng *không* fine-tune tiếng Việt |

Lý do chọn EraX làm ứng viên chính: cùng encoder 32 lớp / d_model 1280 như PhoWhisper-large, nhưng
**decoder 4 lớp thay vì 32**. Decode tự hồi quy là phần đắt nhất trên CPU, nên về lý thuyết nhanh
hơn nhiều mà vẫn được huấn luyện cho tiếng Việt (8 vùng miền, ~1000h). Cùng API `transformers` →
không thêm dependency nào, không rủi ro khi đóng băng Docker (bài học `autoawq` ở v2).

### Đo WER thật trên split `train` (có nhãn `audio_text`, không tốn lượt nộp)

12 mẫu rải đều 1545 mẫu `train`, cùng audio, cùng cấu hình. WER = word-level edit distance sau khi
bỏ dấu câu và lowercase.

| ASR | Params | WER | CPU s/file | GPU s/file |
|---|---|---|---|---|
| **`vinai/PhoWhisper-large`** | 1.543B | **23.59%** | 24.5 | **3.1** |
| `erax-ai/EraX-WoW-Turbo-V1.1` | 809M | 28.17% | 13.1 | 0.9 |
| `openai/whisper-large-v3-turbo` | 809M | 28.87% | — | 1.0 |

**Kết quả âm — và đây mới là phần có giá trị:** EraX nhanh 1.9x (CPU) nhưng WER tệ hơn **4.6 điểm**.
Model đối chứng trả lời dứt điểm vì sao: vanilla turbo 28.87% so với EraX 28.17% → **việc fine-tune
tiếng Việt của EraX chỉ mua được 0.7 điểm**, gần như không có gì. Thứ thật sự tạo ra khoảng cách là
**độ sâu decoder** (32 lớp của PhoWhisper so với 4 lớp của turbo), không phải dữ liệu huấn luyện.

Hệ quả: **không đổi ASR.** Giữ `PhoWhisper-large`. Đổi sang EraX sẽ tiết kiệm ~20% thời gian để đánh
đổi 4.6 điểm WER — mà WER hỏng là mất thẳng tool call và `state_hash`, tức mất điểm mẫu.

Lưu ý khi đọc bảng: turbo viết số bằng chữ số (`50.000`, `8h`) còn PhoWhisper viết bằng chữ, nên WER
phạt turbo oan một phần (vd. `train_1144`: turbo 58.3% vs Pho 25% phần lớn do định dạng số). Dù trừ
đi phần này, PhoWhisper vẫn dẫn rõ ràng.

### Time leak thật: ASR bị đẩy xuống CPU, không phải model sai

Chỗ này là một sai sót của chính tôi ở v3. Comment trong `src/asr.py` ghi *"ASR is only ~1% of a
sample's time"* — đúng ở thời điểm 1 lượt LLM mất 900s, nhưng sau khi tối ưu LLM thì con số thật là:

```
15 mẫu đã chạy: tổng 893s = ASR 357s (40%) + LLM 535s (60%)
```

**ASR đã thành 40% thời gian mà vẫn bị để trên CPU.** Và chi phí gần như **cố định theo số file**,
không theo độ dài audio (~16s/file), vì Whisper luôn pad mel lên 30s — nên mẫu có 2 lượt history
audio đắt gấp ba mẫu không có.

Cùng một transcript, GPU nhanh hơn CPU **8x** (3.1s vs 24.5s/file, WER y hệt 23.59%). Nhưng trên card
4GB thì ASR fp16 (3.1GB) và LLM 4-bit (2.3GB) không thể cùng nằm trong VRAM — ai tràn ra thì chậm 8x.

**Giải pháp: tách thành hai pha.** Transcribe toàn bộ trước, khi ASR được dùng cả card một mình; giải
phóng trọng số; rồi mới nạp LLM. A30 giữ được cả hai cùng lúc nên không cần việc này — `--asr-phase
inline` giữ đúng đường chạy của BTC/notebook.

- `src/asr.py`: thêm `ASR_MODELS` (repo → revision pin, kèm kiểm tra giấy phép), đổi `PhoWhisperASR`
  → `WhisperASR` (giờ phục vụ 3 checkpoint cùng API), thêm `close()` và `CachedASR`.
- `src/pipeline.py`: tách `audio_paths()` thành một nguồn duy nhất cho cả hai pha.
- `scripts/infer.py`: `--asr-phase {auto,inline,prefetch}`, `--transcript-cache` (ghi dần, chạy lại
  không phải transcribe lại), `--asr-model` nhận nhiều checkpoint và **từ chối repo chưa pin revision**.
- Metrics thêm `asr_revision`, `asr_phase`; `seconds` và `over_30_seconds` vẫn cộng cả thời gian ASR
  của pha trước, để không tự lừa mình là đã đạt mốc 30s.

### Kết quả (đúng 5 mẫu của v3, so trực tiếp)

| Mẫu | v3 tổng / ASR | v4 tổng / ASR |
|---|---|---|
| public_0002 | 40.4 / 17.8 | 35.5 / 5.4 |
| public_0003 | 30.2 / 21.0 | 16.5 / 7.0 |
| public_0004 | 40.0 / 16.4 | 28.1 / 2.9 |
| public_0005 | 47.1 / 16.5 | 35.1 / 3.0 |
| public_0006 | 95.8 / 41.3 | 56.0 / 5.6 |
| **trung bình** | **50.7s** | **34.2s** |

ASR nhanh **4.7x** (22.6s → 4.8s/mẫu), tổng nhanh **1.48x** → **600 mẫu: 8.4h → 5.7h**.
**`tool_calls` giống hệt v3 trên cả 5 mẫu** — thuần tốc độ, không đánh đổi chất lượng gì.

### Còn tồn đọng (đừng lặp lại những gì ở trên)

- **ASR là trần điểm thật, không phải routing.** `public_0003` ra "nếu may mới thì dời chạy vốn sáng
  may mốn hai tiếng nhắn nhắt em nhắn báo giờ mới ra hư nha không mới thì thôi" — routing chọn sai
  app (`ride_hailing`) vì transcript không còn thông tin gì để chọn đúng. WER 23.59% trên `train`
  nghĩa là **cứ 4 từ sai 1 từ**; sửa routing không cứu được. Hướng chưa thử: ghép nhiều ASR
  (PhoWhisper + turbo) rồi chọn/hợp nhất, hoặc đưa cả 2 transcript cho LLM tự chọn — đổi thời gian
  lấy độ chính xác, giờ đã có ngân sách thời gian để làm.
- **`public_0006` trả lời bằng văn xuôi rồi dừng** ("Mưa không xảy ra vào ngày mai...") thay vì thực
  hiện phần việc còn lại. Chưa rõ gold label có yêu cầu hành động không → cần chạy `train` có nhãn.
- **Vẫn chưa chạy `train` để đo F1 thật.** Tất cả kết luận về chất lượng tới giờ là suy đoán từ
  transcript và tool call, chưa đối chiếu nhãn. Đây là việc đáng làm nhất tiếp theo, **không tốn lượt nộp**.
- `public_0009` từng `stop_reason=error` trong lần chạy 10 mẫu — chưa truy nguyên.
- Notebook v9 vẫn mang bug parser của v2 (`_parse_decision` nhận tool call dạng văn xuôi là câu trả
  lời cuối → nộp 0 tool call). Chưa port.

---

## local v3 – tối ưu tốc độ để laptop chạy đủ tập <24h (2026-10-31)

Mục tiêu: biến laptop RTX 2050 4GB thành công cụ thử nghiệm thật — chạy đủ 600 mẫu `public_test`
trong dưới 24 giờ. Điểm khởi đầu: **1027s/mẫu ≈ 171 giờ** (v2 bf16, context 4096).

### Đo để tìm nghẽn thật, không đoán

Phân tích metrics v2 cho thấy **ASR chỉ chiếm 1%** (15-17s), LLM chiếm 99% (688-916s **mỗi lượt**).
Giả thuyết đầu tiên của tôi — "trọng số bf16 8GB tràn khỏi 4GB VRAM nên stream qua PCIe mỗi token" —
**đã sai**: chuyển sang 4-bit (2.3GB, vừa VRAM) chỉ nhanh hơn 2x (437s/lượt), không phải 100x.

Benchmark trực tiếp (4-bit Qwen3-4B, 32 token mới, RTX 2050 4GB) mới chỉ ra nghẽn thật:

| Độ dài prompt | Thời gian | Peak VRAM |
|---|---|---|
| 64 token | 3.0s | 2.52 GiB |
| 1.024 | 5.5s | 3.05 GiB |
| 2.048 | 28.3s | 4.11 GiB |
| **3.900** (đang gửi thật) | **252.5s** | **7.54 GiB** |
| 64 token, 128 token mới | 11.3s (~88ms/token) | 2.54 GiB |

**Kết luận: decode không chậm (88ms/token). Prefill nổ theo độ dài prompt** vì attention vượt VRAM.
Và prompt 3.904 token gần như toàn bộ là schema của 34 tool — đúng thứ `docs/context.txt` đã khuyến
nghị cắt từ đầu ("Dynamic Schema Pruning: 6.000 → 1.200 token"), mà code chỉ cắt khi *vượt* context
nên cứ gửi nguyên 3.904.

Phát hiện phụ: **PyTorch bản Windows này không có Flash Attention** (`Torch was not compiled with
flash attention`), chỉ có mem-efficient và math. Math backend materialize toàn bộ ma trận attention
→ giải thích peak 7.54 GiB.

### Thay đổi

1. **Cắt prompt theo budget token** (`--prompt-token-budget`, mặc định 1800) thay vì chỉ cắt khi
   vượt context. Đây là đòn tăng tốc chính.
2. **Xếp hạng theo cấp ứng dụng bằng n-gram ký tự tiếng Việt**: bản cũ so transcript tiếng Việt với
   tên tool tiếng Anh (`alarm.update`) → trùng từ ≈ 0 → chọn gần như ngẫu nhiên, cắt mất đúng tool
   cần dùng. Bản mới dùng `name_vi`/`description` tiếng Việt có sẵn trong catalog, so bằng 3-gram ký
   tự sau khi bỏ dấu (chịu được lỗi ASR: "bảo thích" vs "báo thức" vẫn trùng nhiều 3-gram), và giữ
   **nguyên cả bộ tool của app** để không mất cặp read+write.
3. **Buộc backend SDPA mem-efficient** (vì bản Windows thiếu Flash) để giảm peak VRAM attention.
4. **Nén schema mạnh hơn**: bỏ đuôi `Returns: ...`, mô tả tool cắt còn 60 ký tự → thêm được cả app
   vào cùng budget. Độ phủ app quan trọng hơn văn mô tả từng tool.
5. **SYSTEM_PROMPT**: bỏ câu "Suy luận ngắn gọn trước khi chọn hành động" (nó *mời* model viết dài,
   gây lỗi `LLM output hit token limit` ở 2/3 mẫu) → thay bằng yêu cầu chỉ xuất tool call hoặc DONE.
6. **Sửa bug parser**: model đôi khi xuất tool call dạng văn xuôi
   `sys.ask_missing_info({"tool_name": ...})`; bản cũ nhận đó là "câu trả lời cuối" → nộp **0 tool
   call** và ăn 0 điểm trong im lặng. Nay khôi phục được, và nếu không parse nổi thì báo lỗi rõ.
7. Thêm cờ `--llm-precision {auto,bf16,4bit}` và `--asr-device {auto,cuda,cpu}` — chọn hiện rõ, có
   log cảnh báo, ghi vào metrics; không hạ chất lượng âm thầm.

### Kết quả đo (5 mẫu `public_test`, RTX 2050 4GB, 4-bit + ASR CPU)

| Mẫu | Tổng | ASR | LLM | Lượt | Kết thúc |
|---|---|---|---|---|---|
| public_0002 | 40s | 18s | 23s | 3 | done (`get_quotes` → `book`) |
| public_0003 | 30s | 21s | 9s | 1 | asked_missing |
| public_0004 | 40s | 16s | 24s | 1 | asked_missing |
| public_0005 | 47s | 16s | 31s | 1 | asked_missing |
| public_0006 | 96s | 41s | 54s | 3 | done |

**Trung bình 51s/mẫu → ~8.4 giờ cho 600 mẫu** (xấu nhất 16h). **0 mẫu lỗi.**
Tăng tốc **20x** so với v2 (1027s → 51s). Đạt mục tiêu <24h.

### Còn tồn đọng (việc cho lần chạy sau, đừng lặp lại những gì đã làm ở trên)

- **Routing vẫn sai app ở một số mẫu**: `public_0003` ("nếu mai mưa thì dời chạy bộ... nhắn báo giờ")
  bị route sang `ride_hailing`. Budget 1800 có mẫu chỉ đủ 1 app → sai 1 lần là mất trắng mẫu đó.
  Hướng chưa thử: route 2 bước (hỏi LLM chọn app trước), hoặc ép giữ tối thiểu 3 app, hoặc nén schema
  sang định dạng gọn hơn JSON Schema.
- **Nhiều mẫu ra `asked_missing`** (3/5) — cần đối chiếu gold xem có đúng là thiếu thông tin thật hay
  model bỏ cuộc quá sớm. Chạy trên split `train` (có nhãn) sẽ trả lời được mà không tốn lượt nộp.
- **Cấu hình local (4-bit + prompt cắt 1800) khác bản nộp A30 (bf16, có thể để prompt rộng hơn)** →
  điểm local không dự đoán chính xác điểm leaderboard. Muốn đối chiếu công bằng thì chạy Kaggle.

---

## local/kaggle unified v2 – ASR đổi sang PhoWhisper-large (2026-10-31, đang thử nghiệm)

### Research — vì sao đổi ASR lần 2 (Qwen3-ASR-0.6B → PhoWhisper-large)

Yêu cầu: mỗi lần chạy (local hay Kaggle) phải là một thử nghiệm thật, ảnh hưởng rõ đến điểm, không
chỉ sửa vài dòng không đáng kể. Đã khảo sát 2 hướng:

**Hướng 1 — AWQ-quantize LLM để chạy nhanh trên máy yếu**: thử `autoawq` + checkpoint
`cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit`. **Bỏ** vì: autoawq chính thức *deprecated*, tác giả nói
rõ "no longer be maintained", bản test cuối của họ là Torch 2.6/Transformers 4.51 (máy ta đang
2.10/5.13 — cách 2 major version), cần cài thêm `datasets` mới import được, cảnh báo thiếu
`triton`/`zstandard`. Rủi ro gãy giữa lúc đóng băng Docker nộp bài (hạn 16/11) là không chấp nhận
được, trong khi lợi ích chỉ là tốc độ **trên máy dev yếu** — A30 thật (24GB) đã đủ nhanh với bf16
theo chính ước tính của `docs/context.txt` (~2.5s/mẫu). Không mang lại điểm số, chỉ mang lại rủi ro.

**Hướng 2 — đổi ASR sang model Vietnamese-specific**: tìm thấy benchmark ngoài
([TSPC, arXiv 2509.05983](https://arxiv.org/pdf/2509.05983)) so `vinai/PhoWhisper-base` (74M, train
riêng cho tiếng Việt) với `Qwen/Qwen3-ASR-0.6B` (782M, đa ngôn ngữ): **PhoWhisper-base WER 14.05 vs
Qwen3-ASR-0.6B WER 22.47** trên cùng test Việt — chênh rất lớn dù PhoWhisper-base nhỏ hơn 10 lần.
Giấy của PhoWhisper tự báo cáo bản `-large` là SOTA trên các benchmark tiếng Việt khi phát hành.

**Quyết định**: đổi ASR sang `vinai/PhoWhisper-large` (1.55B, BSD-3-Clause, phát hành 2023-12,
cập nhật lần cuối 2024-11 — chắc chắn trước hạn 26/10/2026). Tổng hệ thống
1.55B (ASR) + 4.02B (LLM) = **5.57B < 6B**, LLM vẫn 4.02B < 5B — đúng luật. Đây là thay đổi tác động
thật đến WER/F1 cuối, không phải chỉnh sửa nhỏ. Không đổi LLM lần này (giữ Qwen3-4B-Instruct-2507
bf16, đã xác minh hỗ trợ tool-calling qua `apply_chat_template`).

### Thay đổi code
- `src/asr.py`: `Qwen3ASR` → `PhoWhisperASR`, dùng kiến trúc Whisper chuẩn
  (`AutoModelForSpeechSeq2Seq` + `AutoProcessor`, API `generate()`/`batch_decode()` quen thuộc, đã
  dùng ổn định nhiều năm trong `transformers`) thay cho API `Qwen3ASRForConditionalGeneration` rất
  mới và hẹp người dùng. Vẫn giữ batch toàn bộ audio 1 mẫu trong 1 lượt GPU, vẫn giữ auto-fallback
  CPU khi VRAM < 6GiB free (máy dev yếu).
- `src/pipeline.py`, `scripts/infer.py`: đổi type hint/import theo tên class mới.
- `lifemate_kaggle.ipynb`: *(đang cập nhật — xem ghi chú dưới, chưa hoàn tất khi viết dòng này)*.

### Trạng thái thử nghiệm — ĐÃ XÁC NHẬN đúng trên local (public_0002)

So trực tiếp transcript cùng 1 mẫu (`public_0002`, audio: "Đặt xe bốn chỗ thử công ty đi khám răng
chiều nay, đặt ngay bây giờ nhé!"):

| ASR | Transcript | Đúng? |
|---|---|---|
| PhoASR-whisper-small (baseline 22.52, đã xoá) | "Đặt xe bốn chỗ thử công ty đi khám **răng** chiều nay, đặt ngay bây giờ nhé!" | Đúng |
| Qwen3-ASR-0.6B-hf (v2 trước khi sửa, đã xoá) | "đặt xe bốn chỗ ở tại công ty đi khám **gan** chiều nay đặt ngay bây giờ nhé" | **Sai** (răng→gan) |
| **PhoWhisper-large (v2 hiện tại)** | "đặt xe bốn chỗ thử công ty đi khám **răng** chiều nay đặt ngay bây giờ nhé." | **Đúng**, khớp cả câu chữ với baseline |

Xác nhận nghiên cứu đúng: PhoWhisper-large sửa đúng lỗi mà Qwen3-ASR-0.6B mắc, khớp hoàn toàn với
baseline PhoASR cũ — không đánh đổi chất lượng ASR để đổi kiến trúc.

- **Bug phát hiện khi tích hợp**: `src/models.py::model_path()` chỉ nhận file `*.safetensors` làm
  "trọng số hợp lệ"; PhoWhisper-large chỉ phát hành `pytorch_model.bin` (không có safetensors) nên
  bị báo "Weights missing" sai. Đã sửa: nhận cả `*.bin`.
- **Local**: ASR chạy đúng (CPU, VRAM dồn cho LLM). LLM 4-bit NF4 trên RTX 2050 4GB vẫn rất chậm
  (920s cho 1 lượt, giống trước khi đổi ASR) — **không phải do PhoWhisper**, là hạn chế phần cứng đã
  biết của máy dev, không đại diện tốc độ thật trên A30 (xem mục trước).
- **Kaggle**: đã đồng bộ (`lifemate_kaggle.ipynb` v9, `SpeechModel` dùng `AutoModelForSpeechSeq2Seq`
  chuẩn thay Qwen3ASR, cùng model `vinai/PhoWhisper-large` với local)
- **Điểm**: *chờ chạy đủ `public_test` và nộp để có điểm leaderboard đối chiếu với baseline 22.52*

### Quyết định bổ sung (2026-10-31, theo yêu cầu người dùng): không hạ precision để "vừa" máy yếu

Ban đầu đã thêm auto-fallback: LLM xuống 4-bit NF4 (bitsandbytes) và ASR chạy CPU khi VRAM <10GiB/
<6GiB free, để tránh OOM trên RTX 2050 4GB. **Người dùng yêu cầu bỏ** — tổng hệ thống 6B tham số
(bf16 ≈ 11GB) vẫn khả thi trên "4GB VRAM + 15GB shared RAM" nếu chấp nhận chậm; không muốn đổi sang
model/độ chính xác yếu hơn chỉ vì máy dev yếu, muốn một thử nghiệm nghiêm túc khớp đúng cấu hình
sẽ dùng trên A30 (bf16 đầy đủ), khác biệt chỉ nên là **thời gian**, không phải **độ chính xác**.

**Đã gỡ**: `BitsAndBytesConfig` 4-bit trong `LLMAgent`, CPU-fallback theo VRAM trong `PhoWhisperASR`,
`bitsandbytes` khỏi `requirements.txt`. Cả 2 model giờ luôn load bf16/fp16 trên GPU.

**Hệ quả kỹ thuật phát hiện (ban đầu nghi sai)**: tổng 2 model bf16 (~11GB) vượt xa 4GB VRAM vật lý.
Ban đầu nghi cần bật driver NVIDIA "CUDA Sysmem Fallback Policy" — **sai một phần**: Windows/driver
**đã tự tràn sang RAM mà không cần chỉnh gì** (load model 11.36GB thành công nhiều lần, đúng như
người dùng phản biện). Thứ thật sự gây `torch.OutOfMemoryError` là **context quá lớn (8192 token)**
khiến 1 lượt attention của LLM cần cấp phát tức thời 5.93GB — vượt ngưỡng cấp phát khả dụng tại thời
điểm đó, lặp lại y hệt dù đã thêm `PYTORCH_ALLOC_CONF=expandable_segments:True` (biến này **không
được hỗ trợ trên Windows**: log báo rõ "expandable_segments not supported on this platform").

### ĐÃ CHẠY THÀNH CÔNG — bf16 đầy đủ, không giảm chất lượng, không cần đổi driver

Fix thật: giảm `--context-size 4096 --max-new-tokens 192` (đúng bằng config Kaggle T4 trong
`lifemate_kaggle.ipynb`) — **đã đổi thành default** trong `scripts/infer.py`. Kết quả 2 mẫu
`public_test` (2026-10-31, máy RTX 2050 4GB + 32GB RAM):

| Mẫu | Kết quả | Thời gian | Nhận xét |
|---|---|---|---|
| `public_0002` | `done`, 2 lượt LLM — gọi `ride_hailing.book` đúng tham số rồi trả lời tự nhiên | 1846s (~31 phút) | ⚠️ gọi tool ngay lượt 1 không `read` trước; địa chỉ đón/trả suy luận từ context, cần đối chiếu gold khi có điểm |
| `public_0003` | `asked_missing`, 1 lượt — gọi đúng `sys.ask_missing_info(alarm.update, [time,label])` | 706s (~12 phút) | Đúng quy tắc: thiếu param bắt buộc → hỏi, không bịa |

**Không có mẫu nào fail.** Xác nhận: hệ thống <6B tham số, bf16 đầy đủ (không 4-bit), chạy được trên
GPU 4GB + RAM 32GB thật — đúng như người dùng kỳ vọng, chỉ khác thời gian so với A30, không khác độ
chính xác. Đã cập nhật default `--context-size`/`--max-new-tokens` để lần chạy sau không cần truyền
tay flag này nữa.

**Việc KHÔNG cần làm** (đã thử, không có tác dụng hoặc không cần thiết):
- ❌ Bật "CUDA Sysmem Fallback Policy" trong NVIDIA Control Panel — chưa thử nhưng có khả năng không
  cần, vì vấn đề là context quá lớn chứ không phải chính sách tràn bộ nhớ
- ❌ `PYTORCH_ALLOC_CONF=expandable_segments:True` — không được hỗ trợ trên Windows, giữ lại trong
  code (vô hại, có thể hữu ích trên Linux/Kaggle) nhưng không phải fix chính

---

## v1–v4 (local, baseline)
- **Pipeline**: Whisper + Qwen2.5-3B llama.cpp
- **Điểm**: chưa ghi nhận / thử nghiệm ban đầu

---

## local – PhoASR-small + Qwen3.5-4B Q8 (llama.cpp)
- **Script**: `python scripts/infer.py --split public_test --output submission.jsonl`
- **ASR**: `Qualcomm-AI-Research/PhoASR-whisper-small` (rev `da2d9ae`)
- **LLM**: `unsloth/Qwen3.5-4B-GGUF` → `Qwen3.5-4B-Q8_0.gguf` (rev `e87f176`) qua llama-server
- **Điểm chính thức**: **22.52000**
- **Thời gian/mẫu**: 50–471 s (trung bình ~167 s) — *vượt giới hạn 30 s; chỉ để đo chất lượng*
- **VRAM dùng**: 787 MB (llama.cpp chạy phần lớn trên CPU, swap RAM)
- **RAM process tree**: 6.6–8.2 GB
- **stop_reason phổ biến**: `done` (37/50), `step_limit` (3), `asked_missing` (5), `empty_response` (4), `repeated_invalid` (1)
- **Ghi chú**: Điểm cải thiện đáng kể so với local trước. LLM rất chậm vì chạy CPU llama.cpp.

---

## kaggle – `qwen_asr06_qwen4_v7` (chưa hoàn chỉnh)
- **Notebook**: `lifemate_kaggle.ipynb`
- **ASR**: `Qwen/Qwen3-ASR-0.6B-hf` (rev `7f1569a`) — bfloat16 GPU
- **LLM**: `Qwen/Qwen3-4B-Instruct-2507` (rev `cdbee75`) — bfloat16 GPU
- **Điểm chính thức**: *Chưa có* (notebook không chạy hoàn chỉnh lần nào)
- **Lỗi phát hiện**:
  1. `torch.use_deterministic_algorithms(True, warn_only=False)` → ném RuntimeError khi warm-up vì nhiều CUDA op không deterministic
  2. `processor.decode(generated_2D_tensor, ...)` → decode toàn bộ batch như một chuỗi, không trả về list → `zip(paths, texts)` lặp qua ký tự thay vì string
  3. `assert counts == {'asr': 782426112, 'llm': 4022468096}` cứng — fail nếu revision thay đổi nhỏ

---

## kaggle – `qwen_asr06_qwen4_v8` (bản mới, đang chạy)
- **Notebook**: `lifemate_kaggle.ipynb` (cập nhật 2026-10-31)
- **ASR / LLM**: giữ nguyên Qwen3-ASR-0.6B + Qwen3-4B-Instruct-2507
- **Sửa so với v7**:
  - `warn_only=True` → không crash khi warm-up
  - Decode ASR theo từng hàng: `[processor.decode(generated[k], ...) for k in range(batch)]`
  - `assert counts ==` → `logger.warning` (soft check, giới hạn 5B/6B vẫn còn)
  - `'code_hash'` tĩnh → `'pipeline_version': 'v8'` (tránh fingerprint lỗi khi notebook thay đổi)
- **Điểm**: *Chờ kết quả*
- **Kỳ vọng**: Qwen3-4B-Instruct-2507 full precision bfloat16 tốt hơn Q8 quantized; ASR 0.6B mới hơn PhoASR-small

---

## local/kaggle unified – `transformers` GPU, Qwen3-ASR-0.6B + Qwen3-4B-Instruct-2507 (2026-10-31)

### Research — vì sao đổi model/backend

Luật cuộc thi (`docs/context.txt`): hệ thống <6B tham số, LLM <5B, context ≤16384, chạy offline,
**30 giây/mẫu trên 1×A30 24GB + Xeon 16 core + 64GB RAM**, decode tất định (greedy/temperature=0),
chỉ dùng model/license mở phát hành trước 26/10/2026, không fine-tune bằng nhãn public/private test.

**Phát hiện quan trọng**: baseline 22.52 điểm (local, llama.cpp CPU) chạy 50–471 giây/mẫu —
**vượt xa giới hạn 30 giây của BTC**. Điểm 22.52 chỉ phản ánh chất lượng tool-calling khi không bị
giới hạn thời gian; nếu nộp đúng luật, toàn bộ mẫu sẽ bị tính 0 điểm vì time-out. Đây là lỗi nghiêm
trọng hơn bất kỳ lỗi notebook Kaggle — pipeline cũ **không khả thi để nộp bài**.

Nguyên nhân: `llm_agent.py` cũ chạy Qwen3.5-4B **Q8 GGUF qua llama-server trên CPU** (chỉ 787MB
VRAM dùng, phần lớn tính toán rơi vào CPU/RAM) → chậm gấp hàng chục lần so với chạy thẳng trên GPU.

**Xác minh model thay thế** (tra trực tiếp model card HuggingFace):

| Model | Tham số | License | Ngày | Hỗ trợ |
|---|---|---|---|---|
| `Qwen/Qwen3-4B-Instruct-2507` | 4.0B (BF16) | Apache-2.0 | kỹ thuật báo cáo 05/2025 | context 262K native, tool-calling qua `apply_chat_template(tools=...)` |
| `Qwen/Qwen3-ASR-0.6B-hf` | 0.8B (BF16) | Apache-2.0 | bài báo 01/2026, bảng WER 06/2026 | 30 ngôn ngữ gồm tiếng Việt, batch nhiều audio 1 lượt qua `apply_transcription_request`, cần `transformers>=5.13.0` |
| `Qualcomm-AI-Research/PhoASR-whisper-small` (cũ) | 0.2B (F32) | BSD-3-Clause-Clear + Qualcomm RAI | EACL 2026 | fine-tune whisper-small trên 3000h, không có WER tiếng Việt công khai |

**Quyết định**: chuyển cả 2 model sang chạy thẳng qua `transformers` trên GPU (bf16), bỏ hoàn toàn
llama.cpp. Tổng tham số 4.8B (< 6B), LLM 4.0B (< 5B) — đúng luật. Thống nhất code local và Kaggle
(cùng 2 model), xoá sự khác biệt từng gây khó bảo trì.

### Thay đổi code
- `src/asr.py`: `WhisperASR` → `Qwen3ASR`, batch toàn bộ audio của 1 mẫu (request + history) trong
  một lượt GPU duy nhất (giảm số lần gọi model, khớp cách Kaggle notebook đã làm).
- `src/llm_agent.py`: bỏ subprocess llama-server, dùng `AutoModelForCausalLM` + KV-cache reuse giữa
  các lượt tool trong cùng 1 mẫu (copy logic từ Kaggle `ToolModel.decide`, đã kiểm tra tính đúng).
- `src/models.py` (mới): tiện ích tải/định vị checkpoint HF dùng chung cho ASR và LLM.
- `src/pipeline.py`: gọi `llm_agent.reset()` đầu mỗi mẫu (bắt buộc vì giờ cache KV là state có thật
  trên GPU, không như llama-server tự quản lý theo slot); ASR giờ gọi `transcribe_batch` 1 lần.
- `scripts/infer.py`: bỏ `--gpu-layers`, `--thinking-tokens`, `--asr-beams` (không còn áp dụng);
  `--max-seconds` mặc định đổi từ 600 → 30 để khớp luật thi thay vì chế độ "nghiên cứu chất lượng".
- GPU/CPU: không cấu hình gì thêm — `.to("cuda")`; máy yếu VRAM sẽ tự swap sang shared GPU memory
  của Windows (chậm hơn nhưng không cần chỉnh layer CPU/GPU thủ công như llama.cpp cũ).

### Baseline cũ để đối chiếu
File metrics gốc (50 mẫu public_test, llama.cpp CPU, 22.52 điểm) đã được chép lại nguyên văn phía
trên mục *"local – PhoASR-small + Qwen3.5-4B Q8 (llama.cpp)"* — dùng mục đó để so sánh **chất lượng
tool-calling** (không so sánh thời gian, vì bản cũ không đúng luật 30s).

- **Điểm chính thức**: *chờ chạy `scripts/infer.py --split public_test` và nộp để lấy điểm leaderboard*
- **Cách đối chiếu khi có kết quả mới**:
  1. So `stop_reason` theo từng `id` giữa `runs/baseline_llamacpp_qwen3.5-4b_q8/submission.metrics.jsonl`
     (nếu còn giữ bản chép) và `submission.metrics.jsonl` mới — tìm mẫu đổi từ `done`→lỗi hoặc ngược lại.
  2. So `transcript` (ASR) giữa hai bản để xem Qwen3-ASR-0.6B có đọc đúng hơn PhoASR-whisper-small không.
  3. Kiểm `seconds` mới phải <30s mọi mẫu trên GPU thật (A30/laptop RTX) — nếu không, giảm
     `--max-new-tokens`/`--context-size` hoặc bớt `--max-steps` trước khi nộp.
  4. Điểm leaderboard công khai (Zalo AI Challenge portal) là thước đo chính thức duy nhất, log local
     chỉ để debug nguyên nhân tăng/giảm.

---

## Bài học rút ra

| Vấn đề | Nguyên nhân | Cách xử lý |
|--------|-------------|------------|
| `use_deterministic_algorithms` crash | Nhiều kernel CUDA không có bản deterministic | `warn_only=True` hoặc bỏ |
| `processor.decode(2D_tensor)` | API nhận 1D, batch cần lặp | Loop `decode(row)` hoặc `batch_decode` |
| Assert cứng trên số tham số | Revision model có thể thay đổi nhẹ | Chuyển thành warning, chỉ giữ kiểm tra giới hạn tổng |
| llama.cpp CPU quá chậm | Llama.cpp không offload VRAM đủ trên Windows | Dùng transformers full GPU (Kaggle A30/T4) |
| `code_hash` tĩnh trong fingerprint | Hash tính tay, lỗi thời khi sửa notebook | Dùng `pipeline_version` string |
| Điểm local cao nhưng không khớp luật | Chỉ đo chất lượng, quên giới hạn 30s/mẫu của BTC | Luôn đối chiếu `seconds` trong metrics với 30s trước khi tin điểm |
| llama.cpp CPU vs transformers GPU | llama.cpp tiện cho máy yếu nhưng sai hướng tối ưu (BTC chạy A30 GPU) | Nhắm đúng hardware chấm điểm (A30), chạy local bằng GPU + shared VRAM tự nhiên |
