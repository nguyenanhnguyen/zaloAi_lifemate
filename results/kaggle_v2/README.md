# kaggle_v2 — gửi cả catalogue tool, không route *(chưa có điểm)*

Notebook: [`notebook/lifemate_kaggle_v2.ipynb`](../../notebook/lifemate_kaggle_v2.ipynb),
`RUN_NAME = 'kaggle_v2'`. Thư mục này trống cho tới khi có kết quả tải về.

## Lịch sử các lần chạy trên Kaggle

| Ver | Thời gian | Kết cục | Nguyên nhân |
|---|---|---|---|
| 11 | 45 s | `FileNotFoundError` checkpoint | `allow_patterns` thiếu `*.bin`; PhoWhisper-large không có safetensors |
| 12 | 225 s | `NameError: TRANSCRIPTS` | `TRANSCRIPTS` nạp sau khi dựng `config` |
| 13 | 238 s | traceback bị cắt | — (không chẩn đoán được từ log) |
| 14 | 227 s | `OutOfMemoryError` 5.00 GiB | `enable_gqa=True` của transformers là đường math kernel, O(N²) |
| 15 | 1499 s | 4/5 mẫu `LLM token limit reached` | bản **Thinking** tiêu hết 1024 token reasoning mà chưa đóng `</tool_call>` |

Từ V15 trở đi hạ tầng đã thông: model nạp xong, prefill ~9160 token chạy qua, 1 mẫu hoàn tất
đúng (`train_0063`, 1 call, `done`). Những gì còn lại là chất lượng, không phải lỗi hệ thống.

### V15 cho hai con số chưa từng có

**Thông lượng sinh token: ~9 token/s trên T4 ở context ~9k.** Mỗi mẫu lỗi đi qua đúng 2 lượt sinh
1024 token (lần đầu cụt → nhắc lại → cụt → dừng), nên `(273 − ~5 s ASR) / 2048 ≈ 0,13 s/token`.
Bốn mẫu lỗi cho 7,6 / 10,2 / 9,1 / 7,9 token/s — nhất quán.

Hệ quả: **một lượt 1024 token mất ~114 s.** Mốc 30 s của vòng Docker loại hẳn cách này, và kết luận
đó không cần phép quy chiếu T4→A30 nào. (Đáng để ghi: 9 token/s thấp hơn ~4 lần trần băng thông của
chính T4, nên nút cổ chai có lẽ là overhead mỗi token chứ không phải băng thông — nếu đúng vậy thì
A30 sẽ **không** cứu được nhiều như tỉ lệ 2,9× gợi ý. Chưa đo, chưa kết luận.)

## Vì sao có V2

`local_v4` chấm được **5.50** so với **22.52** của `local_v1`, và tìm ra nguyên nhân: cắt prompt để
chạy nhanh khiến router chỉ giữ đủ app nhãn cần ở **24% mẫu**, 43% mẫu không có app nào đúng.
`kaggle_v1` cũng route — `MAX_CONTEXT 4096` trừ output còn **3904 token**, trong khi prompt đã nén
cần trung vị **4553**. V2 bỏ hẳn việc route.

## Khác biệt so với V1

Mục tiêu V2: **tối đa hoá điểm**, lấy đủ bài học của `local_v1` (22.52) chứ không chỉ sửa route.

| | V1 = 13.41 | V2 | local_v1 = 22.52 |
|---|---|---|---|
| `MAX_CONTEXT` | 4096 | **12288** | 12288 |
| Prompt tool | nén + route từng tool | **nguyên catalogue** | nguyên catalogue |
| LLM | Qwen3-4B-Instruct-2507 | Qwen3-4B-Instruct-2507 | Qwen3.5-4B |
| Thinking | tắt | tắt *(đã thử bật, xem dưới)* | bật (512 token) |
| `MAX_NEW_TOKENS` | 192 | **256** | 1024 |
| LLM trên GPU | 1 GPU | 1 GPU (cuda:0), ASR ở cuda:1 | CPU |
| Tool call văn xuôi | nộp 0 call (42 mẫu) | khôi phục được | — |
| Hỏi thiếu thông tin | **kết thúc mẫu ngay** | **tiếp tục làm việc** | — |
| ID trong write arg | tin model | **chặn nếu không truy được** | — |
| `context_limit` | giết cả run | chỉ hỏng 1 mẫu | — |
| Đầu ra bị cắt 2 lần | làm lỗi cả mẫu | **`invalid_output`, giữ call đã làm** | — |
| Cổng dừng run | 3 lỗi liên tiếp, mọi split | 3 trên `train`, **10** trên public_test | — |
| Gate preflight | 1 mẫu `step_limit` là chặn | chặn khi lỗi >20% | — |
| `local_train_score` | `null` | **có số thật, 50 mẫu** | — |
| Chuẩn hoá số nói | không có | **có** (`NORMALISE_NUMBERS`) | — |
| Nạp adapter fine-tune | không có đường | **`ADAPTER_DIR`, merge vào trọng số** | — |

### ⚠️ Model: đã thử bản Thinking và đã bỏ

V2 ban đầu đổi sang `Qwen3-4B-Thinking-2507` với lập luận: cấu hình duy nhất có điểm tốt
(`local_v1`, 22.52) chạy reasoning, còn `kaggle_v1` không reasoning và chỉ được 13.41. Đây là thay
đổi duy nhất của V2 **không có số đo nào đứng sau**. V15 đo, và nó thất bại:

```
4/5 mẫu: InvalidDecision: LLM token limit reached; truncated action rejected
          -> nộp 0 call
205-301 s/mẫu
```

Bản Thinking tiêu hết 1024 token reasoning mà chưa đóng `</tool_call>`. Nâng `MAX_NEW_TOKENS` lên
1536 — đúng lời khuyên tôi từng viết ở cuối file này — làm **cả hai** vấn đề tệ hơn: cụt vẫn cụt mà
mỗi lượt đắt thêm. Lời khuyên đó đã bị xoá.

**Lý do quyết định không phải thời gian, mà là tính nhất quán với fine-tune.**
[`scripts/train_sft.py`](../../scripts/train_sft.py) gắn cứng vào **Instruct-2507** vì quỹ đạo gold
không có khối `<think>` nào. Adapter LoRA học trên trọng số Instruct không thể nạp lên trọng số
Thinking. Fine-tune là hướng bắt buộc của dự án, nên inference phải chạy đúng checkpoint đó. Chạy
Thinking lúc inference trong khi fine-tune Instruct là hai việc không ăn khớp nhau.

**Đánh đổi phải nói thật:** bỏ reasoning là bỏ đúng cái thứ tương quan với điểm 22.52. Nhưng
`local_v1` cho reasoning **512** token trên một model **hybrid** (Qwen3.5-4B) biết suy luận *ngắn*.
`Qwen3-4B-Thinking-2507` là checkpoint chuyên suy luận: template luôn mở `<think>` nên
`ENABLE_THINKING = False` **vô hiệu** với nó, và không có cách nào bảo nó suy luận ngắn. Hai thứ đó
không thay thế được nhau, và cái có sẵn trong tầm tay là cái thứ hai.

Nên V2 vẫn là **một thí nghiệm khác hẳn V1**, chỉ không khác ở chỗ model: context 12288 thay vì
4096, nguyên catalogue thay vì route, cộng hai sửa đổi luồng làm việc dưới đây.

Qwen3.5-4B (đúng bản đạt 22.52) dùng linear attention (DeltaNet): `ToolModel` khẳng định
`model_type == 'qwen3'` và phần crop KV-cache chỉ kiểm chứng cho full attention, nên để dành.

### Hai thay đổi luồng làm việc — đều đo trên 1545 mẫu nhãn `train`

**A. Hỏi thiếu thông tin không còn kết thúc mẫu.**

```
sys.ask_missing_info trong nhãn gold:
  là call DUY NHẤT      :  60 mẫu  (3.9%)
  CÓ MẶT trong nhãn     : 217 mẫu  (14.0%)
  trong đó 157 lần nằm SAU các call khác
```

V1 kết thúc mẫu ngay khi model hỏi — **109 mẫu (17.2%)** bị cắt mất toàn bộ call chưa kịp thực hiện.
Gold nói hành vi đúng là: làm hết phần làm được, hỏi, rồi tiếp tục. V2 ghi nhận việc đã hỏi (tối đa
2 lần) và chạy tiếp; `stop_reason` báo về vẫn là `asked_missing` nên cách tính lỗi không đổi.

**B. ID trong đối số write phải truy được nguồn.**

1277 đối số write của gold là ID phải tra cứu (`device_id`, `contact_id`, `bill_id`...). Kiểm trên
toàn bộ nhãn: trong các mẫu mà gold **không có call read nào**, có 211 đối số dạng ID — và **211/211
(100%) xuất hiện trong history/context của chính mẫu đó**. Nên một ID không khớp kết quả tool nào và
cũng không có trong context là ID bịa; thực thi nó làm sai `state_hash` vĩnh viễn (local_v4: 86% mẫu
lệch hash). V2 từ chối call đó và bảo model đi đọc ID thật. **0 false positive trên 1545 mẫu.**

Phụ: `MAX_STEPS = 8` đã đủ — mẫu gold cần nhiều call nhất là **6**, không mẫu nào cần hơn 8.

**C. Đầu ra bị cắt không còn làm lỗi cả mẫu** *(thêm sau V15)*. Trước đây lần `InvalidDecision` thứ
hai `raise` → `stop_reason='error'` → 3 mẫu liên tiếp giết cả run; V15 chết đúng kiểu đó ở mẫu 5/50.
Giờ trả về `'invalid_output'` và **giữ nguyên các call đã thực hiện**: submission chỉ tốt lên hoặc
giữ nguyên, không bao giờ xấu đi, vì `submission_value(env, …)` đã luôn được ghi trong khối
`finally` kể cả khi mẫu lỗi. `'invalid_output'` vẫn tính là `failed`, nên điểm preflight không bị
làm đẹp và cổng `failure_rate ≤ 20%` vẫn chặn nếu model không chịu ra tool call.

Lời nhắc khi đầu ra chưa hợp lệ cũng được viết lại cho gắt hơn: *"chỉ một khối
`<tool_call>…</tool_call>` duy nhất, không giải thích, không suy luận."*

**D. Cổng 3-lỗi-liên-tiếp chỉ còn gắt trên `train`.** Trên `train` ta đang **đo**, nên dừng sớm là
đúng — V15 dừng ở mẫu 5 và tiết kiệm quota. Trên `public_test` không có nhãn để học thêm điều gì
giữa run, mà chết ở mẫu 400/634 sau 6 giờ là mất trắng phiên đó; giới hạn ở đó là **10**. OOM vẫn
giết run ở mọi split, vì nó nghĩa là kế hoạch bộ nhớ sai.

Đo bằng tokenizer trên 100 mẫu `train`, đúng dạng prompt của notebook:

```
limit = 12288 - 256 = 12032
prompt lượt 1, schema nguyên bản:  trung vị 5945   p95 7666   max 7783   -> vừa
số mẫu phải route:                 0/100
```

## Nạp model fine-tune

```python
ADAPTER_DIR = Path('/kaggle/input/lifemate-lora/lora_v1')
NORMALISE_NUMBERS = True      # BẮT BUỘC khi dùng ADAPTER_DIR
```

Adapter được **merge vào trọng số** → còn lại một model Qwen3 thường: số tham số không đổi (luật
<6B nguyên vẹn) và phần crop KV-cache vẫn đúng. Notebook tự thêm `peft` vào lệnh cài khi
`ADAPTER_DIR` khác `None`. `manifest.json` ghi cả đường dẫn adapter và `adapter_config.json` nên lần
chạy vẫn tái lập được.

⚠️ **`NORMALISE_NUMBERS` không phải tuỳ chọn khi fine-tune.** Dataset SFT dựng từ transcript **đã
chuẩn hoá** ("xe 4 chỗ"), nên model mong đợi đầu vào đã chuẩn hoá. Tắt đi là lệch train/inference.
Và vì vậy normalizer **bị đóng băng**: sửa `src/text_norm.py` thì phải build lại dataset SFT, nếu
không adapter sẽ lệch với thứ nó nhận lúc chạy.

Bản sao normalizer trong notebook là **nguyên văn** `src/text_norm.py` (Kaggle không import được từ
repo). Đã kiểm: khớp tuyệt đối trên 236 transcript thô của `local_v4` (35.2% dòng bị đổi) và trên
1933 transcript `public_test` đã chuẩn hoá (0 dòng đổi, tức idempotent). **Sửa thì sửa cả hai chỗ.**

## Ghép với laptop: ASR chạy sẵn ở nhà

V2 đọc được transcript do laptop sinh ra, nên không tốn GPU Kaggle cho ASR.

```bash
# laptop, chỉ ASR, không nạp LLM
python scripts/infer.py --split public_test --asr-only \
  --transcript-cache public_test.transcripts.jsonl
```

Upload file đó thành Kaggle Dataset rồi đặt trong cell cấu hình:

```python
TRANSCRIPT_CACHE = Path('/kaggle/input/lifemate-transcripts/public_test.transcripts.jsonl')
```

Mẫu nào thiếu trong cache **vẫn được ASR như thường**, nên cache chạy dở cũng dùng được.
`metrics.jsonl` ghi `asr_cached` từng mẫu, `manifest.json` ghi đường dẫn và số dòng của cache.

⚠️ Đặt lại `TRANSCRIPT_CACHE = None` trước khi đóng Docker: BTC mount private test rồi chạy trong
container, không có cache nào ở đó.

## Cách chạy — hai phiên Kaggle

1. **Phiên 1** (mặc định của notebook): `SPLIT='train'`, `RUN_FULL=False`. Chạy preflight 50 mẫu có
   nhãn. Đọc `local_train_score` trong `kaggle_v2/preflight/summary.json` — log cũng in ra dòng
   `PREFLIGHT TRAIN SCORE = ...`. **So với 5.50 của `local_v4`.**
2. **Phiên 2**, chỉ khi điểm chấp nhận được: `SPLIT='public_test'`, `RUN_FULL=True`. 634 mẫu với
   prompt dài có thể vượt 12 giờ của Kaggle → dùng `RESUME_FROM` trỏ vào `kaggle_v2` của phiên trước.

**Bộ nhớ một card** — phép tính này tôi đã viết sai **hai lần**, đây là bản có đo:

| | GiB |
|---|---|
| trọng số fp16 | 7.49 |
| KV cache 12288 token (36 lớp × 8 KV head × 128 dim × 2 × 2 byte = 144 KiB/token) | 1.69 |
| workspace attention lúc prefill, **sau** vá GQA (đo trên Qwen3 thật, N=4096: 1792 MiB) | ~4 |
| **tổng** | **~13.2 trên 14.46 trống** |

Lần đầu tôi viết "một card không chứa nổi" mà không tính. Lần hai tôi viết 9.18 GiB và **bỏ sót
dòng thứ ba** — chính dòng đã làm V14 OOM. Bài học đã ghi vào `EXPERIMENTS.md`: đừng viết một con số
vào tài liệu nếu chưa tính nó.

`SHARD_LLM_ACROSS_GPUS = False` là mặc định đúng, ASR ở `cuda:1`. Bật sharding lại là **đường chưa
từng chạy được** — `cache.crop()` chưa bao giờ chạy với các lớp trải trên hai card — và nó **không**
phải thứ chữa V14.

**Vá GQA (bắt buộc, cell 9).** `transformers` bật `enable_gqa=True` mỗi khi `attention_mask is None`
— đúng ca batch=1 không padding của ta — và comment trong code của nó nói làm vậy để *tránh* math
kernel. Đo trên torch 2.11 thì ngược lại: `enable_gqa` **chính là** đường math kernel. V14 chết vì
`32 × 9160² × 2 byte = 5.00 GiB`. Vá bằng cách buộc nó lặp KV head:

```python
from transformers.integrations import sdpa_attention as _sdpa
_sdpa.use_gqa_in_sdpa = lambda *args, **kwargs: False
```

Đo trên Qwen3-4B thật: chưa vá 1657 → 5638 MiB khi N đi từ 2048 → 4096 (×3,4 ≈ N²); đã vá
902 → 1792 MiB (×1,99, đúng N). Đã kiểm thêm ở pha **decode** (q_len=1, N=9160): `repeat_kv` còn
**nhanh hơn** `enable_gqa` 3× và tốn ít bộ nhớ hơn, nên bản vá đúng ở cả hai pha, không chỉ prefill.

**Nếu OOM:** hạ `MAX_CONTEXT`. Sàn là **8192**: dưới đó `MAX_CONTEXT − MAX_NEW_TOKENS` tụt xuống
dưới 7783 và route chạy lại, là lúc điểm sụp. Đừng hạ `MAX_CONTEXT` về 4096 trong mọi trường hợp.

**Nếu nhiều mẫu báo token limit / `InvalidDecision`:** *không* nâng `MAX_NEW_TOKENS` — V15 cho thấy
đó là lối sai. Nguyên nhân là model đang trả lời dài dòng thay vì ra tool call. Xem
`events.jsonl` → `invalid_decision` và `response` để biết nó viết gì. Từ V15, lần cụt thứ hai trả về
`stop_reason = 'invalid_output'` và giữ các call đã thực hiện, thay vì làm lỗi cả mẫu.

## Cần tải về những file nào

Tab **Output** → `lifemate/kaggle_v2/`, chỉ lấy:

| File | Vì sao |
|---|---|
| `preflight/summary.json` | **điểm thật** (`local_train_score`) |
| `preflight/metrics.jsonl` | WER, `state_match`, `matched_calls` từng mẫu |
| `submission.jsonl` | bài nộp (chỉ có ở phiên 2) |
| `metrics.jsonl` | số đo từng mẫu |
| `summary.json` | tổng hợp |
| `run.log` | log |
| `events.jsonl` | **V1 thiếu file này** — chứa `tool_prompt_routed`, `loose_call_recovered` |

Model nằm ở `/kaggle/temp`, ngoài output, nên không cần lọc như lần V1.
