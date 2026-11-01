# kaggle_v2 — gửi cả catalogue tool, không route *(chưa chạy)*

Notebook: [`notebook/lifemate_kaggle_v2.ipynb`](../../notebook/lifemate_kaggle_v2.ipynb),
`RUN_NAME = 'kaggle_v2'`. Thư mục này trống cho tới khi có kết quả tải về.

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
| LLM | Qwen3-4B-**Instruct** | **Qwen3-4B-Thinking-2507** | Qwen3.5-4B |
| Thinking | tắt | **bật** | bật (512 token) |
| `MAX_NEW_TOKENS` | 192 | **1024** | 1024 |
| LLM trên GPU | 1 GPU | 1 GPU (cuda:0), ASR ở cuda:1 | CPU |
| Tool call văn xuôi | nộp 0 call (42 mẫu) | khôi phục được | — |
| Hỏi thiếu thông tin | **kết thúc mẫu ngay** | **tiếp tục làm việc** | — |
| ID trong write arg | tin model | **chặn nếu không truy được** | — |
| `context_limit` | giết cả run | chỉ hỏng 1 mẫu | — |
| Gate preflight | 1 mẫu `step_limit` là chặn | chặn khi lỗi >20% | — |
| `local_train_score` | `null` | **có số thật, 50 mẫu** | — |
| Chuẩn hoá số nói | không có | **có** (`NORMALISE_NUMBERS`) | — |
| Nạp adapter fine-tune | không có đường | **`ADAPTER_DIR`, merge vào trọng số** | — |

### Model: vì sao đổi sang bản Thinking

`Qwen3-4B-Instruct-2507` **không có chế độ thinking** — Qwen phát hành Instruct và Thinking thành hai
checkpoint riêng, nên không thể bật bằng config. Bản Thinking có **cùng số tham số
4.022.468.096**, cùng kiến trúc `qwen3` (36 lớp, full attention) nên phần tái dùng KV-cache vẫn đúng
và tổng hệ thống vẫn 5.565.773.056 < 6B. Apache-2.0, tạo 2025-08-05 — trước hạn 26/10/2026.

Đây là bước gần nhất tới công thức 22.52 mà **không phải sửa kiến trúc**. Qwen3.5-4B (đúng bản đạt
22.52) dùng linear attention (DeltaNet): `ToolModel` khẳng định `model_type == 'qwen3'` và phần crop
KV-cache chỉ kiểm chứng cho full attention, nên để dành.

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

**Bộ nhớ một card, tính trước:** trọng số fp16 7.49 GiB + KV cache 12288 token 1.69 GiB
(36 lớp × 8 KV head × 128 dim × 2 × 2 byte = 144 KiB/token) = **9.18 GiB trên 14.46 GiB trống**.
Nên `SHARD_LLM_ACROSS_GPUS = False` là mặc định đúng; ASR ở `cuda:1`. Bật sharding lại là **đường
chưa từng chạy được** — `cache.crop()` chưa bao giờ chạy với các lớp trải trên hai card.

**Nếu OOM:** hạ `MAX_NEW_TOKENS` xuống 768 trước (reasoning vẫn đủ chỗ), rồi mới hạ `MAX_CONTEXT`.
Sàn là **9216**: dưới đó `MAX_CONTEXT − MAX_NEW_TOKENS` tụt xuống dưới 7783 và route chạy lại, là
lúc điểm sụp. Đừng hạ `MAX_CONTEXT` về 4096 trong mọi trường hợp.

**Nếu nhiều mẫu báo token limit / `InvalidDecision`:** reasoning dài hơn 1024 token. Nâng
`MAX_NEW_TOKENS` lên 1536 — context 12288 vẫn đủ (12288 − 1536 = 10752 > 7783).

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
