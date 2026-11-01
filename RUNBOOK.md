# RUNBOOK — làm gì, theo thứ tự nào

Trạng thái ngày 01/11/2026. Laptop đã làm xong phần của nó; mọi bước dưới đây chạy trên Kaggle.
Điểm thật hiện có: **22.52** (`local_v1`, cũ) và **13.41** (`kaggle_v1`). Mục tiêu: vượt 22.52.

Lý do thứ tự này: xem [EXPERIMENTS.md](EXPERIMENTS.md) mục "Phân tách nguyên nhân lệch `state_hash`".
Tóm lại — chỗ nghẽn là **model thấy công cụ mà không chịu gọi** (48% mẫu gold cần ghi mà ta không
ghi), nên hai can thiệp đáng giá là bản vá luồng của V2 và fine-tune. Bước 2 và 4 đo riêng từng cái.

---

## Mỗi bước làm Ở ĐÂU

| Bước | Máy | Mở cái gì |
|---|---|---|
| 0 | **laptop** | terminal |
| 1 | Kaggle | *Datasets* → *New Dataset* (làm 2 lần) |
| 2 | Kaggle · **session A** | notebook **`lifemate_kaggle_v2.ipynb`** (import vào) |
| 2b | **laptop** | file `metrics.jsonl` tải về |
| 3 | Kaggle · **session B** | notebook **mới, rỗng** — 1 cell, KHÔNG phải notebook V2 |
| 4 | Kaggle · session A | **cùng** notebook V2, sửa cell 1, chạy lại |
| 5 | Kaggle · session A | **cùng** notebook V2, sửa cell 1, chạy lại |
| 6 | **laptop** | `results/<run>/` + git |

Chỉ có **hai** notebook trong cả quy trình: notebook V2 (bước 2, 4, 5 — cùng một file, khác cấu hình
cell 1) và một notebook rỗng chỉ để huấn luyện (bước 3). Laptop không chạy GPU gì nữa từ bước 2.

Dòng dữ liệu đi qua các bước:

```
laptop: 4 file jsonl ──> Bước 1: 2 Kaggle Dataset
                              │
         ┌────────────────────┴────────────────────┐
         ▼ lifemate-transcripts                    ▼ lifemate-sft
   Bước 2 (notebook V2)                      Bước 3 (notebook rỗng)
   -> điểm mốc zero-shot                     -> lora_v1 trong /kaggle/working
         │                                          │
         │                 Notebook Output          │
         └──────────────> Bước 4 (notebook V2) <────┘
                           -> điểm có adapter
                                 │
                       cấu hình thắng
                                 ▼
                          Bước 5 (notebook V2)
                          -> submission.jsonl
                                 ▼
                    laptop: tải về -> nộp lên Zalo
```

---

## ⚠️ Hai điều phải biết trước khi bấm Run

**1. Chạy dài thì KHÔNG để session tương tác.** Kaggle giết session tương tác khi trình duyệt ngắt
hoặc để không một lúc. Mọi bước từ 2 trở đi đều dài hàng giờ, nên bấm ***Save Version* → *Save & Run
All (Commit)*** để nó chạy ngầm, rồi đóng máy đi ngủ. Session tương tác chỉ dùng cho smoke test.

**2. Quota GPU Kaggle là 30 giờ/tuần.** Kế hoạch này tiêu 15–23 giờ, nên **không có chỗ cho sai sót
lặp lại**. Trước mỗi lần chạy dài, smoke test bằng session tương tác:

```python
PREFLIGHT_SAMPLES = 2        # bước 2 và 4
MAX_SAMPLES = 2              # bước 5
```

Mất ~15 phút, xác nhận chạy hết luồng và in ra điểm. Xong rồi mới đặt lại 50 / `None` và *Save & Run
All*. 15 phút này bảo vệ 3 giờ.

---

## Bước 0 — Kiểm tra file ở laptop *(1 phút)*

```bash
ls -la train_all.transcripts.jsonl public_test.transcripts.jsonl sft.train.jsonl sft.eval.jsonl
```

Phải thấy 4 file: 0.7 MB · 0.3 MB · 24.1 MB · 5.1 MB. Thiếu file nào thì xem
[README.md](README.md) mục "Fine-tune" để sinh lại.

---

## Bước 1 — Tạo 2 Kaggle Dataset *(10 phút)*

**Dataset `lifemate-transcripts`** — Kaggle → *Datasets* → *New Dataset*:

| Upload | |
|---|---|
| `train_all.transcripts.jsonl` | ASR 1545 mẫu train, dùng cho preflight |
| `public_test.transcripts.jsonl` | ASR 634 mẫu public_test, dùng cho bước 5 |

**Dataset `lifemate-sft`** — *New Dataset* lần nữa:

| Upload | |
|---|---|
| `sft.train.jsonl` | 1284 hội thoại huấn luyện |
| `sft.eval.jsonl` | 261 hội thoại giữ lại |
| `scripts/train_sft.py` | script huấn luyện (chạy độc lập, không cần `src/`) |

> Không cần upload `sft.tok.*.jsonl` (92 MB). Kaggle tự tokenize mất ~7 phút — không đáng kể so với
> quota 12 giờ, và bớt được một chỗ dễ nhầm.

---

## Bước 2 — Phiên 1: preflight **zero-shot** *(≈2–3 giờ)*

**Ở đâu:** Kaggle, notebook `lifemate_kaggle_v2.ipynb` import từ laptop. Đây cũng là notebook của
bước 4 và 5 — giữ lại, đừng tạo mới mỗi lần.

Đây là mốc so sánh. Chạy **không** adapter là cố ý: không có mốc này thì sau không biết fine-tune
đóng góp bao nhiêu, lại gộp confound như V1.

1. Kaggle → *Code* → *New Notebook* → *File* → *Import Notebook* → chọn
   `notebook/lifemate_kaggle_v2.ipynb` trên laptop
2. Panel phải → **Accelerator: GPU T4 × 2** · **Internet: On**
3. *Add Data*: dataset thi đấu của BTC **và** `lifemate-transcripts`
4. Sửa **cell 1** (chỉ 5 dòng này):

```python
SPLIT = 'train'
RUN_FULL = False
RUN_NAME = 'v2_base'            # tên riêng cho thí nghiệm này
ADAPTER_DIR = None
TRANSCRIPT_CACHE = Path('/kaggle/input/lifemate-transcripts/train_all.transcripts.jsonl')
```

5. *Run All*.

**Đọc về 3 thứ:**

| Ở đâu | Là gì |
|---|---|
| Log, dòng `PREFLIGHT TRAIN SCORE = ...` | **điểm thật trên 50 mẫu có nhãn** |
| `v2_base/preflight/summary.json` | cùng số đó, dạng máy đọc |
| `v2_base/preflight/metrics.jsonl` → cột `llm_seconds` | **ngân sách thời gian** (xem bước 2b) |

**Điểm nghĩa là gì:**

| | |
|---|---|
| **> 15** | Bản vá luồng có tác dụng. Đi tiếp bước 3. |
| **8–15** | Khá hơn `local_v4` (5.50) nhưng chưa đủ. Vẫn đi tiếp bước 3 — fine-tune là đòn còn lại. |
| **< 8** | Có gì hỏng. Mở `preflight/metrics.jsonl` xem `stop_reason` nào chiếm đa số trước khi đổi bất cứ gì — đừng đổi cấu hình theo cảm giác. |

> **Đã làm xong một nhánh của bảng này.** Thinking từng là thay đổi duy nhất của V2 không có phép đo
> bảo vệ. Kaggle V15 đo rồi: 4/5 mẫu nộp 0 call vì đầu ra bị cắt, 205–301 s/mẫu. V2 giờ chạy
> `Qwen3-4B-Instruct-2507`, `ENABLE_THINKING = False`, `MAX_NEW_TOKENS = 256` — cũng là checkpoint
> mà fine-tune bắt buộc phải dùng. Chi tiết: `results/kaggle_v2/README.md`.

### Bước 2b — kiểm ngân sách thời gian *(2 phút, đừng bỏ)*

**Ở đâu:** laptop. Tải `v2_base/preflight/metrics.jsonl` từ tab *Output* của Kaggle về, rồi:

```bash
python -c "import json,statistics as s; print(s.median([json.loads(l)['llm_seconds'] for l in open('metrics.jsonl',encoding='utf-8')]))"
```

Lấy trung vị `llm_seconds` trong `metrics.jsonl`. Giới hạn **30s/mẫu chỉ được thực thi ở vòng
Docker**, nên bỏ qua bước này là nhận rủi ro: điểm public test đẹp mà vòng Docker timeout, mọi mẫu
về 0.

Quy đổi T4 → A30 theo băng thông: `A30 ≈ T4 × 933/320 ≈ 2.9 lần nhanh hơn`.

> ⚠️ **Đừng tin tỉ lệ 2.9× quá nhiều.** V15 đo được **~9 token/s trên T4** ở context ~9k, thấp hơn
> ~4 lần trần băng thông của chính T4. Nếu nút cổ chai là overhead mỗi token chứ không phải băng
> thông — mà con số đó gợi ý đúng như vậy — thì A30 nhanh hơn ít hơn 2.9×. Đây là rủi ro lớn nhất
> còn lại cho vòng Docker, và nó **chưa được đo**.

| Trung vị `llm_seconds` trên T4 | Dự phóng A30 | |
|---|---|---|
| < 60 s | < 21 s | an toàn |
| 60–87 s | 21–30 s | **sát mép** — hạ `MAX_NEW_TOKENS` về 512 (đúng công thức `local_v1`) |
| > 87 s | > 30 s | **sẽ timeout** — phải hạ `MAX_NEW_TOKENS`, hoặc `ENABLE_THINKING = False` |

Đừng hạ `MAX_CONTEXT` để chữa thời gian. Hạ context là quay về chỗ mất điểm cũ.

---

## Bước 3 — Fine-tune *(≈3–5 giờ, chạy SONG SONG với bước 2 ở session khác)*

**Ở đâu:** Kaggle, một notebook **mới và rỗng** — *không* phải notebook V2. Nó chỉ có đúng một cell.

1. Kaggle → *Code* → *New Notebook* (để nguyên, không import gì)
2. Panel phải → **GPU T4 × 2** · **Internet: On** (cần tải base model 7.5 GiB từ Hugging Face)
3. *Add Data* → `lifemate-sft`
4. Dán **một cell** này rồi *Save Version* → *Save & Run All*:

```python
!pip install -q peft bitsandbytes
!cp /kaggle/input/lifemate-sft/train_sft.py .
!python train_sft.py \
    --train /kaggle/input/lifemate-sft/sft.train.jsonl \
    --eval  /kaggle/input/lifemate-sft/sft.eval.jsonl \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --out /kaggle/working/lora_v1
```

**Đọc về:** `eval_loss` sau mỗi epoch.

| | |
|---|---|
| `eval_loss` giảm cả 2 epoch | tốt. Nếu cuối epoch 2 vẫn còn giảm rõ → chạy lại `--epochs 3`. |
| `eval_loss` tăng ở epoch 2 | overfit. Dùng checkpoint epoch 1 trong `/kaggle/working/lora_v1/checkpoint-*`. |

**Nếu OOM:** thêm `--max-seq-len 6144`. Mất 29% hội thoại (67.4% vừa, so với 96.0%) — đánh đổi thật,
không miễn phí. **Đừng xuống 4096**: chỉ 18.1% hội thoại vừa, tập huấn luyện còn ~230 mẫu.

`lora_v1` nằm trong `/kaggle/working` nên tự thành *notebook output* — bước 4 gắn trực tiếp, không
cần tải về rồi upload lại.

---

## Bước 4 — Phiên 1b: preflight **có adapter** *(≈2–3 giờ)*

**Ở đâu:** Kaggle, **mở lại chính notebook của bước 2**. Không import lại, không tạo mới.

Cùng 50 mẫu, cùng `SEED = 42`, chỉ khác adapter → hiệu số chính là đóng góp của fine-tune.

1. Mở notebook bước 2 → *Add Data* → tab *Your Work* / *Notebook Output* → chọn notebook bước 3
2. Sửa cell 1:

```python
RUN_NAME = 'v2_lora'            # PHẢI khác 'v2_base', nếu không output lẫn vào nhau
ADAPTER_DIR = Path('/kaggle/input/<tên-notebook-bước-3>/lora_v1')
NORMALISE_NUMBERS = True        # giữ True. Dataset SFT dựng từ văn bản ĐÃ chuẩn hoá,
                                # tắt đi là lệch train/inference.
```

3. *Run All*, đọc `PREFLIGHT TRAIN SCORE` lần nữa.

Notebook tự cài `peft` và tự merge adapter vào trọng số — số tham số không đổi nên luật <6B nguyên
vẹn. `manifest.json` ghi lại cả `adapter_config.json` để lần chạy tái lập được.

**So hai số:** nếu `v2_lora` > `v2_base` thì dùng adapter cho bước 5. Nếu thấp hơn, đi bước 5 bằng
cấu hình `v2_base` — và ghi kết quả âm đó vào `EXPERIMENTS.md`, đừng xoá.

---

## Bước 5 — Phiên 2: chạy thật `public_test` *(≈8–12 giờ)*

**Ở đâu:** Kaggle, vẫn **chính notebook của bước 2/4**.

Chỉ làm sau khi bước 2 (hoặc 4) cho điểm chấp nhận được. Lấy **cấu hình thắng** ở trên, rồi sửa:

```python
SPLIT = 'public_test'
RUN_FULL = True
RUN_NAME = 'v2_submit'
TRANSCRIPT_CACHE = Path('/kaggle/input/lifemate-transcripts/public_test.transcripts.jsonl')
```

634 mẫu với prompt dài có thể vượt 12 giờ của Kaggle. Nếu bị cắt: tạo session mới, *Add Data* →
*Notebook Output* của lần vừa rồi, đặt `RESUME_FROM = Path('/kaggle/input/<notebook>/v2_submit')`.
Nó đọc tiếp những mẫu đã xong, không chạy lại.

**Tải về:** `v2_submit/submission.jsonl` → nộp lên Zalo. Kèm theo, để lưu trữ:
`summary.json`, `metrics.jsonl`, `run.log`, `events.jsonl`, `manifest.json`.

---

## Bước 6 — Lưu trữ *(15 phút, đừng bỏ)*

**Ở đâu:** laptop.

Tạo `results/<tên-run>/` cho mỗi lần chạy có kết quả, đặt file tải về vào đó, và viết **một** file
`README.md` ghi: cấu hình, số đo, điểm leaderboard, bài học. Đó là cách duy nhất để lần sau không
lặp lại thí nghiệm đã có kết luận. Xem [results/README.md](results/README.md).

---

## Việc còn treo, chưa làm

- **Dockerfile: chưa có file nào.** Hạn 16/11 yêu cầu nộp cả giải pháp dạng Docker gồm code huấn
  luyện + inference + dữ liệu. Chỉ 13.26 GiB trong 25.35 GiB cache là thật sự cần
  (`Qwen3-4B-Instruct` 7.51 + `PhoWhisper-large` 5.75); phần còn lại là thí nghiệm đã bỏ.
  Code nạp model đã đúng offline (`local_files_only=True` khắp nơi).
- Chuẩn hoá số: hiệu quả end-to-end vẫn là **null trung thực** ở n=22, cần đo lại sau khi
  `state_hash` đỡ hỏng. Nó vẫn bật vì fine-tune cần nhất quán, không phải vì đã chứng minh có lợi.
