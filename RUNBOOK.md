# RUNBOOK — làm gì, theo thứ tự nào

Trạng thái ngày 01/11/2026. Laptop đã làm xong phần của nó; mọi bước dưới đây chạy trên Kaggle.
Điểm thật hiện có: **22.52** (`local_v1`, cũ) và **13.41** (`kaggle_v1`). Mục tiêu: vượt 22.52.

Lý do thứ tự này: xem [EXPERIMENTS.md](EXPERIMENTS.md) mục "Phân tách nguyên nhân lệch `state_hash`".
Tóm lại — chỗ nghẽn là **model thấy công cụ mà không chịu gọi** (48% mẫu gold cần ghi mà ta không
ghi), nên hai can thiệp đáng giá là bản vá luồng của V2 và fine-tune. Bước 2 và 4 đo riêng từng cái.

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

Đây là mốc so sánh. Chạy **không** adapter là cố ý: không có mốc này thì sau không biết fine-tune
đóng góp bao nhiêu, lại gộp confound như V1.

1. *New Notebook* → *File* → *Import Notebook* → `notebook/lifemate_kaggle_v2.ipynb`
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
| **< 8** | Có gì hỏng. Đổi `ENABLE_THINKING = False`, `RUN_NAME = 'v2_nothink'`, chạy lại. Thinking là thay đổi duy nhất trong V2 không có phép đo nào bảo vệ. |

### Bước 2b — kiểm ngân sách thời gian *(2 phút, đừng bỏ)*

Lấy trung vị `llm_seconds` trong `metrics.jsonl`. Giới hạn **30s/mẫu chỉ được thực thi ở vòng
Docker**, nên bỏ qua bước này là nhận rủi ro: điểm public test đẹp mà vòng Docker timeout, mọi mẫu
về 0.

Quy đổi T4 → A30 theo băng thông: `A30 ≈ T4 × 933/320 ≈ 2.9 lần nhanh hơn`.

| Trung vị `llm_seconds` trên T4 | Dự phóng A30 | |
|---|---|---|
| < 60 s | < 21 s | an toàn |
| 60–87 s | 21–30 s | **sát mép** — hạ `MAX_NEW_TOKENS` về 512 (đúng công thức `local_v1`) |
| > 87 s | > 30 s | **sẽ timeout** — phải hạ `MAX_NEW_TOKENS`, hoặc `ENABLE_THINKING = False` |

Đừng hạ `MAX_CONTEXT` để chữa thời gian. Hạ context là quay về chỗ mất điểm cũ.

---

## Bước 3 — Fine-tune *(≈3–5 giờ, chạy SONG SONG với bước 2 ở session khác)*

1. *New Notebook* → **GPU T4 × 2** · **Internet: On** (cần tải base model 7.5 GiB)
2. *Add Data*: `lifemate-sft`
3. Một cell, chạy:

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

Cùng 50 mẫu, cùng `SEED = 42`, chỉ khác adapter → hiệu số chính là đóng góp của fine-tune.

1. Mở lại notebook bước 2 → *Add Data* → tab *Notebook Output* → chọn notebook bước 3
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
