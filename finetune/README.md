# `lora_V4` — adapter đầu tiên của dự án, và cách dùng nó

Thư mục này lưu **trọng số fine-tune** và là tài liệu duy nhất về chúng. Trọng số bị
`.gitignore` (nặng, tải lại được từ Kaggle); file `.md` này thì không — mất nó là mất cách
dùng.

```
finetune/
  README.md        <- bạn đang đọc; tài liệu DUY NHẤT của thư mục này
  lora_V4/         <- adapter_config.json + adapter_model.safetensors (~264 MB)
```

## Nó là gì

| | |
|---|---|
| base | `Qwen/Qwen3-4B-Instruct-2507` rev `cdbee75f17c01a7cc42f958dc650907174af0554` |
| LoRA | r=32, alpha=64, 7 target module → 66,060,288 tham số huấn luyện (1.62%) |
| dữ liệu | 320 hội thoại từ `sft.train.jsonl`, nhãn gold trajectory |
| cấu hình | fp16, `--device balanced` (2×T4), seq 8192, 80 bước, 1 epoch |
| thời gian | **9 giờ** trên Kaggle 2×T4 |

**Đây là adapter đầu tiên tồn tại.** `lora_V1`/`V2`/`V3` đều chết không ra trọng số nào, nên
mọi điểm của dự án từ 22.52 đến 13.41 đều là **model gốc**. Chưa có con số nào nói adapter
này tốt hay tệ.

Ngân sách học rất mỏng và phải nói trước: chỉ **1.5%** token được tính loss (96,168 trên
6,547,604), vì chỉ phần model *phải sinh ra* mới được giám sát. Nên kỳ vọng đúng là **chỉnh
hành vi** — biết gọi lệnh ghi, biết dừng khi không cần ghi — chứ không phải kiến thức mới.

## Lấy trọng số về

Sau khi notebook train xong, panel Output của nó có `/kaggle/working/lora_V4`. Tải hai file
`adapter_config.json` và `adapter_model.safetensors` vào `finetune/lora_V4/`. **Không** lấy
các `checkpoint-*` — đó là bản giữa đường của `--save-steps 10`.

```powershell
dir finetune\lora_V4
```

Phải thấy đúng hai file, file `.safetensors` khoảng 264 MB. Thiếu file nào thì mọi lệnh dưới
đây sẽ nạp base và bạn sẽ chấm model gốc trong khi tin là đang chấm adapter.

## Chạy thử trên laptop (RTX 2050, 4 GB)

Máy này chạy `local_v6` được ở 4-bit / context 10240 / 50 mẫu held-out, nên **không cần hạ
model**. Và không hạ được: LoRA gắn chặt vào hình học của base — đổi sang Qwen2.5-1.5B thì
adapter vô dụng.

`src/llm_agent.py` tự xử lý phần khó: bf16 thì **gộp** adapter vào trọng số, 4-bit thì
**gắn kèm** (gộp vào trọng số 4-bit sẽ lượng tử hoá cả delta). Không phải truyền cờ gì thêm.

Lệnh `local_v7`, đổi **đúng một biến** so với `local_v6` là `--adapter`:

```powershell
env\Scripts\python.exe scripts\infer.py --split train ^
  --sample-ids (cat results\local_v6\ids50.txt) ^
  --transcript-cache train_all.transcripts.jsonl ^
  --adapter finetune\lora_V4 ^
  --llm-precision 4bit --context-size 10240 --max-seconds 600 --xref-gate ^
  --vram-fraction 0.85 ^
  --output results\local_v7\adapter.jsonl ^
  --metrics-output results\local_v7\adapter.metrics.jsonl
```

Trong log phải có `Loading LoRA adapter finetune\lora_V4`. Không có dòng đó thì đang chạy base.

`--vram-fraction` là **bắt buộc**, đừng bỏ. Card 4 GB tràn VRAM không báo OOM mà làm sập máy;
adapter thêm ~132 MB vào một cấu hình vốn đã sát mép.

## Chấm, và đọc đúng dòng

```powershell
env\Scripts\python.exe scripts\score.py results\local_v7\adapter.jsonl --split train
env\Scripts\python.exe scripts\diag\action_buckets.py results\local_v7\adapter.jsonl ^
  --baseline results\local_v6\xref_gate.jsonl
```

Dòng phán xét **không** phải điểm F1 mà là:

```
thoát khỏi 'KHÔNG ghi': ?   rơi vào 'KHÔNG ghi': ?
```

Toàn bộ lý do fine-tune là bucket đó — `local_v4` đo được 48% mẫu gold cần ghi mà ta không
ghi. Adapter không kéo được số đó thì 9 giờ không mua được gì, bất kể F1 nhúc nhích thế nào.

## Vì sao số local KHÔNG phải phán quyết

Adapter được huấn luyện trên base **fp16**. Local gắn nó lên base **nf4**, tức lệch
train/inference: nó học delta so với trọng số đầy đủ, rồi được áp lên trọng số đã lượng tử.
Local trả lời được "dây đã nối đúng chưa" và "bucket có dịch không"; nó **không** trả lời
được "adapter đáng giữ không".

Phán quyết nằm ở Kaggle, nơi 2×T4 = 29 GiB đủ chứa base fp16 và adapter được **gộp** —
đúng điều kiện đã huấn luyện. Local 50 mẫu / nf4 / ctx 10240, Kaggle 243 mẫu / fp16 /
ctx 12288: hai thí nghiệm khác nhau, cố ý.

## Hai thứ phải kiểm trước khi tin bất kỳ con số nào

1. **`NORMALISE_NUMBERS` / không `--raw-numbers`.** Dataset SFT dựng từ văn bản **đã chuẩn
   hoá** (`'sáu mươi phút'` → `'60 phút'`). Tắt chuẩn hoá là lệch train/inference, và nó lệch
   âm thầm. Mặc định của `infer.py` đã đúng — chỉ đừng thêm `--raw-numbers`.
2. **Mẫu phải là held-out.** 42/50 mẫu preflight từng nằm trong `sft.train.jsonl`. Chấm
   adapter trên dữ liệu nó đã học thuộc thì điểm sẽ tăng và con số đó vô nghĩa. `ids50.txt`
   của `local_v6` đã lọc; nếu bạn tự chọn mẫu mới thì phải giao `sft.train.jsonl` = 0.

## Rủi ro lớn hơn cả điểm số

Luật Docker cho **30 giây/mẫu**, quá thì mẫu đó 0 điểm. Local đang đo 151–165 s/mẫu, và lệnh
trên dùng `--max-seconds 600` nên nó *không* đo được điều này. Adapter làm model nói đúng
hơn, không làm nó nhanh hơn. Xem `--max-seconds` và trường thời gian trong
`*.metrics.jsonl` khi đến lúc siết.
