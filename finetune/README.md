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

## Chạy thử trên laptop (RTX 2050, 4 GB) — vì sao nó crash, và cách tránh

**Không hạ model, và cũng không hạ được.** LoRA gắn chặt vào hình học base: đổi sang
Qwen2.5-1.5B thì adapter thành vô dụng. Đòn phải xoay là **context**, không phải model.

Thứ làm nổ VRAM **không** phải KV cache mà là **độ dài prompt lúc prefill**. Bản PyTorch
này không có kernel flash-attention, nên đỉnh attention tăng siêu tuyến tính. `fit_tools()`
trong `src/llm_agent.py` đã đo sẵn trên đúng card RTX 2050 4 GB này, 4-bit Qwen3-4B:

| prompt | thời gian | đỉnh VRAM |
|---|---|---|
| 1k token | 5.5 s | **3.05 GiB** |
| 2k token | 28.3 s | **4.11 GiB** ← đã vượt 4 GB |
| 3.9k token | 252.5 s | **7.54 GiB** |

Vì vậy `LLMAgent` mặc định `prompt_token_budget=1800`. Nhưng `infer.py` truyền cờ
`--prompt-token-budget` với mặc định **100000**, tức ghi đè mất con số đó: ngân sách thật
thành `context_size - max_new_tokens` = **7936 token**. Prefill 7936 token trên card 4 GB.

`local_v6` sống sót ở đó là vì nó **tràn sang shared memory** (`local_v5` ghi "bám sát 4096,
tràn shared memory") — đổi lại là 160 s/mẫu. Còn `--vram-fraction 0.85` đặt một trần **cứng**
3.40 GiB, chặn đúng cái tràn đó, nên nó báo OOM ngay thay vì bò.

OOM, chứ không sập máy — `--vram-fraction` làm đúng việc của nó. Nhưng cũng cho thấy kết
luận trước của tôi sai chỗ: tôi tính KV cache và hạ context, trong khi cache ở đây là
**động** (`self.cache = None` rồi mới mọc), nên `--context-size` gần như không đổi gì.
Đòn đúng là `--prompt-token-budget`.

Adapter chỉ là giọt cuối: nhánh LoRA trong `peft/tuners/lora/bnb.py` giữ output của base và
tạm của `lora_A`/`lora_B` cùng lúc trong MLP, nên nó đòi thêm vài trăm MiB ở đúng chỗ vốn đã
sát mép — traceback dừng ở `lora_B(lora_A(dropout(x)))`, xin 256 MiB khi còn 0 byte.

### Nhưng 1400 token là một cái bẫy — đo rồi mới biết

Smoke ở `--prompt-token-budget 1400` chạy xanh (0 failed, 5.8 s/mẫu) nhưng log khai hỏng:

```
train_0001  "nếu mai mưa thì dời..."  → routed to 2/50 tools from apps ['gold_price']
train_0002  "nếu mai mưa thì huỷ..."  → routed to 5/44 tools from apps ['tv']
```

Cả hai mẫu nói về **mưa** và cả hai bị `fit_tools` định tuyến sai app, rồi model gọi
`sys.ask_missing_info` cho `weather.get_weather` — tool **không có trong prompt**. Chạy A/B ở
đây thì cả hai lượt đều không thấy tool cần dùng, cùng rơi vào `KHÔNG ghi`, và phép so đo
được **con số không**.

Đo ngân sách thật cần cho 20 mẫu đầu của `ids50.txt` (thuần CPU, `_prompt_tokens` + catalog):

| | token |
|---|---|
| nén, giữ **mọi** tool — nhỏ nhất | 1,263 |
| nén — trung vị | 3,790 |
| nén — **lớn nhất** (`train_0093`, 60 tool) | **5,152** |

**Để không bị định tuyến app, ngân sách phải ≥ 5152 token.** Không có ngân sách nào vừa dưới
3.40 GiB vừa trên 5152 — tức trên card này, với trần VRAM cứng, **không tồn tại** cấu hình
cho A/B công bằng. Đó là kết luận, không phải tham số cần dò thêm.

### Thiết kế đúng: lặp lại y nguyên `local_v6`, thêm một cờ

`local_v6` đã chạy được 50 mẫu ở ctx 10240, **không** `--vram-fraction` (mặc định 0.90),
**không** `--prompt-token-budget` (nên ngân sách = 9984 > 5152, không mẫu nào bị định tuyến).
Giá phải trả là 160 s/mẫu. Bài nộp của nó **chính là baseline** — không cần chạy lại.

Nên `local_v7` = lệnh của `local_v6` cộng **đúng một cờ**:

```powershell
env\Scripts\python.exe scripts\infer.py --split train --sample-ids (cat results\local_v6\ids50.txt) --transcript-cache train_all.transcripts.jsonl --adapter finetune\lora_V4 --llm-precision 4bit --context-size 10240 --max-seconds 600 --xref-gate --resume --output results\local_v7\lora.jsonl --metrics-output results\local_v7\lora.metrics.jsonl
```

Giữ `--xref-gate` dù `local_v6` kết luận nó đắt: bỏ đi là đổi **hai** biến, và khi đó hiệu số
không quy được cho adapter. `--resume` để dừng giữa đường không mất gì. 50 mẫu × ~160 s ≈
**2.2 giờ**.

Không truyền `--vram-fraction` ở lượt này — `local_v6` không truyền, và thêm vào là đổi thêm
một biến. Nó cũng chính là thứ đã chặn lượt smoke: trần cứng 3.40 GiB cắt đúng đường tràn
sang shared memory mà `local_v6` dựa vào để sống.

Rủi ro đã biết: adapter thêm ~132 MiB tĩnh và ~250 MiB tạm trong MLP. Nếu OOM, thêm
`--prompt-token-budget 5200` — vừa trên mức 5152 nên vẫn không mẫu nào bị định tuyến, mà cắt
được phần schema nguyên bản (đầy đủ 7,051 token ở `train_0093` xuống còn nén 5,152).
**Nếu phải dùng cờ đó thì baseline `local_v6` không còn dùng được**, và phải chạy lại nó với
cùng cờ — hai lượt, 4.4 giờ.

## Chấm, và đọc đúng dòng

```powershell
env\Scripts\python.exe scripts\score.py results\local_v7\lora.jsonl --split train
env\Scripts\python.exe scripts\diag\action_buckets.py results\local_v7\lora.jsonl --baseline results\local_v6\xref_gate.jsonl
```

Dòng phán xét **không** phải điểm F1 mà là:

```
thoát khỏi 'KHÔNG ghi': ?   rơi vào 'KHÔNG ghi': ?
```

Toàn bộ lý do fine-tune là bucket đó — `local_v4` đo được 48% mẫu gold cần ghi mà ta không
ghi. Adapter không kéo được số đó thì 9 giờ không mua được gì, bất kể F1 nhúc nhích thế nào.

Trên 50 mẫu, dịch chuyển 1–2 mẫu là nhiễu. Chỉ tin khi nó đủ lớn để không cần tin.

### Lượt smoke vẫn có giá trị, chỉ không phải giá trị tôi tưởng

Nó chứng minh ba điều: adapter **nạp và gắn** đúng (`Loading LoRA adapter`, và không có dòng
`merged` — đúng, vì 4-bit thì gắn kèm chứ không gộp), pipeline chạy hết không lỗi, và
`--vram-fraction` báo OOM thay vì làm sập máy. Nó **không** chứng minh cấu hình đó đo được gì.

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
