# local_v10 — bài nộp đầy đủ đầu tiên có adapter, chạy trên laptop 4 GB

Đây là lần đầu dự án có **634/634 mẫu `public_test` chạy qua adapter**. Mọi lượt local trước
(`local_v5`–`local_v9`) đều là 50 mẫu `train` để A/B; lượt này nhằm ra **file nộp được**.

```
results/local_v10/
  README.md                     <- tài liệu DUY NHẤT của thư mục này
  submission.jsonl              <- 634 dòng, file đem nộp
  submission.metrics.jsonl      <- số đo từng mẫu
  run.log / run2.log            <- log lượt đầu (chết vì OOM) và lượt chạy thật
```

## Cấu hình

| | |
|---|---|
| tập | `public_test`, **634/634** mẫu |
| model | `Qwen/Qwen3-4B-Instruct-2507` 4-bit (nf4) |
| adapter | `finetune/lora_V4` **gắn kèm**, không gộp (gộp vào trọng số 4-bit sẽ lượng tử cả delta) |
| ASR | cache sẵn `public_test.transcripts.jsonl`, phủ **1933/1933** audio |
| `--prompt-token-budget` | 2000 |
| `--vram-fraction` | **0.95** → trần 3.80 GiB (xem bên dưới, đây là chỗ phải sửa) |
| `--max-seconds` | 30 — đúng trần luật |
| `--ask-retries` | **1** — kết luận của [`local_v9`](../local_v9/README.md) |
| `--xref-gate` | bật |

## Kết quả

File nộp đã kiểm: 634 dòng, đủ ID, **đúng thứ tự**, không trùng, mọi dòng có `state_hash`,
schema đúng `id` / `tool_calls` / `state_hash`.

| `stop_reason` | mẫu | % |
|---|---|---|
| `done` | 468 | **73.8%** |
| `asked_missing` | 109 | 17.2% |
| `error` (OOM) | **30** | 4.7% |
| `time_limit` | 24 | 3.8% |
| `repeated_invalid_call` | 3 | 0.5% |

| | |
|---|---|
| giây/mẫu (LLM) trung vị | **13.3** — tối đa 32.5 |
| quá 30 giây | **24 (3.8%)** |
| mẫu nộp 0 tool call | **62** |
| tổng tool call | 1144, trung vị 2/mẫu |
| `xref` chặn | 42 lần |

## Không chấm được ở đây, và vì sao

**`public_test` không có gold ở local.** `scripts/score.py` cần nhãn, nên con số điểm duy
nhất của file này sẽ do portal trả về. Mọi thứ trên đây là **số đo vận hành**, không phải
điểm.

Mốc để so khi có điểm: [`kaggle_v1`](../kaggle_v1/README.md) = **13.41** trên cùng 634 mẫu,
cùng pipeline, **model gốc** và **đủ tool**. Ở đó 42 mẫu nộp 0 call + 33 failed. Lượt này
**62** mẫu 0 call + 30 failed — tệ hơn ở khoản 0 call, đúng như phải vậy khi ngân sách prompt
bị ép xuống 2000.

Nên đọc kết quả thế này: nếu nó **vượt** 13.41 *dù* bị cắt tool, đó là tin rất lớn về
adapter. Nếu nó thấp hơn, chưa kết luận được gì về adapter — vì hai lượt khác nhau ở *hai*
biến (adapter, và ngân sách tool).

## Hai cái bẫy tìm ra trong lượt này

### 1. `--prompt-token-budget` KHÔNG chặn tổng prompt

Nó chỉ chặn phần **schema tool**. Tổng prompt còn cộng `SYSTEM_PROMPT`, transcript và
history. Đo được: ở budget 2000, prompt thật là **2335–2986 token**.

Và hạ budget **vô tác dụng** — prompt giữ nguyên 2335 token ở cả 2000, 1800 và 1600:

```
budget 2000 : prompt 2335 tok (7 tools)  OOM
budget 1800 : prompt 2335 tok (7 tools)  OOM
budget 1600 : prompt 2335 tok (7 tools)  OOM
```

Lý do: `fit_tools` định tuyến theo **app**, và nó đã xuống tới một app duy nhất
(`smart_home`, 7 tool). Dưới đó không còn gì để cắt. **2335 là sàn cứng, không phải tham số
dò được.**

### 2. OOM đến từ mép VRAM hẹp, không từ mẫu nặng

Hai mẫu OOM đầu tiên (`public_0015`, `public_0034`) là mẫu có prompt **nhỏ nhất** trong lượt
— 7/57 và 7/42 tool. Và chúng OOM **cả khi chạy riêng**, ở mẫu đầu tiên sau khi nạp model,
nên không có gì tích lại để phân mảnh.

```
3.60 GiB allowed; 3.23 GiB allocated, 243-314 MiB reserved-but-unallocated
Tried to allocate 88.00 MiB
```

Model 4-bit + adapter chiếm **3.23 GiB thường trú** — phần lớn là `lm_head` với vocab 151k
*không* được lượng tử (151k × 2560 × 2 byte ≈ 775 MiB). Dưới trần 0.90 chỉ còn ~370 MiB cho
prefill, và nhánh LoRA trong MLP (`peft/tuners/lora/bnb.py:549`) xin 88 MiB thì hết chỗ.

Cách sửa, đo chứ không đoán: nhấc trần lên **0.95** = 3.80 GiB. Đúng hai mẫu đó: **2 failed
→ 0 failed, 0 over 30 s**.

Nhưng nó chỉ đẩy mép ra chứ không bỏ mép: cả lượt vẫn còn **29 OOM**. `whole_gpu_used_mb`
đỉnh = **4096**, tức card bão hoà. Trần 0.95 trên card 4 GB *cũng* chạy màn hình là vùng rủi
ro — ở đây máy không sập, nhưng đừng nhấc cao hơn.

## Ba lần tôi đoán sai về bộ nhớ trên card này

Ghi lại vì cả ba đều tốn thời gian thật, và đều cùng một kiểu lỗi: đưa giả thuyết trước khi
đọc con số.

1. **KV cache** (`local_v7`): tôi tính 0.1406 MiB/token rồi hạ `--context-size`. Cache là
   **động** (`self.cache = None` rồi mới mọc) nên gần như không đổi gì.
2. **Phân mảnh** (lượt này): tôi thấy 88 MiB lặp lại và đoán phân mảnh tích lại giữa các mẫu.
   Thử `PYTORCH_ALLOC_CONF=expandable_segments:True` — số liệu **trùng khít từng byte**. Và
   hai mẫu fail cả khi chạy riêng, nên không thể là tích lũy.
3. **Hạ ngân sách prompt** (lượt này): prompt không giảm một token nào, vì sàn là một app.

Con số phải đọc trước khi đoán là dòng `allocated` / `allowed` trong chính thông báo OOM.
Nó nói thẳng còn bao nhiêu headroom.

## Giới hạn của file này

**ASR không nằm trong số giây.** `asr_seconds` tổng = 0 vì transcript lấy từ cache. Luật tính
30 giây cho **cả** ASR và LLM. Trung vị 13.3 s ở đây là *chỉ LLM* — cộng ASR thật vào thì số
mẫu quá trần sẽ cao hơn 24. Muốn đo đúng như ban tổ chức thì dùng `--asr-phase inline`.

**4-bit, không phải bf16.** A30 24 GB của vòng Docker chạy bf16 và adapter được **gộp** —
đúng điều kiện đã huấn luyện. Không con số nào ở đây dự đoán được vòng đó.

## Lệnh tái lập

```powershell
$env:PYTHONIOENCODING='utf-8'
env\Scripts\python.exe scripts\infer.py --split public_test --transcript-cache public_test.transcripts.jsonl --adapter finetune\lora_V4 --llm-precision 4bit --context-size 10240 --prompt-token-budget 2000 --vram-fraction 0.95 --max-seconds 30 --xref-gate --ask-retries 1 --resume --output results\local_v10\submission.jsonl --metrics-output results\local_v10\submission.metrics.jsonl
```

Lượt chạy thoát với **exit code 1** vì có mẫu lỗi — đó *không* phải crash. Dòng phán xét là
`Wrote 634 samples ...; 30 failed; 24 over 30 s`.

`--resume` có một màng chắn đáng biết: output cũ phải là **tiền tố** của thứ tự chạy. Tôi xoá
hai dòng lỗi ở *giữa* file để chạy lại chúng, và nó từ chối với `Existing output is not a
prefix of this run`. Đúng. Cách làm là **cắt về tiền tố** (`head -n`), không phải lọc giữa.
