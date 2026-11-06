# local_v7 — adapter `lora_V4` lần đầu được đo, và cái nó thật sự đổi

A/B hai lượt trên **cùng** 50 mẫu held-out, khác **đúng một cờ** `--adapter`. Đây là lần đầu
tiên một adapter của dự án được đo: `lora_V1`/`V2`/`V3` chết không ra trọng số, nên mọi điểm
trước đây đều là model gốc.

| | |
|---|---|
| adapter | `finetune/lora_V4` — r=32, alpha=64, 7 module, 320 mẫu, 9 giờ trên 2×T4 |
| máy | RTX 2050 4 GB, 4-bit, adapter **gắn kèm** (không gộp) |
| mẫu | 50 ID của [`local_v6/ids50.txt`](../local_v6/ids50.txt), split `train` |
| cấu hình | ctx 10240, `--prompt-token-budget 2000`, `--xref-gate`, `--max-seconds 600` |

## Kết quả

| | base | lora |
|---|---|---|
| **điểm** | 2.57 | **4.00** |
| perfect | 1 | 1 |
| state_hash mismatch | 45 | 44 |
| **giây/mẫu trung vị** | 20.7 | **9.1** |
| **quá 30 giây** | **14/50** | **0/50** |
| lượt LLM trung vị | 3.0 | **1.0** |
| failed | 3 | 1 |

### Bucket — và đây là dòng phán xét

```
đúng hẳn    1 →  2
ghi thiếu   2 →  1
ghi sai    14 → 13
KHÔNG ghi  29 → 30
dừng đúng   4 →  4

thoát khỏi 'KHÔNG ghi': 6   rơi vào 'KHÔNG ghi': 7
```

**Net −1 trên 50 mẫu: nhiễu.** 14/50 mẫu đổi bucket, nhưng gần hết là đổi chỗ cho nhau
(`ghi sai → KHÔNG ghi` 7, `KHÔNG ghi → ghi sai` 5). Đúng một mẫu đi lên thật
(`KHÔNG ghi → đúng hẳn`). Điểm 2.57 → 4.00 cũng chỉ là một mẫu.

→ **Ở cấu hình này, adapter không kéo được bucket `KHÔNG ghi`** — tức không làm được cái
việc duy nhất nó được huấn luyện để làm.

## Nhưng nó đổi một thứ khác, và thứ đó là ràng buộc đang chặn cả dự án

`quá 30 giây: 14/50 → 0/50`. Luật vòng Docker là 30 giây/mẫu, quá thì mẫu đó **0 điểm** bất
kể gọi lệnh đúng hay sai. Base để mất 14/50 = 28% số mẫu theo đường đó. Lora để mất 0.

Cơ chế nằm ở `stop_reason`, và nó **không** hoàn toàn là tin tốt:

| | base | lora |
|---|---|---|
| `done` | 24 | **18** |
| `asked_missing` | 23 | **31** |
| `error` | 3 | 1 |

Adapter làm model **hỏi lại rồi dừng** nhiều hơn, thay vì hành động. Lượt LLM trung vị
3 → 1. Nó nhanh gấp đôi vì **bỏ cuộc sớm hơn**, không phải vì làm xong sớm hơn.

Hai cách đọc, và dữ liệu ở đây **không** tách được:

1. Adapter học được phản xạ "thiếu thông tin thì hỏi" từ gold trajectory — đúng hành vi,
   nhưng áp quá rộng.
2. Ở ngân sách 2000 token, model thường **không thấy** tool cần dùng, nên hỏi lại là phản
   ứng *đúng* với một prompt bị cắt. Khi đó adapter chỉ đang phản ứng hợp lý hơn base.

## Vì sao ngân sách chỉ có 2000 token

Đo được, không phải chọn. Trên `train_0093` (mẫu nặng nhất, 60 tool):

| ngân sách | VRAM | tool giữ lại |
|---|---|---|
| 4000 | **OOM** | 33/60 |
| 3200 | **OOM** | 24/60 |
| 2600 | **OOM** | 18/60 |
| **2000** | OK | **11/60** |

Adapter thêm ~132 MiB tĩnh và ~190–250 MiB tạm trong MLP (`lora_B(lora_A(x))` giữ cùng lúc
với output của base). Trên card 4 GB, đó là phần đẩy mọi ngân sách ≥ 2600 qua mép.

Nên **phép đo này là phép đo dưới mức**, và phải đọc như vậy: mẫu nào bị cắt mất app đúng thì
**cả hai** lượt cùng mù và không góp gì vào hiệu số. Nó chưa phủ định adapter — nó chỉ nói
adapter không tự cứu được một prompt đã bị cắt còn 11/60 tool.

## Lệnh chạy

```powershell
$ids = (Get-Content results\local_v6\ids50.txt) -split '\s+'
env\Scripts\python.exe scripts\infer.py --split train --sample-ids $ids --transcript-cache train_all.transcripts.jsonl --llm-precision 4bit --context-size 10240 --prompt-token-budget 2000 --max-seconds 600 --xref-gate --output results\local_v7\base.jsonl --metrics-output results\local_v7\base.metrics.jsonl
env\Scripts\python.exe scripts\infer.py --split train --sample-ids $ids --transcript-cache train_all.transcripts.jsonl --adapter finetune\lora_V4 --llm-precision 4bit --context-size 10240 --prompt-token-budget 2000 --max-seconds 600 --xref-gate --output results\local_v7\lora.jsonl --metrics-output results\local_v7\lora.metrics.jsonl
env\Scripts\python.exe scripts\diag\action_buckets.py results\local_v7\lora.jsonl --baseline results\local_v7\base.jsonl
```

`(cat file)` **không** dùng được: `ids50.txt` nằm trên một dòng nên cả 50 ID thành một
argument và `infer.py` báo `Unknown sample IDs`.

## Việc tiếp theo mà kết quả này chỉ ra

1. **Chạy lại A/B trên Kaggle** ở ngân sách không cắt tool (≥ 9984) và base fp16 + adapter
   **gộp** — đúng điều kiện huấn luyện. Đây mới là phán quyết. Local đã hết đường: không
   ngân sách nào vừa 4 GB mà đủ lớn.
2. **Đo `asked_missing` tăng là tốt hay xấu** bằng cách so với gold: trong 31 mẫu
   `asked_missing` của lora, bao nhiêu mẫu gold thật sự cần hỏi lại? Nếu phần lớn gold yêu
   cầu *hành động*, adapter đang hỏng theo hướng dễ nhận.
3. **Mang `0/50 quá 30 giây` sang vòng Docker.** Dù adapter không ăn điểm F1, riêng việc
   không mẫu nào timeout đã là giá trị thật — base để mất 28% mẫu theo đường đó.
