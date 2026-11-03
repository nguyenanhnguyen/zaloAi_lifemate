# local_v5 — baseline CÓ NHÃN với catalogue đầy đủ, trên laptop

Không phải bài nộp. Đây là phép đo mà dự án đang **thiếu**: điểm thật, từng mẫu, trên dữ liệu
có nhãn, với cấu hình **không cắt catalogue**.

Vì sao thiếu: `local_v1` không cắt gì và đạt 22.52, nhưng chạy `public_test` không nhãn nên
không chấm lại được mẫu nào. `local_v4` có nhãn nhưng **cắt prompt về 1800 token** và chỉ được
5.50, nên vô dụng để làm mốc. Chưa có lần chạy nào vừa có nhãn vừa không cắt.

Và nó là mốc để phán xét adapter fine-tune: cùng máy, cùng 50 mẫu, cùng transcript, cùng cấu
hình, khác **đúng một biến** là `--adapter`.

## Cấu hình

| | |
|---|---|
| LLM | `Qwen/Qwen3-4B-Instruct-2507` (rev `cdbee75f`), **4-bit NF4** |
| ASR | **không chạy** — đọc `train_all.transcripts.jsonl` qua `--transcript-cache` |
| Context | **10240** |
| Cắt prompt | **không** (`--prompt-token-budget 100000`) |
| `--max-seconds` | 600 — đo chất lượng, không đo tốc độ thi |
| Tập | 50 mẫu đầu của `sft.eval.jsonl`, giao với `sft.train.jsonl` = **0** |
| Máy | RTX 2050, 4095 MiB (bám sát 4096, tràn shared memory) |

Bỏ ASR khỏi đường đi là điều khiến phép đo này khả thi trên 4 GB: toàn bộ VRAM cho LLM, và
transcript giống **chính xác** thứ Kaggle dùng, nên hai bên so được với nhau.

## Số đo từ phép thử 1 mẫu (`probe3.*`)

| | |
|---|---|
| `train_0011` | **151.5 giây** (LLM 151.5, ASR 0.0), 4 lượt, `done` |
| GPU đỉnh | 4096 MiB — bão hoà |

→ 50 mẫu ≈ **2.1 giờ**. 151.5 giây/mẫu không phải tốc độ thi được (luật là 30 giây trên A30),
và không cố là: lượt này đo **chất lượng quyết định**, đúng như `local_v1` từng làm.

## Kết quả

*chờ chạy — mục này điền sau khi có `score.py` và `action_buckets.py`*

| | baseline (không adapter) | có `lora_V3` |
|---|---|---|
| Điểm | — | — |
| `đúng hẳn` | — | — |
| `ghi sai` | — | — |
| `KHÔNG ghi` | — | — |
| `dừng đúng` | — | — |

Mốc để so, từ `local_v4` (100 mẫu, **có cắt prompt**): `đúng hẳn` 0%, `ghi sai` 36%,
`KHÔNG ghi` 48%, `dừng đúng` 12%.

## File

| | |
|---|---|
| `ids50.txt` | 50 id đã chạy; 50 dòng đầu của `sft.eval.jsonl` |
| `probe3.*` | phép thử định cỡ |
| `baseline.jsonl` + `.metrics.jsonl` | lượt không adapter |
| `lora_v3.jsonl` + `.metrics.jsonl` | lượt có adapter |
