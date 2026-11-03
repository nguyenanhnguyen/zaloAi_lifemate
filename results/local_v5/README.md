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

→ 50 mẫu ≈ **2.1 giờ**; thực tế trung vị 165 giây/mẫu. 151.5 giây/mẫu không phải tốc độ thi được (luật là 30 giây trên A30),
và không cố là: lượt này đo **chất lượng quyết định**, đúng như `local_v1` từng làm.

## Kết quả

**Điểm 0.0390 — leaderboard đọc là 3.90.** 1/50 mẫu hoàn hảo, 47/50 ăn 0, và cả 47 đều vì
**`state_hash` lệch** — lệch hash thì điểm là 0 bất kể gọi đúng lệnh nào.

| bucket | local_v5 (50 mẫu, **không cắt**) | local_v4 (100 mẫu, **cắt 1800 token**) |
|---|---|---|
| `đúng hẳn` | **2%** | 0% |
| `ghi thiếu` | 16% | 1% |
| `ghi sai` | **58%** | 36% |
| `KHÔNG ghi` | **16%** | 48% |
| `ghi thừa` | 6% | 3% |
| `dừng đúng` | 2% | 12% |

Đây là phép đo đáng giá nhất của lượt này, và nó **không phải thứ tôi đoán trước**. Cho đủ
catalogue thì `KHÔNG ghi` sụp từ 48% xuống 16%: model chịu hành động. Nhưng `ghi sai` phình từ
36% lên 58%. Tức là **cắt prompt không tạo ra lỗi "không ghi" — nó chỉ làm model tê liệt**, còn
cái dốt nằm bên dưới thì vẫn nguyên. Fine-tune phải nhắm vào `ghi sai`, không phải `KHÔNG ghi`.

## Vì sao `ghi sai` — đã truy ra, không phải phỏng đoán

Đo trên chính 50 mẫu này:

| | |
|---|---|
| Lệnh có tham số `x-ref` | 48 |
| ...thiếu một lệnh **đọc** đi trước | **0 (0%)** |
| Lệnh `x-ref` **đúng tool** | 31 |
| ...nhưng **sai ID hoàn toàn** | **11 (35%)** |
| Lệnh `x-ref` **sai cả tool** | 20 |

`0%` ở dòng thứ hai giết một hướng đi: bản kế hoạch phụ lục trong `docs/context.txt` đề xuất
luật "read-before-write cho `x-ref`", nhưng model **đã luôn đọc trước khi ghi**. Luật đó sẽ
không bắt trúng một lệnh nào.

Lỗi thật là model **bịa ra định dạng ID**:

| ta gọi | ID thật |
|---|---|
| `cont_002` | `c_04` |
| `contact_01` | `c_03` |
| `ct_001` | `c_01` |
| `bill_001` | `bill_01` |
| `light_bedroom` | `light_bedroom2` |

Hai điều khiến chuyện này sửa được bằng code, không cần GPU:

1. `SYSTEM_PROMPT` **đã** ghi "Lấy ID từ kết quả công cụ... không tự tạo ID". Nhắc bằng lời
   không đủ.
2. `env.validate()` **không** kiểm tra `x-ref` có tồn tại. Submission lấy từ
   `env.get_tool_calls()`, nên `cont_002` đã **chạy thật**, chứ không phải bị chặn rồi lọt vào
   file.

→ Màng chắn `x-ref` (commit `548ca96`), A/B ở `results/local_v6/`.

## File

| | |
|---|---|
| `ids50.txt` | 50 id đã chạy; 50 dòng đầu của `sft.eval.jsonl` |
| `probe3.*` | phép thử định cỡ |
| `baseline.jsonl` + `.metrics.jsonl` | lượt không adapter — **3.90** |

Lượt có adapter sẽ nằm ở folder riêng khi `lora_V3` về, không ghi đè vào đây.
