# local_v8 — `sys.ask_missing_info` là cửa một chiều, và mở nó ra chưa ăn điểm

Tìm ra chỗ rò lớn nhất của pipeline, sửa, đo — và **kết luận là chưa dùng được**, vì giá thời
gian ăn hết phần lợi. Ghi lại để không ai đi lại đường này mà tưởng là mới.

## Chỗ rò: đo trước, không đoán

So `stop_reason` của `local_v7` với nhãn gold (tool `kind == "write"`):

| | base | lora_V4 |
|---|---|---|
| mẫu dừng vì `asked_missing` | 23 | 31 |
| trong đó **gold CẦN ghi** → hỏi lại là **sai** | **21 (91%)** | **28 (90%)** |
| gold không cần ghi → hỏi lại có lý | 2 | 3 |

**Base sai 91%.** 21/50 = 42% toàn bộ mẫu mất chỉ vì model hỏi lại rồi dừng trong khi gold
hành động. Đây cùng là phần lớn bucket `KHÔNG ghi` (58%) — một hiện tượng, không phải hai.

Nó cũng giải thích adapter `lora_V4`: đẩy thêm 13 mẫu vào `asked_missing`, **12 trong đó gold
cần ghi**; thoát ra 5, cả 5 cũng cần ghi. Net hỏng thêm 7. Cái "nhanh gấp đôi" của adapter ở
`local_v7` chính là 13 mẫu bỏ cuộc sớm.

Nguyên nhân là một dòng: `src/pipeline.py` `break` ngay khi `sys.ask_missing_info` chạy thành
công. Một lần hỏi non là mất cả mẫu, dù `max_steps` còn 7 lượt chưa dùng.

## Bản sửa: `--ask-retries N`

Lần hỏi đầu bị đẩy lại kèm lời nhắc dùng công cụ đọc, rồi mẫu được tiếp tục. Mặc định **0**
= hành vi cũ, nên mọi kết quả cũ tái tạo được.

Giá đã biết trước: lệnh `ask` thừa nằm trong call log làm giảm precision (`P = M/(M+U)`).
Nhưng `state_hash` là all-or-nothing nên đổi precision lấy cơ hội ghi đúng là đổi có lợi —
*nếu* nó ghi đúng.

## Kết quả, 50 mẫu, cùng cấu hình `local_v7` (ctx 10240, budget 2000, `--xref-gate`)

Ở `--max-seconds 600` (chạy chất lượng):

| | ask 0 | ask 1 |
|---|---|---|
| điểm | 2.57 | 2.50 |
| `KHÔNG ghi` | 29 | **26** |
| `ghi sai` | 14 | **17** |
| **thoát `KHÔNG ghi`** | — | **3, rơi vào 0** |
| `asked_missing` | 23 | **5** |
| giây/mẫu trung vị | 20.7 | **30.0** |
| **quá 30 giây** | 14 | **25** |
| failed | 3 | 7 |

Ở `--max-seconds 30` (**trần thật của luật**):

| | ask 0 | ask 1 |
|---|---|---|
| điểm | **2.57** | 2.50 |
| `time_limit` | 9 | **23 (46%)** |
| giây/mẫu trung vị | 15.8 | 29.0 |

## Kết luận: cơ chế đúng, bản sửa chưa dùng được **ở đây**

Dịch chuyển bucket là một chiều và sạch — thoát 3, rơi 0 — nên cửa một chiều có thật và
đóng nó lại đang làm mất mẫu. `asked_missing` 23 → 5 cho thấy bản sửa hoạt động đúng ý.

Nhưng cả 3 mẫu thoát ra đều rơi vào `ghi sai`, và dưới trần 30 giây thì **23/50 mẫu chạm
`time_limit`** — mỗi mẫu đó 0 điểm theo luật. Đổi "bỏ cuộc sớm" thành "hết thời gian" không
ăn điểm nào.

**Nên để `--ask-retries 0` cho bài nộp, và đo lại trên A30.** Hai lý do con số này không
chuyển sang hardware chấm:

1. **Thời gian mỗi lượt khác hẳn.** Laptop chạy nf4 trên card 4 GB, prefill 2000 token mất
   hàng giây và không ngắt được giữa đường (`StopAtDeadline` là `StoppingCriteria`, nó chỉ
   cắt lúc sinh token, không cắt được prefill). A30 bf16 nhanh hơn nhiều bậc, nên 4 lượt ở
   đó có thể vẫn vừa 30 giây trong khi ở đây thì không.
2. **Ngân sách 2000 token.** Model chỉ thấy 11/60 tool ở mẫu nặng nhất, nên "được phép hành
   động" chỉ dẫn tới hành động bằng tool sai. 18/21 mẫu không thoát được `KHÔNG ghi` gần như
   chắc chắn vì mù tool, không phải vì chưa chịu hành động.

## Lệnh chạy

```powershell
$ids = (Get-Content results\local_v6\ids50.txt) -split '\s+'
env\Scripts\python.exe scripts\infer.py --split train --sample-ids $ids --transcript-cache train_all.transcripts.jsonl --llm-precision 4bit --context-size 10240 --prompt-token-budget 2000 --max-seconds 30 --xref-gate --ask-retries 1 --output results\local_v8\s30_ask1.jsonl --metrics-output results\local_v8\s30_ask1.metrics.jsonl
env\Scripts\python.exe scripts\diag\action_buckets.py results\local_v8\ask1.jsonl --baseline results\local_v7\base.jsonl
```

Đổi `--ask-retries` giữa `0` và `1`, giữ nguyên mọi cờ khác.

## Việc tiếp theo mà kết quả này chỉ ra

1. **Đo `--ask-retries 1` trên Kaggle/A30** ở ngân sách không cắt tool. Đây là phép đo duy
   nhất còn ý nghĩa; local đã nói hết phần nó biết.
2. **Nếu vẫn chạm `time_limit`**, đòn tiếp theo là giảm *số lượt*, không phải cấm hỏi: gộp
   bước đọc và bước ghi, hoặc cho model thấy kết quả đọc ngay trong prompt đầu.
3. **Bỏ adapter khỏi danh sách ưu tiên.** `local_v7` + phép đo ở đây cho thấy nó làm hỏng
   thêm 7 mẫu theo đúng đường `asked_missing` này. Vẫn nên đo trên A30 một lần, nhưng nó là
   ứng viên yếu nhất trong ba.
