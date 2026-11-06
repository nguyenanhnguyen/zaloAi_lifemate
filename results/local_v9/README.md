# local_v9 — đẩy lại *đúng* lần hỏi non: miễn phí, dịch một chiều, nhưng chưa ăn điểm

Tiếp [`local_v8`](../local_v8/README.md). Ở đó bản sửa đẩy lại **mọi** lần
`sys.ask_missing_info` và trả giá 23/50 mẫu chạm `time_limit`. Ở đây nó chỉ bắn khi model hỏi
mà **chưa gọi lệnh đọc nào** — "hỏi trước khi nhìn".

## Vì sao nhắm vào đúng nhóm đó

Đếm trên bài nộp của `local_v7`, lệnh nào xuất hiện **trước** lần hỏi:

| hỏi lại khi | base | lora_V4 |
|---|---|---|
| **chưa đọc gì** | 7/23 | **24/31** |
| đã đọc rồi | 16/23 | 7/31 |
| trong nhóm "chưa đọc", gold cần ghi | **7 (100%)** | 21 |

Trên base, luật bắn vào đúng 7 mẫu và **cả 7 gold đều cần ghi** — không mẫu nào bị bắn oan.

Và nó là lời giải thích đầy đủ cho `lora_V4`: **adapter dạy model hỏi ngay mà không nhìn**
(24 so với 7 của base). Cái "nhanh gấp đôi" ở `local_v7` chính là chỗ đó.

## Kết quả, 50 mẫu, `--max-seconds 30` (trần thật của luật)

| | base ask0 | base ask1 | lora ask0 | lora ask1 |
|---|---|---|---|---|
| **điểm** | 2.57 | 2.57 | 4.00 | 4.00 |
| `asked_missing` | 23 | **16** | 31 | **15** |
| `done` | 18 | 22 | 18 | **34** |
| `time_limit` | 9 | 10 | 0 | 0 |
| giây/mẫu trung vị | 15.8 | 16.4 | 9.1 | 14.5 |
| **quá 30 giây** | 9 | 10 | 0 | 0 |

Chuyển dịch bucket:

| | thoát `KHÔNG ghi` | rơi vào |
|---|---|---|
| base ask0 → ask1 | 0 | 0 |
| **lora ask0 → ask1** | **5** | **0** |

## Đọc kết quả

**Miễn phí.** Khác hẳn `local_v8`: giây/mẫu gần như không đổi trên base (15.8 → 16.4), quá
30 giây 9 → 10, `time_limit` 9 → 10. Luật nhắm đúng nên nó chỉ động vào 7 mẫu thay vì 23.

**Dịch một chiều.** Trên lora: 5 mẫu thoát `KHÔNG ghi`, **0 rơi vào**. Không có mẫu nào tệ đi.

**Nhưng chưa ăn điểm.** Cả 5 mẫu thoát ra đều vào `ghi sai` — model chịu hành động, nhưng ghi
sai thứ. Điểm không đổi: 2.57 và 4.00. Đúng như phải vậy ở ngân sách 2000 token, nơi model
chỉ thấy 11/60 tool ở mẫu nặng nhất: "được phép hành động" chỉ dẫn tới hành động bằng tool sai.

Trên base thì 0/50 mẫu đổi bucket dù `asked_missing` giảm 7 — 7 mẫu đó hỏi lại lần hai hoặc
hết lượt, vẫn không ghi.

→ **Bật `--ask-retries 1` khi chạy Kaggle.** Nó không tốn thời gian, không làm mẫu nào tệ đi,
và biến "không ghi" thành "có thử" — ở nơi tool đầy đủ, lần thử đó mới có cơ hội đúng.

## Hai lỗi của tôi ghi lại ở đây, vì cả hai đều tạo kết luận sai

**1. Bản sửa chưa hề chạy mà trông như vô dụng.** Lượt `local_v9` đầu tiên cho 0/50 mẫu đổi
bucket, điểm trùng khớp hoàn hảo với baseline. Nguyên nhân: nhánh `elif` đếm `reads_done` bắt
mọi thứ không phải `"write"`, mà `sys.ask_missing_info` có `kind = 'meta'` — nên **chính lệnh
hỏi** được đếm là một lần đọc, `reads_done` thành 1, và điều kiện `reads_done == 0` không bao
giờ đúng. Sửa thành `elif kinds[name] == "read"`. Nếu không đếm số lần đẩy lại thì kết luận
sẽ là "luật nhắm đúng cũng không ăn điểm" — một phán quyết về thứ chưa được thử.

**2. Giả thuyết KV cache của tôi sai.** Tôi đoán `fit_tools` refit mỗi lượt làm vỡ prefix và
phải prefill lại cả prompt. Đo ra thì ngược: lượt 2 dùng lại 1719/1743 token, prefill 24.
Tool giữ ổn định trong một mẫu; chỉ vỡ ở ngoại lệ (`train_0014` lượt 3, tool 9 → 7, reused
1705 → 1189). Thời gian mỗi lượt ~3.5 s là **decode**, không phải prefill. Dòng log
`prompt N tok (M tools): reused R, prefill P` trong `src/llm_agent.py` giữ lại để lần sau
không phải đoán nữa.

## Lệnh chạy

```powershell
$ids = (Get-Content results\local_v6\ids50.txt) -split '\s+'
env\Scripts\python.exe scripts\infer.py --split train --sample-ids $ids --transcript-cache train_all.transcripts.jsonl --adapter finetune\lora_V4 --llm-precision 4bit --context-size 10240 --prompt-token-budget 2000 --max-seconds 30 --xref-gate --ask-retries 1 --output results\local_v9\lora_ask1.jsonl --metrics-output results\local_v9\lora_ask1.metrics.jsonl
env\Scripts\python.exe scripts\diag\action_buckets.py results\local_v9\lora_ask1.jsonl --baseline results\local_v7\lora.jsonl
```

Bỏ `--adapter` để có lượt base. `--ask-retries 0` là mặc định và là hành vi cũ.
