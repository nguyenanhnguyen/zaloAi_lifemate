# local_v6 — A/B màng chắn `x-ref`

Đổi **đúng một biến** so với [`local_v5`](../local_v5/README.md): bật màng chắn `x-ref` trong
`src/pipeline.py` (commit `548ca96`). Cùng máy, cùng 50 mẫu held-out, cùng transcript cache,
cùng model 4-bit, cùng context 10240, cùng `--max-seconds 600`, **không** adapter.

Mục đích: tách phần `ghi sai` mà **code** sửa được ra khỏi phần phải chờ fine-tune.

## Màng chắn làm gì

Thu mọi giá trị vô hướng từ `completed_history` và từ **mọi** kết quả `env.call` vào một tập.
Lệnh nào có tham số đánh cờ `x-ref` mang giá trị **ngoài** tập đó thì bị từ chối **trước khi
chạy**, kèm danh sách ID thật đã thấy để model chọn lại.

Cần nó vì `env.validate()` không kiểm tra `x-ref` có tồn tại, và `SYSTEM_PROMPT` đã dặn
"không tự tạo ID" mà model vẫn bịa — xem bảng ID trong README của `local_v5`.

## Lỗ đã biết trước khi chạy

Nếu giá trị sai nhưng **có** xuất hiện ở đâu đó thì màng chắn không bắt được. `train_0011` nhồi
số điện thoại `0912345678` vào `contact_id`, và số đó nằm trong kết quả `contacts.search`, nên
nó lọt. Màng chắn chỉ bắt lớp lỗi **bịa định dạng** (`cont_002`, `bill_001`, `ct_001`).

Vậy nên con số cần đọc không phải "sửa hết 11 ca" mà là **bắt được bao nhiêu trong 11**.

## Rủi ro đã xử lý

Lỗi `x-ref` **không** kích hoạt quy tắc "lặp lại lệnh sai thì dừng". Nếu dừng, mẫu rơi vào
bucket `KHÔNG ghi` — đúng thứ ta đang cố thoát ra. `max_steps` 8 đã chặn vòng lặp.

## Kết quả

*chờ chạy*

| | local_v5 (tắt) | local_v6 (bật) |
|---|---|---|
| Điểm | **3.90** | — |
| `đúng hẳn` | 2% | — |
| `ghi sai` | 58% | — |
| `KHÔNG ghi` | 16% | — |
| `dừng đúng` | 2% | — |
| Lệnh bị màng chắn từ chối | 0 | — |

Con số phán xét: `ghi sai` phải **giảm** mà `KHÔNG ghi` **không tăng** bù lại. Nếu lệnh chỉ
chuyển từ `ghi sai` sang `KHÔNG ghi` thì màng chắn vô ích và phải tắt.

## Lệnh chạy

```
env\Scripts\python.exe scripts\infer.py --split train ^
  --sample-ids (cat results\local_v6\ids50.txt) ^
  --transcript-cache train_all.transcripts.jsonl ^
  --llm-precision 4bit --context-size 10240 --max-seconds 600 ^
  --output results\local_v6\xref_gate.jsonl ^
  --metrics-output results\local_v6\xref_gate.metrics.jsonl
```

Chấm và so bucket:

```
env\Scripts\python.exe scripts\score.py results\local_v6\xref_gate.jsonl --split train
env\Scripts\python.exe scripts\diag\action_buckets.py results\local_v6\xref_gate.jsonl ^
  --baseline results\local_v5\baseline.jsonl
```

Tắt màng chắn để quay lại hành vi cũ: `--no-xref-gate`.
