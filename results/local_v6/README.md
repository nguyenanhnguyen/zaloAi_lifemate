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

## Kết quả — màng chắn KHÔNG ăn điểm, đã tắt mặc định

| | local_v5 (tắt) | local_v6 (bật) |
|---|---|---|
| **Điểm** | **3.90** | **3.90** |
| `state_hash` lệch | 47/50 | 47/50 |
| `đúng hẳn` | 2% | 2% |
| `ghi thiếu` | 16% | 16% |
| `ghi sai` | 58% | **56%** |
| `KHÔNG ghi` | 16% | **18%** |
| `dừng đúng` | 2% | 2% |
| Lệnh bị từ chối | 0 | **10, trên 10 mẫu** |
| Quỹ đạo đổi | — | 10/50 |

Chuyển dịch bucket: **1 mẫu, `ghi sai` → `KHÔNG ghi`**. Tức là tệ hơn một chút. Theo đúng tiêu
chí đã đặt ra trước khi chạy ("nếu chỉ chuyển bucket thì tắt"), màng chắn giờ mặc định **TẮT**,
bật lại bằng `--xref-gate`.

## Nhưng nó sửa đúng hành vi — chỗ nghẽn chỉ không nằm ở đó

Giữ lại code vì phép đo cho thấy cơ chế **chạy đúng ý định**:

```
train_0283 step 1: ewallet.get_balance {}
train_0283 rejected: sms.send  ID không tồn tại: 12345. Bạn đang TỰ TẠO ID...
train_0283 step 3: contacts.search {'query': 'mẹ'}
train_0283 step 4: sms.send {'contact_id': 'c_01', ...}     <- ID THẬT
```

Model bị chặn, đi đọc, rồi dùng ID thật. `train_0214` cũng vậy. Đây là bằng chứng trực tiếp
rằng model **biết** phải làm gì khi bị buộc, và việc bịa ID chỉ là đường tắt nó chọn khi không
ai chặn.

Vậy vì sao điểm không nhúc nhích: **ID đúng không phải điều kiện còn thiếu.** 47/50 mẫu lệch
`state_hash` vì chuyện khác — chọn sai tool (20/51 lệnh `x-ref` sai cả tool), sai tham số không
phải ID, thiếu hẳn một tác vụ. Sửa ID trong một lệnh của một mẫu vẫn lệch hash thì vẫn 0 điểm.

## Giá phải trả

Mỗi lần chặn tốn một lượt LLM. Trong 10 mẫu bị bắn trúng: **2 chạm `time_limit`, 2 chạm
`step_limit`** (`train_0214` dùng hết 8 lượt). Ở ngân sách 30 giây của vòng Docker, giá này
đắt hơn ở đây nhiều.

→ Bật lại khi `ghi sai` đã giảm vì lý do khác, và khi còn dư lượt.

## Lệnh chạy

## Lệnh chạy

```
env\Scripts\python.exe scripts\infer.py --split train ^
  --sample-ids (cat results\local_v6\ids50.txt) ^
  --transcript-cache train_all.transcripts.jsonl ^
  --llm-precision 4bit --context-size 10240 --max-seconds 600 --xref-gate ^
  --output results\local_v6\xref_gate.jsonl ^
  --metrics-output results\local_v6\xref_gate.metrics.jsonl
```

Chấm và so bucket:

```
env\Scripts\python.exe scripts\score.py results\local_v6\xref_gate.jsonl --split train
env\Scripts\python.exe scripts\diag\action_buckets.py results\local_v6\xref_gate.jsonl ^
  --baseline results\local_v5\baseline.jsonl
```

Bỏ `--xref-gate` để quay lại hành vi `local_v5` (mặc định hiện tại).
