# kaggle_v3_fix — `public_test`, 10/11/2026

**Điểm portal: 20.93.** Thấp hơn cả `kaggle_v3` (21.25) và `kaggle_v2` C (**21.64**).

## Ba điểm dữ liệu, một chiều

| lượt | can thiệp | điểm |
|---|---|---|
| `kaggle_v2` C | **không gate nào** | **21.64** |
| `kaggle_v3` | ask-gate + bare-read `'all'` | 21.25 |
| `kaggle_v3_fix` | ask-gate **tắt** + bare-read `'safe'` + ID_HINT + **ID_SNAP** | 20.93 |

Mỗi lần thêm một cơ chế can thiệp vào quỹ đạo, điểm xuống. Cấu hình tốt nhất đã đo được
vẫn là **không can thiệp gì**.

## Lỗi của tôi: đổi ba biến trong một lượt

Lượt này đổi **ba** thứ cùng lúc (ask-gate tắt, bare-read hạ xuống `'safe'`, ID_SNAP bật).
Hai trong ba là đi *về phía* cấu hình 21.64, nên đáng ra điểm phải lên. Nó xuống, tức thứ
thứ ba đủ xấu để lật cả hai — nhưng không lượt nào tách được từng cái ra. Chính điều tôi đã
viết là không nên làm.

## ID_SNAP: bắn 3 mẫu, sai ít nhất 2

Chỉ **21/634** mẫu đổi `state_hash` so với lượt trước. Trong đó 3 call có ID bị đổi:

| mẫu | tool | model muốn | snap sửa thành |
|---|---|---|---|
| `public_0012` | `banking.transfer` | `acc_01` | **`acc_02`** |
| `public_0083` | `banking.transfer` | `acc_01` | **`acc_02`** |
| `public_0415` | `shopping_list.remove` | `item_772` | **`item_06`** |

Nó **chuyển tiền từ tài khoản khác**. Ba mẫu bị xoá điểm = tối đa 0.47 điểm, đúng bậc độ
lớn của khoảng sụt.

**Vì sao giả định sai:** họ `acc_*` chỉ có một thành viên trong `seen_ids` vì ta **chưa đọc
hết**, không phải vì đó là cái model cần. "Ứng viên duy nhất" là phát biểu về *những gì ta
đã thấy*, không phải về *cái đúng*. Phép đo 87.7% họ-một-thành-viên trên gold đã được ghi
là "trần trên" — nhưng tôi vẫn bật mặc định thay vì để `False` chờ A/B.

→ `ID_SNAP = False`. Giữ `ID_HINT` (chỉ đổi chữ trong thông báo lỗi, không viết lại call).

## Đổi khác đo được

| | lượt này | lượt trước |
|---|---|---|
| `state_hash` giống lượt trước | 613/634 | — |
| call | 1667 | 1671 |
| `sys.ask_missing_info` | 79 | 63 |
| mẫu nộp rỗng | 7 | 10 |
| 2 tool rủi ro: trống / có đối số | 53 / 9 | 56 / 0 |

`ask` tăng 63 → 79 là hệ quả trực tiếp của việc tắt ask-gate, như dự đoán.

## Kết luận

Bốn lượt `public_test` nói cùng một điều: **chỗ nghẽn không nằm ở lớp can thiệp.** Mọi gate,
màng chắn, phép sửa đều hoặc trung tính hoặc âm. Việc tiếp theo không phải cơ chế thứ năm —
là **`lora_V5`** (dạy model định dạng ID) hoặc **Docker** (hạn 16/11, chưa build).
