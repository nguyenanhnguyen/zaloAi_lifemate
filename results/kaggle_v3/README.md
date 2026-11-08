# kaggle_v3 — lượt một-shot, `public_test`, 2026-11-08

**Điểm portal: 21.25.** Mốc trước `kaggle_v2` C: **21.64**. Kỷ lục cá nhân: 22.52.

→ **Ba gate của v3 cộng lại là net −0.39.** Không phải "không thay đổi": thay đổi theo
chiều xấu, nhỏ nhưng đo được.

## Cấu hình

`SPLIT=public_test`, `RUN_FULL=True`, `ASK_REQUIRE_READ=True`, `BARE_READ_GATE='all'`,
`STRATIFY_PREFLIGHT=True`, hai dataset, không transcript cache. 14974 s (4 h 10).

## Vận hành

| | |
|---|---|
| completed / split | 634 / 634 |
| failed | 28 (= 15 `step_limit` + 13 `repeated_invalid`, **trùng khít**) |
| stop_reasons | done 545, asked_missing 61, step_limit 15, repeated_invalid 13 |
| mean / p95 / max giây | 21.13 / 39.36 / 86.51 |
| **quá 30 giây** | **96 (15.1%)** — `kaggle_v2` C là 70 (11%) |
| call nộp | 1671 = 2.64/mẫu (gold train 2.29) |
| mẫu nộp rỗng | 10 |

`failed` == `step_limit` + `repeated_invalid` chính xác, nên nhãn `failed` ở đây là **nhãn
của ta**, không phải phán quyết của BTC. Và vì portal chấm `submission.jsonl` chứ không
chạy lại code, **96 mẫu quá 30 giây KHÔNG bị trừ ở con số 21.25**. Chúng là khoản lỗ của
vòng Docker trên A30, chưa phải khoản lỗ hôm nay.

## Hai gate: cơ chế chạy đúng, hiệu quả âm

**ask-gate — chạy đúng, và chi phí đúng như đã dự đoán.**

- `sys.ask_missing_info` ở **vị trí 0: 0 mẫu**. Gate chặn sạch.
- mẫu **chỉ** có `ask`: **0**. Nhưng gold có 60/1545 (3.9%) mẫu mà `ask` *là cả câu trả lời*.
  Ở đó F1 rơi 1.00 → 0.67. Trên 634 mẫu đó là **tối đa −1.3 điểm**, và không có khoản thu
  nào bù lại nhìn thấy được.
- `ask` vẫn còn 63 call (3.8% tổng call) — gate không bỏ nó, chỉ **đẩy lùi** nó: vị trí
  1 ×29, 2 ×18, còn lại rải tới vị trí 7. Mỗi lần đẩy lùi là một lượt nữa trong trần
  `MAX_STEPS=8` và thêm vài giây.

**bare-read gate — 191 call bare trong submission, 56 thuộc hai tool rủi ro.**

| tool | call bare | |
|---|---|---|
| bill_payment.list_bills | 37 | |
| smart_home.get_status | 31 | **rủi ro** (gold bare 86.2%) |
| chat_ott.list_chats | 28 | |
| banking.get_balance | 27 | |
| banking.list_beneficiaries | 26 | |
| calendar.list_events | 25 | **rủi ro** (gold bare 81.9%) |
| food_delivery.list_restaurants | 14 | |
| ewallet.get_balance | 3 | |

Con số này là *kết quả*, không phải *số lần gate bắn* — một phần trong đó model vốn đã gọi
bare. Số lần bắn thật nằm ở `bare_read_strips` trong bảng preflight.

## Kết luận đổi hướng

1. Mức `'all'` không đáng giữ: 56 strip trên hai tool có 14–18% gold **có** đối số, và net
   tổng là âm. Hạ về `'safe'`.
2. ask-gate có **chi phí đo được** (−1.3 tối đa) và **không có khoản thu đo được**. Tắt.
3. Nhưng không nên tiêu thêm lượt 4 giờ để tinh chỉnh ±0.4. Khoảng cách tới 22.52 là chất
   lượng, và 15.1% mẫu quá 30 giây là khoản lỗ *tương lai* của vòng Docker — hạn
   **2026-11-16**, Docker **chưa dựng**. Docker chưa xong thì 21 hay 23 đều thành 0.
