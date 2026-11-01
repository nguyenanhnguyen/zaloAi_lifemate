# local_v4 — ASR thành pha riêng trên GPU + **lần đầu chấm điểm thật**

Không phải bài nộp. Đây là lần chạy **100 mẫu split `train` (có nhãn)** để lần đầu tiên trong dự án
biết điểm thật của pipeline thay vì suy đoán từ transcript.

**Điểm: 0.0550 → leaderboard scale 5.50.** So với `local_v1` đạt **22.52**, bản này **tệ hơn 4 lần**.
Lần chạy này có giá trị không phải vì điểm, mà vì nó **tìm ra nguyên nhân**.

## Cấu hình

| | |
|---|---|
| ASR | `vinai/PhoWhisper-large` (rev `b9136a44`), **pha riêng trên GPU** (`--asr-phase prefetch`) |
| LLM | `Qwen/Qwen3-4B-Instruct-2507` (rev `cdbee75f`), **4-bit NF4** |
| Prompt | cắt về **1800 token** bằng định tuyến theo app |
| Context / output | **4096** / 256 token |
| Tập | `train`, 100 mẫu đầu, `--max-seconds 3600` |

## Số đo

| | |
|---|---|
| Thời gian/mẫu | trung bình **71.7s**, trung vị 43.5s, max 638.5s |
| Phân bổ | ASR 18.1s (25%) · LLM phần còn lại |
| Vượt 30s | 65/100 |
| Thất bại | 2 mẫu |
| `stop_reason` | done 54, **asked_missing 43**, step_limit 1, error 2 |

Chậm hơn con số 34s/mẫu đo trên `public_test` ở v3/v4 vì mẫu `train` có nhiều audio history và nhiều
lượt LLM hơn. Pha ASR riêng vẫn đúng như kỳ vọng: ASR tụt từ 40-45% xuống 25% tổng thời gian.

## Điểm chi tiết

```
SCORE 0.0550  trên 100 mẫu
  hoàn hảo (1.0):      5
  điểm 0:             94
  lệch state_hash:    86   (mỗi mẫu như vậy là 0 bất kể gọi tool gì)
  gold rỗng:           5
```

## Nguyên nhân: **router prompt bỏ mất chính tool cần dùng**

Đây là phát hiện quan trọng nhất của dự án tới giờ. Đo trực tiếp bằng cách đối chiếu app mà nhãn
yêu cầu với app mà router thực sự đưa vào prompt (phân tích từ `train_check.log`):

| | 93 mẫu có gold call |
|---|---|
| Mọi app cần thiết **đều** vào prompt | **22 (24%)** |
| Thiếu một phần | 31 (33%) |
| **Không app nào cần thiết vào prompt** | **40 (43%)** |

Ví dụ rõ nhất, `train_0117`: transcript nói về **giá vàng và chuyển tiền**, router đưa vào
`7/48 tool từ app ['smart_home']`. Model sau đó gọi `financial.get_gold_price` — một tool không có
trong prompt — nên rơi về `sys.ask_missing_info`. **43 mẫu `asked_missing` chính là dấu vết của lỗi
này**: model không "bỏ cuộc sớm", nó không được nhìn thấy tool nó cần.

### Router này không sửa được bằng cách tính điểm tốt hơn

Thử offline 4 hàm xếp hạng trên đúng 93 mẫu đó (recall@k = tỉ lệ mẫu có **mọi** app gold nằm trong
top-k):

| Hàm xếp hạng | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 |
|---|---|---|---|---|---|---|
| Dice trên văn bản ghép (đang dùng) | 17% | 31% | 35% | 40% | 46% | 55% |
| Dice max theo từng field | 22% | 27% | **41%** | 47% | 55% | 60% |
| Containment | 11% | 17% | 20% | 27% | 40% | 51% |
| Containment có IDF | 13% | 18% | 25% | 29% | 38% | 48% |

Hàm tốt nhất chỉ hơn hàm hiện tại 6 điểm ở k=3 và **không hàm nào vượt 60% kể cả k=6**. So khớp từ
vựng giữa transcript WER 23.6% và mô tả app là **hướng chết** — không phải lỗi cài đặt.

### Gốc rễ thật: `context_size 4096` nhỏ hơn prompt đầy đủ

Đo số token prompt thật bằng tokenizer trên 100 mẫu này:

| | min | trung vị | p90 | max |
|---|---|---|---|---|
| Số tool trong prompt | 13 | 44 | 53 | 60 |
| Số app trong prompt | 4 | 13 | 16 | 18 |
| Prompt **nguyên bản** | 2198 | **5882** | 7233 | 7956 |
| Prompt **đã nén** | 1741 | **4354** | 5245 | 5790 |

Nén schema (bỏ `Returns:`, cắt description còn 60 ký tự) tiết kiệm 26% — **không đủ**. Ngân sách cần
để giữ **mọi** app:

| Ngân sách | 1800 | 2500 | 3000 | 3500 | 4000 | 5000 | **6000** |
|---|---|---|---|---|---|---|---|
| % mẫu giữ đủ app | 2% | 6% | 11% | 17% | 41% | 84% | **100%** |

Nên `context_size 4096` **tự nó bắt buộc phải cắt prompt** — router không phải lựa chọn tối ưu hoá,
nó là hệ quả của context quá nhỏ. Luật BTC cho phép tới 16384 token, nên **context 8192 là hợp lệ**
và đủ chỗ cho prompt nén đầy đủ.

## Bài học

**"Tối ưu hoá" lớn nhất của v3 chính là thứ phá điểm nặng nhất.** Cắt prompt về 1800 token đưa laptop
từ 171h xuống 8.4h cho 600 mẫu — và đồng thời kéo điểm xuống còn 1/4. `local_v1` đạt 22.52 **không
cắt prompt**. Tốc độ đo được mà không chấm điểm là tốc độ của một kết quả sai.

**Phải chấm điểm mọi lần chạy.** Ba phiên bản (v2, v3, v4) tối ưu tốc độ liên tiếp mà chưa lần nào
biết điểm. `scripts/score.py` viết ở bản này lẽ ra phải có từ v2.

**`state_hash` không phải nguyên nhân độc lập.** 86 mẫu lệch hash là *hệ quả* của việc gọi sai tool,
không phải lỗi riêng cần sửa riêng. 49/93 mẫu không khớp được **dù chỉ một tên tool** với nhãn.

## File

| File | Nội dung |
|---|---|
| `train_check.jsonl` | 100 mẫu dự đoán trên split `train` |
| `train_check.metrics.jsonl` | số đo từng mẫu, kèm transcript và cấu hình |
| `train_check.transcripts.jsonl` | cache transcript của pha ASR (tái dùng được cho lần chạy sau) |
| `train_check.log` | log đầy đủ — chứa dòng `Tool prompt routed to ...` dùng để phân tích router |

Lệnh tái lập:

```bash
python scripts/infer.py --split train --max-samples 100 --max-seconds 3600 --output train_check.jsonl
python scripts/score.py train_check.jsonl --split train
```

Lưu ý: lần chạy kết thúc với **exit code 1** dù hoàn tất 100/100 mẫu — `scripts/infer.py` cố ý
`raise SystemExit(1)` khi có mẫu thất bại (ở đây 2 mẫu). Hoàn tất bình thường vẫn báo lỗi cho job
nền, cần biết để không tưởng là crash.
