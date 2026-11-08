# kaggle_v2 — phép A/B công bằng đầu tiên về adapter

Hai lượt Kaggle chạy ngày 2026-11-07, khác nhau **đúng một cờ** `USE_ADAPTER`. Mọi thứ còn
lại giữ nguyên: cùng 50 mẫu held-out từ `sft.eval.jsonl` (cùng `SEED`, cùng thứ tự ID), fp16
trên Tesla T4, đủ **147 tool trong 46 app** (không route prompt), `context=12288`,
`MAX_NEW_TOKENS=256`, `MAX_STEPS=8`, ASR chạy inline (`TRANSCRIPT_CACHE=None`).

Vì sao điều này đáng kể: `local_v5`–`local_v10` đều lẫn thêm biến. Trên laptop 4 GB adapter
chỉ **gắn kèm** được (gộp vào trọng số 4-bit sẽ lượng tử cả delta) và ngân sách prompt bị VRAM
ép xuống 2000 token, nên mọi so sánh ở đó đều lệch *hai* biến. Lượt này adapter được **gộp**
vào fp16 — đúng điều kiện đã huấn luyện — và tool không bị cắt.

```
554.1s  LoRA adapter merged: /kaggle/input/datasets/anhnguyen7508/lifemate-lora
554.2s  Parameter audit: {'asr': 1543304960, 'llm': 4022468096}; dtype=torch.float16
```

## Kết quả

| | A — adapter `lora_V4` | B — model gốc | |
|---|---|---|---|
| **điểm (`local_train_score`)** | **5.50** | **0.4444** | A = **12.4×** B |
| giây/mẫu trung bình | **23.84** | 33.31 | B chậm **1.40×** |
| p95 | **38.61** | 57.75 | |
| max | **53.32** | 60.08 | |
| **quá 30 giây** | **9/50 (18%)** | 26/50 (52%) | |
| thất bại | **0** | 3 | |
| `done` | 46 | 46 | |
| `asked_missing` | 4 | 1 | |
| `step_limit` | **0** | 3 | |
| tổng thời gian lượt | 1766.5s | 2372.1s | |

Adapter thắng ở **mọi** cột. Không có đánh đổi nào phải cân.

### Cơ chế: ít token mỗi bước, không ít bước

Trên 39 mẫu đầu đo được từ log, số call trung bình gần như **bằng nhau** (A 3.13 vs B 3.21)
trong khi A nhanh hơn 1.40×. Adapter không đi đường ngắn hơn — nó **sinh ít token hơn mỗi
lượt**, quyết đoán hơn, bớt lan man. Khớp với thứ `local_v7` đã thấy (quá 30 giây 14/50 → 0/50)
nhưng lần này đo sạch biến.

Base còn ba dấu hiệu lan man mà adapter không có: `train_0081` 8 call `step_limit`,
`train_0229` 6 call `step_limit`, và `train_0945` nộp **0 call**.

### Vì sao chuyện 30 giây không phải chuyện phụ

Luật: mỗi mẫu phải xong trong **30 giây**, quá hạn **ăn 0 điểm**. Base vượt hạn ở **52%** mẫu,
adapter ở 18%.

Lượt này không ai bị chặn vì notebook đặt deadline T4 = 600s, và chính nó in cảnh báo:
`T4/P100 diagnostic run: 600s/sample. This does NOT establish the 30s A30 rule.` T4 chậm hơn
A30 24 GB của vòng Docker đáng kể, nên **không chuyển trực tiếp** con số 18% sang vòng đó
được. Nhưng nó nói biên thời gian mỏng, và đây là thứ phải đo lại khi dựng image.

Lưu ý khi so với `local_v10`: ở đây ASR **nằm trong** số giây (`TRANSCRIPT_CACHE=None`), còn
13.3 s/mẫu của `local_v10` là *chỉ LLM*. Hai con số không so được.

## `done` KHÔNG có nghĩa là đúng

Đây là thứ quan trọng nhất học được, và nó không nằm ở cột nào trong bảng trên.

Cả hai lượt đều `done: 46/50` — mô hình **hoàn thành** 92% mẫu — rồi ăn 5.5 và 0.44 điểm.

Bóc số ra: điểm = trung bình F1 × 100. A được 5.50 nghĩa là tổng F1 trên 50 mẫu chỉ **2.75**.
B được 0.4444 = **2/9** chính xác, tức gần như đúng **một** mẫu có điểm lẻ trong cả 50
(P=1/4, R=1/5 cho ra đúng 2/9).

Nên `done` nghĩa là mô hình **tự tin kết thúc**, không phải làm đúng. Nó gọi sai tool, hoặc
sai tham số, hoặc để `state_hash` lệch. Luật chấm all-or-nothing trên `state_hash`: lệch một
tham số là mất trắng cả mẫu. Đừng bao giờ đọc `stop_reasons` như một thước đo chất lượng.

## Hai điều KHÔNG kết luận được từ đây

**5.50 trùng khít `local_v4` = 5.50.** Rất có thể là trùng hợp — điểm là tỉ số của các số
nguyên nhỏ nên hay chạm nhau. Nhưng nếu một lượt khác cấu hình cũng ra đúng 5.50 thì phải
nghi hàm chấm trước khi nghi model.

**5.50 không so được với 13.41 (`kaggle_v1`) hay 11.18 (`local_v10`).** Đó là 50 mẫu held-out
`train`; hai số kia là 634 mẫu `public_test`. Tập khác nhau. Mốc duy nhất đúng nghĩa trong
tài liệu này là **B = 0.44** — và adapter hơn nó 12.4 lần.

## Ba lượt chết trước khi tới được đây

Đáng ghi vì cả ba đều là lỗi **môi trường Python**, không lỗi dữ liệu, và hai trong ba do
chính màng chắn gây ra.

1. **Version 18, giây 586** — `ImportError: Found an incompatible version of torchao. Found
   version 0.10.0`. `peft/import_utils.is_torchao_available()` **raise** khi thấy torchao cũ
   thay vì trả `False`, và nó nằm trong chuỗi dispatcher LoRA. Chết **sau** khi đã tải 13.4
   GiB model. Sửa: gỡ torchao (không nâng — nâng có thể kéo theo torch khác), xoá
   `sys.modules`, `invalidate_caches()`, rồi gọi đúng hàm đó để kiểm.
2. **Version 19, giây 514** — `RuntimeError: Restart kernel before installing a different
   Transformers version.` Mục `[9]` của màng chắn `import peft`, peft kéo theo `transformers`
   5.16.1 vào `sys.modules`, nên cell cài đặt (đúng) từ chối làm việc. Màng chắn sập vào đúng
   cái bẫy nó được viết ra để tránh, bằng chính hành động kiểm tra. Sửa: gọi peft trong
   **subprocess**, thêm mục `[10]` chặn sớm nếu kernel đã nhiễm.
3. **Một lượt trước đó** bị màng chắn **chặn oan**: phép so base của adapter lấy
   `base.split('/')[-1]` — tức lấy **mã revision** rồi đem so với **tên model**.

Bài học thành tính chất: **màng chắn phải read-only với `sys.modules`, không chỉ read-only
với đĩa.** Nó chạy trước mọi thứ, nên mỗi `import` nó làm là một ràng buộc nó áp lên phần còn
lại của notebook.

Và: **4 dataset tách rời không gây lỗi nào.** Log cả ba lượt đều xác nhận mount đủ —
`lifemate-lora` 0.25 GiB, `lifemate-sft` 0.03 GiB, `lifemate-transcripts`,
`zaic2026-track-lifemate` 1.05 GiB, `train.jsonl` 1545 mẫu, `147 tool trong 46 app`,
`audio: đủ 1545/1545 file`.

## Lượt C — bài nộp `public_test` 634/634

Chạy 2026-11-07, **13624.8s (3 giờ 47 phút)**, `Successfully ran`. Cấu hình = lượt A thắng,
đổi bốn dòng: `SPLIT='public_test'`, `RUN_FULL=True`, `USE_ADAPTER=True`, và
`TRANSCRIPT_CACHE` trỏ vào `public_test.transcripts.jsonl`.

```
Summary: {'completed': 634, 'selected': 634, 'split_size': 634,
          'complete_selection': True, 'full_submission': True, 'failed': 28,
          'stop_reasons': {'done': 530, 'asked_missing': 76,
                           'step_limit': 17, 'repeated_invalid': 11},
          'mean_seconds': 18.29, 'p95_seconds': 36.40, 'max_seconds': 87.21,
          'over_30_seconds': 70}
```

File nộp: `results/kaggle_v2/submission.jsonl` (ngoài git, 227.6 KB). Đã kiểm:

```
634/634 dòng · đủ ID · ĐÚNG thứ tự · 0 trùng · 0 thiếu state_hash
schema ['id','state_hash','tool_calls'] · 1684 tool call (2.66/mẫu)
```

| `stop_reason` | mẫu | % |
|---|---|---|
| `done` | 530 | **83.6%** |
| `asked_missing` | 76 | 12.0% |
| `step_limit` | 17 | 2.7% |
| `repeated_invalid` | 11 | 1.7% |

### So với `local_v10` = 11.18 điểm

Cùng 634 mẫu `public_test`, cùng adapter. Khác: fp16 + adapter **gộp** + đủ 147 tool, thay vì
nf4 + gắn kèm + ngân sách 2000 token.

| | local_v10 (**11.18**) | kaggle_v2 C |
|---|---|---|
| mẫu nộp **0 tool call** | 62 | **7** |
| tổng tool call | 1144 | **1684** |
| `done` | 468 (73.8%) | **530 (83.6%)** |
| thất bại | 30 (OOM) | 28 (`step_limit` + `repeated_invalid`) |

**62 → 7 mẫu nộp rỗng.** Đó đúng là thứ tôi đã chỉ ra là nguyên nhân chính của khoảng cách
11.18 vs 13.41: `--prompt-token-budget 2000` trên laptop 4 GB cắt mất tool cần dùng, và sàn
2335 token là một app nên không hạ thêm được. Ở đây không có giới hạn đó.

### ĐIỂM PORTAL: **21.64**

Cao nhất trên pipeline hiện tại, và cao thứ hai của cả dự án.

| | điểm | so với 21.64 | |
|---|---|---|---|
| **kaggle_v2 C** | **21.64** | — | fp16 + adapter gộp + đủ 147 tool |
| `local_v10` | 11.18 | **+10.46 (1.94×)** | cùng adapter, nhưng nf4 + gắn kèm + cắt tool |
| `kaggle_v1` | 13.41 | **+8.23 (1.61×)** | đủ tool nhưng **không** adapter, có route prompt |
| `local_v1` | 22.52 | **−0.88** | kỷ lục cũ — đường đi khác: llama.cpp CPU + Qwen3.5-4B Q8 |

Gần gấp đôi bản trước. Hai thứ cùng đóng góp và không tách được từ *riêng* cặp số này
(`kaggle_v1` khác cả adapter *lẫn* route prompt), nhưng phép A/B ở trên **đã** tách: trên 50
mẫu held-out ở đủ tool, adapter một mình cho **5.50 vs 0.44**. Nên adapter là phần lớn.

### 28 mẫu thất bại chỉ đáng 1.0 điểm — phần còn lại ở chất lượng

Đây là con số quan trọng hơn 21.64, và nó đổi hướng việc phải làm tiếp.

```
điểm/mẫu            = 21.64 / 634 = 0.0341   (tổng F1 = 137.2 / 634)
trần nếu sửa HẾT 28 mẫu thất bại = 22.64     <- vượt local_v1 = 22.52
  trong đó 17 mẫu step_limit     ≤ 0.61
  trong đó  7 mẫu 0-call         ≤ 0.24
```

Tức toàn bộ phần "đường ống" — OOM, quá bước, nộp rỗng — cộng lại đáng **1.0 điểm**. Còn
**~78 điểm** nằm ở 606 mẫu *chạy xong* mà trung bình mỗi mẫu chỉ ăn **0.0357**.

`done` 530 mẫu (83.6%) nhưng điểm 21.64. Đúng như phần A/B đã cảnh báo: **hoàn thành không
phải làm đúng.** Mô hình tự tin kết thúc rồi gọi sai tool hoặc sai tham số, và `state_hash`
chấm all-or-nothing nên lệch một tham số là mất trắng cả mẫu.

Nên việc tiếp theo **không phải** nâng `MAX_STEPS` (≤ 0.61 điểm). Việc tiếp theo là tìm ra
*sai ở đâu* trên 606 mẫu đó — và chỗ duy nhất đo được việc này là `train`, nơi có gold: chạy
một lượt held-out rồi so từng `tool_calls` với gold trajectory, phân loại sai tool / sai tham
số / thiếu call / thừa call. Chưa có bảng phân loại đó thì mọi thay đổi tiếp theo là đoán.

### Chỗ rò còn lại, đã khoanh được

**28 mẫu thất bại và 17 trong đó là `step_limit`.** Nhìn danh sách mẫu chậm nhất thì rõ: 8
trong 10 mẫu chậm nhất có `llm_turns: 8` — tức **chạm đúng trần `MAX_STEPS=8`** rồi bị cắt.

```
public_0025  87.21s  llm_turns 8  step_limit
public_0007  60.20s  llm_turns 8  step_limit
public_0133  56.74s  llm_turns 8  step_limit
public_0031  53.28s  llm_turns 8  step_limit
public_0008  51.79s  llm_turns 8  step_limit
```

Đây là một **tham số**, không phải một giới hạn vật lý — khác hẳn 29 mẫu OOM của `local_v10`
mà tôi phải bỏ dở vì card 4 GB. Nhưng nâng `MAX_STEPS` **chưa chắc ăn điểm**: có thể model
đang lặp vô ích chứ không đang thiếu bước. Phải đo trước: lấy 17 mẫu đó, chạy lại với
`MAX_STEPS=12`, xem `state_hash` có khớp gold thêm mẫu nào. Chưa đo thì chưa đổi.

**70 mẫu (11%) quá 30 giây, max 87.2s.** Vòng public ta tự nộp nên không bị chặn. Vòng Docker
trên A30 thì **đây là điểm** — mỗi mẫu quá hạn ăn 0. A30 nhanh hơn T4 nên con số sẽ nhỏ hơn,
nhưng 87 giây thì không có GPU nào cứu được: `MAX_STEPS` phải xuống, hoặc phải có màng chắn
thời gian cắt sớm và nộp những gì đã gọi.

## Phụ lục — cấu hình lượt C

Đổi so với lượt A:

```python
SPLIT      = 'public_test'
RUN_FULL   = True
USE_ADAPTER = True          # giữ - đã thắng 12.4x
TRANSCRIPT_CACHE = Path('/kaggle/input/datasets/anhnguyen7508/'
                        'lifemate-transcripts/public_test.transcripts.jsonl')
```

Ước lượng của tôi trước khi chạy là **4.2 giờ** (634 × 23.8 s). Thực tế **3 giờ 47 phút** —
transcript cache bỏ ASR khỏi đường đi nên giây/mẫu xuống 18.29 (`asr_seconds` ~5e-06, tức
bằng 0). Hạn session GPU 9 giờ, thừa thoải mái.

Phóng bằng **Save & Run All (Commit)** để đóng tab không giết kernel.
