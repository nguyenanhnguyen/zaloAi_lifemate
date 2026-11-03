# baseline_2252 — bản chạy đạt 22.52, truy lại từ git

Đây là **bản phục dựng** của lần chạy ghi 22.52 trên leaderboard (`results/local_v1`), do
Codex làm trước khi dự án chuyển sang Qwen3-4B. Folder này tồn tại để bản đó không bị mất,
và để so sánh được với mọi bản sau.

## Truy vết: vì sao chắc chắn là commit này

`pipeline/` là cây làm việc tại **`7002744`** (31/10/2026, "update gitignore"), lấy ra bằng
`git archive`. Bằng chứng, chứ không phải suy đoán:

| | |
|---|---|
| `submission.jsonl` tại `7002744` | md5 `8e65b65a4b7c0a17bab102a1b3ad8ba1` |
| `results/local_v1/submission.jsonl` | md5 `8e65b65a4b7c0a17bab102a1b3ad8ba1` |
| `submission.metrics.jsonl` tại `7002744` | md5 `b1dff0b2a5932d9993866fad6a11ab86` |
| `results/local_v1/submission.metrics.jsonl` | md5 `b1dff0b2a5932d9993866fad6a11ab86` |

Khớp byte-đúng cả hai file. Và `git diff f71bc40 7002744 -- src/` **rỗng**, nên code suy luận
không đổi giữa lần chạy và commit — không có khoảng mờ nào để nghi ngờ.

Hai commit sau (`0498c10`) có cùng md5 nhưng chỉ khác `.gitignore`; `7002744` là commit sớm
nhất mang đủ bằng chứng.

## Không có model đã train

Cần nói rõ, vì đây là điều hay bị nhớ sai: **lần chạy 22.52 không fine-tune gì cả.** Nó là
Qwen3.5-4B Q8 nguyên bản qua llama.cpp, zero-shot, có bật thinking. Không tồn tại checkpoint
hay adapter nào để truy lại — không phải vì bị xoá, mà vì chưa từng có.

Việc fine-tune trong dự án này bắt đầu ở `319423a` (01/11), trên một model **khác**
(Qwen3-4B-Instruct-2507), và chưa ra điểm.

## Trọng số: đã bị xoá khỏi đĩa, nhưng tải lại được chính xác

Tìm toàn máy không còn file `.gguf` nào, và cache HuggingFace không còn
`models--unsloth--Qwen3.5-4B-GGUF` lẫn `models--Qualcomm-AI-Research--PhoASR-whisper-small`.
Không sao: code pin cả repo lẫn revision, nên tải lại cho ra đúng bit cũ.

| | Repo | Revision | File |
|---|---|---|---|
| LLM | `unsloth/Qwen3.5-4B-GGUF` | `e87f176479d0855a907a41277aca2f8ee7a09523` | `Qwen3.5-4B-Q8_0.gguf` (~4.2 GB) |
| ASR | `Qualcomm-AI-Research/PhoASR-whisper-small` | `da2d9ae0ac0a6a6faba20add61c2e980ccd85037` | snapshot |

Chạy `python restore.py` trong folder này để tải cả hai về `models/`.

Ngoài ra cần **`llama-server`** của llama.cpp (không có trong git, không tải tự động được
bằng script này). Code tìm nó theo thứ tự: biến môi trường `LLAMA_SERVER` → đệ quy trong
`models/llama.cpp/` → `PATH`.

## Cấu hình: cái gì chứng minh được, cái gì chỉ suy ra

**Chứng minh được từ `submission.metrics.jsonl` của chính lần chạy** (mỗi dòng tự ghi lại):

| | |
|---|---|
| LLM | `Qwen3.5-4B-Q8_0.gguf` |
| ASR | `Qualcomm-AI-Research/PhoASR-whisper-small` |
| Thiết bị | **llama.cpp trên CPU** — `whole_gpu_used_mb` chỉ 787 MB |

**Chỉ suy ra, không có trong file đo nào.** `metrics.jsonl` của bản này *không* ghi
`context_size`, `max_new_tokens` hay `thinking_tokens` (các bản sau mới ghi). Những số dưới
đây là **mặc định của `scripts/infer.py`**, cộng với lệnh chạy đầy đủ mà
`pipeline/README.md` ghi là `python scripts/infer.py --split public_test --output
submission.jsonl` — không truyền thêm cờ nào:

| | Mặc định | Khớp với số đo? |
|---|---|---|
| `--context-size` | 12288 | không kiểm được |
| `--max-new-tokens` | 1024 | không kiểm được |
| `--thinking-tokens` | 512 (thinking **bật**) | không kiểm được |
| `--max-seconds` | 600 | ✔ `max_seconds` quan sát 529.7 < 600 |
| `--max-steps` | 8 | ✔ `llm_turns` trung bình 4.18, không mẫu nào vượt 8 |
| `--asr-beams` | 5 | không kiểm được |

Hai dòng có dấu ✔ là kiểm tra nhất quán, không phải bằng chứng — chúng loại trừ việc cờ bị
ghi đè, chứ không khẳng định. Ba dòng đầu thì **không có cách nào xác nhận** từ dữ liệu còn
lại.

Cờ llama.cpp trong `llm_agent.py`: `--flash-attn on --jinja --ctx-size 12288 --parallel 1
--batch-size 512 --ubatch-size 128 --fit on --fit-target 384 --cache-ram 512
--reasoning-format deepseek --no-context-shift`.

`transformers` pin **4.57.6** (repo hiện tại dùng 5.13.0 — phải pin lại để chạy bản này).

## Khoảng mờ không gỡ được

1. **Cây làm việc lúc chạy có thể khác commit.** Git không thấy sửa chưa commit. Bằng chứng
   gián tiếp là mạnh — `metrics.jsonl` tự ghi đúng hai model mà code pin, và `src/` không
   đổi giữa `f71bc40` và `7002744` — nhưng nó là bằng chứng gián tiếp.
2. **Ba tham số context/token/thinking** như bảng trên.
3. **Phiên bản llama.cpp.** Xem mục dưới; đây là lỗ hổng nặng nhất.

## Cảnh báo quan trọng: 22.52 không phải điểm hợp lệ

Từ `submission.metrics.jsonl` của chính lần chạy đó:

| | |
|---|---|
| Thời gian/mẫu, trung vị | **149.3 giây** |
| p90 | 294.1 giây |
| Tối đa | 529.7 giây |
| Mẫu vượt 30 giây | **634/634 (100%)** |

Luật là 30 giây/mẫu trên 1×A30 24GB, timeout tính 0 điểm. Mốc đó **chỉ được áp ở vòng
Docker**, nên ở vòng public test bản này nộp được và ghi 22.52 thật. Nhưng trong container
nộp ngày 16/11 thì cả 634 mẫu đều timeout, và điểm là **0**.

Nên con số 22.52 dùng được để làm gì, và không dùng được để làm gì:

- **Dùng được:** nó chứng minh pipeline này (parser, định tuyến, vòng gọi tool) *đúng về
  logic* — 22.52 so với 13.41 của `kaggle_v1` là chênh thật về chất lượng quyết định.
- **Không dùng được:** làm mục tiêu cần vượt. Điểm hợp lệ tốt nhất ta có là **13.41**.
  Không có cách nào ép 149 giây xuống 30 giây bằng cách tắt thinking hay đổi quant — đó là
  gấp 5 lần.

## Hai câu còn phải kiểm trước khi dùng lại Qwen3.5

1. **Giấy phép.** Luật loại thẳng mọi licence NonCommercial. `unsloth/Qwen3.5-4B-GGUF` là
   bản quantize lại của người khác; phải đọc licence của cả repo gốc và repo quantize.
2. **Ngày phát hành.** Chỉ model công khai trước **26/10/2026** được dùng. Chưa ai trong dự
   án kiểm ngày của Qwen3.5-4B.

Nếu một trong hai câu trả lời là không, thì folder này thuần là tư liệu, và mọi so sánh với
22.52 chỉ còn giá trị chẩn đoán.

## Đã mất gì — thứ đồng đội muốn dùng lại mà không còn

Xếp theo mức độ đau, nặng nhất trước.

**1. Phiên bản `llama-server` của llama.cpp. Mất hẳn, và không pin ở đâu cả.**
Đây là lỗ hổng nặng nhất của bản phục dựng. Không `requirements.txt`, `README`, log hay
commit nào ghi build number. Mà các cờ đang dùng — `--fit on --fit-target 384 --cache-ram
512 --reasoning-budget` — là cờ mới và llama.cpp đổi tên/ngữ nghĩa cờ rất nhanh. Tải một
build hôm nay thì cờ có thể không còn tồn tại, hoặc còn tên mà khác hành vi. **Hệ quả: bản
22.52 không tái lập bit-đúng được, chỉ tái lập gần đúng.**

**2. `models/llama.cpp/server.log`.** Chưa từng được commit. Nó chứa đúng thứ mục 1 cần:
build number, cách llama.cpp chia GPU/CPU, tốc độ token. Mất một file log đã xoá mục 1.

**3. Điểm từng mẫu của `local_v1`.** `public_test` không có nhãn, nên 22.52 là **một con số
tổng duy nhất**. Không thể phân tách mẫu nào đúng mẫu nào sai trên lần chạy tốt nhất của dự
án — tức không làm được phân tích lỗi trên chính bản đáng học nhất. Không gỡ được, kể cả có
đủ model, trừ khi BTC phát hành nhãn.

**4. `train_quality.jsonl` và `train_quality.metrics.jsonl`: 0 byte.** Một lượt chạy trên
split `train` (**có nhãn**) với cấu hình này đã được khởi động và không bao giờ ra kết quả.
Đó chính là mảnh ghép sẽ cho phép so Qwen3.5-4B với Qwen3-4B-Instruct **một cách hợp lệ,
trên dữ liệu có nhãn, cùng ngân sách thời gian**. Hiện ta không có cách nào biết 9 điểm chênh
giữa 22.52 và 13.41 là do model hay do 149 giây. Cái này **chạy lại được** — và nên chạy.

**5. Trọng số GGUF + PhoASR.** Đã xoá khỏi đĩa (chỉ còn thư mục `.locks` trong cache HF làm
chứng), nhưng **tải lại được chính xác** nhờ revision đã pin, nên chỉ mất nếu repo thượng
nguồn biến mất. `EXPERIMENTS.md:675` đã ghi nhận việc xoá này từ trước.

**Thứ KHÔNG mất, và đáng giá nhất cho đồng đội:** `pipeline/submission.jsonl` — đủ 634 dự
đoán của cấu hình điểm cao nhất. Không có nhãn để chấm, nhưng **so từng mẫu với bài nộp của
bất kỳ bản sau** thì được, và đó là cách dùng được ngay: chỗ nào bản mới gọi tool khác bản
22.52, chỗ đó là nghi phạm.

## Việc nên làm ngay để khỏi mất thêm

1. `python restore.py` rồi **lưu trữ** GGUF + snapshot ASR ra ngoài cache HF.
2. Khi nào chạy lại bản này, ghi `llama-server --version` vào `pipeline/` **trước** khi chạy.
3. Chạy lượt `--split train` còn thiếu (mục 4) — nó trả lời câu model-hay-thời-gian.

## Nội dung folder

```
baseline_2252/
  README.md     file này
  restore.py    tải lại hai model theo đúng revision đã pin
  pipeline/     cây code tại 7002744, nguyên vẹn, kèm submission + metrics của lần chạy
  models/       chỗ restore.py đặt trọng số (gitignore)
```
