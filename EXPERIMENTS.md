# Lịch sử thử nghiệm — ZAC 2026 LifeMate

Ghi lại từng lần chạy: tên run, mô hình, điểm chính thức và nhận xét. Chỉ thêm vào, không xoá.

---

## Bảng điểm leaderboard — hai điểm thật duy nhất của dự án

| Lần chạy | Điểm | Context | Prompt tool | LLM | Thinking |
|---|---|---|---|---|---|
| `local_v1` | **22.52** | 12288 | **nguyên catalogue** | Qwen3.5-4B Q8 | **bật** |
| `kaggle_v1` | **13.41** | 4096 | route từng tool | Qwen3-4B bf16 | tắt |

**Bài học trung tâm của dự án:** ASR tốt hơn (`PhoWhisper-large` 1.543B thay `PhoASR-small`), GPU
thay CPU, LLM instruct mới hơn — và **mất 9 điểm**. Thứ quyết định không phải model, mà là **model
có được nhìn thấy đủ công cụ hay không**, và **tool call có được đọc đúng hay không**.

Phần mất điểm của `kaggle_v1` đo được rất gọn: 42 mẫu nộp 0 tool call (cả 42 đều `stop_reason: done`
— bug parser) + 33 mẫu `failed`, hai nhóm không trùng nhau = **75 mẫu, 11.8 điểm mất trắng**.
`13.41 + 11.8 = 25.2`, tức riêng hai nhóm này đã đủ giải thích cả khoảng cách, chưa tính 109 mẫu
`asked_missing`.

---

## local v4 (chấm điểm) – **phát hiện: router prompt là chỗ mất điểm lớn nhất** (2026-10-31)

Chạy 100 mẫu `train` có nhãn và chấm bằng `scripts/score.py` — lần đầu dự án biết điểm thật.
Số đo và bảng chi tiết: [`results/local_v4/README.md`](results/local_v4/README.md).

**Điểm 5.50 so với `local_v1` 22.52.** Ba phiên bản liên tiếp (v2 → v3 → v4) tối ưu tốc độ từ
1027s xuống 34s/mẫu mà không lần nào chấm điểm; hoá ra đòn tăng tốc lớn nhất của v3 — cắt prompt về
1800 token bằng router theo app — **là thứ phá điểm nặng nhất**.

Đo trực tiếp: chỉ **24%** mẫu có đủ mọi app mà nhãn cần trong prompt; **43% không có app nào**.
43 mẫu `asked_missing` không phải model bỏ cuộc sớm (giả thuyết tôi nêu lúc chạy được 14/100 — **sai**),
mà là model không được nhìn thấy tool nó cần.

**Router không sửa được bằng hàm xếp hạng tốt hơn.** Thử offline 4 hàm (Dice trên văn bản ghép,
Dice max theo field, containment, containment có IDF): hàm tốt nhất đạt 41% recall@3 so với 35% hiện
tại, và **không hàm nào vượt 60% kể cả khi cho 6 app**. So khớp từ vựng giữa transcript WER 23.6% và
mô tả app là hướng chết — đừng thử lại, đừng viết router thứ năm.

**Gốc rễ là `context_size 4096`.** Prompt đầy đủ tốn trung vị 5882 token (nén còn 4354), nên context
4096 **bắt buộc** phải cắt — router là hệ quả, không phải lựa chọn. Cần ngân sách **6000 token** để
giữ 100% app; luật BTC cho tới 16384 nên **context 8192 là hợp lệ**.

### Laptop **không chạy được** cấu hình đúng — đã đo, đừng thử lại

> ⚠️ **Kết luận này đang bị nghi vấn (2026-11-02).** Nó rất có thể là bug `enable_gqa` ở mục
> `kaggle_v2 chạy thật`, không phải trần cứng của card: context 8192 nghĩa là
> `32 × 8192² × 2 byte = 4 GiB` ma trận attention — một mình nó đã quá card 4 GB, và con số
> "còn 21 MiB" dưới đây khớp chính xác với kiểu tràn đó. Bản vá đã vào `src/llm_agent.py`.
> **Phải đo lại trước khi tin đoạn dưới.** Chưa đo lại vì việc trên Kaggle gấp hơn.

Dò 3 mẫu với `--context-size 8192 --prompt-token-budget 6000`: sau 20 phút **chưa xong mẫu đầu**.
`nvidia-smi` lúc đó: **3945/3966 MiB, còn 21 MiB, GPU 100%** — attention ở ~6k token tràn hết card và
spill sang RAM. Khớp với lịch sử notebook: v3 OOM ở context 6144 trên **một T4 15 GiB**, phải hạ về
4096. Một card 4 GB không có cửa. Đã dừng, không để nó đốt 3 giờ cho 3 mẫu `time_limit`.

Kết luận phân vai: **laptop** làm nghiên cứu không cần prompt dài (so sánh ASR, thí nghiệm offline
bằng tokenizer — hai thí nghiệm router/budget ở trên không tốn một giây GPU nào). **Kaggle** đo cấu
hình đúng.

---

## local v5 – nâng chất lượng ASR: beam search **thua**, chuẩn hoá số **thắng** (2026-10-31)

Công cụ mới `scripts/asr_bench.py`: đo WER thật trên split `train` (có `audio_text`), nhiều chiến
lược decode trên **cùng một bộ audio**, không tốn lượt nộp.

### Beam search: loại

| Cấu hình | WER (n=15) | s/file |
|---|---|---|
| `beam5` | **24.61%** | 22.4 |
| `greedy` | 25.23% | **1.9** |
| `beam5` + `no_repeat_ngram_size=4` | 26.17% | 21.9 |

Beam search mua **0.62 điểm WER bằng 12 lần thời gian** — không đáng. Chặn n-gram lặp lại làm *tệ
hơn*: tiếng Việt lặp từ hợp pháp ("nhắc tôi tôi"), chặn đi là cắt mất chữ đúng. Giữ greedy.

### Lỗi thật của ASR là gì

Soi 5 mẫu WER cao nhất: lỗi chia hai loại khác hẳn nhau.

- **Lỗi thật** — giọng vùng miền và tên riêng: "vàng nhẫn"→"vàng dẫn", "Sống xanh"→"sóng sơn",
  "podcast"→"post khác", "hàng chờ"→"hành trời". Đây là trần của model, không sửa bằng code.
- **Không phải lỗi** — khác định dạng: nhãn ghi `18`, `60`, `SMS`; Whisper ghi "mười tám",
  "sáu mươi", "em ét". WER tính là sai, nhưng nghĩa thì đúng.

### Chuẩn hoá số: WER nói không, điểm nói có

`src/text_norm.py` viết lại số nói thành chữ số. Hai phép đo cho hai kết luận trái ngược, và **phép
đo thứ hai mới là phép đo đúng**:

| Phép đo | Kết quả |
|---|---|
| WER (chỉ chuẩn hoá hypothesis) | **+0.31pp — tệ hơn** |
| Gold numeric argument có mặt trong transcript | **0% → 22%** (10 đạt thêm, 0 mất) |

BTC chấm **đối số tool**, không chấm từ. Một `{"amount": 50000}` phải ra đúng con số; transcript thô
không chứa chuỗi `50000` ở **bất kỳ** mẫu nào trong 46 đối số numeric của 100 mẫu `train`. Sau khi
chuẩn hoá thì 10 đối số xuất hiện nguyên văn, kể cả số điện thoại `09505360367`.

**Tự sửa hai lần trong lúc làm — ghi lại để không lặp:**

1. Lần đầu tôi đo được WER **giảm 1.24pp** và tưởng đã thắng. Sai: lúc đó tôi chuẩn hoá **cả
   reference**, không được phép — reference là nhãn, phải giữ nguyên. Đo lại đúng cách thì WER *tăng*.
2. Phần lớn "cải thiện" ảo đó đến từ `không` → `0`. Nhãn `train` **tự nó** chuẩn hoá sai như vậy
   ("0 mưa thì thôi" = "không mưa thì thôi"), nên bắt chước nó thì đẹp WER mà làm transcript khó đọc
   hơn cho LLM. Đã bỏ `không` khỏi bảng số. **Đừng tối ưu theo quirk của nhãn.**

**Ba lỗi correctness đã bắt được bằng unit test** (17/17 case hiện pass, trong `src/text_norm.py`):

- Số điện thoại đọc từng chữ số bị **cộng lại**: "không chín một chín ba bảy..." → `41` thay vì
  `0919377320`. Chuỗi ≥3 chữ số đơn giờ được **nối**, không cộng. Sai số điện thoại là mất cả mẫu.
- "hai ba" → `5`. Hai chữ số đơn liền nhau là nhập nhằng (23 hay "2 3"), cộng lại là cách chắc chắn
  sai duy nhất → từ chối chuyển.
- "bác ba" → "bác 3". Người Việt đặt tên theo thứ: *Bác Ba*, *Cô Tư* là người, không phải số. Gold
  cần `query: "Bác Ba"`, mà token-F1 của `{bác, 3}` với `{bác, ba}` = 0.5 < ngưỡng 0.6 → **mất đối
  số vì chính bản vá**. Giờ sau từ chỉ quan hệ/chức danh thì một chữ số đơn giữ nguyên là chữ, trừ
  khi nó mở đầu số lớn hơn ("bác ba mươi tuổi" vẫn ra 30).

Bật mặc định trong pipeline cục bộ, tắt bằng `--raw-numbers` để đo hai chiều. **Chưa đưa vào
notebook V2** — V2 đang kiểm chứng việc bỏ route, đổi hai thứ cùng lúc thì không đọc được kết quả nào
ra kết quả nào.

### Đo end-to-end: **kết quả null**, và vì sao chưa kết luận được

Chạy lại **22 mẫu mà router giữ đúng toàn bộ app gold** (chỗ duy nhất hiệu ứng không bị lỗi routing
nhấn chìm), so với chính v4 trên đúng 22 mẫu đó:

| | v4 (số nguyên văn) | v5 (chuẩn hoá số) |
|---|---|---|
| Điểm | **11.36** | **11.36** |
| Mẫu hoàn hảo | 2 | 2 |
| Lệch `state_hash` | 18 | 18 |

Phép thử **có hiệu lực**, không phải không có gì xảy ra: 18/41 file âm cho transcript khác nhau,
**8/22 mẫu ra `tool_calls` khác**, 6/22 khác `state_hash`. Nhưng tổng điểm bằng nhau — có mẫu tốt
lên, có mẫu xấu đi, bù trừ hết.

**Đây không phải bằng chứng phủ định, mà là phép đo thiếu lực.** 18/22 mẫu (82%) mất điểm vì lệch
`state_hash` do chọn sai tool/đối số khác — thứ mà chuẩn hoá số không chạm tới. Phải sửa xong nguyên
nhân đó rồi mới đo lại được hiệu ứng này. Giữ bật vì cơ chế rõ (0% → 22% đối số numeric có mặt, 0 cái
bị phá), nhưng **ghi rõ là chưa chứng minh end-to-end**.

Phép thử này còn bắt được lỗi thứ tư của normalizer: `"tắt đèn thằng hai giùm chị"` →
`"đèn thằng 2"`. `thằng Hai` cũng là tên gọi theo thứ. Đã thêm `thằng, con, cháu, bé, mẹ, bố, ba, má`
vào danh sách chức danh (16/16 unit case pass).

---

## kaggle_v2 – gửi cả catalogue tool, không route prompt (2026-10-31)

Notebook mới `notebook/lifemate_kaggle_v2.ipynb`. Chi tiết cấu hình, cách chạy 2 phiên và danh sách
file cần tải: [`results/kaggle_v2/README.md`](results/kaggle_v2/README.md).

**`kaggle_v1` cũng route, không phải "không cắt prompt" như tôi ghi lúc đầu.** `MAX_CONTEXT 4096` trừ
192 token output còn **3904**, trong khi prompt đã nén cần trung vị **4553**. `metrics.jsonl` không ghi
số token nên không thấy trực tiếp, nhưng phép tính thì rõ, và `asked_missing 109/634` (17%) là đúng
dấu vết đó — nhẹ hơn 43% của local v4 vì ngân sách gấp đôi.

> ⚠️ **Khẳng định sai, đã sửa (2026-11-01).** Đoạn dưới nói V2 "chia LLM lên cả 2 T4 (một card không
> chứa nổi context đó)". Tôi khẳng định mà **không hề tính**. Phép tính: trọng số fp16 7.49 GiB + KV
> cache 12288 token 1.69 GiB (36 lớp × 8 KV head × 128 dim × 2 × 2 byte = **144 KiB/token**) =
> **9.18 GiB trên 14.46 GiB trống** — một card dư 5.28 GiB. *(Con số 9.18 này vẫn **thiếu** workspace
> attention lúc prefill, đúng thứ làm V14 OOM; tổng thật sau khi vá GQA là ~13.2 GiB. Xem mục
> `kaggle_v2 chạy thật` phía dưới. Tức tôi viết con số chưa tính **hai lần liên tiếp**.)* Sharding
> không mang lại gì, lại là khác
> biệt cấu trúc **duy nhất** so với mọi cấu hình đã chạy được (V1 và laptop đều 1 GPU) trong khi
> `cache.crop()` chưa bao giờ chạy với các lớp trải trên hai card, và lần chạy Kaggle đầu tiên chết
> đúng trong `model.generate()`. `SHARD_LLM_ACROSS_GPUS` giờ mặc định `False`; GPU thứ hai vẫn dùng
> cho ASR. Bài học: **đừng viết một con số vào tài liệu nếu chưa tính nó.**

Thay đổi: context 4096 → **12288**, bỏ hẳn route, ~~chia LLM lên cả 2 T4 (một card không chứa nổi
context đó)~~, port bản vá parser văn xuôi (42 mẫu nộp 0 call ở V1), `context_limit` không còn giết cả
run, và gate preflight chặn theo **tỉ lệ** lỗi >20% thay vì chặn khi có **một** mẫu `step_limit` —
V1 đặt gate đó sẽ chặn oan một lần chạy 10 tiếng.

Đo trước bằng tokenizer, đúng dạng prompt của notebook, trên 100 mẫu `train`: prompt lượt 1 schema
**nguyên bản** trung vị 5945 / max 7783 < limit 12032 → **0/100 mẫu phải route**, và vẫn còn ~4200
token cho tool result tích luỹ. Nên V2 gửi schema đầy đủ cả phần mô tả, không nén.

Phiên 1 chạy preflight 50 mẫu `train` để lấy **điểm thật** (`local_train_score`) — V1 chưa bao giờ có
số này. Chỉ sau khi điểm chấp nhận được mới chạy `public_test`.

---

## kaggle_v2 chạy thật – 5 lần thất bại, và điều chúng dạy (2026-11-01 → 02)

| Ver | Thời gian | Kết cục | Nguyên nhân |
|---|---|---|---|
| 11 | 45 s | `FileNotFoundError` checkpoint | `allow_patterns` thiếu `*.bin`; PhoWhisper-large không có safetensors |
| 12 | 225 s | `NameError: TRANSCRIPTS` | `TRANSCRIPTS` nạp sau khi dựng `config` |
| 13 | 238 s | traceback bị cắt | — (không chẩn đoán được từ log) |
| 14 | 227 s | `OutOfMemoryError` 5.00 GiB | `enable_gqa=True` của transformers là đường math kernel, O(N²) |
| 15 | 1499 s | 4/5 mẫu `LLM token limit reached` | bản **Thinking** tiêu hết 1024 token reasoning mà chưa đóng `</tool_call>` |
| 16 | — | **chạy thông** | thời gian đã ổn (trung vị ~17–19 s); lộ ra chỗ mất điểm thật |

**Lỗi chung của V11–V13 là của tôi, không phải của dữ liệu người dùng.** Tôi kiểm cú pháp code mình
viết nhưng không kiểm **những khẳng định** mình dựng lên trên đó. Hai công cụ ra đời để chặn đúng lớp
lỗi ấy mà không cần GPU: [`scripts/diag/check_notebook_order.py`](scripts/diag/check_notebook_order.py)
(AST, dùng-trước-khi-gán; đã kiểm ngược trên bản chưa vá → đúng 1 phát hiện là bug thật) và
[`scripts/diag/verify_notebook.py`](scripts/diag/verify_notebook.py) (14 hạng mục).

### V14: `enable_gqa` là đường math kernel, không phải đường tránh nó

`5.00 GiB = 32 head × N² × 2 byte → N ≈ 9160 token`. `transformers` bật `enable_gqa=True` mỗi khi
`attention_mask is None` — đúng ca batch=1 không padding — và **comment trong code của nó nói làm vậy
để *tránh* math kernel**. Đo trên torch 2.11 thì ngược lại.

Giả thuyết đầu của tôi (`torch.use_deterministic_algorithms` ép math backend) **sai, và tôi tự chứng
minh nó sai bằng phép đo**: cờ đó không đổi gì ở N=1024/2048/3072. Rồi benchmark 3 chiều:

```
SDPA trần, N=1024/2048/3072:  enable_gqa  372 / 1312 / 2844 MiB     (≈ N²)
                              repeat_kv     8 /   16 /   24 MiB     (≈ N)   -> 118x ở N=3072
Qwen3-4B thật,  N=2048→4096:  chưa vá    1657 → 5638 MiB  (×3,4)
                              đã vá       902 → 1792 MiB  (×1,99)
```

Vá: `_sdpa.use_gqa_in_sdpa = lambda *a, **k: False` (notebook cell 9 + `src/llm_agent.py`).
Kiểm thêm ở pha **decode** (q_len=1, N=9160): `repeat_kv` còn **nhanh hơn** `enable_gqa` 3× và tốn ít
bộ nhớ hơn — tức bản vá đúng ở cả hai pha. (Ở đây tôi lại đoán sai một lần nữa: tôi dự là vá làm
decode chậm đi vì phải copy K,V mỗi token, và phép đo nói ngược lại.)

**Hệ quả chưa kiểm:** kết luận "laptop có trần VRAM cứng ở context lớn" (mục `local v4` bên dưới) có
thể chính là bug này — context 8192 nghĩa là `32 × 8192² × 2 = 4 GiB` trên một card 4 GB. **Phải đo
lại trước khi tin kết luận đó.**

### V15: hạ tầng thông, và bản Thinking bị loại bằng số đo

Model nạp xong, prefill ~9160 token chạy qua, `train_0063` hoàn tất đúng (1 call, `done`). Nhưng
4/5 mẫu nộp **0 call** vì đầu ra bị cắt. Đây là thay đổi duy nhất của V2 **không có số đo nào đứng
sau** khi tôi làm nó, và giờ có số đo thì nó thất bại.

**Hai con số V15 cho, chưa từng có:**

1. **~9 token/s trên T4 ở context ~9k.** Mỗi mẫu lỗi đi qua đúng 2 lượt sinh 1024 token, nên
   `(273 − ~5 s ASR)/2048 ≈ 0,13 s/token`; bốn mẫu cho 7,6 / 10,2 / 9,1 / 7,9 — nhất quán. Một lượt
   1024 token mất **~114 s**, nên mốc 30 s của vòng Docker loại hẳn cách này *không cần* phép quy
   chiếu T4→A30 nào.
2. 9 token/s thấp hơn ~4 lần trần băng thông của chính T4 → nút cổ chai có lẽ là **overhead mỗi
   token**, không phải băng thông. Nếu đúng vậy, A30 **không** cứu được nhiều như tỉ lệ 2,9× gợi ý.
   Chưa đo, chưa kết luận — nhưng đây là rủi ro lớn nhất còn lại cho vòng Docker.

**Quyết định: về `Qwen3-4B-Instruct-2507`, `ENABLE_THINKING = False`, `MAX_NEW_TOKENS = 256`.** Lý do
quyết định không phải thời gian mà là **tính nhất quán với fine-tune**: `scripts/train_sft.py` gắn
cứng vào Instruct vì quỹ đạo gold không có khối `<think>`, và adapter LoRA học trên trọng số Instruct
không nạp lên trọng số Thinking được. Fine-tune là hướng bắt buộc → inference phải cùng checkpoint.

Nói thật đánh đổi: bỏ reasoning là bỏ đúng cái tương quan với 22.52. Nhưng `local_v1` cho reasoning
**512** token trên model **hybrid** Qwen3.5-4B biết suy luận *ngắn*; `Qwen3-4B-Thinking-2507` là
checkpoint chuyên suy luận, template luôn mở `<think>` nên `ENABLE_THINKING=False` **vô hiệu** với nó
và không có cách bảo nó suy luận ngắn. Hai thứ không thay thế được nhau.

> ⚠️ **Lời khuyên cũ của tôi đã bị xoá.** `results/kaggle_v2/README.md` từng viết: "nếu nhiều mẫu báo
> token limit thì nâng `MAX_NEW_TOKENS` lên 1536". Nó làm **cả hai** vấn đề tệ hơn. Lời khuyên đó
> được viết khi tôi chưa có một số đo thời gian nào.

Hai sửa đổi độ bền kèm theo: lần cắt thứ hai trả về `stop_reason='invalid_output'` và **giữ các call
đã thực hiện** thay vì làm lỗi cả mẫu (submission chỉ tốt lên hoặc giữ nguyên); và cổng
3-lỗi-liên-tiếp chỉ còn gắt trên `train` (đang đo → dừng sớm là đúng), nới lên 10 trên `public_test`
(chết ở mẫu 400/634 sau 6 giờ là mất trắng phiên đó). OOM vẫn giết run ở mọi split.

### V16: thời gian hết là vấn đề — và model **đọc dữ liệu rồi nói dối bằng văn xuôi**

Lần đầu pipeline chạy thông. **Thời gian đã giải quyết:** trung vị ~17–19 s/mẫu trên T4 (V1: 30,6 s),
đỉnh `cuda:0` ~9100 MiB trong 14912 (V1: 12869), quy đổi A30 còn ~6 s. Phép tính "~13,2 GiB" của tôi
vẫn bi quan quá — prefill thật ~6500 token, không phải 9160.

Thông lượng vẫn ~8 token/s (`train_0080` lượt 8: 256 token / 31,4 s = 123 ms/token), và **không đổi**
giữa context 6500 và 9160 → nghẽn là **overhead mỗi token**, không phải băng thông. Nghi vấn ở V15
được xác nhận, nhưng giờ nó không còn quan trọng vì đầu ra chỉ 15–35 token.

**Điểm 0,00 trên cả 7 mẫu đọc được, `state_match` sai cả 7.** Nguyên nhân áp đảo:

| mẫu | gold | ta | **write** | kết cục | model nói gì |
|---|---|---|---|---|---|
| `train_0015` | 4 | 3 | **0** | `asked_missing` | "Tôi không thể tìm thấy liên hệ…" |
| `train_0063` | 1 | 1 | 1 | `done` (văn xuôi) | "Đã gửi tin nhắn…" — ASR sai nội dung |
| `train_0066` | 1 | 2 | 1 | `done` (văn xuôi) | `alarm.create`, gold có lẽ tool khác |
| `train_0075` | 3 | 2 | **0** | `done` (văn xuôi) | **"Đã bật đèn phòng khách với độ sáng 50%"** |
| `train_0080` | 3 | 6 | **0** | `step_limit` | liệt kê 6 loại hoá đơn rồi tóm tắt, **không trả** |
| `train_0106` | 3 | 1 | **0** | `done` (văn xuôi) | **"sẽ đặt ô tô 4 chỗ…"** |
| `train_0203` | — | — | **0** | (đang chạy) | `bill_payment.pay` từ chối vì thiếu `bill_id` |

5/7 kết thúc bằng `plain_final`; **4/7 không gọi một call `write` nào** trong khi gold có write.

**Đây là xác nhận độc lập cho mục "chỗ nghẽn là HÀNH VI, không phải nhận thức" ở trên** — nhưng lần
này ở context 12288, không route. Giả thuyết "model không thấy tool vì prompt bị cắt" bị loại hẳn:
nó thấy tool, tìm đúng thiết bị, rồi **không bấm**. Và đó chính là lý do fine-tune là đúng hướng:
mọi quỹ đạo gold đều gọi tool write.

**Cơ chế chính xác** (`parse_decision` dòng 350): `allow_plain_done = bool(env.get_tool_calls())` bật
ngay khi đã có một call **bất kỳ, kể cả read**, nên văn xuôi sau một lần `smart_home.get_status` được
coi là `DONE` → mẫu đóng với 0 write → `state_hash` không đổi → 0 điểm.

**Vá, ba chốt** — mỗi chốt chặn một cách làm hỏng thêm:
1. Đẩy lại **đúng một lần** khi văn xuôi xuất hiện và `state_version == 0`.
2. **Chỉ khi `state_version == 0`**: thúc sau khi đã có write dễ sinh write thừa, mà write thừa làm
   sai `state_hash` vĩnh viễn — tệ hơn không làm gì.
3. **Chỉ khi `step < MAX_STEPS`**: đẩy ở lượt cuối thì vòng lặp thành `step_limit` → `failed`, biến
   một mẫu chỉ-tra-cứu đang được tính `done` thành 0 điểm.

Cộng một dòng `SYSTEM_PROMPT`: *"Đọc dữ liệu xong là để hành động, không phải để tóm tắt lại."*
Event mới `no_write_pushback` để đo xem nó có bắn và có cứu được mẫu nào.

**Hai vấn đề độc lập, nhỏ hơn:** ASR kém hơn tưởng (WER từng mẫu 0,60 / 0,33 / 0,17 / 0,14 / 0,13 /
0,12; `train_0063` mất điểm *chỉ vì* ASR) — nhưng không phải nguyên nhân chính, `train_0075` thất bại
không liên quan gì tới ASR. Và `MAX_STEPS = 8` bị tiêu hoang: `train_0080` dùng 6 lượt gọi
`bill_payment.list_bills` lần lượt cho 6 loại. Chưa sửa — nâng `MAX_STEPS` làm xấu thời gian xấu nhất
(8 lượt đã là 66 s), mà bản vá trên có thể làm model hành động sớm hơn thay vì liệt kê. Đo lại trước.

---

## Phân tách nguyên nhân lệch `state_hash` – **chỗ nghẽn là HÀNH VI, không phải nhận thức** (2026-11-01)

Đo trên 100 mẫu `local_v4`, không dùng GPU. Chỉ call `kind: "write"` đổi được `state_hash`, nên phân
loại theo đúng tập call ghi:

| | % | |
|---|---|---|
| **C. gold ghi, ta KHÔNG ghi** | **48%** | **bỏ việc** |
| D. ghi sai tool | 23% | |
| F. đúng tool, sai đối số | 14% | |
| A. không bên nào ghi | 12% | hash phải khớp |
| B. gold không ghi, ta ghi | 3% | bịa thao tác |

88% mẫu lệch ở tầng ghi, khớp với 86% lệch `state_hash` đã ghi nhận. Bucket C lớn gấp đôi mọi thứ
khác. Nội dung của nó: `bill_payment.pay` 18, `banking.transfer` 8 — **model chùn tay đúng ở chỗ
tiền chuyển đi**.

**Bucket C: không thể hay không chịu?** Tool ghi của gold **được phép gọi ở 48/48 mẫu** — chưa một
lần là vấn đề giấy phép. 72.9% dừng ở `asked_missing`, 25% dừng ở `done` với 0 call. Chia tiếp bằng
phép thử "model có chạm vào đúng app của gold không" (route là theo app, nên chạm được app nghĩa là
tool ghi có trong prompt):

| | |
|---|---|
| Đã chạm đúng app rồi dừng → **thấy tool mà không gọi** | **20.8%** |
| Gọi app khác hẳn → chọn/route sai app | 29.2% |
| Không gọi tool read/write nào → không điều tra gì | **50.0%** |

Bằng chứng sạch: `train_0039` gọi `chat_ott.list_chats` (app có trong prompt → tool gửi tin cũng có)
rồi `sys.ask_missing_info` và dừng; gold chỉ cần gửi tin. `train_0014` gọi `smart_home.get_status`
hai lần rồi `done`, không bật đèn.

**Phụ 1 — model thêm đối số, gần như không bao giờ thiếu.** Trên 78 call trùng tên gold: **thêm 56
khoá, thiếu 2**. Một chiều rõ rệt. Trên tool write (`ride_hailing.book.payment` 6,
`banking.transfer.note` 4, `alarm.create.repeat`/`label` 6) nó phá `state_hash`; trên tool read nó
phá F1 tool-call. Đây là thứ học từ ví dụ, khó giải thích bằng prompt → đúng việc của SFT.

**Phụ 2 — dạng thời gian.** Gold dùng ISO đầy đủ 455 lần / giờ treo tường 20 lần (95.8% ISO). Ta:
10 ISO / 4 treo tường. Sai tỉ lệ cao hơn nhưng tuyệt đối chỉ 4 ca — thật nhưng nhỏ, **không phải chỗ
nghẽn**. Đừng tiêu thời gian vào đây trước.

**Sửa lại khẳng định cũ của tài liệu này:** mục `local v4` nói routing là nguyên nhân lớn nhất.
Chính xác hơn — routing làm prompt thiếu app, có thật, nhưng bucket lớn nhất là **không hành động**,
và trong đó 0/48 là thiếu giấy phép, chỉ 29% có dấu vết sai app. Routing là một tầng riêng và nhỏ
hơn đã ghi.

**Hệ quả: nâng chất lượng ASR là đòn sai.** Kết quả null của chuẩn hoá số không phải do thiếu may
mắn — nó nhắm vào tầng không phải chỗ nghẽn.

⚠️ **Giới hạn phải nhớ:** `local_v4` chạy context 4096 / budget 1800 — cấu hình bị bóp. Một model
không thấy tool đúng cũng sẽ đi hỏi, nên **không tách sạch được** "không chịu làm" khỏi "không thấy
nên phải hỏi"; chỉ nhóm 20.8% là bằng chứng chắc. Thứ tách được hai tầng là **phiên 1 của Kaggle V2**
(context 12288, không route): hành vi bỏ việc còn → là hành vi, fine-tune là câu trả lời; sụp xuống →
là nạn đói prompt từ đầu. Chạy nó trước khi thêm thí nghiệm nào ở laptop.

Script: `scratchpad/why_hash.py`, `bucket_c.py`, `couldnt_or_wouldnt.py`, `argfmt.py` (tạm, chạy lại
được từ `results/local_v4/train_check.jsonl` + nhãn `train`).

---

## finetune v1 – QLoRA trên trajectory gold (2026-11-01, dữ liệu xong, **chưa huấn luyện**)

`scripts/build_sft.py` + `scripts/train_sft.py`. Vận hành: [`results/finetune_v1/README.md`](results/finetune_v1/README.md).

**Vì sao buộc phải fine-tune — một con số.** Trong phép thử chuẩn hoá số, 22 mẫu được đem so là
những mẫu router giữ **đủ toàn bộ app gold**. Vậy mà chỉ 2/22 mẫu được điểm, 18/22 lệch `state_hash`.
Route đúng cho gấp đôi điểm (11.36 vs 5.50) nhưng vẫn cách rất xa 22.52. Điều này sửa lại một khẳng
định trước đó của chính tài liệu này: routing là nguyên nhân **lớn nhất đã đo được**, nhưng sửa nó
một mình **không đủ** — model vẫn chọn sai tool và sai đối số khi đã có đủ công cụ trong tay. Prompt
đúng chỉ là điều kiện cần.

**Dữ liệu không cần tổng hợp gì.** Nhãn `train` đã là trajectory: `label.tool_calls` khớp 1:1 với
`label.tool_results` ở cả 1434 mẫu có nhãn. Mỗi mẫu replay thành hội thoại nhiều lượt, lượt cuối
`DONE` — quan trọng ngang các call, vì 111 mẫu có gold rỗng và model không học dừng sẽ luôn bịa thêm
một call rồi mất precision.

Huấn luyện trên **transcript ASR của chính ta**, không phải `audio_text` sạch — lúc chạy thật model
chỉ thấy đầu ra ASR ở ~24% WER. Chi phí gần như bằng 0:

| | `audio_text` (nhãn sạch) | **ASR thật** |
|---|---|---|
| Trung vị token | 5281 | **5306** |
| p95 / max | — / 9503 | 8011 / **9554** |
| Vừa seq 8192 | 96.4% | **96.0%** |

1284 train / 261 eval, 0 mẫu bị loại, chia theo sha256 của id nên build lại không rò rỉ eval.

**Mask loss là điểm sống còn, không phải tinh chỉnh.** Đo thật:

```
sft.train.jsonl: 1233 hội thoại, 51 bị bỏ vì >8192 token; giám sát 96168/6547604 token (1.5%)
sft.eval.jsonl:   243 hội thoại, 18 bị bỏ vì >8192 token; giám sát 18632/1327915 token (1.4%)
```

Không mask thì **98.5% gradient đổ vào học thuộc lại kết quả tool** — văn bản model không bao giờ
phải sinh ra, vì lúc chạy thật nó đến từ `ToolEnv` của BTC.

Mặt khác của cùng con số: ngân sách tín hiệu thật chỉ **96k token giám sát**, 2 epoch ≈ 192k. Ít.
Nên trông đợi là **chỉnh hành vi** (gọi đúng tool, đúng dạng đối số, biết dừng), không phải kiến
thức mới. `encode()` khôi phục các đoạn assistant
bằng cách render prefix tăng dần rồi diff, và **raise nếu template không prefix-stable** thay vì âm
thầm train sai. Kiểm bằng `--inspect N`, không cần GPU hay `peft`.

Mẫu dài hơn `--max-seq-len` bị **bỏ, không cắt**: cắt là mất đuôi, mà đuôi là các call cuối và `DONE`.

**Base là Instruct, không phải Thinking** — trajectory không có khối `<think>` nào, còn template bản
Thinking mở khối reasoning vô điều kiện và checkpoint được post-train để luôn suy luận; SFT trên
target không-reasoning là dạy nó dập đúng hành vi nó được huấn luyện. Cũng giữ Kaggle V2 (Thinking,
zero-shot) và bản fine-tune là **hai thí nghiệm khác nhau thật sự**.

Chưa có điểm. Đừng ghi kết luận vào đây trước khi `scripts/score.py` cho số.

---

## local v4 – thử ASR khác hẳn + sửa time leak thật của ASR (2026-10-31)

Mục tiêu người dùng đặt ra: **laptop và Kaggle không được chạy trùng nhau**. Notebook v9 đang chạy
`PhoWhisper-large`, nên lần này laptop phải thử một ASR **khác hẳn** để biết chênh lệch thật, chứ
không lặp lại chính model đã biết.

### Luật trước tiên — loại ngay những ứng viên không đủ điều kiện

Ngân sách tham số: hệ thống <6B, LLM đã chiếm 4.022B → **ASR phải <1.978B**. Mọi checkpoint phải
công khai trước 26/10/2026 và có giấy phép cho phép dùng.

| Ứng viên | Kết luận |
|---|---|
| `Qwen/Qwen3-ASR-1.7B` | **Loại** — tên ghi 1.7B nhưng thực tế **2.349B** → tổng 6.37B > 6B |
| `hynt/Zipformer-30M-RNNT-6000h` | **Loại** — `cc-by-nc-nd-4.0` (NonCommercial + NoDerivatives) |
| `nguyenvulebinh/wav2vec2-large-vi-vlsp2020` | **Loại** — `cc-by-nc-4.0` |
| `nvidia/parakeet-tdt-0.6b-v3` | **Loại** — 25 thứ tiếng châu Âu, **không có tiếng Việt** |
| `mistralai/Voxtral-Mini-4B` | **Loại** — 4B, cộng LLM là 8B |
| `erax-ai/EraX-WoW-Turbo-V1.1` | **Nhận** — MIT, 809M, 2025-03-31, fine-tune tiếng Việt từ whisper-large-v3-turbo |
| `openai/whisper-large-v3-turbo` | **Nhận làm đối chứng** — MIT, 809M, cùng kiến trúc nhưng *không* fine-tune tiếng Việt |

Lý do chọn EraX làm ứng viên chính: cùng encoder 32 lớp / d_model 1280 như PhoWhisper-large, nhưng
**decoder 4 lớp thay vì 32**. Decode tự hồi quy là phần đắt nhất trên CPU, nên về lý thuyết nhanh
hơn nhiều mà vẫn được huấn luyện cho tiếng Việt (8 vùng miền, ~1000h). Cùng API `transformers` →
không thêm dependency nào, không rủi ro khi đóng băng Docker (bài học `autoawq` ở v2).

### Đo WER thật trên split `train` (có nhãn `audio_text`, không tốn lượt nộp)

12 mẫu rải đều 1545 mẫu `train`, cùng audio, cùng cấu hình. WER = word-level edit distance sau khi
bỏ dấu câu và lowercase.

| ASR | Params | WER | CPU s/file | GPU s/file |
|---|---|---|---|---|
| **`vinai/PhoWhisper-large`** | 1.543B | **23.59%** | 24.5 | **3.1** |
| `erax-ai/EraX-WoW-Turbo-V1.1` | 809M | 28.17% | 13.1 | 0.9 |
| `openai/whisper-large-v3-turbo` | 809M | 28.87% | — | 1.0 |

**Kết quả âm — và đây mới là phần có giá trị:** EraX nhanh 1.9x (CPU) nhưng WER tệ hơn **4.6 điểm**.
Model đối chứng trả lời dứt điểm vì sao: vanilla turbo 28.87% so với EraX 28.17% → **việc fine-tune
tiếng Việt của EraX chỉ mua được 0.7 điểm**, gần như không có gì. Thứ thật sự tạo ra khoảng cách là
**độ sâu decoder** (32 lớp của PhoWhisper so với 4 lớp của turbo), không phải dữ liệu huấn luyện.

Hệ quả: **không đổi ASR.** Giữ `PhoWhisper-large`. Đổi sang EraX sẽ tiết kiệm ~20% thời gian để đánh
đổi 4.6 điểm WER — mà WER hỏng là mất thẳng tool call và `state_hash`, tức mất điểm mẫu.

Lưu ý khi đọc bảng: turbo viết số bằng chữ số (`50.000`, `8h`) còn PhoWhisper viết bằng chữ, nên WER
phạt turbo oan một phần (vd. `train_1144`: turbo 58.3% vs Pho 25% phần lớn do định dạng số). Dù trừ
đi phần này, PhoWhisper vẫn dẫn rõ ràng.

### Time leak thật: ASR bị đẩy xuống CPU, không phải model sai

Chỗ này là một sai sót của chính tôi ở v3. Comment trong `src/asr.py` ghi *"ASR is only ~1% of a
sample's time"* — đúng ở thời điểm 1 lượt LLM mất 900s, nhưng sau khi tối ưu LLM thì con số thật là:

```
15 mẫu đã chạy: tổng 893s = ASR 357s (40%) + LLM 535s (60%)
```

**ASR đã thành 40% thời gian mà vẫn bị để trên CPU.** Và chi phí gần như **cố định theo số file**,
không theo độ dài audio (~16s/file), vì Whisper luôn pad mel lên 30s — nên mẫu có 2 lượt history
audio đắt gấp ba mẫu không có.

Cùng một transcript, GPU nhanh hơn CPU **8x** (3.1s vs 24.5s/file, WER y hệt 23.59%). Nhưng trên card
4GB thì ASR fp16 (3.1GB) và LLM 4-bit (2.3GB) không thể cùng nằm trong VRAM — ai tràn ra thì chậm 8x.

**Giải pháp: tách thành hai pha.** Transcribe toàn bộ trước, khi ASR được dùng cả card một mình; giải
phóng trọng số; rồi mới nạp LLM. A30 giữ được cả hai cùng lúc nên không cần việc này — `--asr-phase
inline` giữ đúng đường chạy của BTC/notebook.

- `src/asr.py`: thêm `ASR_MODELS` (repo → revision pin, kèm kiểm tra giấy phép), đổi `PhoWhisperASR`
  → `WhisperASR` (giờ phục vụ 3 checkpoint cùng API), thêm `close()` và `CachedASR`.
- `src/pipeline.py`: tách `audio_paths()` thành một nguồn duy nhất cho cả hai pha.
- `scripts/infer.py`: `--asr-phase {auto,inline,prefetch}`, `--transcript-cache` (ghi dần, chạy lại
  không phải transcribe lại), `--asr-model` nhận nhiều checkpoint và **từ chối repo chưa pin revision**.
- Metrics thêm `asr_revision`, `asr_phase`; `seconds` và `over_30_seconds` vẫn cộng cả thời gian ASR
  của pha trước, để không tự lừa mình là đã đạt mốc 30s.

### Kết quả (đúng 5 mẫu của v3, so trực tiếp)

| Mẫu | v3 tổng / ASR | v4 tổng / ASR |
|---|---|---|
| public_0002 | 40.4 / 17.8 | 35.5 / 5.4 |
| public_0003 | 30.2 / 21.0 | 16.5 / 7.0 |
| public_0004 | 40.0 / 16.4 | 28.1 / 2.9 |
| public_0005 | 47.1 / 16.5 | 35.1 / 3.0 |
| public_0006 | 95.8 / 41.3 | 56.0 / 5.6 |
| **trung bình** | **50.7s** | **34.2s** |

ASR nhanh **4.7x** (22.6s → 4.8s/mẫu), tổng nhanh **1.48x** → **600 mẫu: 8.4h → 5.7h**.
**`tool_calls` giống hệt v3 trên cả 5 mẫu** — thuần tốc độ, không đánh đổi chất lượng gì.

### Còn tồn đọng (đừng lặp lại những gì ở trên)

- **ASR là trần điểm thật, không phải routing.** `public_0003` ra "nếu may mới thì dời chạy vốn sáng
  may mốn hai tiếng nhắn nhắt em nhắn báo giờ mới ra hư nha không mới thì thôi" — routing chọn sai
  app (`ride_hailing`) vì transcript không còn thông tin gì để chọn đúng. WER 23.59% trên `train`
  nghĩa là **cứ 4 từ sai 1 từ**; sửa routing không cứu được. Hướng chưa thử: ghép nhiều ASR
  (PhoWhisper + turbo) rồi chọn/hợp nhất, hoặc đưa cả 2 transcript cho LLM tự chọn — đổi thời gian
  lấy độ chính xác, giờ đã có ngân sách thời gian để làm.
- **`public_0006` trả lời bằng văn xuôi rồi dừng** ("Mưa không xảy ra vào ngày mai...") thay vì thực
  hiện phần việc còn lại. Chưa rõ gold label có yêu cầu hành động không → cần chạy `train` có nhãn.
- **Vẫn chưa chạy `train` để đo F1 thật.** Tất cả kết luận về chất lượng tới giờ là suy đoán từ
  transcript và tool call, chưa đối chiếu nhãn. Đây là việc đáng làm nhất tiếp theo, **không tốn lượt nộp**.
- `public_0009` từng `stop_reason=error` trong lần chạy 10 mẫu — chưa truy nguyên.
- Notebook v9 vẫn mang bug parser của v2 (`_parse_decision` nhận tool call dạng văn xuôi là câu trả
  lời cuối → nộp 0 tool call). Chưa port.

---

## local v3 – tối ưu tốc độ để laptop chạy đủ tập <24h (2026-10-31)

Mục tiêu: biến laptop RTX 2050 4GB thành công cụ thử nghiệm thật — chạy đủ 600 mẫu `public_test`
trong dưới 24 giờ. Điểm khởi đầu: **1027s/mẫu ≈ 171 giờ** (v2 bf16, context 4096).

### Đo để tìm nghẽn thật, không đoán

Phân tích metrics v2 cho thấy **ASR chỉ chiếm 1%** (15-17s), LLM chiếm 99% (688-916s **mỗi lượt**).
Giả thuyết đầu tiên của tôi — "trọng số bf16 8GB tràn khỏi 4GB VRAM nên stream qua PCIe mỗi token" —
**đã sai**: chuyển sang 4-bit (2.3GB, vừa VRAM) chỉ nhanh hơn 2x (437s/lượt), không phải 100x.

Benchmark trực tiếp (4-bit Qwen3-4B, 32 token mới, RTX 2050 4GB) mới chỉ ra nghẽn thật:

| Độ dài prompt | Thời gian | Peak VRAM |
|---|---|---|
| 64 token | 3.0s | 2.52 GiB |
| 1.024 | 5.5s | 3.05 GiB |
| 2.048 | 28.3s | 4.11 GiB |
| **3.900** (đang gửi thật) | **252.5s** | **7.54 GiB** |
| 64 token, 128 token mới | 11.3s (~88ms/token) | 2.54 GiB |

**Kết luận: decode không chậm (88ms/token). Prefill nổ theo độ dài prompt** vì attention vượt VRAM.
Và prompt 3.904 token gần như toàn bộ là schema của 34 tool — đúng thứ `docs/context.txt` đã khuyến
nghị cắt từ đầu ("Dynamic Schema Pruning: 6.000 → 1.200 token"), mà code chỉ cắt khi *vượt* context
nên cứ gửi nguyên 3.904.

Phát hiện phụ: **PyTorch bản Windows này không có Flash Attention** (`Torch was not compiled with
flash attention`), chỉ có mem-efficient và math. Math backend materialize toàn bộ ma trận attention
→ giải thích peak 7.54 GiB.

### Thay đổi

1. **Cắt prompt theo budget token** (`--prompt-token-budget`, mặc định 1800) thay vì chỉ cắt khi
   vượt context. Đây là đòn tăng tốc chính.
2. **Xếp hạng theo cấp ứng dụng bằng n-gram ký tự tiếng Việt**: bản cũ so transcript tiếng Việt với
   tên tool tiếng Anh (`alarm.update`) → trùng từ ≈ 0 → chọn gần như ngẫu nhiên, cắt mất đúng tool
   cần dùng. Bản mới dùng `name_vi`/`description` tiếng Việt có sẵn trong catalog, so bằng 3-gram ký
   tự sau khi bỏ dấu (chịu được lỗi ASR: "bảo thích" vs "báo thức" vẫn trùng nhiều 3-gram), và giữ
   **nguyên cả bộ tool của app** để không mất cặp read+write.
3. **Buộc backend SDPA mem-efficient** (vì bản Windows thiếu Flash) để giảm peak VRAM attention.
4. **Nén schema mạnh hơn**: bỏ đuôi `Returns: ...`, mô tả tool cắt còn 60 ký tự → thêm được cả app
   vào cùng budget. Độ phủ app quan trọng hơn văn mô tả từng tool.
5. **SYSTEM_PROMPT**: bỏ câu "Suy luận ngắn gọn trước khi chọn hành động" (nó *mời* model viết dài,
   gây lỗi `LLM output hit token limit` ở 2/3 mẫu) → thay bằng yêu cầu chỉ xuất tool call hoặc DONE.
6. **Sửa bug parser**: model đôi khi xuất tool call dạng văn xuôi
   `sys.ask_missing_info({"tool_name": ...})`; bản cũ nhận đó là "câu trả lời cuối" → nộp **0 tool
   call** và ăn 0 điểm trong im lặng. Nay khôi phục được, và nếu không parse nổi thì báo lỗi rõ.
7. Thêm cờ `--llm-precision {auto,bf16,4bit}` và `--asr-device {auto,cuda,cpu}` — chọn hiện rõ, có
   log cảnh báo, ghi vào metrics; không hạ chất lượng âm thầm.

### Kết quả đo (5 mẫu `public_test`, RTX 2050 4GB, 4-bit + ASR CPU)

| Mẫu | Tổng | ASR | LLM | Lượt | Kết thúc |
|---|---|---|---|---|---|
| public_0002 | 40s | 18s | 23s | 3 | done (`get_quotes` → `book`) |
| public_0003 | 30s | 21s | 9s | 1 | asked_missing |
| public_0004 | 40s | 16s | 24s | 1 | asked_missing |
| public_0005 | 47s | 16s | 31s | 1 | asked_missing |
| public_0006 | 96s | 41s | 54s | 3 | done |

**Trung bình 51s/mẫu → ~8.4 giờ cho 600 mẫu** (xấu nhất 16h). **0 mẫu lỗi.**
Tăng tốc **20x** so với v2 (1027s → 51s). Đạt mục tiêu <24h.

### Còn tồn đọng (việc cho lần chạy sau, đừng lặp lại những gì đã làm ở trên)

- **Routing vẫn sai app ở một số mẫu**: `public_0003` ("nếu mai mưa thì dời chạy bộ... nhắn báo giờ")
  bị route sang `ride_hailing`. Budget 1800 có mẫu chỉ đủ 1 app → sai 1 lần là mất trắng mẫu đó.
  Hướng chưa thử: route 2 bước (hỏi LLM chọn app trước), hoặc ép giữ tối thiểu 3 app, hoặc nén schema
  sang định dạng gọn hơn JSON Schema.
- **Nhiều mẫu ra `asked_missing`** (3/5) — cần đối chiếu gold xem có đúng là thiếu thông tin thật hay
  model bỏ cuộc quá sớm. Chạy trên split `train` (có nhãn) sẽ trả lời được mà không tốn lượt nộp.
- **Cấu hình local (4-bit + prompt cắt 1800) khác bản nộp A30 (bf16, có thể để prompt rộng hơn)** →
  điểm local không dự đoán chính xác điểm leaderboard. Muốn đối chiếu công bằng thì chạy Kaggle.

---

## local/kaggle unified v2 – ASR đổi sang PhoWhisper-large (2026-10-31, đang thử nghiệm)

### Research — vì sao đổi ASR lần 2 (Qwen3-ASR-0.6B → PhoWhisper-large)

Yêu cầu: mỗi lần chạy (local hay Kaggle) phải là một thử nghiệm thật, ảnh hưởng rõ đến điểm, không
chỉ sửa vài dòng không đáng kể. Đã khảo sát 2 hướng:

**Hướng 1 — AWQ-quantize LLM để chạy nhanh trên máy yếu**: thử `autoawq` + checkpoint
`cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit`. **Bỏ** vì: autoawq chính thức *deprecated*, tác giả nói
rõ "no longer be maintained", bản test cuối của họ là Torch 2.6/Transformers 4.51 (máy ta đang
2.10/5.13 — cách 2 major version), cần cài thêm `datasets` mới import được, cảnh báo thiếu
`triton`/`zstandard`. Rủi ro gãy giữa lúc đóng băng Docker nộp bài (hạn 16/11) là không chấp nhận
được, trong khi lợi ích chỉ là tốc độ **trên máy dev yếu** — A30 thật (24GB) đã đủ nhanh với bf16
theo chính ước tính của `docs/context.txt` (~2.5s/mẫu). Không mang lại điểm số, chỉ mang lại rủi ro.

**Hướng 2 — đổi ASR sang model Vietnamese-specific**: tìm thấy benchmark ngoài
([TSPC, arXiv 2509.05983](https://arxiv.org/pdf/2509.05983)) so `vinai/PhoWhisper-base` (74M, train
riêng cho tiếng Việt) với `Qwen/Qwen3-ASR-0.6B` (782M, đa ngôn ngữ): **PhoWhisper-base WER 14.05 vs
Qwen3-ASR-0.6B WER 22.47** trên cùng test Việt — chênh rất lớn dù PhoWhisper-base nhỏ hơn 10 lần.
Giấy của PhoWhisper tự báo cáo bản `-large` là SOTA trên các benchmark tiếng Việt khi phát hành.

**Quyết định**: đổi ASR sang `vinai/PhoWhisper-large` (1.55B, BSD-3-Clause, phát hành 2023-12,
cập nhật lần cuối 2024-11 — chắc chắn trước hạn 26/10/2026). Tổng hệ thống
1.55B (ASR) + 4.02B (LLM) = **5.57B < 6B**, LLM vẫn 4.02B < 5B — đúng luật. Đây là thay đổi tác động
thật đến WER/F1 cuối, không phải chỉnh sửa nhỏ. Không đổi LLM lần này (giữ Qwen3-4B-Instruct-2507
bf16, đã xác minh hỗ trợ tool-calling qua `apply_chat_template`).

### Thay đổi code
- `src/asr.py`: `Qwen3ASR` → `PhoWhisperASR`, dùng kiến trúc Whisper chuẩn
  (`AutoModelForSpeechSeq2Seq` + `AutoProcessor`, API `generate()`/`batch_decode()` quen thuộc, đã
  dùng ổn định nhiều năm trong `transformers`) thay cho API `Qwen3ASRForConditionalGeneration` rất
  mới và hẹp người dùng. Vẫn giữ batch toàn bộ audio 1 mẫu trong 1 lượt GPU, vẫn giữ auto-fallback
  CPU khi VRAM < 6GiB free (máy dev yếu).
- `src/pipeline.py`, `scripts/infer.py`: đổi type hint/import theo tên class mới.
- `lifemate_kaggle.ipynb`: *(đang cập nhật — xem ghi chú dưới, chưa hoàn tất khi viết dòng này)*.

### Trạng thái thử nghiệm — ĐÃ XÁC NHẬN đúng trên local (public_0002)

So trực tiếp transcript cùng 1 mẫu (`public_0002`, audio: "Đặt xe bốn chỗ thử công ty đi khám răng
chiều nay, đặt ngay bây giờ nhé!"):

| ASR | Transcript | Đúng? |
|---|---|---|
| PhoASR-whisper-small (baseline 22.52, đã xoá) | "Đặt xe bốn chỗ thử công ty đi khám **răng** chiều nay, đặt ngay bây giờ nhé!" | Đúng |
| Qwen3-ASR-0.6B-hf (v2 trước khi sửa, đã xoá) | "đặt xe bốn chỗ ở tại công ty đi khám **gan** chiều nay đặt ngay bây giờ nhé" | **Sai** (răng→gan) |
| **PhoWhisper-large (v2 hiện tại)** | "đặt xe bốn chỗ thử công ty đi khám **răng** chiều nay đặt ngay bây giờ nhé." | **Đúng**, khớp cả câu chữ với baseline |

Xác nhận nghiên cứu đúng: PhoWhisper-large sửa đúng lỗi mà Qwen3-ASR-0.6B mắc, khớp hoàn toàn với
baseline PhoASR cũ — không đánh đổi chất lượng ASR để đổi kiến trúc.

- **Bug phát hiện khi tích hợp**: `src/models.py::model_path()` chỉ nhận file `*.safetensors` làm
  "trọng số hợp lệ"; PhoWhisper-large chỉ phát hành `pytorch_model.bin` (không có safetensors) nên
  bị báo "Weights missing" sai. Đã sửa: nhận cả `*.bin`.
- **Local**: ASR chạy đúng (CPU, VRAM dồn cho LLM). LLM 4-bit NF4 trên RTX 2050 4GB vẫn rất chậm
  (920s cho 1 lượt, giống trước khi đổi ASR) — **không phải do PhoWhisper**, là hạn chế phần cứng đã
  biết của máy dev, không đại diện tốc độ thật trên A30 (xem mục trước).
- **Kaggle**: đã đồng bộ (`lifemate_kaggle.ipynb` v9, `SpeechModel` dùng `AutoModelForSpeechSeq2Seq`
  chuẩn thay Qwen3ASR, cùng model `vinai/PhoWhisper-large` với local)
- **Điểm**: *chờ chạy đủ `public_test` và nộp để có điểm leaderboard đối chiếu với baseline 22.52*

### Quyết định bổ sung (2026-10-31, theo yêu cầu người dùng): không hạ precision để "vừa" máy yếu

Ban đầu đã thêm auto-fallback: LLM xuống 4-bit NF4 (bitsandbytes) và ASR chạy CPU khi VRAM <10GiB/
<6GiB free, để tránh OOM trên RTX 2050 4GB. **Người dùng yêu cầu bỏ** — tổng hệ thống 6B tham số
(bf16 ≈ 11GB) vẫn khả thi trên "4GB VRAM + 15GB shared RAM" nếu chấp nhận chậm; không muốn đổi sang
model/độ chính xác yếu hơn chỉ vì máy dev yếu, muốn một thử nghiệm nghiêm túc khớp đúng cấu hình
sẽ dùng trên A30 (bf16 đầy đủ), khác biệt chỉ nên là **thời gian**, không phải **độ chính xác**.

**Đã gỡ**: `BitsAndBytesConfig` 4-bit trong `LLMAgent`, CPU-fallback theo VRAM trong `PhoWhisperASR`,
`bitsandbytes` khỏi `requirements.txt`. Cả 2 model giờ luôn load bf16/fp16 trên GPU.

**Hệ quả kỹ thuật phát hiện (ban đầu nghi sai)**: tổng 2 model bf16 (~11GB) vượt xa 4GB VRAM vật lý.
Ban đầu nghi cần bật driver NVIDIA "CUDA Sysmem Fallback Policy" — **sai một phần**: Windows/driver
**đã tự tràn sang RAM mà không cần chỉnh gì** (load model 11.36GB thành công nhiều lần, đúng như
người dùng phản biện). Thứ thật sự gây `torch.OutOfMemoryError` là **context quá lớn (8192 token)**
khiến 1 lượt attention của LLM cần cấp phát tức thời 5.93GB — vượt ngưỡng cấp phát khả dụng tại thời
điểm đó, lặp lại y hệt dù đã thêm `PYTORCH_ALLOC_CONF=expandable_segments:True` (biến này **không
được hỗ trợ trên Windows**: log báo rõ "expandable_segments not supported on this platform").

### ĐÃ CHẠY THÀNH CÔNG — bf16 đầy đủ, không giảm chất lượng, không cần đổi driver

Fix thật: giảm `--context-size 4096 --max-new-tokens 192` (đúng bằng config Kaggle T4 trong
`lifemate_kaggle.ipynb`) — **đã đổi thành default** trong `scripts/infer.py`. Kết quả 2 mẫu
`public_test` (2026-10-31, máy RTX 2050 4GB + 32GB RAM):

| Mẫu | Kết quả | Thời gian | Nhận xét |
|---|---|---|---|
| `public_0002` | `done`, 2 lượt LLM — gọi `ride_hailing.book` đúng tham số rồi trả lời tự nhiên | 1846s (~31 phút) | ⚠️ gọi tool ngay lượt 1 không `read` trước; địa chỉ đón/trả suy luận từ context, cần đối chiếu gold khi có điểm |
| `public_0003` | `asked_missing`, 1 lượt — gọi đúng `sys.ask_missing_info(alarm.update, [time,label])` | 706s (~12 phút) | Đúng quy tắc: thiếu param bắt buộc → hỏi, không bịa |

**Không có mẫu nào fail.** Xác nhận: hệ thống <6B tham số, bf16 đầy đủ (không 4-bit), chạy được trên
GPU 4GB + RAM 32GB thật — đúng như người dùng kỳ vọng, chỉ khác thời gian so với A30, không khác độ
chính xác. Đã cập nhật default `--context-size`/`--max-new-tokens` để lần chạy sau không cần truyền
tay flag này nữa.

**Việc KHÔNG cần làm** (đã thử, không có tác dụng hoặc không cần thiết):
- ❌ Bật "CUDA Sysmem Fallback Policy" trong NVIDIA Control Panel — chưa thử nhưng có khả năng không
  cần, vì vấn đề là context quá lớn chứ không phải chính sách tràn bộ nhớ
- ❌ `PYTORCH_ALLOC_CONF=expandable_segments:True` — không được hỗ trợ trên Windows, giữ lại trong
  code (vô hại, có thể hữu ích trên Linux/Kaggle) nhưng không phải fix chính

---

## v1–v4 (local, baseline)
- **Pipeline**: Whisper + Qwen2.5-3B llama.cpp
- **Điểm**: chưa ghi nhận / thử nghiệm ban đầu

---

## local – PhoASR-small + Qwen3.5-4B Q8 (llama.cpp)
- **Script**: `python scripts/infer.py --split public_test --output submission.jsonl`
- **ASR**: `Qualcomm-AI-Research/PhoASR-whisper-small` (rev `da2d9ae`)
- **LLM**: `unsloth/Qwen3.5-4B-GGUF` → `Qwen3.5-4B-Q8_0.gguf` (rev `e87f176`) qua llama-server
- **Điểm chính thức**: **22.52000**
- **Thời gian/mẫu**: 50–471 s (trung bình ~167 s) — *vượt giới hạn 30 s; chỉ để đo chất lượng*
- **VRAM dùng**: 787 MB (llama.cpp chạy phần lớn trên CPU, swap RAM)
- **RAM process tree**: 6.6–8.2 GB
- **stop_reason phổ biến**: `done` (37/50), `step_limit` (3), `asked_missing` (5), `empty_response` (4), `repeated_invalid` (1)
- **Ghi chú**: Điểm cải thiện đáng kể so với local trước. LLM rất chậm vì chạy CPU llama.cpp.

---

## kaggle – `qwen_asr06_qwen4_v7` (chưa hoàn chỉnh)
- **Notebook**: `lifemate_kaggle.ipynb`
- **ASR**: `Qwen/Qwen3-ASR-0.6B-hf` (rev `7f1569a`) — bfloat16 GPU
- **LLM**: `Qwen/Qwen3-4B-Instruct-2507` (rev `cdbee75`) — bfloat16 GPU
- **Điểm chính thức**: *Chưa có* (notebook không chạy hoàn chỉnh lần nào)
- **Lỗi phát hiện**:
  1. `torch.use_deterministic_algorithms(True, warn_only=False)` → ném RuntimeError khi warm-up vì nhiều CUDA op không deterministic
  2. `processor.decode(generated_2D_tensor, ...)` → decode toàn bộ batch như một chuỗi, không trả về list → `zip(paths, texts)` lặp qua ký tự thay vì string
  3. `assert counts == {'asr': 782426112, 'llm': 4022468096}` cứng — fail nếu revision thay đổi nhỏ

---

## kaggle – `qwen_asr06_qwen4_v8` (bản mới, đang chạy)
- **Notebook**: `lifemate_kaggle.ipynb` (cập nhật 2026-10-31)
- **ASR / LLM**: giữ nguyên Qwen3-ASR-0.6B + Qwen3-4B-Instruct-2507
- **Sửa so với v7**:
  - `warn_only=True` → không crash khi warm-up
  - Decode ASR theo từng hàng: `[processor.decode(generated[k], ...) for k in range(batch)]`
  - `assert counts ==` → `logger.warning` (soft check, giới hạn 5B/6B vẫn còn)
  - `'code_hash'` tĩnh → `'pipeline_version': 'v8'` (tránh fingerprint lỗi khi notebook thay đổi)
- **Điểm**: *Chờ kết quả*
- **Kỳ vọng**: Qwen3-4B-Instruct-2507 full precision bfloat16 tốt hơn Q8 quantized; ASR 0.6B mới hơn PhoASR-small

---

## local/kaggle unified – `transformers` GPU, Qwen3-ASR-0.6B + Qwen3-4B-Instruct-2507 (2026-10-31)

### Research — vì sao đổi model/backend

Luật cuộc thi (`docs/context.txt`): hệ thống <6B tham số, LLM <5B, context ≤16384, chạy offline,
**30 giây/mẫu trên 1×A30 24GB + Xeon 16 core + 64GB RAM**, decode tất định (greedy/temperature=0),
chỉ dùng model/license mở phát hành trước 26/10/2026, không fine-tune bằng nhãn public/private test.

**Phát hiện quan trọng**: baseline 22.52 điểm (local, llama.cpp CPU) chạy 50–471 giây/mẫu —
**vượt xa giới hạn 30 giây của BTC**. Điểm 22.52 chỉ phản ánh chất lượng tool-calling khi không bị
giới hạn thời gian; nếu nộp đúng luật, toàn bộ mẫu sẽ bị tính 0 điểm vì time-out. Đây là lỗi nghiêm
trọng hơn bất kỳ lỗi notebook Kaggle — pipeline cũ **không khả thi để nộp bài**.

Nguyên nhân: `llm_agent.py` cũ chạy Qwen3.5-4B **Q8 GGUF qua llama-server trên CPU** (chỉ 787MB
VRAM dùng, phần lớn tính toán rơi vào CPU/RAM) → chậm gấp hàng chục lần so với chạy thẳng trên GPU.

**Xác minh model thay thế** (tra trực tiếp model card HuggingFace):

| Model | Tham số | License | Ngày | Hỗ trợ |
|---|---|---|---|---|
| `Qwen/Qwen3-4B-Instruct-2507` | 4.0B (BF16) | Apache-2.0 | kỹ thuật báo cáo 05/2025 | context 262K native, tool-calling qua `apply_chat_template(tools=...)` |
| `Qwen/Qwen3-ASR-0.6B-hf` | 0.8B (BF16) | Apache-2.0 | bài báo 01/2026, bảng WER 06/2026 | 30 ngôn ngữ gồm tiếng Việt, batch nhiều audio 1 lượt qua `apply_transcription_request`, cần `transformers>=5.13.0` |
| `Qualcomm-AI-Research/PhoASR-whisper-small` (cũ) | 0.2B (F32) | BSD-3-Clause-Clear + Qualcomm RAI | EACL 2026 | fine-tune whisper-small trên 3000h, không có WER tiếng Việt công khai |

**Quyết định**: chuyển cả 2 model sang chạy thẳng qua `transformers` trên GPU (bf16), bỏ hoàn toàn
llama.cpp. Tổng tham số 4.8B (< 6B), LLM 4.0B (< 5B) — đúng luật. Thống nhất code local và Kaggle
(cùng 2 model), xoá sự khác biệt từng gây khó bảo trì.

### Thay đổi code
- `src/asr.py`: `WhisperASR` → `Qwen3ASR`, batch toàn bộ audio của 1 mẫu (request + history) trong
  một lượt GPU duy nhất (giảm số lần gọi model, khớp cách Kaggle notebook đã làm).
- `src/llm_agent.py`: bỏ subprocess llama-server, dùng `AutoModelForCausalLM` + KV-cache reuse giữa
  các lượt tool trong cùng 1 mẫu (copy logic từ Kaggle `ToolModel.decide`, đã kiểm tra tính đúng).
- `src/models.py` (mới): tiện ích tải/định vị checkpoint HF dùng chung cho ASR và LLM.
- `src/pipeline.py`: gọi `llm_agent.reset()` đầu mỗi mẫu (bắt buộc vì giờ cache KV là state có thật
  trên GPU, không như llama-server tự quản lý theo slot); ASR giờ gọi `transcribe_batch` 1 lần.
- `scripts/infer.py`: bỏ `--gpu-layers`, `--thinking-tokens`, `--asr-beams` (không còn áp dụng);
  `--max-seconds` mặc định đổi từ 600 → 30 để khớp luật thi thay vì chế độ "nghiên cứu chất lượng".
- GPU/CPU: không cấu hình gì thêm — `.to("cuda")`; máy yếu VRAM sẽ tự swap sang shared GPU memory
  của Windows (chậm hơn nhưng không cần chỉnh layer CPU/GPU thủ công như llama.cpp cũ).

### Baseline cũ để đối chiếu
File metrics gốc (50 mẫu public_test, llama.cpp CPU, 22.52 điểm) đã được chép lại nguyên văn phía
trên mục *"local – PhoASR-small + Qwen3.5-4B Q8 (llama.cpp)"* — dùng mục đó để so sánh **chất lượng
tool-calling** (không so sánh thời gian, vì bản cũ không đúng luật 30s).

- **Điểm chính thức**: *chờ chạy `scripts/infer.py --split public_test` và nộp để lấy điểm leaderboard*
- **Cách đối chiếu khi có kết quả mới**:
  1. So `stop_reason` theo từng `id` giữa `runs/baseline_llamacpp_qwen3.5-4b_q8/submission.metrics.jsonl`
     (nếu còn giữ bản chép) và `submission.metrics.jsonl` mới — tìm mẫu đổi từ `done`→lỗi hoặc ngược lại.
  2. So `transcript` (ASR) giữa hai bản để xem Qwen3-ASR-0.6B có đọc đúng hơn PhoASR-whisper-small không.
  3. Kiểm `seconds` mới phải <30s mọi mẫu trên GPU thật (A30/laptop RTX) — nếu không, giảm
     `--max-new-tokens`/`--context-size` hoặc bớt `--max-steps` trước khi nộp.
  4. Điểm leaderboard công khai (Zalo AI Challenge portal) là thước đo chính thức duy nhất, log local
     chỉ để debug nguyên nhân tăng/giảm.

---

## Bài học rút ra

| Vấn đề | Nguyên nhân | Cách xử lý |
|--------|-------------|------------|
| `use_deterministic_algorithms` crash | Nhiều kernel CUDA không có bản deterministic | `warn_only=True` hoặc bỏ |
| `processor.decode(2D_tensor)` | API nhận 1D, batch cần lặp | Loop `decode(row)` hoặc `batch_decode` |
| Assert cứng trên số tham số | Revision model có thể thay đổi nhẹ | Chuyển thành warning, chỉ giữ kiểm tra giới hạn tổng |
| llama.cpp CPU quá chậm | Llama.cpp không offload VRAM đủ trên Windows | Dùng transformers full GPU (Kaggle A30/T4) |
| `code_hash` tĩnh trong fingerprint | Hash tính tay, lỗi thời khi sửa notebook | Dùng `pipeline_version` string |
| Điểm local cao nhưng không khớp luật | Chỉ đo chất lượng, quên giới hạn 30s/mẫu của BTC | Luôn đối chiếu `seconds` trong metrics với 30s trước khi tin điểm |
| llama.cpp CPU vs transformers GPU | llama.cpp tiện cho máy yếu nhưng sai hướng tối ưu (BTC chạy A30 GPU) | Nhắm đúng hardware chấm điểm (A30), chạy local bằng GPU + shared VRAM tự nhiên |
| 5 lần chạy Kaggle thất bại liên tiếp (V11–V15) | Tôi kiểm **code** mình viết nhưng không kiểm **khẳng định** mình dựng trên đó: con số chưa tính, pattern chép tay thiếu, cấu hình chọn theo lập luận chứ không theo phép đo | Đo trên model thật trước khi đưa người dùng chạy. Hai checker offline (`check_notebook_order.py`, `verify_notebook.py`) chặn lớp lỗi "chỉ lộ sau khi tải 13 GiB trọng số" |
| Con số sai trong tài liệu, hai lần liên tiếp (5.28 GiB dư → thực ra OOM) | Viết phép tính thiếu một hạng tử (workspace attention) rồi dùng nó để ra quyết định | **Đừng viết một con số vào tài liệu nếu chưa tính nó**, và khi đã tính thì liệt kê đủ hạng tử dưới dạng bảng để chỗ thiếu lộ ra |
| Tin comment trong thư viện | Comment của `transformers` nói `enable_gqa` *tránh* math kernel; đo được điều ngược lại | Benchmark, đừng đọc comment. Ba lần trong dự án này tôi đoán sai cơ chế và phép đo nói ngược lại |
| Chọn model theo lập luận tương quan | "22.52 có reasoning, 13.41 không reasoning → bật reasoning" bỏ qua việc hai model reasoning rất khác nhau (hybrid 512 token vs checkpoint chuyên suy luận) | Ràng buộc cứng trước (fine-tune dùng checkpoint nào?), tương quan sau |
