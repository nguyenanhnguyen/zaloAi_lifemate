# Chạy trên Kaggle — hướng dẫn và suy luận sau điểm 11.18

Tài liệu **duy nhất** của thư mục này. Notebook thì `.gitignore` không bỏ, nhưng cách chạy
thì không nằm trong notebook.

```
notebook/
  README.md                     <- bạn đang đọc
  lifemate_kaggle_v2.ipynb      <- notebook suy luận, có MÀNG CHẮN ở cell 3
  lifemate_kaggle_train.ipynb   <- notebook huấn luyện (đã sinh ra lora_V4)
  lifemate_kaggle_v1.ipynb      <- bản đã nộp được 13.41, giữ để tham chiếu
```

## Suy luận lại sau 11.18

[`local_v10`](../results/local_v10/README.md) nộp 634 mẫu `public_test` với adapter và được
**11.18**, thấp hơn [`kaggle_v1`](../results/kaggle_v1/README.md) = **13.41** là 2.23 điểm.

Bóc tách khoảng cách đó bằng số đã có:

| | |
|---|---|
| điểm/mẫu trung bình | 0.0176 |
| điểm/mẫu trên 604 mẫu **không** lỗi | 0.0185 |
| trần nếu sửa **hết** 30 mẫu OOM | **11.74** |
| khoảng cách còn lại so với 13.41 | **1.67** |

**OOM chỉ giải thích được 0.56 trong 2.23.** Còn 1.67 thuộc về chỗ khác, và nghi phạm chính
là **prompt bị cắt tool**: `local_v10` có 62 mẫu nộp 0 tool call, `kaggle_v1` chỉ 42. Ngân
sách 2000 token ép `fit_tools` xuống còn một app, nên nhiều mẫu không hề *thấy* tool nó cần.

Đó là tin tốt cho Kaggle, vì ràng buộc ấy **không tồn tại** ở đó: T4 ×2 = 30 GiB, notebook
V2 gửi **cả** catalog và không route (`MAX_CONTEXT = 12288`). Biến nghi ngờ nhất bị loại bỏ
hoàn toàn.

Và phải nói rõ điều ngược lại: **11.18 KHÔNG chứng minh adapter tệ.** Hai lượt khác nhau ở
*ba* biến — adapter, ngân sách tool, và 4-bit so với fp16. Quy một hiệu số cho một biến khi
ba biến cùng đổi là đúng kiểu lỗi đã ghi ở `local_v7`. Dự án **vẫn chưa có** một A/B công
bằng nào cho adapter. Hai lượt ngắn dưới đây chính là để có nó.

### Một thứ KHÔNG cần port sang Kaggle

`local_v9` kết luận "bật `--ask-retries 1` trên Kaggle". **Bỏ kết luận đó** — notebook không
có cái lỗi mà `src/pipeline.py` có. Cell 14 ghi `asked_missing = True` rồi **chạy tiếp** với
`ASK_LIMIT = 2`, chứ không `break` ở lần hỏi đầu. Cửa một chiều chỉ tồn tại ở đường local.
Notebook đã làm điều đó ở dạng mạnh hơn từ trước.

## Ràng buộc quyết định cách chia lượt

`kaggle_v1` đo **34.9 s/mẫu** (p95 68.9, max 90.2) → 634 mẫu ≈ **6.1 giờ**. V2 gửi cả catalog
nên prompt dài hơn và sẽ chậm hơn. Session GPU của Kaggle **tối đa 9 giờ**.

Nên lượt nộp phải có `TRANSCRIPT_CACHE` để bỏ hẳn ASR, và phải biết `RESUME_FROM` nếu đứt.

## Bốn dataset cần đóng

Làm một lần, dùng cho cả ba lượt chạy. Vào <https://www.kaggle.com/datasets> → **New Dataset**
→ kéo file vào → đặt slug như bảng.

| slug | file từ repo | dùng ở lượt |
|---|---|---|
| (của BTC) `zaic2026-track-lifemate` | — Add Input trực tiếp | A, B, C |
| `lifemate-lora` | `finetune/lora_V4/adapter_config.json` + `adapter_model.safetensors` (252 MiB) | A, C |
| `lifemate-sft` | `sft.eval.jsonl` | A, B |
| `lifemate-transcripts` | `public_test.transcripts.jsonl` | C |

**Tránh đúng cái bug lần trước:** dataset chỉ chứa **dữ liệu**, không chứa `.py`. Lần trước
notebook gọi vào một file `.py` nằm trong dataset và nghẽn ở đó. Mọi mã nằm trong notebook
hoặc trong repo clone; dataset chỉ là trọng số và JSONL.

**Và không đổi tên file.** Màng chắn tìm `adapter_config.json`, `adapter_model.safetensors`,
`sft.eval.jsonl` bằng **glob theo tên**, không theo đường dẫn — vì Kaggle mount ở
`/kaggle/input/<slug>/` hoặc `/kaggle/input/datasets/<user>/<slug>/` tuỳ cách Add Data, và
đường dẫn cứng sai thì `ADAPTER_DIR` thành `None` một cách **âm thầm**: model gốc chạy, và ta
chấm một bài nộp base trong khi tin là đang chấm adapter. Đổi tên file là phá cơ chế đó.

Không lấy `checkpoint-*` — đó là bản giữa đường của `--save-steps 10`. Chỉ hai file cuối.

## Ba lượt chạy, theo thứ tự này

Hai lượt đầu ngắn và **quyết định**; lượt thứ ba dài và chỉ để **sinh file nộp**. Chạy ngược
thứ tự là đốt 6 giờ quota vào một cấu hình chưa biết tốt hay tệ.

### Lượt A — adapter, 50 mẫu held-out (~1 giờ)

Add Input: `zaic2026-track-lifemate`, `lifemate-lora`, `lifemate-sft`.

Trong cell cấu hình, chỉ sửa:

```python
SPLIT = 'train'
RUN_FULL = False
PREFLIGHT_SAMPLES = 50
USE_ADAPTER = True
TRANSCRIPT_CACHE = None
```

### Lượt B — base, **đúng 50 mẫu đó** (~1 giờ)

Add Input: giữ nguyên, bỏ `lifemate-lora` ra cũng được nhưng không cần.

```python
USE_ADAPTER = False      # đổi DUY NHẤT dòng này
```

Đổi một biến và chỉ một biến. `PREFLIGHT_SAMPLES` và `SEED` giữ nguyên nên hai lượt chấm
cùng 50 mẫu, và `HOLDOUT_IDS` đảm bảo chúng không nằm trong `sft.train.jsonl`.

**Hiệu số A − B là A/B công bằng đầu tiên của dự án cho adapter.** Nó trả lời câu mà 9 giờ
huấn luyện đặt ra, và không lượt nào trước đây trả lời được.

### Lượt C — bên thắng, cả `public_test` (~6 giờ)

Add Input: thêm `lifemate-transcripts`.

```python
SPLIT = 'public_test'
RUN_FULL = True
USE_ADAPTER = <bên thắng ở A/B>
TRANSCRIPT_CACHE = Path('/kaggle/input/lifemate-transcripts/public_test.transcripts.jsonl')
```

Bật **Save Version → Save & Run All (Commit)** cho lượt này, đừng chạy trong cửa sổ tương
tác: tab đóng là mất kernel. Output lấy ở panel Output của version.

Nếu đứt giữa đường: đóng `RUN_DIR` của lượt đó thành dataset, Add Input, rồi đặt
`RESUME_FROM` trỏ vào đó. Nó chỉ chứa kết quả từng mẫu, không chứa model.

## Trước mỗi lượt: ba dòng phải đọc

Màng chắn ở **cell 3** chạy trong vài giây, không nạp model, không tốn GPU. Thiếu thứ bắt
buộc thì nó `raise SystemExit` ngay thay vì để bạn phát hiện ở giờ thứ 8.

Nó kiểm: GPU có mặt; dataset nào đã gắn; thư mục split tìm được và **đủ** `<split>.jsonl`
(đúng số mẫu), `audio/`, `zaic_tools/`, `<split>_environment/`, catalog; adapter có **cả hai**
file và `base_model_name_or_path` **khớp** `LLM_ID`; `sft.eval.jsonl` có khi dùng adapter;
transcript cache phủ đủ audio; có mạng hay model đã gắn sẵn; và cấu hình không tự mâu thuẫn.

### Cái bẫy torchao — đã sửa, nhưng phải biết nó tồn tại

Lượt Kaggle đầu chết ở **giây 579**, sau khi đã tải xong 13.4 GiB model:

```
ImportError: Found an incompatible version of torchao.
Found version 0.10.0, but only versions above 0.16.0 are supported
  peft/tuners/lora/torchao.py:160 -> peft/import_utils.py:147
```

`peft/import_utils.is_torchao_available()` **raise** khi thấy torchao cũ thay vì trả về
`False`, và hàm đó nằm trong chuỗi dispatcher mà PEFT đi qua để nạp **mọi** lớp LoRA. Image
Kaggle có torchao 0.10.0; peft 0.20.0 đòi ≥ 0.16.0. Lỗi không liên quan gì tới adapter.

Cell cài đặt giờ **gỡ** torchao ra khi dùng adapter — gỡ chứ không nâng cấp, vì nâng torchao
có thể kéo theo một bản torch khác, mà torch là thứ duy nhất ở đây không được đổi. Ta không
dùng torchao ở đâu: model chạy fp16, không lượng tử bằng torchao.

Gỡ khỏi đĩa một mình là **chưa đủ**: `importlib.util.find_spec` đọc `sys.modules` trước và
trả về `module.__spec__`, nên nếu torchao đã được import ở đâu đó thì hàm vẫn thấy nó và vẫn
raise. Cell cài đặt vì vậy xoá cả `sys.modules` và gọi `invalidate_caches()`, rồi **gọi đúng
hàm đã làm chết lượt trước** để kiểm — chứ không chỉ kiểm số phiên bản.

Màng chắn cũng kiểm việc này ở mục `[9]`, nhưng **chỉ cảnh báo**, không chặn: cell cài đặt tự
sửa được, nên chặn ở đó là báo oan lần thứ hai.

### Ba lỗi của chính màng chắn, đã sửa ở lần chạy thật đầu tiên

Lần chạy đầu trên Kaggle, màng chắn **chặn sai** một adapter hoàn toàn đúng. Ghi lại vì cả ba
đều là lỗi của phép kiểm, không của dữ liệu — và một phép kiểm sai thì tệ hơn không kiểm.

1. **`base_model_name_or_path` có thể là đường dẫn cache, không phải tên repo.**
   `train_sft.py` ghi `/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/
   snapshots/<rev>`. Tôi so `base.split('/')[-1]`, tức lấy **mã revision** rồi đem so với
   **tên model** — không bao giờ khớp. Giờ khớp khi tên repo xuất hiện ở bất kỳ đâu trong
   chuỗi, hoặc khi đúng `LLM_REV` xuất hiện.
2. **Catalog in "3 tool" cho một catalog 147 tool.** Nó là
   `{schema_version, note, apps:[{..., tools:[...]}]}` — 46 app; tôi đếm `len(dict)` = 3 key.
   Con số sai mà vẫn mang nhãn `OK` còn tệ hơn im lặng, vì nó làm người đọc tin là đã kiểm.
   Giờ cộng `len(app['tools'])` và chặn nếu dưới 50.
3. **Mục [2] chỉ in một dòng `datasets`.** Kaggle mount theo hai kiểu:
   `/kaggle/input/<slug>/` và `/kaggle/input/datasets/<user>/<slug>/`. Liệt kê tầng một che
   mất mọi dataset thật. Giờ nó xuống thêm một tầng khi tầng một là `datasets`.

Hai mục nó kiểm mà dễ bị bỏ qua nhất:

- **`zaic_tools` là THƯ MỤC package**, không phải `zaic_tools.py`. Kiểm sai chỗ này từng sinh
  ra cảnh báo "thiếu zaic_tools" trong khi `ToolEnv` nạp bình thường và `state_hash` vẫn ra.
- **`<split>_environment/` chỉ được kiểm tồn tại.** Luật cấm đọc, copy hay parse bất kỳ file
  nào dưới đó, kể cả `state.json`, nên màng chắn không mở gì bên trong.

Sau khi nó in `ĐỦ ĐIỀU KIỆN CHẠY`, đọc tiếp hai dòng này ở đầu output cell cấu hình:

```
ADAPTER_DIR: /kaggle/input/.../lora_V4 (1 ứng viên tìm thấy)
HOLDOUT_IDS: /kaggle/input/.../sft.eval.jsonl
```

`ADAPTER_DIR: None` khi `USE_ADAPTER=True` thì màng chắn đã chặn. `HOLDOUT_IDS: None` ở lượt
A hoặc B cũng bị chặn — vì nếu không, preflight lấy mẫu từ cả split và 42/50 mẫu nằm trong
`sft.train.jsonl`: điểm sẽ tăng và con số đó vô nghĩa.

**Màng chắn không chứng minh pipeline chạy được** — nó chỉ chứng minh file đã đủ. Bằng chứng
chạy được là mẫu **thật** đầu tiên ra `state_hash` ở cell preflight. Dự án đã mất hai lượt vì
nhầm "nạp model xong" với "chạy được"; đừng rời máy trước mẫu đầu tiên.

## Những gì Kaggle KHÔNG trả lời được

Vòng Docker chạy A30 24 GB một card, **bf16**, adapter được **gộp** (`merge_and_unload()`),
và **30 giây/mẫu** cho *cả* ASR và LLM. Notebook này chạy T4 ×2, fp16, hai card, với
`T4_TEST_DEADLINE_SECONDS = 600`. Không con số thời gian nào ở đây dự đoán được vòng đó.

Và `TRANSCRIPT_CACHE` phải trả về `None` trước khi đóng image: bản Docker buộc phải tự chạy
ASR trong máy, nếu không nó sẽ không chạy nổi khi BTC mount private test.

## Cái bẫy thứ hai: màng chắn tự giết lượt chạy

Lượt Kaggle thứ hai (Version 19) chết ở **giây 514**, và không ở chỗ nào liên quan tới
torchao hay dataset. Nó chết ở cell cài đặt:

```
Exception encountered at "In [3]":
RuntimeError: Restart kernel before installing a different Transformers version.
```

Nguyên nhân là **mục [9] của chính màng chắn**. Nó `import peft` để kiểm dispatcher LoRA;
`peft` kéo theo `transformers`, nên sau khi màng chắn chạy xong, `sys.modules` của kernel đã
có `transformers` **5.16.1** của Kaggle. Cell cài đặt muốn 5.13.0, thấy module đã sống trong
kernel, và từ chối làm việc — **đúng**, vì `pip install` không thay được một module đã nạp:
nếu nó cứ cài rồi chạy tiếp thì model sẽ chạy trên bản cũ mà không ai biết.

Tức là màng chắn đã sập vào đúng cái bẫy nó được viết ra để tránh, và bằng chính hành động
"kiểm tra". Bài học là một tính chất, không phải một bản vá: **màng chắn phải read-only với
`sys.modules`, không chỉ read-only với đĩa.** Nó chạy trước mọi thứ, nên mỗi `import` nó làm
là một ràng buộc nó áp lên phần còn lại của notebook.

Hai thứ đã sửa:

- Mục **[9]** giờ gọi `peft` trong **subprocess** (`subprocess.run([sys.executable, '-c', …])`)
  và chỉ đọc một dòng `GOOD`/`BAD` từ stdout. Đã thử ở local: in `GOOD 0.20.0`, và
  `'transformers' in sys.modules` của tiến trình mẹ vẫn là `False`.
- Mục **[10]** mới kiểm ngược: nếu `transformers` đã nằm trong `sys.modules` ở phiên bản khác
  thì **chặn ngay**, với chỉ dẫn `Restart & Clear Cell Outputs` rồi chạy lại từ cell đầu. Thà
  chặn ở giây 10 với thông báo đúng chỗ hơn là chết ở giây 514 với một thông báo nói về
  transformers trong khi lỗi là một `import peft` cách đó mười mục.

### Không phải lỗi dataset

Việc tách thành 4 dataset **không** gây ra lượt chết nào. Log của Version 19 xác nhận cả bốn
mount đúng và đủ: `lifemate-lora` 0.25 GiB, `lifemate-sft` 0.03 GiB, `lifemate-transcripts`,
`zaic2026-track-lifemate` 1.05 GiB, `train.jsonl` 1545 mẫu, `147 tool trong 46 app`,
`audio: đủ 1545/1545 file`, `base khớp LLM_ID` và khớp cả revision. Hai lượt chết vì **hai
lỗi môi trường Python khác nhau**, cả hai đều ở phía phiên bản thư viện, không ở phía dữ liệu.

---

# kaggle_v3 — chạy thuần, và mỗi lượt trả về bốn số thay vì một

`lifemate_kaggle_v3.ipynb` (21 cell, cùng số cell với v2). Bốn thay đổi, **mỗi cái nhắm một
thứ đã đo được**, không cái nào là dọn dẹp cho đẹp.

## 1. Từ 4 dataset về 2 — và bỏ luôn một chế độ thất bại

`scripts/build_sft.py:holdout()` chia train/eval bằng **một hàm thuần của `sample_id`**:

```python
sha256(id)[:8] / 0xFFFFFFFF < 0.15
```

Nên notebook **tự dựng lại** tập held-out bằng ba dòng, không cần `sft.eval.jsonl`. Đã kiểm ở
local: 1545 mẫu `train` → **261 mẫu held-out, khớp chính xác** con số màng chắn v2 đọc từ
dataset.

Cái bỏ được không chỉ là một mount. v2 dò file bằng `glob`, và **khi glob không thấy thì nó âm
thầm rơi về lấy mẫu từ cả split** — nơi 42/50 mẫu nằm trong `sft.train.jsonl`, tức đo adapter
trên dữ liệu nó đã học thuộc. v3 không còn nhánh đó: thiếu tập held-out thì `assert` chết ngay.

`lifemate-transcripts` cũng bỏ: bản Docker **bắt buộc** tự chạy ASR, nên chạy sẵn ASR ở đây là
đo một cấu hình **khác** cấu hình sẽ bị chấm. Giá phải trả đã biết: `kaggle_v2` C có cache chạy
18.29 s/mẫu, lượt A không cache chạy 23.84 — khoảng +1 giờ cho 634 mẫu.

Còn lại **hai** dataset: của BTC, và adapter (252 MiB, không nhúng vào notebook được). Màng
chắn giờ cảnh báo nếu có hơn hai.

> Nói thẳng một điều: 4 dataset **chưa bao giờ** gây ra lỗi nào. Ba lượt chết đều do phiên bản
> thư viện Python, và log luôn xác nhận đủ 4 mount. Việc gộp này mua được *ít chỗ để sai*, chứ
> không sửa một lỗi đang có.

## 2. Lấy mẫu phân tầng — vì thước đo đang lệch

Đo được 2026-11-08: pool 261 mẫu held-out có gold **2.30 call/mẫu**, khớp cả tập train (2.29).
Nhưng `random.sample(pool, 50)` của v2 rút ra một tập **2.86** — nặng hơn **1.244×**.

v3 phân tầng theo số gold call. Chạy đúng đoạn code đã nhúng, trên dữ liệu thật:

| | mẫu | gold call/mẫu | lệch so với pool |
|---|---|---|---|
| pool held-out | 261 | 2.30 | — |
| v2 `random.sample` | 50 | 2.86 | **1.244×** |
| **v3 phân tầng** | 50 | **2.28** | **0.992×** |

Phân bố theo số gold call:

```
 k   pool%   v2%    v3%
 0     8.8    2.0    8.0     <- v2 gan nhu khong thu nhanh nay
 1    24.5   16.0   24.0
 2    24.9   26.0   28.0
 3    22.2   24.0   22.0
 4     8.8   14.0    8.0
 5    10.7   18.0   10.0
```

Cột `k=0` đáng để ý riêng: v2 chỉ rút **2%** mẫu có gold rỗng trong khi thực tế là **8.8%**.
Nhánh chấm `float(not predicted) if not gold` — "gold rỗng, ta cũng không gọi gì → 1.0 điểm" —
gần như chưa bao giờ được thử. Cùng `SEED` nên vẫn tái lập được y nguyên.

## 3. Ask-gate: đòi đọc trước khi hỏi

`sys.ask_missing_info` là **33/111 = 30% tất cả call** ta nộp, trong khi gold muốn nó ở 3.9%
mẫu. Nó lãng phí hai lần: tốn một lượt trong trần `MAX_STEPS=8`, và làm `state_hash` lệch.
`local_v8` đã thấy base hỏi sai ở 91% mẫu; adapter **không** sửa được.

v3 từ chối `ask_missing_info` khi chưa đọc được gì, và đẩy model đi gọi một tool đọc.

**Đánh đổi đã định lượng trên cả 1545 mẫu gold, không phỏng đoán:**

| | mẫu | gate |
|---|---|---|
| gold có `ask` | 217 (14.0%) | |
| ask **sau** một tool đọc | 153 | **không chạm** |
| ask ở **vị trí đầu** | 60 | **chặn** |

Và đúng 60 mẫu đó là **toàn bộ** số mẫu mà gold *chỉ* có `ask`. Tức gate chỉ bao giờ chặn đúng
trường hợp ask-là-cả-câu-trả-lời. Ở đó ta bị buộc đọc trước nên F1 từ 1.00 → `2·1/(2+1)` =
**0.67**, mất **tối đa 1.29 điểm** trên thang 100 — và chỉ mất nếu hiện giờ ta đang làm *đúng*
những mẫu đó, điều rất có thể không.

Net dương là **khả năng, không phải chắc chắn**. Nên nó là cờ `ASK_REQUIRE_READ` để A/B đúng
một biến, không phải một bản vá mặc định mãi mãi.

## 4. Mỗi lượt trả về bảng phân lớp, không chỉ một số

Đây là thay đổi đáng giá nhất về workflow. v2 trả về **đúng một số** sau 30 phút đến 4 giờ
GPU. Một con số không nói được nên sửa gì.

v3 in bốn lớp, cộng lại bằng toàn bộ gold call, kèm tool bỏ sót/thừa hay gặp nhất:

```
PHAN LOP LOI  (bon lop cong lai = toan bo gold call)
  gold 114 call (2.28/mau) | ta nop NN call
  exact      ...   ..% cua gold   <- RECALL thuc
  args_wrong ...   ..%
  missing    ...   ..%
  extra      ...   ..% cua pred
  BO SOT nhieu nhat : contacts.search x18, calendar.list_events x11, ...
  goi THUA nhieu nhat: sys.ask_missing_info x33, ...
```

Nó dùng lại chính `matched` của phép ghép đôi hai phía trong `train_score`, nên **cùng ngữ
nghĩa chấm điểm thật**, không phải một phép so xấp xỉ khác. Số cũng ghi vào `summary.json` tại
khoá `error_classes`, và `scripts/diag/error_classes.py` cho cùng bảng đó từ một file
submission **đã có** — không GPU, vài giây.

Dòng `sys.ask_missing_info` ở phần THỪA là thước đo trực tiếp của thay đổi số 3.

## Kế hoạch chạy v3

Hai lượt, khác nhau **đúng một cờ**, mỗi lượt ~35 phút (không còn transcript cache):

| | `SPLIT` | `RUN_FULL` | `ASK_REQUIRE_READ` |
|---|---|---|---|
| **D** | `train` | `False` | `True` |
| **E** | `train` | `False` | `False` |

D − E là phán quyết về ask-gate. Mốc `kaggle_v2` A = **5.50** đo trên tập lệch 1.244× nên
**không so trực tiếp được** — E là baseline đúng cho v3.

Rồi lượt F: cờ thắng, `SPLIT='public_test'`, `RUN_FULL=True`, ~4.7 giờ (634 × 23.8 s + nạp
model), phóng bằng **Save & Run All (Commit)**.

## Những gì v3 KHÔNG sửa

- **Notebook vẫn tự dựng lại pipeline, không import gì từ `src/`.** Nên `kaggle_v3` và
  `local_v*` khác nhau ở *code*, không chỉ cấu hình — mọi so sánh giữa hai môi trường vẫn lẫn
  biến. Đây là nợ kỹ thuật đắt nhất còn lại.
- **Tải 13.4 GiB model mỗi lượt (~9 phút).** Với lượt 35 phút đó là 25% thời gian. Sửa bằng
  cách đẩy model thành một Kaggle dataset — và việc này **phải làm cho Docker dù sao**, vì luật
  đòi chạy hoàn toàn offline.
- **126 call `missing` (bỏ sót bước tra cứu).** Khối lớn nhất. Màng chắn ID bịa đã có sẵn
  (Workflow fix B, chặn 42 lần ở `local_v10`) nhưng **chưa ai đo** nó có bắn đúng 18 mẫu
  `contacts.search` này không. Việc tiếp theo, sau khi có D và E.
- **70 mẫu (11%) quá 30 giây ở `kaggle_v2` C.** Vòng Docker trên A30 thì đó *là* điểm.
