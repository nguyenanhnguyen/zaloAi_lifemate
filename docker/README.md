# `docker/` — image nộp bài, và mọi thứ việc build cần

Tài liệu **duy nhất** của thư mục này. Tất cả thứ ta viết cho vòng Docker nằm ở đây; mọi thứ
việc build *sinh ra* nằm trong `docker/out/` (đã gitignore).

```
docker/
  README.md       <- bạn đang đọc
  Dockerfile
  entrypoint.py   <- dò layout /data rồi gọi scripts/infer.py
  build.ps1       <- build + smoke, ghi mọi log vào docker/out/
  out/            <- build.log, smoke.log, image.txt, smoke/submission.jsonl
```

Có **một** file bắt buộc nằm ngoài: `.dockerignore` ở gốc repo. Docker chỉ đọc nó ở gốc build
context và không đọc được từ thư mục con. Không có nó thì `docker build` nhét cả `env/`
(~8 GB), `data/` (~10 GB) và các model không dùng (~20 GB) vào context trước khi bắt đầu.

## Chạy

```powershell
.\docker\build.ps1                 # build + smoke 3 mẫu
.\docker\build.ps1 -SkipBuild      # chỉ smoke lại image đã có
.\docker\build.ps1 -Full           # chạy trọn 634 mẫu public_test
```

Build mất ~20–40 phút lần đầu (tải wheel + copy 13.4 GB trọng số); các lần sau dùng cache.

## Luật mà image này phải thoả

| luật | cách thoả |
|---|---|
| BTC mount private test vào `/data` | `entrypoint.py` dò `*_tools_catalog.json`, lấy tên split từ tên file |
| lấy kết quả ở `result/submission.jsonl` | `WORKDIR /app`, ghi `/app/result/submission.jsonl`; đổi bằng `RESULT_DIR` |
| **offline hoàn toàn** | trọng số COPY vào `/opt/hf`, đặt `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` |
| **30 giây/mẫu** | `--max-seconds 30` cứng trong entrypoint (không phải cờ chất lượng) |
| cả hệ < 6B, LLM < 5B | Qwen3-4B (4.0B) + PhoWhisper-large (1.5B) = 5.5B |
| giải mã tất định | `do_sample=False, num_beams=1` ở cả LLM và ASR; seed 42 cho random/numpy/torch |
| gói nộp gồm code huấn luyện | `scripts/` vào trọn, gồm `train_sft.py` |
| không đọc `<split>_environment/` | không có đường dẫn nào tới nó; `/data` mount `:ro` |

## Hai chỗ `entrypoint.py` phải dò thay vì đoán

Luật chỉ nói "mount the private test folder (same structure as public_test) into the container
at `/data`". Hai cách đọc đều hợp lệ:

```
/data/private_test/private_test.jsonl     ← mount thư mục CHA
/data/private_test.jsonl                  ← mount CHÍNH thư mục split
```

`infer.py` ghép `data_root/split`, nên đoán sai là bài nộp ra rỗng — và ta chỉ được chấm một
lần. Entrypoint tìm `*_tools_catalog.json`, lấy split từ tên file, và nếu catalog nằm ngay
trong `/data` thì dựng symlink `/tmp/zaic_root/<split> -> /data` để phép ghép vẫn đúng.

Nó cũng **không** chết khi `infer.py` trả về khác 0: một mẫu lỗi thì mất điểm mẫu đó, còn
container chết là mất cả lượt. Chỉ trả lỗi khi `submission.jsonl` rỗng.

## Biến môi trường

| biến | mặc định | dùng khi |
|---|---|---|
| `DATA_DIR` | `/data` | BTC mount chỗ khác |
| `RESULT_DIR` | `/app/result` | BTC chờ kết quả chỗ khác |
| `MAX_SECONDS` | `30` | chạy thử chất lượng, **không** dùng cho bài nộp |
| `USE_ADAPTER` | `1` | `0` để chạy base làm baseline A/B |
| `INFER_ARGS` | *(rỗng)* | thêm cờ cho `infer.py`, ví dụ `--max-samples 3` |

A/B adapter ngay trên image nộp bài:

```powershell
docker run --rm --gpus all -v "${split}:/data:ro" -v "${out}\base:/app/result" -e USE_ADAPTER=0 lifemate:submit
docker run --rm --gpus all -v "${split}:/data:ro" -v "${out}\lora:/app/result" lifemate:submit
```

## Chưa kiểm được ở đây

Máy dev là RTX 2050 4 GB, còn hardware chấm là **A30 24 GB**. Trên A30 `infer.py` tự chọn
`bf16` + `--asr-phase inline` (card lớn), tức **đường đi khác** với mọi số đã đo ở
`results/local_*`. Cụ thể:

- adapter được **gộp** vào trọng số bf16 (đúng điều kiện huấn luyện), không phải gắn kèm nf4;
- không có bức tường ngân sách prompt 2000 token mà `results/local_v7/README.md` ghi;
- 30 giây/mẫu chưa từng được đo trên A30 — `local_v7` đo 0/50 mẫu quá 30 giây nhưng ở
  prompt bị cắt và trên card khác.

Smoke trên 4 GB sẽ chạy nhánh 4-bit và **sẽ** chật. Nó chứng minh image hợp lệ (dò `/data`
đúng, offline được, ghi đúng chỗ, schema đúng), **không** chứng minh điểm hay tốc độ.
