# finetune_v1 — QLoRA trên trajectory gold *(dữ liệu xong, chưa huấn luyện)*

Code: [`scripts/build_sft.py`](../../scripts/build_sft.py), [`scripts/train_sft.py`](../../scripts/train_sft.py).
Thư mục này trống cho tới khi có adapter và điểm thật.

## Vì sao fine-tune, bằng một con số

Phép thử chuẩn hoá số so 22 mẫu mà router giữ **đủ toàn bộ app gold**. Chỉ 2/22 mẫu được điểm,
18/22 lệch `state_hash`. Route đúng cho **gấp đôi** điểm (11.36 vs 5.50) nhưng vẫn cách 22.52 rất xa.

Nghĩa là: routing là nguyên nhân lớn nhất **đã đo được**, nhưng sửa nó một mình không đủ. Khi đã có
đủ công cụ trong tay, model vẫn chọn sai tool và sai đối số. Prompt đúng chỉ là điều kiện cần.

## Dữ liệu — không tổng hợp gì

| | Số |
|---|---|
| Mẫu `train` có nhãn | 1545 |
| `tool_calls` khớp 1:1 `tool_results` | 1434/1434 |
| Hội thoại train / eval | **1284 / 261** |
| Mẫu bị loại | **0** |
| Id nằm cả hai split | **0** (chia theo sha256 của id) |

Mỗi mẫu replay thành hội thoại nhiều lượt: lượt assistant = một gold call, lượt cuối = `DONE`. Lượt
`DONE` quan trọng ngang các call — 111 mẫu có gold rỗng, model không học dừng sẽ luôn bịa thêm một
call rồi mất precision.

Nguồn văn bản là **transcript ASR của chính ta**, không phải `audio_text` sạch, vì lúc chạy thật
model chỉ thấy đầu ra ASR ở ~24% WER. Chi phí gần như bằng 0:

| | `audio_text` (nhãn sạch) | **ASR thật** |
|---|---|---|
| Trung vị token | 5281 | **5306** |
| p95 | — | 8011 |
| max | 9503 | **9554** |
| Vừa seq 8192 | 96.4% | **96.0%** |

## Mask loss — điểm sống còn

Đo trên toàn bộ dataset ở seq 8192:

| | train | eval |
|---|---|---|
| Hội thoại giữ lại | 1233 | 243 |
| Bỏ vì >8192 token | 51 | 18 |
| Token được giám sát | **96.168 / 6.547.604 = 1.5%** | 18.632 / 1.327.915 = 1.4% |

Văn bản được giám sát của `train_0001` (7437 token, 205 giám sát = 2.8%), decode ra để kiểm bằng mắt:

```
<tool_call>{"name": "weather.get_forecast", ...}</tool_call><|im_end|>
<tool_call>{"name": "calendar.list_events", ...}</tool_call><|im_end|>
<tool_call>{"name": "calendar.update_event", ...}</tool_call><|im_end|>
<tool_call>{"name": "contacts.search", ...}</tool_call><|im_end|>
<tool_call>{"name": "sms.send", ...}</tool_call><|im_end|>
DONE<|im_end|>
```

Không một token nào của system prompt, của JSON yêu cầu, hay của **kết quả tool**. Không mask thì
**98.5% gradient** đổ vào học thuộc lại kết quả tool — văn bản model không bao giờ phải sinh ra, vì
lúc chạy thật nó đến từ `ToolEnv` của BTC.

**Hệ quả phải biết trước:** ngân sách tín hiệu thật chỉ là **96k token giám sát**, 2 epoch ≈ 192k.
Đó là ít. Nên trông đợi ở lần này là **chỉnh hành vi** — gọi đúng tool, đúng dạng đối số, biết dừng —
chứ không phải dạy model kiến thức mới. Nếu eval loss còn giảm đều ở cuối epoch 2 thì tăng
`--epochs 3`, đừng tăng `--lora-r`.

`encode()` khôi phục các đoạn assistant bằng cách render prefix tăng dần rồi diff, và **raise nếu
template không prefix-stable** thay vì âm thầm train sai. Kiểm bất cứ lúc nào, không cần GPU hay
`peft`:

```bash
python scripts/train_sft.py --train sft.train.jsonl --inspect 2
```

Mẫu dài hơn `--max-seq-len` bị **bỏ, không cắt** — cắt là mất đuôi, mà đuôi là các call cuối và
`DONE`, phần giám sát giá trị nhất.

## Cấu hình

| | |
|---|---|
| Base | `Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f...` |
| Lượng tử hoá | 4-bit NF4, double quant, compute fp16 |
| LoRA | r=32, alpha=64, dropout 0.05 |
| `target_modules` | q,k,v,o,gate,up,down |
| seq len | 8192 (giữ nguyên 96% hội thoại) |
| batch | 1 × grad accum 8 |
| LR | 1e-4, cosine, warmup 3% |
| epoch | 2 |
| optim | `paged_adamw_8bit` |
| GPU | 2× T4, `device_map="balanced"` |

**Base là Instruct, không phải Thinking.** Trajectory không có khối `<think>` nào; template bản
Thinking mở khối reasoning **vô điều kiện** và checkpoint được post-train để luôn suy luận, nên SFT
trên target không-reasoning là dạy nó dập đúng hành vi nó được huấn luyện. Cũng giữ Kaggle V2
(Thinking, zero-shot) và bản này là **hai thí nghiệm khác nhau thật sự**.

**fp16 chứ không bf16, sdpa chứ không flash-attention:** T4 là sm75, không có bf16 và không có
flash-attn. Script tự phát hiện bằng `torch.cuda.is_bf16_supported()`.

**Vì sao chia 2 GPU** — không phải vì trọng số (4-bit chỉ 2.4 GiB) mà vì **loss**: tensor logit
8192 × 151936 là 2.5 GiB ở fp16 và bị upcast lên fp32 để tính cross-entropy. Chia để `lm_head` và
đỉnh bộ nhớ đó nằm trên card thứ hai.

## Cách chạy

**Laptop (đã xong):**

```bash
python scripts/infer.py --split train --asr-only --transcript-cache train_all.transcripts.jsonl
python scripts/build_sft.py --transcripts train_all.transcripts.jsonl --text-source asr --out sft --count-tokens
python scripts/train_sft.py --train sft.train.jsonl --eval sft.eval.jsonl --cache sft.tok --tokenize-only
```

Upload `sft.train.jsonl`, `sft.eval.jsonl`, `sft.tok.*.jsonl` thành một Kaggle Dataset. Có `sft.tok.*`
thì Kaggle bỏ hẳn bước tokenize (~15 phút CPU) — đáng, vì quota 12 giờ.

**Kaggle:**

```bash
pip install -q peft bitsandbytes
python scripts/train_sft.py --train sft.train.jsonl --eval sft.eval.jsonl --cache sft.tok --out lora_v1
```

**Chấm điểm (laptop hoặc Kaggle):**

```bash
python scripts/infer.py --split train --max-samples 100 --adapter lora_v1 --output ft_check.jsonl
python scripts/score.py ft_check.jsonl --split train
```

**So với 5.50 của `local_v4` trên đúng 100 mẫu đó.** Không so với 22.52 — khác split, khác cấu hình.

bf16 thì adapter được merge vào trọng số → còn lại một model Qwen3 thường: giữ nguyên số tham số
(luật <6B không bị ảnh hưởng) và phần tái dùng KV-cache prefix vẫn đúng. 4-bit giữ adapter rời, vì
merge vào trọng số 4-bit sẽ lượng tử hoá cả phần delta.

## Nếu OOM

1. `--grad-accum` không ảnh hưởng bộ nhớ, đừng sửa.
2. Hạ `--max-seq-len` xuống **6144** — mất 29% hội thoại (67.4% vừa, so với 96.0%). Đây là đánh đổi
   thật, không miễn phí.
3. `--lora-r 16`. Tiết kiệm ít, thử sau cùng.
4. Đừng hạ xuống 4096: chỉ 18.1% hội thoại vừa, tập huấn luyện còn ~230 mẫu.

## Đúng luật BTC

- Chỉ đọc split `train` — split duy nhất có nhãn. Gán nhãn `public_test`/`private_test` (bằng tay
  hay bằng model/dịch vụ ngoài) để train hay tune là **bị cấm**.
- Transcript dùng để huấn luyện sinh từ **audio train của chính BTC**, nên không phải dữ liệu tổng
  hợp từ ngoài.
- Code huấn luyện **nằm trong repo**, không nằm trong notebook: BTC yêu cầu nộp code huấn luyện kèm
  trong Docker ngày 16/11.
- Adapter merge không đổi số tham số: ASR 1.543B + LLM 4.022B = **5.566B < 6B**, LLM < 5B.
