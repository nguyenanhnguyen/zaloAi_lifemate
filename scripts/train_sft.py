"""QLoRA supervised fine-tuning on the gold agent trajectories.

The organisers require the training code to ship inside the Docker image, so this lives in the repo
and takes only local paths; nothing here reaches the network when `--model` points at a local
snapshot.

Base checkpoint is **Qwen3-4B-Instruct-2507**, not the Thinking sibling. The trajectories built by
scripts/build_sft.py contain no `<think>` block -- they are gold tool calls and their results. The
Thinking checkpoint's chat template opens a reasoning block unconditionally and the weights are
trained to always reason, so fine-tuning it on reasoning-free targets teaches it to suppress the one
behaviour it was post-trained for. The Instruct template matches the data exactly.

Loss is computed on assistant tokens only, and this is not a refinement. The median trajectory is
5306 tokens of which the assistant turns are a few hundred: train on everything and over 95% of the
gradient goes into memorising the tool RESULTS, text the model never has to produce and which at
inference comes from the organisers' ToolEnv. `encode()` recovers the assistant spans by rendering
growing message prefixes and diffing, and refuses to run if the template turns out not to be
prefix-stable, because a silently wrong mask trains a silently wrong model.

    # Kaggle, 2x T4. Ghim transformers đúng bản notebook suy luận dùng: huấn luyện và suy
    # luận trên hai bản thư viện khác nhau là một khác biệt không ai kiểm soát, và bản mới
    # hơn của Kaggle đã bỏ cả warmup_ratio lẫn group_by_length khỏi TrainingArguments.
    pip install -q 'transformers==5.13.0' 'accelerate>=1.10,<2' peft bitsandbytes
    python scripts/train_sft.py --train sft.train.jsonl --eval sft.eval.jsonl --out lora_v1

    # mask check only, no GPU and no peft needed:
    python scripts/train_sft.py --train sft.train.jsonl --inspect 2
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Attention in blocks allocates a differently shaped score tensor per block (1024 x 1024,
# 1024 x 2048, ...), which fragments the caching allocator: the run that OOMed at step 2 had
# 2.51 GiB reserved but unallocated while asking for 704 MiB. Set before torch initialises
# CUDA, which is why it sits at import time rather than next to the other GPU settings.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

IGNORE = -100
QUERY_BLOCK = 1024
LOSS_CHUNK = 1024


def chunked_cross_entropy(head, hidden, labels, chunk: int = LOSS_CHUNK):
    """Sum of the token losses, without ever holding logits for the whole sequence.

    `hidden` is (B, L, H) straight from the transformer body and `labels` is (B, L) already
    shifted; both are sliced into chunks of `chunk` positions, and each chunk's lm_head is
    wrapped in gradient checkpointing so its logits are recomputed in the backward pass
    instead of being kept.

    The vocabulary is 151936 wide, so the full-sequence version costs, for one 6133-token
    sample: logits in fp16 1.74 GiB, the fp32 copy cross-entropy upcasts to 3.47 GiB, a
    second fp32 copy from accelerate's convert_to_fp32 wrapper 3.47 GiB, and the gradient
    3.47 GiB. Trainer reads only `outputs.loss`, so three of those four are pure waste --
    and the gradient is what took out GPU 0 once attention stopped being the bottleneck.
    Per chunk the same figures are 311 MiB and 622 MiB, one chunk live at a time.
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    def piece(block, target):
        logits = head(block).float()
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
            ignore_index=IGNORE, reduction="sum")

    total = None
    for start in range(0, hidden.shape[1], chunk):
        stop = min(start + chunk, hidden.shape[1])
        part = checkpoint(piece, hidden[:, start:stop], labels[:, start:stop],
                          use_reentrant=False)
        total = part if total is None else total + part
    return total


def block_attention_forward(module, query, key, value, attention_mask,
                            dropout: float = 0.0, scaling: float | None = None,
                            is_causal: bool | None = None, **kwargs):
    """Causal attention computed in blocks of QUERY_BLOCK queries.

    Below sm80 SDPA has no flash kernel, and on the 2x T4 run it also declined the
    memory-efficient one, leaving the math fallback -- which materialises the entire
    (1, 32, L, L) score matrix: 1.1 GiB at L=4335, 4.0 GiB at L=8192. Rather than keep
    guessing why a kernel was rejected, this sidesteps the question: each block only ever
    scores QUERY_BLOCK queries against the keys up to its own end, so the peak is
    32 x 1024 x L instead of 32 x L x L -- 512 MiB at L=8192, and flat in L per block.

    Slower than a fused kernel, but a T4 at 8192 tokens is compute-bound anyway, and the
    arithmetic is plain SDPA per block so the result is bit-comparable with the sdpa path
    (checked in scripts/diag/check_block_attention.py). Autograd handles the blocks; each
    decoder layer is already wrapped in gradient checkpointing above this.
    """
    import torch
    from transformers.integrations.sdpa_attention import repeat_kv

    groups = getattr(module, "num_key_value_groups", 1)
    if groups > 1:
        key = repeat_kv(key, groups)
        value = repeat_kv(value, groups)

    length = query.shape[2]
    pieces = []
    for start in range(0, length, QUERY_BLOCK):
        stop = min(start + QUERY_BLOCK, length)
        if attention_mask is not None:
            mask = attention_mask[:, :, start:stop, :stop]
        else:
            # Query at absolute position i may attend to keys j <= i. is_causal cannot express
            # that here: within a block the diagonal sits at column start+row, not at row.
            rows_ = torch.arange(start, stop, device=query.device)
            cols = torch.arange(stop, device=query.device)
            mask = cols[None, :] <= rows_[:, None]
        pieces.append(torch.nn.functional.scaled_dot_product_attention(
            query[:, :, start:stop], key[:, :, :stop], value[:, :, :stop],
            attn_mask=mask, dropout_p=dropout, scale=scaling, is_causal=False))

    return torch.cat(pieces, dim=2).transpose(1, 2).contiguous(), None


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def encode(tokenizer, row: dict, max_len: int) -> dict | None:
    """Tokenise one conversation, supervising only the assistant turns.

    Returns None when the conversation does not fit; the caller drops it. Truncating instead would
    cut the tail, and the tail is the last tool calls and the closing DONE -- the most valuable
    supervision in the sample.
    """
    messages, tools = row["messages"], row["tools"]

    def render(upto: int, generation: bool) -> list[int]:
        return tokenizer.apply_chat_template(
            messages[:upto], tools=tools, tokenize=True,
            add_generation_prompt=generation, return_dict=True)["input_ids"]

    ids: list[int] = []
    labels: list[int] = []
    for index, message in enumerate(messages):
        if message["role"] == "assistant":
            # Rendering the prefix WITH a generation prompt ends exactly at "<|im_start|>assistant\n",
            # so the supervised span starts after the header: the header is supplied by the decoder
            # at inference and spending loss on it teaches nothing.
            header = render(index, generation=True)
            full = render(index + 1, generation=False)
            if header[:len(ids)] != ids or full[:len(header)] != header:
                raise RuntimeError(
                    f"chat template is not prefix-stable at message {index} of {row['id']}; "
                    "the assistant mask cannot be trusted -- fix the template handling before "
                    "training, do not fall back to training on every token")
            labels.extend([IGNORE] * (len(header) - len(ids)))
            labels.extend(full[len(header):])
            ids = full
        else:
            full = render(index + 1, generation=False)
            if full[:len(ids)] != ids:
                raise RuntimeError(f"chat template is not prefix-stable at message {index} "
                                   f"of {row['id']}")
            labels.extend([IGNORE] * (len(full) - len(ids)))
            ids = full

    if len(ids) > max_len or not any(label != IGNORE for label in labels):
        return None
    return {"input_ids": ids, "labels": labels, "id": row["id"]}


def build(tokenizer, path: Path, max_len: int, cache: Path | None) -> list[dict]:
    if cache and cache.exists():
        kept = [json.loads(line) for line in cache.open(encoding="utf-8") if line.strip()]
        print(f"{path.name}: {len(kept)} conversations from cache {cache.name}")
        return kept
    kept, dropped = [], 0
    for row in rows(path):
        encoded = encode(tokenizer, row, max_len)
        if encoded is None:
            dropped += 1
            continue
        kept.append(encoded)
    supervised = sum(sum(1 for label in item["labels"] if label != IGNORE) for item in kept)
    total = sum(len(item["input_ids"]) for item in kept)
    print(f"{path.name}: {len(kept)} conversations, {dropped} dropped over {max_len} tokens; "
          f"supervised {supervised}/{total} tokens ({supervised/max(1,total):.1%})")
    if cache:
        with cache.open("w", encoding="utf-8") as handle:
            for item in kept:
                handle.write(json.dumps(item) + "\n")
    return kept


class Collator:
    """Right-pad a batch; padded positions and prompt positions are both ignored by the loss."""

    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        import torch
        width = max(len(item["input_ids"]) for item in batch)
        return {
            "input_ids": torch.tensor([item["input_ids"] + [self.pad_id] * (width - len(item["input_ids"]))
                                       for item in batch]),
            "attention_mask": torch.tensor([[1] * len(item["input_ids"]) + [0] * (width - len(item["input_ids"]))
                                            for item in batch]),
            "labels": torch.tensor([item["labels"] + [IGNORE] * (width - len(item["labels"]))
                                    for item in batch]),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", default="sft.train.jsonl")
    parser.add_argument("--eval", default="sft.eval.jsonl")
    parser.add_argument("--model", help="local snapshot or hub id; defaults to the pipeline's LLM")
    parser.add_argument("--revision",
                        help="commit to pin when --model is a hub id. Without it the hub serves "
                             "whatever main points at today, so the adapter can end up trained on "
                             "weights the notebook does not load at inference -- and the rules "
                             "require a deterministic, offline-reproducible solution.")
    parser.add_argument("--out", default="lora_sft")
    parser.add_argument("--max-seq-len", type=int, default=8192,
                        help="8192 keeps 96%% of conversations whole; see build_sft.py --count-tokens")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quant", choices=["nf4", "fp16"], default="nf4",
                        help="fp16 = no 4-bit, 8.06 GiB of weights split across the cards. On a "
                             "T4 the nf4 dequantisation runs at 2.8%% of peak, so this trades "
                             "memory we have for speed we do not")
    parser.add_argument("--limit", type=int,
                        help="train on this many conversations, sampled with --seed. 310 steps "
                             "measured 738 s each, so a full epoch is 31 h against a 12 h "
                             "session limit; this is what buys a run that finishes")
    parser.add_argument("--eval-limit", type=int,
                        help="likewise for the eval set; 0 turns evaluation off. 243 samples of "
                             "eval cost about 1.5 h, which is budget better spent on training")
    parser.add_argument("--save-steps", type=int, default=10,
                        help="lưu adapter mỗi bao nhiêu optimizer step. Mặc định 10 (~50 phút) "
                             "vì một lượt chết ở bước 46/80 đã mất trắng 4 giờ do chỉ lưu theo "
                             "epoch")
    parser.add_argument("--resume", nargs="?", const="latest",
                        help="tiếp tục huấn luyện. Không tham số = checkpoint mới nhất trong "
                             "--out; hoặc truyền thẳng đường dẫn checkpoint-N")
    parser.add_argument("--device", choices=["single", "balanced"], default="single",
                        help="single = một card, không có hook chuyển tensor của accelerate "
                             "(chỗ nổ illegal memory access). balanced = chia trên mọi card, "
                             "chỉ cần khi trọng số không vừa một card")
    parser.add_argument("--attn", choices=["sdpa", "block"], default="sdpa",
                        help="block = attention in query blocks, for cards where SDPA falls back "
                             "to the math kernel and its (L, L) score matrix does not fit")
    parser.add_argument("--cache", help="prefix for the tokenised cache, e.g. sft.tok")
    parser.add_argument("--inspect", type=int, metavar="N",
                        help="print the mask for N conversations and exit; no GPU, no peft")
    parser.add_argument("--tokenize-only", action="store_true",
                        help="write the --cache files and stop; lets a CPU box prepare for Kaggle")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    if args.model:
        # Stand-alone: this file plus the two JSONL datasets is enough to train on a fresh box
        # (Kaggle, or the Docker build), with no need to carry src/ along.
        source, revision = args.model, args.revision
        if revision is None and not Path(source).exists():
            print(f"⚠️  --model {source} là hub id mà không có --revision: huấn luyện sẽ bám bản "
                  f"main hôm nay, có thể khác bản notebook nạp lúc suy luận.")
    else:
        from src.llm_agent import LLM_REPO, LLM_REVISION
        source, revision = LLM_REPO, LLM_REVISION
    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)

    if args.inspect:
        for row in rows(Path(args.train))[:args.inspect]:
            encoded = encode(tokenizer, row, args.max_seq_len)
            if encoded is None:
                print(f"{row['id']}: dropped, over {args.max_seq_len} tokens")
                continue
            supervised = [i for i, label in enumerate(encoded["labels"]) if label != IGNORE]
            print(f"\n=== {row['id']}: {len(encoded['input_ids'])} tokens, "
                  f"{len(supervised)} supervised ({len(supervised)/len(encoded['input_ids']):.1%})")
            print("supervised text:",
                  repr(tokenizer.decode([encoded["labels"][i] for i in supervised]))[:700])
        return

    random.seed(args.seed)
    cache_train = Path(f"{args.cache}.train.jsonl") if args.cache else None
    cache_eval = Path(f"{args.cache}.eval.jsonl") if args.cache else None
    train_set = build(tokenizer, Path(args.train), args.max_seq_len, cache_train)
    eval_set = build(tokenizer, Path(args.eval), args.max_seq_len, cache_eval) \
        if Path(args.eval).exists() else None
    if args.tokenize_only:
        return

    def shrink(items, keep, name):
        if keep is None or keep >= len(items):
            return items
        picked = random.Random(args.seed).sample(items, keep) if keep else []
        tokens = sum(len(item["input_ids"]) for item in picked)
        supervised = sum(sum(1 for x in item["labels"] if x != IGNORE) for item in picked)
        print(f"{name}: lấy {len(picked)}/{len(items)} hội thoại, {tokens:,} token "
              f"({supervised:,} có loss)")
        return picked

    train_set = shrink(train_set, args.limit, "train")
    eval_set = shrink(eval_set, args.eval_limit, "eval") if eval_set else None
    if not train_set:
        raise SystemExit("tập train rỗng")

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments

    # T4 (sm75) has no bfloat16 and no flash-attention, so fp16 compute and sdpa attention.
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    # Below sm80 the only memory-sane kernel SDPA can offer is the memory-efficient one: the
    # math fallback materialises the whole (1, 32, L, L) score matrix, 1.1 GiB at L=4340 and
    # 4.0 GiB at L=8192, which is what took out a 14.5 GiB card on the first step.
    #
    # Two things keep the memory-efficient kernel out. First, transformers hands SDPA
    # enable_gqa=True whenever the causal mask is None (sdpa_attention.use_gqa_in_sdpa), and
    # only the flash and math backends accept that flag; forcing the repeat_kv path instead
    # costs one expanded K and V, 64 MiB at L=8192. Removing enable_gqa alone did NOT move the
    # run off math, so something else is rejecting the kernel too and we do not yet know what.
    # Hence the second half: with math disabled, SDPA raises instead of silently falling back,
    # and the warnings it prints name the reason each kernel was turned down. Diagnosing a
    # crash we can read beats diagnosing an OOM we have to reverse-engineer.
    old_gpu = (args.attn == "sdpa" and torch.cuda.is_available()
               and torch.cuda.get_device_capability()[0] < 8)
    if old_gpu:
        from transformers.integrations import sdpa_attention
        sdpa_attention.use_gqa_in_sdpa = lambda *_: False
        print("sm75 hoặc cũ hơn: tắt enable_gqa, và chỉ cho phép backend mem_efficient")
    if args.attn == "block":
        # Registered under its own name rather than patched over "sdpa", so the log and the
        # saved adapter config both say which attention the weights were trained with.
        from transformers.modeling_utils import AttentionInterface
        AttentionInterface.register("block", block_attention_forward)
        print(f"attention theo khối {QUERY_BLOCK} query: không dựng ma trận (L, L)")
    dtype = torch.bfloat16 if bf16 else torch.float16
    quant = None
    if args.quant == "nf4":
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    # Sharding across both T4s is not about the 4-bit weights (2.4 GiB) but the loss: a
    # 8192 x 151936 logit tensor is 2.5 GiB in fp16 and is upcast to fp32 for cross-entropy.
    # Splitting puts lm_head and that peak on the second card.
    count = torch.cuda.device_count()
    # Chia trên hai card bắt accelerate gắn hook pre_forward vào mỗi lớp để chuyển tensor qua
    # card khác, và đó chính là chỗ "illegal memory access" nổ ra ở bước 46/80. fp16 là 7.49
    # GiB, một T4 có 14.56 GiB, nên một card vừa -- và một card thì không có hook nào để nổ.
    if args.device == "single" or count <= 1:
        device_map = {"": 0} if count else None
    else:
        device_map = "balanced"
    if count > 1:
        print(f"{count} card, dùng {'MỘT card (cuda:0)' if device_map != 'balanced' else 'cả hai, chia balanced'}")
    model = AutoModelForCausalLM.from_pretrained(
        source, revision=revision, quantization_config=quant, dtype=dtype,
        device_map=device_map, attn_implementation=args.attn)
    model.config.use_cache = False
    if quant is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        # Same two things prepare_model_for_kbit_training does that we still need. Its third --
        # upcasting every unquantised parameter to fp32 -- is what made embed_tokens cost
        # 1.56 GiB instead of 778 MiB, and with no 4-bit weights there is nothing to protect.
        model.gradient_checkpointing_enable({"use_reentrant": False})
        print(f"fp16, không lượng tử hoá: {sum(p.numel() for p in model.parameters()) * 2 / 2**30:.2f} "
              f"GiB trọng số chia trên {count} card")
    # peft thử từng dispatcher để chọn lớp LoRA. Với nf4 thì dispatcher bitsandbytes khớp
    # trước; với nn.Linear thường thì tới lượt dispatch_torchao, và nó gọi
    # is_torchao_available() -- hàm này RAISE ImportError khi phiên bản không đủ mới thay vì
    # trả False. Kaggle có torchao 0.10.0, peft đòi > 0.16.0, nên fp16 vỡ ngay lúc dựng
    # adapter. Ta không dùng torchao chút nào, nên nói thẳng là không có.
    import peft.import_utils
    import peft.tuners.lora.torchao as lora_torchao
    peft.import_utils.is_torchao_available = lambda: False
    lora_torchao.is_torchao_available = lambda: False

    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    class ChunkedLossTrainer(Trainer):
        """Runs the body, then the loss in chunks; never calls the full model's forward.

        Going through the top-level forward is what produces the whole-sequence logits and
        accelerate's fp32 copy of them. Calling the body directly skips both, at the cost of
        applying autocast here by hand -- that wrapper is where it came from.
        """

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # Bóc mọi lớp vỏ trước khi tìm lm_head. DataParallel và DistributedDataParallel bọc
            # model vào `.module`, và PeftModel thì mở bằng get_base_model() -- thứ tự vỏ không
            # cố định, nên cứ bóc đến khi thấy lm_head. Lượt lora_V3 chết ngay bước 0 vì chỉ thử
            # get_base_model() rồi gặp DataParallel.
            causal = model
            for _ in range(4):
                if hasattr(causal, "lm_head"):
                    break
                if hasattr(causal, "get_base_model"):
                    causal = causal.get_base_model()
                elif hasattr(causal, "module"):
                    causal = causal.module
                else:
                    break
            if not hasattr(causal, "lm_head"):
                raise RuntimeError(f"không tìm thấy lm_head trên {type(model).__name__}")
            labels = inputs["labels"][:, 1:]
            with torch.autocast("cuda", dtype=dtype, enabled=torch.cuda.is_available()):
                hidden = causal.model(input_ids=inputs["input_ids"],
                                      attention_mask=inputs.get("attention_mask")).last_hidden_state
                total = chunked_cross_entropy(causal.lm_head, hidden[:, :-1], labels)
            # Mirrors ForCausalLMLoss: a sum divided by the token count of the whole
            # accumulation window, so short and long samples carry their true weight. When
            # Trainer withholds that count it divides by grad_accum itself, and dividing by
            # the local count here is then the matching halves of the same normalisation.
            if num_items_in_batch is None:
                loss = total / (labels != IGNORE).sum().clamp(min=1)
            else:
                loss = total / num_items_in_batch
            return (loss, {"loss": loss}) if return_outputs else loss

    trainer = ChunkedLossTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,          # one 8k conversation already fills the card
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=10,
            eval_strategy="epoch" if eval_set else "no",
            # compute_loss returns no logits, and eval only needs eval_loss to pick a
            # checkpoint; without this Trainer would try to collect predictions it cannot get.
            prediction_loss_only=True,
            # Theo BƯỚC, không theo epoch. Lượt 320 mẫu chết ở bước 46/80 sau 4 giờ 12 phút và
            # không để lại gì, vì save_strategy="epoch" với --epochs 1 chỉ lưu ở cuối. Với
            # --save-steps 10 thì mất nhiều nhất ~50 phút, và luôn có adapter dùng được.
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=2,
            bf16=bf16,
            fp16=not bf16,
            optim="paged_adamw_8bit",
            report_to=[],
            seed=args.seed,
            data_seed=args.seed,
            remove_unused_columns=False,
            # group_by_length đã bị bỏ khỏi TrainingArguments ở transformers 5.x, và ở
            # per_device_train_batch_size=1 thì nó vốn không làm gì: không có batch để gom
            # theo độ dài. Bỏ hẳn thay vì đặt điều kiện theo phiên bản.
            dataloader_num_workers=0,
        ),
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=Collator(tokenizer.pad_token_id or tokenizer.eos_token_id),
    )
    # Trainer skips its own divide-by-grad_accum exactly when this flag is set and it has a
    # token count to hand us. It normally infers the flag from the model's forward signature,
    # which our compute_loss no longer calls -- so pin it, or the loss gets normalised twice
    # and the effective learning rate is silently off by a factor of grad_accum.
    trainer.model_accepts_loss_kwargs = True
    # --device single đặt trọng số lên cuda:0, nhưng Trainer KHÔNG nhìn device_map: nó nhìn
    # torch.cuda.device_count(), thấy 2 card nên đặt n_gpu=2 và bọc model vào nn.DataParallel.
    # Hai hậu quả, cả hai đều phá lượt chạy: lớp vỏ làm compute_loss không tìm ra lm_head, và
    # DataParallel sẽ rải batch sang card thứ hai -- đúng kiểu chuyển tensor qua card mà
    # --device single được thêm vào để tránh. Ghim n_gpu về 1 là cách nói với Trainer đúng thứ
    # device_map đã làm.
    if device_map != "balanced" and count > 1:
        trainer.args._n_gpu = 1
        print("ghim n_gpu=1: không bọc DataParallel")
    import contextlib
    from torch.nn.attention import SDPBackend, sdpa_kernel
    attention = sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]) if old_gpu else contextlib.nullcontext()
    # "latest" = tiếp tục checkpoint mới nhất trong --out nếu có, khởi động mới nếu chưa có.
    # Không dùng True vì True làm Trainer báo lỗi khi output_dir rỗng.
    resume = args.resume
    if resume == "latest":
        found = sorted(Path(args.out).glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[1]))
        resume = str(found[-1]) if found else None
        print(f"tiếp tục từ {resume}" if resume else "không thấy checkpoint, bắt đầu từ đầu")
    with attention:
        trainer.train(resume_from_checkpoint=resume)
    # Con số để định cỡ lượt sau. Không có nó thì mỗi lần đổi --max-seq-len là một lần đoán,
    # và đoán bộ nhớ trên card 4 GB thì chỉ biết sai sau khi đã mất một giờ.
    if torch.cuda.is_available():
        print(f"VRAM đỉnh: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB "
              f"(cấp phát {torch.cuda.max_memory_reserved() / 1024**3:.2f} GiB) "
              f"ở max_seq_len={args.max_seq_len}, quant={args.quant}")
        model.save_pretrained(args.out)
        tokenizer.save_pretrained(args.out)
        print(f"adapter written to {args.out}")
        if eval_set:
            print("final eval:", trainer.evaluate())


if __name__ == "__main__":
    main()
