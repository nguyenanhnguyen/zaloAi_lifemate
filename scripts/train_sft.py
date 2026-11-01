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

    # Kaggle, 2x T4:
    pip install -q peft bitsandbytes
    python scripts/train_sft.py --train sft.train.jsonl --eval sft.eval.jsonl --out lora_v1

    # mask check only, no GPU and no peft needed:
    python scripts/train_sft.py --train sft.train.jsonl --inspect 2
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

IGNORE = -100


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
    parser.add_argument("--out", default="lora_sft")
    parser.add_argument("--max-seq-len", type=int, default=8192,
                        help="8192 keeps 96%% of conversations whole; see build_sft.py --count-tokens")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
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
        source, revision = args.model, None
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

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments

    # T4 (sm75) has no bfloat16 and no flash-attention, so fp16 compute and sdpa attention.
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    # Sharding across both T4s is not about the 4-bit weights (2.4 GiB) but the loss: a
    # 8192 x 151936 logit tensor is 2.5 GiB in fp16 and is upcast to fp32 for cross-entropy.
    # Splitting puts lm_head and that peak on the second card.
    count = torch.cuda.device_count()
    device_map = "balanced" if count > 1 else ({"": 0} if count else None)
    model = AutoModelForCausalLM.from_pretrained(
        source, revision=revision, quantization_config=quant, dtype=dtype,
        device_map=device_map, attn_implementation="sdpa")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    trainer = Trainer(
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
            save_strategy="epoch",
            save_total_limit=2,
            bf16=bf16,
            fp16=not bf16,
            optim="paged_adamw_8bit",
            report_to=[],
            seed=args.seed,
            data_seed=args.seed,
            remove_unused_columns=False,
            group_by_length=False,
            dataloader_num_workers=0,
        ),
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=Collator(tokenizer.pad_token_id or tokenizer.eos_token_id),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"adapter written to {args.out}")
    if eval_set:
        print("final eval:", trainer.evaluate())


if __name__ == "__main__":
    main()
