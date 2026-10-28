"""Fine-tune Whisper (Giai đoạn 2) trên tập train của ZAC 2026.

Dùng LoRA (peft) với whisper-large-v3-turbo từ HuggingFace.

Chạy từ thư mục gốc project:
    python scripts/train_asr.py --split train --epochs 3
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "openai/whisper-large-v3-turbo"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Whisper cho ZAC 2026")
    p.add_argument("--data-root", default="data/train+public_test/train+public_test",
                   help="Thư mục chứa các split (default: data/train+public_test/train+public_test)")
    p.add_argument("--split", default="train",
                   help="Split dùng để train (default: train)")
    p.add_argument("--output-dir", default="checkpoints/whisper-finetuned",
                   help="Thư mục lưu checkpoint (default: checkpoints/whisper-finetuned)")
    p.add_argument("--epochs", type=int, default=3,
                   help="Số epoch (default: 3)")
    p.add_argument("--model-id", default=MODEL_ID,
                   help=f"HuggingFace model ID (default: {MODEL_ID})")
    p.add_argument("--local-model", default=None,
                   help="Đường dẫn model local (bỏ qua HuggingFace hub)")
    return p.parse_args()


# ── Dataset ──────────────────────────────────────────────────────────────────

class ZaloWhisperDataset(torch.utils.data.Dataset):
    """Dataset đọc train.jsonl và load audio WAV."""

    def __init__(self, jsonl_path: Path, audio_root: Path, processor) -> None:
        self.processor = processor
        self.audio_root = audio_root
        self.items: list[dict] = []

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                # Chỉ giữ sample có audio_text (label cho ASR)
                if "audio_text" in item and item["audio_text"]:
                    self.items.append(item)

        logger.info("Dataset: %d samples có audio_text label", len(self.items))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        audio_path = self.audio_root / item["audio"]

        import soundfile as sf  # type: ignore
        audio_array, sr = sf.read(str(audio_path), dtype="float32")

        # WhisperProcessor yêu cầu 16kHz mono
        inputs = self.processor(
            audio_array,
            sampling_rate=sr,
            return_tensors="pt",
        )
        input_features = inputs.input_features[0]  # (80, 3000)

        # Label: lấy audio_text của lượt cuối (câu lệnh hiện tại)
        audio_text = item["audio_text"]
        if isinstance(audio_text, list):
            text = audio_text[-1] if audio_text else ""
        else:
            text = str(audio_text)

        labels = self.processor.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=448,
        ).input_ids[0]

        return {"input_features": input_features, "labels": labels}


def collate_fn(batch: list[dict], processor) -> dict:
    """Padding batch về cùng độ dài."""
    input_features = [b["input_features"] for b in batch]
    labels = [b["labels"] for b in batch]

    # Pad input features
    input_features_padded = torch.stack(input_features)  # đã cùng shape (80,3000)

    # Pad labels với -100 (ignored in loss)
    max_len = max(l.size(0) for l in labels)
    labels_padded = torch.full((len(labels), max_len), -100, dtype=torch.long)
    for i, l in enumerate(labels):
        labels_padded[i, : l.size(0)] = l

    return {
        "input_features": input_features_padded,
        "labels": labels_padded,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    split_dir = data_root / args.split
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.local_model or args.model_id
    logger.info("Load processor & model từ: %s", model_path)

    from transformers import WhisperForConditionalGeneration, WhisperProcessor  # type: ignore
    from peft import LoraConfig, get_peft_model, TaskType  # type: ignore
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments  # type: ignore
    import functools

    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    # ── LoRA ─────────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Đặt ngôn ngữ và task cho decoder
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="vi", task="transcribe"
    )
    model.config.suppress_tokens = []

    # ── Dataset ──────────────────────────────────────────────────────────────
    jsonl_path = split_dir / f"{args.split}.jsonl"
    audio_root = split_dir
    dataset = ZaloWhisperDataset(jsonl_path, audio_root, processor)

    collate = functools.partial(collate_fn, processor=processor)

    # ── Training Arguments ────────────────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        fp16=torch.cuda.is_available(),
        learning_rate=1e-4,
        num_train_epochs=args.epochs,
        save_steps=500,
        logging_steps=50,
        warmup_steps=100,
        dataloader_num_workers=2,
        predict_with_generate=False,
        generation_max_length=448,
        report_to="none",
        seed=42,
        data_seed=42,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate,
    )

    logger.info("Bắt đầu training...")
    trainer.train()

    logger.info("Lưu model vào: %s", output_dir)
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    logger.info("Hoàn thành fine-tuning!")


if __name__ == "__main__":
    main()
