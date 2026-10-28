"""Download các model về offline cho ZAC 2026 LifeMate.

Chạy một lần trước khi inference (cần kết nối internet):
    python scripts/download_models.py --models all
    python scripts/download_models.py --models whisper
    python scripts/download_models.py --models qwen
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WHISPER_MODEL_ID = "openai/whisper-large-v3-turbo"
QWEN_MODEL_ID = "Qwen/Qwen3-4B-Instruct"

MODELS_DIR = Path("models")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download models về local")
    p.add_argument(
        "--models",
        default="all",
        choices=["whisper", "qwen", "all"],
        help="Model cần download (default: all)",
    )
    return p.parse_args()


def download_whisper(local_dir: Path) -> None:
    """Download whisper-large-v3-turbo."""
    from huggingface_hub import snapshot_download  # type: ignore

    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s → %s ...", WHISPER_MODEL_ID, local_dir)
    snapshot_download(
        repo_id=WHISPER_MODEL_ID,
        local_dir=str(local_dir),
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
    )
    logger.info("✅ Whisper đã lưu tại: %s", local_dir.resolve())


def download_qwen(local_dir: Path) -> None:
    """Download Qwen3-4B-Instruct."""
    from huggingface_hub import snapshot_download  # type: ignore

    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s → %s ...", QWEN_MODEL_ID, local_dir)
    snapshot_download(
        repo_id=QWEN_MODEL_ID,
        local_dir=str(local_dir),
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
    )
    logger.info("✅ Qwen đã lưu tại: %s", local_dir.resolve())


def print_next_steps() -> None:
    whisper_dir = (MODELS_DIR / "whisper").resolve()
    qwen_dir = (MODELS_DIR / "qwen3-4b").resolve()
    print("\n" + "=" * 60)
    print("✅ Download hoàn tất!")
    print()
    print("📁 Đường dẫn model:")
    print(f"   Whisper : {whisper_dir}")
    print(f"   Qwen    : {qwen_dir}")
    print()
    print("🚀 Bước tiếp theo:")
    print("   1. Cài dependencies:")
    print("      pip install torch --index-url https://download.pytorch.org/whl/cu124")
    print("      pip install -r requirements.txt")
    print()
    print("   2. Chạy inference:")
    print("      python scripts/infer.py --split public_test")
    print()
    print("   3. Evaluate local (train split):")
    print("      python scripts/evaluate.py --split train --max-samples 50")
    print("=" * 60)


def main() -> None:
    args = parse_args()

    if args.models in ("whisper", "all"):
        download_whisper(MODELS_DIR / "whisper")

    if args.models in ("qwen", "all"):
        download_qwen(MODELS_DIR / "qwen3-4b")

    print_next_steps()


if __name__ == "__main__":
    main()
