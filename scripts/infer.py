"""Script chạy inference chính cho ZAC 2026 LifeMate Agent.

Chạy từ thư mục gốc project:
    python scripts/infer.py --split public_test --output submission.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Cố định seed ─────────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

try:
    import torch
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
except ImportError:
    pass

# ── Đường dẫn gốc ────────────────────────────────────────────────────────────
DATA_ROOT = Path("data/train+public_test/train+public_test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ZAC 2026 LifeMate — Inference")
    p.add_argument("--split", default="public_test",
                   choices=["public_test", "train", "private_test"],
                   help="Split cần chạy (default: public_test)")
    p.add_argument("--output", default="submission.jsonl",
                   help="File output JSONL (default: submission.jsonl)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Giới hạn số sample (debug)")
    return p.parse_args()


def setup_zaic_tools(split: str) -> None:
    """Thêm zaic_tools của split vào sys.path."""
    zaic_path = str(DATA_ROOT / split)
    if zaic_path not in sys.path:
        sys.path.insert(0, zaic_path)
    logger.info("zaic_tools path: %s", zaic_path)


def load_pipeline():
    """Load ASR và LLM một lần, tái sử dụng cho tất cả sample."""
    from src.asr import _load_model as load_whisper  # type: ignore
    from src.pipeline import Pipeline  # type: ignore

    logger.info("Đang load Whisper model...")
    whisper_model = load_whisper()
    logger.info("Đang load LLM pipeline...")
    pipeline = Pipeline(whisper_model=whisper_model)
    return pipeline


def run_inference(
    pipeline,
    item: dict,
    split: str,
    env_cls,
    output_fh,
) -> None:
    """Chạy inference cho một sample, ghi kết quả ra file."""
    from zaic_tools import ToolEnv  # type: ignore  # noqa: F401

    sample_id = item["id"]
    env = ToolEnv(split, sample_id, root=DATA_ROOT.parent.parent)

    tool_calls, state_hash = pipeline.infer(item, env, split=split, data_root=DATA_ROOT)

    record = {
        "id": sample_id,
        "tool_calls": tool_calls,
        "state_hash": state_hash,
    }
    output_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    output_fh.flush()


def main() -> None:
    args = parse_args()
    split = args.split

    # Cần setup trước khi import zaic_tools
    setup_zaic_tools(split)

    pipeline = load_pipeline()

    jsonl_path = DATA_ROOT / split / f"{split}.jsonl"
    logger.info("Đọc dữ liệu từ: %s", jsonl_path)

    items = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    if args.max_samples is not None:
        items = items[: args.max_samples]
        logger.info("Debug mode: chỉ chạy %d sample đầu", len(items))

    logger.info("Tổng số sample: %d", len(items))

    # Import sau khi setup_zaic_tools
    from zaic_tools import ToolEnv  # type: ignore  # noqa: F401

    times: list[float] = []
    output_path = Path(args.output)

    try:
        from tqdm import tqdm
        iterator = tqdm(items, desc=f"Inference [{split}]", unit="sample")
    except ImportError:
        logger.warning("tqdm không cài — chạy không có progress bar")
        iterator = items  # type: ignore[assignment]

    with open(output_path, "w", encoding="utf-8") as fout:
        for item in iterator:
            t0 = time.perf_counter()
            try:
                run_inference(pipeline, item, split, ToolEnv, fout)
            except Exception as exc:  # noqa: BLE001
                logger.error("Lỗi sample %s: %s", item.get("id"), exc, exc_info=True)
                # Ghi dòng rỗng để không mất sample
                fout.write(json.dumps({"id": item["id"], "tool_calls": [], "state_hash": ""}, ensure_ascii=False) + "\n")
                fout.flush()
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

    # ── Thống kê cuối ──────────────────────────────────────────────────────
    total = len(times)
    avg = sum(times) / total if total else 0.0
    logger.info("=" * 50)
    logger.info("Hoàn thành!")
    logger.info("  Số sample    : %d", total)
    logger.info("  TB / sample  : %.2f giây", avg)
    logger.info("  Tổng thời gian: %.1f giây", sum(times))
    logger.info("  Output       : %s", output_path.resolve())


if __name__ == "__main__":
    main()
