"""Run LifeMate inference and write the organiser's JSONL submission format."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.asr import WhisperASR
from src.llm_agent import LLMAgent
from src.pipeline import Pipeline

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="ZAC 2026 LifeMate inference")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/train+public_test/train+public_test")
    parser.add_argument("--split", choices=("train", "public_test", "private_test"), default="public_test")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "submission.jsonl")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-steps", type=int, default=3, help="Maximum LLM calls per sample (1-3)")
    parser.add_argument("--max-seconds", type=float, default=28, help="Sample decoding time budget, at most 30 seconds")
    parser.add_argument("--allow-shared-memory", action="store_true",
                        help="Skip the free VRAM guard; Windows may use shared GPU memory")
    parser.add_argument("--metrics-output", type=Path,
                        help="Write per-sample timing and memory to a separate JSONL file")
    parser.add_argument("--whisper-model", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--low-memory", action="store_true",
                        help="Use Whisper Small and Qwen2.5-3B-Instruct")
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 3 or not 0 < args.max_seconds <= 30:
        parser.error("Use 1-3 steps and a positive time budget up to 30 seconds")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    random.seed(42)
    try:
        import numpy as np
        np.random.seed(42)
        import torch
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    split_dir = args.data_root / args.split
    sys.path.insert(0, str(split_dir))
    from zaic_tools import ToolEnv

    with (split_dir / f"{args.split}_tools_catalog.json").open(encoding="utf-8") as handle:
        catalog = json.load(handle)

    import torch
    if not torch.cuda.is_available():
        raise SystemExit(
            f"PyTorch {torch.__version__} không hỗ trợ CUDA trong env này. "
            "GPU có rảnh sau khi train xong cũng sẽ không được dùng bởi bản PyTorch này."
        )
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    logger.info(
        "GPU: %s; VRAM còn %.1f / %.1f GB",
        torch.cuda.get_device_name(0), free_bytes / 1024**3, total_bytes / 1024**3,
    )
    if free_bytes < 2.5 * 1024**3 and not args.allow_shared_memory:
        raise SystemExit(
            "GPU hiện còn dưới 2.5 GB VRAM. Hãy chạy lại khi tác vụ train khác đã giải phóng GPU."
        )
    if args.allow_shared_memory:
        logger.warning("Allowing Windows shared GPU memory; measure speed against the 30 s/sample limit")
    whisper_name = args.whisper_model or "openai/whisper-small"
    llm_name = args.llm_model or "Qwen/Qwen2.5-3B-Instruct"
    torch.set_num_threads(4)
    pipeline = Pipeline(WhisperASR(whisper_name), LLMAgent(llm_name),
                        max_steps=args.max_steps, max_seconds=args.max_seconds)
    count = 0
    elapsed_total = 0.0
    slow_count = 0
    process = None
    try:
        import psutil
        process = psutil.Process()
    except ImportError:
        pass
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics = None
    if args.metrics_output:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        metrics = args.metrics_output.open("w", encoding="utf-8")
    try:
      with (split_dir / f"{args.split}.jsonl").open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            if args.max_samples is not None and count >= args.max_samples:
                break
            item = json.loads(line)
            env = ToolEnv(args.split, item["id"], root=args.data_root)
            torch.cuda.reset_peak_memory_stats(0)
            started = time.perf_counter()
            failed = False
            try:
                calls, state_hash = pipeline.infer(
                    item, env, catalog=catalog, audio_path=split_dir / item["audio"]
                )
            except Exception:
                logger.exception("Sample %s failed", item["id"])
                calls, state_hash = env.get_tool_calls(), env.get_state_hash()
                failed = True
            elapsed = time.perf_counter() - started
            elapsed_total += elapsed
            slow_count += elapsed > 30
            target.write(json.dumps({"id": item["id"], "tool_calls": calls, "state_hash": state_hash}, ensure_ascii=False) + "\n")
            target.flush()
            if metrics is not None:
                record = {
                    "id": item["id"], "seconds": round(elapsed, 3),
                    "asr_seconds": round(pipeline.last_asr_seconds, 3),
                    "llm_seconds": round(pipeline.last_llm_seconds, 3),
                    "process_ram_mb": round(process.memory_info().rss / 1024**2) if process else None,
                    "cuda_peak_allocated_mb": round(torch.cuda.max_memory_allocated(0) / 1024**2),
                    "over_30_seconds": elapsed > 30, "failed": failed,
                    "stop_reason": pipeline.stop_reason,
                }
                metrics.write(json.dumps(record, ensure_ascii=False) + "\n")
                metrics.flush()
            count += 1
            logger.info("Processed %d: %s (%.2fs, %.2fs average)", count, item["id"], elapsed, elapsed_total / count)
    finally:
        if metrics is not None:
            metrics.close()

    logger.info("Wrote %d samples to %s; %d over 30 seconds", count, args.output, slow_count)


if __name__ == "__main__":
    main()
