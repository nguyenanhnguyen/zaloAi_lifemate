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

from src.asr import _load_model
from src.llm_agent import LLMAgent
from src.pipeline import Pipeline

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="ZAC 2026 LifeMate inference")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/train+public_test/train+public_test")
    parser.add_argument("--split", choices=("train", "public_test", "private_test"), default="public_test")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "submission.jsonl")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--whisper-model", default="turbo")
    parser.add_argument("--llm-model", default="Qwen/Qwen3-4B-Instruct-2507")
    args = parser.parse_args()
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

    pipeline = Pipeline(_load_model(args.whisper_model), LLMAgent(args.llm_model))
    count = 0
    elapsed_total = 0.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (split_dir / f"{args.split}.jsonl").open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            if args.max_samples is not None and count >= args.max_samples:
                break
            item = json.loads(line)
            env = ToolEnv(args.split, item["id"], root=args.data_root)
            started = time.perf_counter()
            try:
                calls, state_hash = pipeline.infer(
                    item, env, catalog=catalog, audio_path=split_dir / item["audio"]
                )
            except Exception:
                logger.exception("Sample %s failed", item["id"])
                calls, state_hash = env.get_tool_calls(), env.get_state_hash()
            elapsed_total += time.perf_counter() - started
            target.write(json.dumps({"id": item["id"], "tool_calls": calls, "state_hash": state_hash}, ensure_ascii=False) + "\n")
            target.flush()
            count += 1
            logger.info("Processed %d: %s (%.2fs average)", count, item["id"], elapsed_total / count)

    logger.info("Wrote %d samples to %s", count, args.output)


if __name__ == "__main__":
    main()
