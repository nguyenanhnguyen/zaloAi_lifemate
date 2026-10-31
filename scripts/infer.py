"""Single entry point: prepare models, run offline inference, write submission JSONL."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import random
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.asr import ASR_REPO, ASR_REVISION, PhoWhisperASR
from src.llm_agent import LLM_REPO, LLM_REVISION, LLMAgent
from src.models import model_path
from src.pipeline import Pipeline

logger = logging.getLogger(__name__)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="ZAC LifeMate: Qwen3-ASR-0.6B + Qwen3-4B-Instruct-2507, GPU")
    parser.add_argument("--prepare-only", action="store_true", help="Download pinned models without running inference")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/train+public_test/train+public_test")
    parser.add_argument("--split", choices=("train", "public_test", "private_test"), default="public_test")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "submission.jsonl")
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--sample-ids", nargs="+", help="Run selected IDs, in original dataset order")
    parser.add_argument("--resume", action="store_true", help="Continue an output containing a valid completed prefix")
    parser.add_argument("--max-steps", type=int, default=8, help="Bounded LLM turns per sample (1-16)")
    parser.add_argument("--max-seconds", type=float, default=30, help="Sample budget; 30 matches the competition's A30 rule")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--context-size", type=int, default=8192, help="Total input/output context, at most 16384")
    parser.add_argument("--asr-max-new-tokens", type=int, default=256)
    parser.add_argument("--asr-model", default=ASR_REPO, help="HF repo id or local directory")
    parser.add_argument("--llm-model", help="Local HF directory; default is pinned Qwen3-4B-Instruct-2507")
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 16 or not 0 < args.max_seconds <= 3600:
        parser.error("Use 1-16 turns and 0 < --max-seconds <= 3600")
    if not 1024 <= args.context_size <= 16384 or not 64 <= args.max_new_tokens < args.context_size:
        parser.error("Context must be 1024-16384 tokens; output must fit inside context")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("Invalid sample count")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.prepare_only:
        model_path(ASR_REPO, ASR_REVISION, download=True)
        model_path(LLM_REPO, LLM_REVISION, download=True)
        logger.info("Models are ready")
        return

    # This run must not silently fetch models or use remote inference.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import psutil
    import torch
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    else:
        logger.warning("No CUDA GPU detected; running on CPU will be far slower than the 30 s/sample budget")
    if args.max_seconds > 30:
        logger.warning("Local quality run: %.0f s/sample. Competition limit is 30 s on A30.", args.max_seconds)
    split_dir = args.data_root / args.split
    sys.path.insert(0, str(split_dir))
    from zaic_tools import ToolEnv
    with (split_dir / f"{args.split}_tools_catalog.json").open(encoding="utf-8") as handle:
        catalog = json.load(handle)
    with (split_dir / f"{args.split}.jsonl").open(encoding="utf-8") as source:
        items = [json.loads(line) for line in source if line.strip()]
    if args.sample_ids:
        wanted = set(args.sample_ids)
        missing = wanted - {item["id"] for item in items}
        if missing:
            parser.error(f"Unknown sample IDs: {sorted(missing)}")
        items = [item for item in items if item["id"] in wanted]
    if args.max_samples:
        items = items[:args.max_samples]
    completed = 0
    if args.resume and args.output.exists():
        with args.output.open(encoding="utf-8") as handle:
            previous = [json.loads(line) for line in handle if line.strip()]
        if len(previous) > len(items) or [x.get("id") for x in previous] != [x["id"] for x in items[:len(previous)]]:
            parser.error("Existing output is not a prefix of this run; choose another output path")
        if any(set(x) != {"id", "tool_calls", "state_hash"} for x in previous):
            parser.error("Existing output has an invalid submission schema")
        completed = len(previous)
    items = items[completed:]
    if not items:
        logger.info("All requested samples already present: %s", args.output)
        return
    logger.info("Will process %d audio samples; already completed %d", len(items), completed)
    asr_path = args.asr_model if Path(args.asr_model).is_dir() else model_path(args.asr_model, ASR_REVISION)
    asr = PhoWhisperASR(asr_path, max_new_tokens=args.asr_max_new_tokens)
    llm = LLMAgent(args.llm_model, context_size=args.context_size, max_new_tokens=args.max_new_tokens)
    pipeline = Pipeline(asr, llm, max_steps=args.max_steps, max_seconds=args.max_seconds)
    metrics_path = args.metrics_output or args.output.with_suffix(".metrics.jsonl")
    if metrics_path.resolve() == args.output.resolve():
        llm.close()
        parser.error("Metrics and submission must have different paths")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    total_seconds = 0.0
    failures = slow = 0
    mode = "a" if args.resume else "w"
    try:
        with args.output.open(mode, encoding="utf-8") as target, metrics_path.open(mode, encoding="utf-8") as metrics:
            for count, item in enumerate(items, 1):
                env = ToolEnv(args.split, item["id"], root=args.data_root)
                started = time.perf_counter()
                failed = False
                try:
                    calls, state_hash = pipeline.infer(item, env, catalog=catalog,
                                                       audio_path=split_dir / item["audio"], split_dir=split_dir)
                except Exception as exc:
                    logger.exception("Sample %s failed", item["id"])
                    pipeline.stop_reason = "time_limit" if isinstance(exc, TimeoutError) else "error"
                    calls, state_hash = env.get_tool_calls(), env.get_state_hash()
                    failed = True
                elapsed = time.perf_counter() - started
                failures += failed
                slow += elapsed > 30
                total_seconds += elapsed
                target.write(json.dumps({"id": item["id"], "tool_calls": calls, "state_hash": state_hash}, ensure_ascii=False) + "\n")
                target.flush()
                ram = process.memory_info().rss
                for child in process.children(recursive=True):
                    try:
                        ram += child.memory_info().rss
                    except psutil.Error:
                        pass
                free, total = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
                record = {"id": item["id"], "seconds": round(elapsed, 3),
                          "asr_seconds": round(pipeline.last_asr_seconds, 3),
                          "llm_seconds": round(pipeline.last_llm_seconds, 3),
                          "transcript": pipeline.last_transcript, "llm_turns": pipeline.steps,
                          "process_tree_ram_mb": round(ram / 1024**2),
                          "whole_gpu_used_mb": round((total-free) / 1024**2),
                          "over_30_seconds": elapsed > 30, "failed": failed,
                          "stop_reason": pipeline.stop_reason,
                          "asr_model": args.asr_model, "llm_model": args.llm_model or LLM_REPO}
                metrics.write(json.dumps(record, ensure_ascii=False) + "\n")
                metrics.flush()
                logger.info("Processed %d/%d: %s (%.1f s; average %.1f s)",
                            count, len(items), item["id"], elapsed, total_seconds/count)
    finally:
        llm.close()
    logger.info("Wrote %d samples to %s; %d failed; %d over 30 s", completed+len(items), args.output, failures, slow)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
