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
# Must be set before CUDA initializes: reduces allocator fragmentation that otherwise OOMs on a
# near-full small GPU even when mem_get_info reports enough free. torch >=2.7 renamed the env var
# to PYTORCH_ALLOC_CONF; set both names since this venv's exact torch version may vary.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from src.asr import ASR_MODELS, ASR_REPO, ASR_REVISION, CachedASR, WhisperASR, asr_revision
from src.llm_agent import LLM_REPO, LLM_REVISION, LLMAgent
from src.models import model_path
from src.pipeline import Pipeline, audio_paths

logger = logging.getLogger(__name__)


def _asr_rev(parser: argparse.ArgumentParser, repo: str) -> str:
    """Refuse an unpinned repo id: the competition requires a reproducible, offline checkpoint."""
    revision = asr_revision(repo)
    if not revision:
        parser.error(f"--asr-model {repo} has no pinned revision. Add it to src/asr.py::ASR_MODELS "
                     f"(check its licence and release date first), or pass a local directory. "
                     f"Known: {', '.join(ASR_MODELS)}")
    return revision


def _prefetch_transcripts(asr_path: str, *, items: list[dict], split_dir: Path, cache_path: Path,
                          device: str, max_new_tokens: int) -> tuple[dict[str, str], dict[str, float]]:
    """Transcribe every sample's audio with the ASR alone on the card, then free its weights.

    Results are appended to cache_path as they are produced, so an interrupted run resumes without
    redoing ASR. Returns (path -> transcript, sample id -> seconds spent).
    """
    transcripts: dict[str, str] = {}
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    transcripts[record["path"]] = record["text"]
        logger.info("Reusing %d cached transcripts from %s", len(transcripts), cache_path)
    pending = [item for item in items
               if any(str(path) not in transcripts
                      for path in audio_paths(item, split_dir / item["audio"], split_dir))]
    seconds: dict[str, float] = {item["id"]: 0.0 for item in items}
    if not pending:
        return transcripts, seconds
    logger.info("ASR phase: %d of %d samples still to transcribe on %s", len(pending), len(items), device)
    asr = WhisperASR(asr_path, max_new_tokens=max_new_tokens, device=device)
    started_phase = time.perf_counter()
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a", encoding="utf-8") as cache:
            for count, item in enumerate(pending, 1):
                paths = audio_paths(item, split_dir / item["audio"], split_dir)
                started = time.perf_counter()
                texts = asr.transcribe_batch(paths)
                seconds[item["id"]] = time.perf_counter() - started
                for path, text in zip(paths, texts):
                    transcripts[str(path)] = text
                    cache.write(json.dumps({"path": str(path), "text": text}, ensure_ascii=False) + "\n")
                cache.flush()
                if count % 25 == 0 or count == len(pending):
                    done = time.perf_counter() - started_phase
                    logger.info("ASR phase %d/%d (%.1f s elapsed, %.1f s/sample, ~%.0f min left)",
                                count, len(pending), done, done / count,
                                (len(pending) - count) * done / count / 60)
    finally:
        asr.close()
    return transcripts, seconds


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="ZAC LifeMate: PhoWhisper-large + Qwen3-4B-Instruct-2507, GPU")
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
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--prompt-token-budget", type=int, default=1800,
                        help="Route/compact tool schemas down to about this many prompt tokens. "
                             "Prefill cost explodes past ~2k tokens on a small GPU (measured: 1k "
                             "prompt 5.5 s vs 3.9k prompt 252 s), so this is the main speed lever.")
    parser.add_argument("--context-size", type=int, default=4096,
                        help="Total input/output context, at most 16384. Matches the Kaggle notebook's "
                             "T4 value; raising it noticeably increases peak attention memory per LLM turn.")
    parser.add_argument("--asr-max-new-tokens", type=int, default=256)
    parser.add_argument("--asr-model", default=ASR_REPO,
                        help="A pinned Whisper checkpoint or a local directory. Known: "
                             + ", ".join(ASR_MODELS))
    parser.add_argument("--llm-model", help="Local HF directory; default is pinned Qwen3-4B-Instruct-2507")
    parser.add_argument("--llm-precision", choices=("auto", "bf16", "4bit"), default="auto",
                        help="bf16 matches the organisers' A30. 4bit (~2.3 GB vs 8.0 GB) keeps the "
                             "weights inside a small card's VRAM instead of streaming them over "
                             "PCIe every token. auto: 4bit when the GPU has under 10 GiB VRAM.")
    parser.add_argument("--asr-device", choices=("auto", "cuda", "cpu"), default="auto",
                        help="auto: GPU in the prefetch phase, else CPU on a GPU under 10 GiB VRAM")
    parser.add_argument("--asr-phase", choices=("auto", "inline", "prefetch"), default="auto",
                        help="prefetch: transcribe every sample first with the card to itself, free "
                             "the ASR weights, then load the LLM. Measured 3.1 s/file on GPU vs "
                             "24.5 s on CPU for the same transcript, and the two models cannot both "
                             "be resident on a small card. inline matches the A30/notebook path, "
                             "where per-sample time includes ASR. auto: prefetch under 10 GiB VRAM.")
    parser.add_argument("--transcript-cache", type=Path,
                        help="Where the prefetch phase stores transcripts; reused on a later run "
                             "so an interrupted LLM phase does not redo ASR. "
                             "Default: <output>.transcripts.jsonl")
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 16 or not 0 < args.max_seconds <= 86400:
        parser.error("Use 1-16 turns and 0 < --max-seconds <= 86400")
    if not 1024 <= args.context_size <= 16384 or not 64 <= args.max_new_tokens < args.context_size:
        parser.error("Context must be 1024-16384 tokens; output must fit inside context")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("Invalid sample count")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.prepare_only:
        if not Path(args.asr_model).is_dir():
            model_path(args.asr_model, _asr_rev(parser, args.asr_model), download=True)
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
    # Small cards cannot hold the 4B LLM in bf16 (8.0 GB); spilled weights stream over PCIe every
    # token and cost ~900 s per LLM turn instead of seconds. Pick the resident configuration.
    small_gpu = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 10 * 1024**3
    precision = args.llm_precision if args.llm_precision != "auto" else ("4bit" if small_gpu else "bf16")
    asr_phase = args.asr_phase if args.asr_phase != "auto" else ("prefetch" if small_gpu else "inline")
    if args.asr_device != "auto":
        asr_device = args.asr_device
    elif asr_phase == "prefetch":
        asr_device = "cuda" if torch.cuda.is_available() else "cpu"  # the card is ours this phase
    else:
        asr_device = "cpu" if small_gpu else "cuda"
    if precision != "bf16":
        logger.warning("LLM precision=%s (not the A30's bf16): local speed configuration, scores may "
                       "differ slightly from a bf16 submission. Pass --llm-precision bf16 to match.", precision)
    if asr_phase == "prefetch":
        logger.warning("ASR runs as a separate phase before the LLM, so per-sample seconds below "
                       "cover the LLM only; asr_seconds comes from the ASR phase. The competition's "
                       "30 s/sample budget applies to both together -- use --asr-phase inline to "
                       "measure a sample end to end as the organisers do.")
    asr_path = args.asr_model if Path(args.asr_model).is_dir() else model_path(args.asr_model, _asr_rev(parser, args.asr_model))
    prefetched: dict[str, float] = {}
    if asr_phase == "prefetch":
        transcripts, prefetched = _prefetch_transcripts(
            asr_path, items=items, split_dir=split_dir,
            cache_path=args.transcript_cache or args.output.with_suffix(".transcripts.jsonl"),
            device=asr_device, max_new_tokens=args.asr_max_new_tokens)
        asr = CachedASR(transcripts)
    else:
        asr = WhisperASR(asr_path, max_new_tokens=args.asr_max_new_tokens, device=asr_device)
    llm = LLMAgent(args.llm_model, context_size=args.context_size, max_new_tokens=args.max_new_tokens,
                   precision=precision, prompt_token_budget=args.prompt_token_budget)
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
                # In prefetch mode this sample's ASR already happened, so fold it back in: the
                # 30 s rule covers ASR plus LLM, not whichever half this loop happened to run.
                whole_sample = elapsed + prefetched.get(item["id"], 0.0)
                failures += failed
                slow += whole_sample > 30
                total_seconds += whole_sample
                target.write(json.dumps({"id": item["id"], "tool_calls": calls, "state_hash": state_hash}, ensure_ascii=False) + "\n")
                target.flush()
                ram = process.memory_info().rss
                for child in process.children(recursive=True):
                    try:
                        ram += child.memory_info().rss
                    except psutil.Error:
                        pass
                free, total = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
                record = {"id": item["id"], "seconds": round(whole_sample, 3),
                          "asr_seconds": round(prefetched.get(item["id"], pipeline.last_asr_seconds), 3),
                          "llm_seconds": round(pipeline.last_llm_seconds, 3),
                          "transcript": pipeline.last_transcript, "llm_turns": pipeline.steps,
                          "process_tree_ram_mb": round(ram / 1024**2),
                          "whole_gpu_used_mb": round((total-free) / 1024**2),
                          "over_30_seconds": whole_sample > 30, "failed": failed,
                          "stop_reason": pipeline.stop_reason,
                          "asr_model": args.asr_model, "asr_revision": asr_revision(args.asr_model),
                          "llm_model": args.llm_model or LLM_REPO,
                          "llm_precision": precision, "asr_device": asr_device, "asr_phase": asr_phase,
                          "context_size": args.context_size, "max_new_tokens": args.max_new_tokens,
                          "prompt_token_budget": args.prompt_token_budget}
                metrics.write(json.dumps(record, ensure_ascii=False) + "\n")
                metrics.flush()
                logger.info("Processed %d/%d: %s (%.1f s; average %.1f s)",
                            count, len(items), item["id"], whole_sample, total_seconds/count)
    finally:
        llm.close()
        asr.close()
    logger.info("Wrote %d samples to %s; %d failed; %d over 30 s", completed+len(items), args.output, failures, slow)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
