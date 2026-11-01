"""Compare ASR decoding strategies by WER on the labelled `train` split.

Why this exists: the `train` split ships `audio_text`, so transcription quality can be measured
exactly, on the real audio, without spending a submission. ASR WER is the ceiling on the whole
pipeline's score — at 23.6% roughly one word in four reaching the LLM is wrong, and `results/local_v4`
showed samples whose transcript is unusable cannot be rescued downstream.

Every strategy here must stay legal for the competition: deterministic decoding only (no sampling),
fixed seeds, pinned revisions. Beam search with `do_sample=False` is deterministic and allowed.

    python scripts/asr_bench.py --samples 15
    python scripts/asr_bench.py --samples 15 --configs greedy beam5 --device cuda
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.asr import ASR_MODELS, ASR_REPO, WhisperASR  # noqa: E402
from src.models import model_path  # noqa: E402

# name -> (repo, WhisperASR kwargs). Keep every entry runnable on one 4 GB card.
CONFIGS: dict[str, tuple[str, dict]] = {
    "greedy": (ASR_REPO, {"beams": 1}),
    "beam5": (ASR_REPO, {"beams": 5}),
    # Beam search tends to favour short hypotheses; >1 pushes it towards longer, more complete ones.
    "beam5_len1.1": (ASR_REPO, {"beams": 5, "generate_overrides": {"length_penalty": 1.1}}),
    # Whisper loops on noisy Vietnamese audio ("nhắc tôi nhắc tôi nhắc tôi"); blocking repeated
    # 4-grams removes that failure mode without touching anything else.
    "beam5_norepeat": (ASR_REPO, {"beams": 5, "generate_overrides": {"no_repeat_ngram_size": 4}}),
    "turbo_beam5": ("erax-ai/EraX-WoW-Turbo-V1.1", {"beams": 5}),
}


def words(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", str(text)).lower()
    return re.sub(r"[^\w\s]", " ", text).split()


def wer(reference: str, hypothesis: str) -> tuple[int, int]:
    """Levenshtein distance over words, and the reference length, so a corpus WER can be pooled."""
    ref, hyp = words(reference), words(hypothesis)
    previous = list(range(len(hyp) + 1))
    for i, word in enumerate(ref, 1):
        current = [i]
        for j, other in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (word != other)))
        previous = current
    return previous[-1], len(ref)


def load_samples(data_root: Path, count: int) -> tuple[Path, list[dict]]:
    split_dir = data_root / "train"
    items = [json.loads(line) for line in (split_dir / "train.jsonl").open(encoding="utf-8")
             if line.strip()]
    items = [x for x in items if x.get("audio_text") and x.get("audio")]
    if not items:
        raise SystemExit("No train sample carries audio_text")
    # Even stride over the whole split, not the first N: the first samples are not representative
    # and a fixed stride keeps every config on exactly the same audio.
    stride = max(1, len(items) // count)
    return split_dir, items[::stride][:count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root",
                        default="data/train+public_test/train+public_test",
                        help="directory holding train/ and public_test/")
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--configs", nargs="+", default=["greedy", "beam5"],
                        help=f"any of: {', '.join(CONFIGS)}")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output", default="asr_bench.json")
    args = parser.parse_args()

    unknown = [name for name in args.configs if name not in CONFIGS]
    if unknown:
        parser.error(f"unknown config(s) {unknown}; choose from {list(CONFIGS)}")

    split_dir, samples = load_samples(Path(args.data_root), args.samples)
    print(f"{len(samples)} samples from {split_dir}")

    results = {}
    for name in args.configs:
        repo, kwargs = CONFIGS[name]
        revision = ASR_MODELS.get(repo, "")
        path = model_path(repo, revision) if revision else repo
        asr = WhisperASR(str(path), device=args.device, **kwargs)
        errors = length = 0
        seconds = 0.0
        rows = []
        try:
            for item in samples:
                audio = split_dir / item["audio"]
                started = time.perf_counter()
                text = asr.transcribe_batch([audio])[0]
                elapsed = time.perf_counter() - started
                distance, reference_words = wer(item["audio_text"], text)
                errors += distance
                length += reference_words
                seconds += elapsed
                rows.append({"id": item["id"], "wer": distance / max(1, reference_words),
                             "seconds": round(elapsed, 2), "hypothesis": text,
                             "reference": item["audio_text"]})
                print(f"  {name:16} {item['id']}  WER {distance/max(1,reference_words):6.1%}  {elapsed:5.1f}s")
        finally:
            asr.close()
        results[name] = {"repo": repo, "revision": revision, "kwargs": repr(kwargs),
                         "wer": errors / max(1, length), "seconds_per_file": seconds / len(samples),
                         "samples": rows}
        print(f"{name}: corpus WER {errors/max(1,length):.2%}, {seconds/len(samples):.1f}s/file\n")

    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    best = min(results, key=lambda k: results[k]["wer"])
    print(f"{'config':18} {'WER':>8} {'s/file':>8}")
    for name, value in sorted(results.items(), key=lambda kv: kv[1]["wer"]):
        mark = " <-- best" if name == best else ""
        print(f"{name:18} {value['wer']:7.2%} {value['seconds_per_file']:8.1f}{mark}")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
