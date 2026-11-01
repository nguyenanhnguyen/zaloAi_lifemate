"""Build the supervised fine-tuning dataset from the labelled `train` split.

The labels are already agent trajectories: `label.tool_calls` aligns 1:1 with `label.tool_results`
in all 1434 labelled samples, so each sample replays as a multi-turn conversation where every
assistant turn is a gold tool call and the final assistant turn is DONE. Nothing is invented.

Rules this stays inside (docs/context.txt):
  - trains only on `train`, which ships labels. Labelling public_test/private_test to train or tune
    a solution is forbidden, so neither split is read here.
  - the organisers require the training code to be submitted with the solution, so this lives in
    the repo rather than in a notebook.

Why train on OUR ASR transcript rather than the clean `audio_text`: at inference the model only ever
sees ASR output, at ~24% WER. Training on the clean text teaches it to rely on words it will never
receive. Pass --transcripts to use a cache produced by scripts/infer.py's ASR phase; --text-source
controls the choice, and `both` emits each sample twice so the model sees clean and noisy wording for
the same gold calls. The transcripts come from the organisers' own train audio, so this is not
external synthetic data.

    python scripts/build_sft.py --transcripts train_all.transcripts.jsonl --text-source asr
    python scripts/build_sft.py --text-source audio_text --out sft            # no ASR run needed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import SYSTEM_PROMPT, _app_hints, _available_tools, _native_tools  # noqa: E402,F401


def holdout(sample_id: str, fraction: float) -> bool:
    """Deterministic split by id hash, so a rebuild never leaks eval samples into training."""
    digest = hashlib.sha256(sample_id.encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF < fraction


def agent_input(item: dict, request_text: str, history_texts: list[str]) -> dict:
    """The exact user payload the inference pipeline builds; train and infer must match."""
    context = item.get("context", {})
    history = []
    pending = iter(history_texts)
    for turn in item.get("history", []):
        if turn.get("role") == "user":
            history.append({"role": "user", "text": next(pending, "")})
        elif turn.get("role") == "assistant":
            history.append({key: turn[key] for key in ("role", "text", "tool_calls", "tool_results")
                            if key in turn})
    return {"context": {key: context.get(key) for key in
                        ("current_time", "timezone", "location", "user_profile")},
            "completed_history": history,
            "latest_request": request_text}


def trajectory(item: dict, catalog: dict, request_text: str, history_texts: list[str]) -> dict | None:
    """One sample -> one chat conversation whose assistant turns are the gold calls, then DONE."""
    label = item.get("label") or {}
    calls = label.get("tool_calls") or []
    results = label.get("tool_results") or []
    if calls and len(calls) != len(results):
        return None
    tools = _available_tools(catalog, item.get("context", {}).get("installed_apps", {}))
    if not tools:
        return None
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(
                    agent_input(item, request_text, history_texts),
                    ensure_ascii=False, separators=(",", ":"))}]
    for index, (call, result) in enumerate(zip(calls, results)):
        payload = {"name": call["name"], "arguments": call.get("arguments", {})}
        messages.append({"role": "assistant",
                         "content": f"<tool_call>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_call>"})
        messages.append({"role": "tool", "tool_call_id": f"call_{index}",
                         "content": json.dumps(result, ensure_ascii=False, separators=(",", ":"))})
    # The closing turn is as important as the calls: 111 samples have an empty gold list, and a
    # model that never learns to stop will always invent one more call and lose precision.
    messages.append({"role": "assistant", "content": "DONE"})
    return {"id": item["id"], "messages": messages, "tools": _native_tools(tools),
            "gold_calls": len(calls)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data/train+public_test/train+public_test")
    parser.add_argument("--transcripts", help="JSONL cache of {path, text} from the ASR phase")
    parser.add_argument("--text-source", choices=["asr", "audio_text", "both"], default="asr")
    parser.add_argument("--out", default="sft", help="prefix: writes <out>.train.jsonl / <out>.eval.jsonl")
    parser.add_argument("--eval-fraction", type=float, default=0.15)
    parser.add_argument("--count-tokens", action="store_true",
                        help="report the token-length distribution with the LLM's own tokenizer")
    args = parser.parse_args()

    split_dir = Path(args.data_root) / "train"
    items = [json.loads(line) for line in (split_dir / "train.jsonl").open(encoding="utf-8")
             if line.strip()]
    catalog = json.loads((split_dir / "train_tools_catalog.json").read_text(encoding="utf-8"))

    transcripts: dict[str, str] = {}
    if args.transcripts and Path(args.transcripts).exists():
        for line in Path(args.transcripts).open(encoding="utf-8"):
            if not line.strip():
                continue
            record = json.loads(line)
            # "audio" is the portable split-relative key; "path" is the older absolute one.
            key = str(split_dir / record["audio"]) if "audio" in record else record["path"]
            transcripts[key] = record["text"]
    if args.text_source in {"asr", "both"} and not transcripts:
        parser.error("--text-source asr needs --transcripts; run the ASR phase first "
                     "(scripts/infer.py --asr-phase prefetch) or use --text-source audio_text")

    sources = ["asr", "audio_text"] if args.text_source == "both" else [args.text_source]
    rows: dict[str, list[dict]] = {"train": [], "eval": []}
    skipped = missing_asr = 0
    for item in items:
        if not (item.get("label") or {}).get("tool_calls") and (item.get("label") or {}).get("tool_calls") != []:
            skipped += 1
            continue
        own = str(split_dir / item["audio"])
        history_paths = [str(split_dir / turn["audio"]) for turn in item.get("history", [])
                         if turn.get("role") == "user" and turn.get("audio")]
        for source in sources:
            if source == "asr":
                if own not in transcripts:
                    missing_asr += 1
                    continue
                request = transcripts[own]
                history_texts = [transcripts.get(path, "") for path in history_paths]
            else:
                request = item.get("audio_text") or ""
                # Only the sample's own clean transcript is published; history falls back to ASR.
                history_texts = [transcripts.get(path, "") for path in history_paths]
            if not request:
                continue
            row = trajectory(item, catalog, request, history_texts)
            if row is None:
                skipped += 1
                continue
            row["text_source"] = source
            rows["eval" if holdout(item["id"], args.eval_fraction) else "train"].append(row)

    for name, bucket in rows.items():
        path = Path(f"{args.out}.{name}.jsonl")
        with path.open("w", encoding="utf-8") as handle:
            for row in bucket:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{path}: {len(bucket)} conversations")
    print(f"skipped {skipped} samples without usable labels; {missing_asr} without an ASR transcript")
    ids = {row["id"] for row in rows["train"]} & {row["id"] for row in rows["eval"]}
    if ids:
        sys.exit(f"Split leak: {len(ids)} ids in both files")
    print("no id appears in both splits")

    if args.count_tokens:
        from transformers import AutoTokenizer
        from src.llm_agent import LLM_REPO, LLM_REVISION
        tokenizer = AutoTokenizer.from_pretrained(LLM_REPO, revision=LLM_REVISION)
        lengths = []
        for row in rows["train"]:
            # return_dict, then index input_ids: len(BatchEncoding) counts its FIELDS, not tokens,
            # which is the same mismeasurement the Kaggle notebook's v2 made.
            encoded = tokenizer.apply_chat_template(row["messages"], tools=row["tools"],
                                                    tokenize=True, return_dict=True,
                                                    return_tensors="pt")
            lengths.append(int(encoded["input_ids"].shape[1]))
        lengths.sort()
        if lengths:
            import statistics
            print(f"\ntrajectory tokens: min {lengths[0]}  median {statistics.median(lengths):.0f}"
                  f"  p95 {lengths[int(0.95*len(lengths))]}  max {lengths[-1]}")
            for limit in (4096, 6144, 8192, 12288, 16384):
                fits = sum(1 for value in lengths if value <= limit)
                print(f"  seq len {limit:6}: {fits/len(lengths):5.1%} of conversations fit whole")


if __name__ == "__main__":
    main()
