"""Đánh giá local pipeline trên train split (có label).

So sánh tool_calls dự đoán vs label["tool_calls"] và tính P/R/F1.

Chạy từ thư mục gốc project:
    python scripts/evaluate.py --split train --max-samples 50
    python scripts/evaluate.py --submission submission.jsonl  # evaluate file sẵn
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import unicodedata
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_ROOT = Path("data/train+public_test/train+public_test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ZAC 2026 LifeMate — Local Evaluation")
    p.add_argument("--split", default="train",
                   choices=["train", "public_test"],
                   help="Split cần evaluate (default: train)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Giới hạn số sample (debug)")
    p.add_argument("--submission", default=None,
                   help="Evaluate file submission.jsonl có sẵn thay vì chạy pipeline")
    return p.parse_args()


# ── Token F1 ─────────────────────────────────────────────────────────────────

def _normalize(text: str) -> list[str]:
    """NFC + lowercase + split thành tokens."""
    text = unicodedata.normalize("NFC", text).lower()
    return text.split()


def token_f1(pred: str, gold: str) -> float:
    """Tính token-F1 giữa hai chuỗi."""
    pred_tokens = _normalize(str(pred))
    gold_tokens = _normalize(str(gold))
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    # Đếm tần suất overlap
    from collections import Counter
    pc, gc = Counter(pred_tokens), Counter(gold_tokens)
    overlap = sum((pc & gc).values())
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Tool call matching ────────────────────────────────────────────────────────

def _is_fuzzy_arg(arg_name: str, tools_catalog: dict | None) -> bool:
    """Heuristic: coi tất cả string arg có thể là fuzzy (thực tế cần catalog)."""
    # Nếu có catalog, có thể kiểm tra x-match: "fuzzy"
    # Ở đây dùng heuristic đơn giản: không fuzzy nếu arg_name là *_id / enum-like
    non_fuzzy_hints = {"id", "_id", "phone", "power", "mode", "type", "status", "floor"}
    return not any(hint in arg_name for hint in non_fuzzy_hints)


def args_match(pred_args: dict, gold_args: dict, tools_catalog: dict | None = None) -> bool:
    """Kiểm tra hai bộ arguments có khớp nhau không.

    - Non-fuzzy args: identical
    - Fuzzy args (string text): token-F1 > 0.6
    """
    if set(pred_args.keys()) != set(gold_args.keys()):
        return False
    for key in gold_args:
        pv = pred_args.get(key)
        gv = gold_args[key]
        if isinstance(gv, str) and _is_fuzzy_arg(key, tools_catalog):
            if token_f1(str(pv), gv) <= 0.6:
                return False
        else:
            if pv != gv:
                return False
    return True


def match_calls(
    pred_calls: list[dict],
    gold_calls: list[dict],
    tools_catalog: dict | None = None,
) -> int:
    """Đếm số matched calls (mỗi call chỉ match tối đa một lần)."""
    used_gold = set()
    matched = 0
    for pc in pred_calls:
        for gi, gc in enumerate(gold_calls):
            if gi in used_gold:
                continue
            if pc.get("name") == gc.get("name") and args_match(
                pc.get("arguments", {}),
                gc.get("arguments", {}),
                tools_catalog,
            ):
                matched += 1
                used_gold.add(gi)
                break
    return matched


# ── Scoring per sample ────────────────────────────────────────────────────────

def score_sample(pred: dict, item: dict) -> dict[str, Any]:
    """Tính điểm cho một sample."""
    label = item.get("label", {})
    gold_calls = label.get("tool_calls", [])
    pred_calls = pred.get("tool_calls", [])
    state_match = pred.get("state_hash", "") == label.get("state_hash", "GOLD_HASH")

    # Out-of-scope
    if not gold_calls:
        score = 1.0 if not pred_calls else 0.0
        return {
            "id": item["id"],
            "score": score,
            "state_match": state_match,
            "precision": score,
            "recall": score,
            "f1": score,
            "gold_n": 0,
            "pred_n": len(pred_calls),
            "matched": 0,
            "out_of_scope": True,
        }

    M = match_calls(pred_calls, gold_calls)
    G = len(gold_calls)
    U = len(pred_calls) - M  # unmatched predicted

    precision = M / len(pred_calls) if pred_calls else 0.0
    recall = M / G
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "id": item["id"],
        "score": f1,
        "state_match": state_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gold_n": G,
        "pred_n": len(pred_calls),
        "matched": M,
        "out_of_scope": False,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    split = args.split

    # Tải data (labels)
    jsonl_path = DATA_ROOT / split / f"{split}.jsonl"
    items: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    if args.max_samples:
        items = items[: args.max_samples]

    logger.info("Tổng sample: %d", len(items))

    # Lấy predictions
    predictions: list[dict] = []

    if args.submission:
        # Evaluate file submission có sẵn
        sub_path = Path(args.submission)
        pred_map: dict[str, dict] = {}
        with open(sub_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    pred_map[r["id"]] = r
        for item in items:
            predictions.append(pred_map.get(item["id"], {"id": item["id"], "tool_calls": [], "state_hash": ""}))

    else:
        # Chạy pipeline
        zaic_path = str(DATA_ROOT / split)
        if zaic_path not in sys.path:
            sys.path.insert(0, zaic_path)
        from src.pipeline import Pipeline  # type: ignore
        from src.asr import _load_model as load_whisper  # type: ignore

        logger.info("Load pipeline...")
        whisper_model = load_whisper()
        pipeline = Pipeline(whisper_model=whisper_model)

        from zaic_tools import ToolEnv  # type: ignore

        try:
            from tqdm import tqdm
            iterator = tqdm(items, desc="Evaluating", unit="sample")
        except ImportError:
            iterator = items  # type: ignore[assignment]

        for item in iterator:
            env = ToolEnv(split, item["id"], root=DATA_ROOT.parent.parent)
            tool_calls, state_hash = pipeline.infer(item, env, split=split, data_root=DATA_ROOT)
            predictions.append({"id": item["id"], "tool_calls": tool_calls, "state_hash": state_hash})

    # ── Tính điểm ──────────────────────────────────────────────────────────
    results = [score_sample(pred, item) for pred, item in zip(predictions, items)]

    logger.info("\n%s", "=" * 60)
    logger.info("KẾT QUẢ TỪNG SAMPLE:")
    for r in results:
        flag = "✓" if r["state_match"] else "✗"
        logger.info(
            "[%s] %s | state=%s | P=%.3f R=%.3f F1=%.3f | match %d/%d (gold) pred=%d",
            flag, r["id"], r["state_match"],
            r["precision"], r["recall"], r["f1"],
            r["matched"], r["gold_n"], r["pred_n"],
        )

    total = len(results)
    state_match_n = sum(1 for r in results if r["state_match"])
    avg_precision = sum(r["precision"] for r in results) / total
    avg_recall = sum(r["recall"] for r in results) / total
    avg_f1 = sum(r["f1"] for r in results) / total

    logger.info("\n%s", "=" * 60)
    logger.info("TỔNG KẾT:")
    logger.info("  Số sample      : %d", total)
    logger.info("  State match    : %d / %d (%.1f%%)", state_match_n, total, 100 * state_match_n / total)
    logger.info("  Avg Precision  : %.4f", avg_precision)
    logger.info("  Avg Recall     : %.4f", avg_recall)
    logger.info("  Avg F1         : %.4f", avg_f1)


if __name__ == "__main__":
    main()
