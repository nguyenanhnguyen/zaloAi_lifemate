"""Score a submission against the labels of a split, using the organisers' formula.

Only `train` carries labels, so this is how a change is judged without spending a leaderboard
submission. The gold `state_hash` is not in the label file: as the rules describe, it is obtained by
replaying the gold `tool_calls` in a fresh ToolEnv and reading `env.get_state_hash()`.

Rules implemented (docs/context.txt, "Metrics"):
  - state_hash must equal the gold hash, otherwise the sample scores 0;
  - an empty gold list scores 1 only when the submission is also empty, else 0;
  - otherwise a submitted call matches a gold call when the tool name is equal, every non-fuzzy
    argument is identical, and every `x-match: "fuzzy"` argument has token-F1 > 0.6 after
    lowercasing and NFC; each call matches at most one call and order is ignored;
  - P = M/(M+U), R = M/|G|, F1 = 2PR/(P+R), and the score is the mean over samples.
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FUZZY_THRESHOLD = 0.6


def fuzzy_params(catalog: dict) -> set[str]:
    """{"sms.send.content", ...}: parameters the rules score with token-F1 instead of equality."""
    return {f"{tool['name']}.{name}"
            for app in catalog.get("apps", []) for tool in app.get("tools", [])
            for name, spec in (tool.get("parameters", {}).get("properties") or {}).items()
            if spec.get("x-match") == "fuzzy"}


def tokens(value: object) -> list[str]:
    return unicodedata.normalize("NFC", str(value)).lower().split()


def token_f1(left: object, right: object) -> float:
    a, b = tokens(left), tokens(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    shared = 0
    pool = list(b)
    for token in a:
        if token in pool:
            pool.remove(token)
            shared += 1
    if not shared:
        return 0.0
    precision, recall = shared / len(a), shared / len(b)
    return 2 * precision * recall / (precision + recall)


def calls_match(submitted: dict, gold: dict, fuzzy: set[str]) -> bool:
    name = submitted.get("name")
    if name != gold.get("name"):
        return False
    mine = submitted.get("arguments") or {}
    theirs = gold.get("arguments") or {}
    if not isinstance(mine, dict) or set(mine) != set(theirs):
        return False
    for key, want in theirs.items():
        if f"{name}.{key}" in fuzzy:
            if token_f1(mine[key], want) <= FUZZY_THRESHOLD:
                return False
        elif mine[key] != want:
            return False
    return True


def max_matching(submitted: list[dict], gold: list[dict], fuzzy: set[str]) -> int:
    """Largest set of pairs where each submitted and each gold call is used at most once.

    Greedy would undercount when one submitted call is eligible for several gold calls, so this
    walks augmenting paths (Kuhn's algorithm). The lists hold a handful of calls, so cost is moot.
    """
    options = [[j for j, want in enumerate(gold) if calls_match(mine, want, fuzzy)]
               for mine in submitted]
    paired: dict[int, int] = {}

    def assign(i: int, seen: set[int]) -> bool:
        for j in options[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in paired or assign(paired[j], seen):
                paired[j] = i
                return True
        return False

    return sum(assign(i, set()) for i in range(len(submitted)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a submission against a labelled split")
    parser.add_argument("submission", type=Path)
    parser.add_argument("--split", default="train", help="Only train has labels")
    parser.add_argument("--data-root", type=Path,
                        default=PROJECT_ROOT / "data/train+public_test/train+public_test")
    parser.add_argument("--show", type=int, default=10, help="List this many lowest-scoring samples")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    split_dir = args.data_root / args.split
    sys.path.insert(0, str(split_dir))
    from zaic_tools import ToolEnv

    with (split_dir / f"{args.split}_tools_catalog.json").open(encoding="utf-8") as handle:
        fuzzy = fuzzy_params(json.load(handle))
    with (split_dir / f"{args.split}.jsonl").open(encoding="utf-8") as handle:
        labels = {}
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if "label" not in item:
                    parser.error(f"Split {args.split} has no labels; only train can be scored")
                labels[item["id"]] = item["label"]
    with args.submission.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    scored = []
    hash_mismatch = empty_gold = 0
    for row in rows:
        gold_calls = labels[row["id"]]["tool_calls"]
        mine = row.get("tool_calls") or []
        # Replay the gold calls to obtain the gold hash, exactly as the organisers reproduce ours.
        env = ToolEnv(args.split, row["id"], root=args.data_root)
        for call in gold_calls:
            env.call(call["name"], call.get("arguments") or {})
        gold_hash = env.get_state_hash()
        if row.get("state_hash") != gold_hash:
            hash_mismatch += 1
            scored.append((row["id"], 0.0, "state_hash mismatch"))
            continue
        if not gold_calls:
            empty_gold += 1
            ok = not mine
            scored.append((row["id"], 1.0 if ok else 0.0,
                           "empty gold, empty submission" if ok else "empty gold, we called tools"))
            continue
        matched = max_matching(mine, gold_calls, fuzzy)
        unmatched = len(mine) - matched
        precision = matched / (matched + unmatched) if matched + unmatched else 0.0
        recall = matched / len(gold_calls)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scored.append((row["id"], f1, f"matched {matched}/{len(gold_calls)} gold, {unmatched} extra"))

    total = sum(score for _, score, _ in scored)
    count = len(scored) or 1
    print(f"\nSCORE {total / count:.4f}  over {len(scored)} samples  "
          f"(leaderboard reports this x100: {100 * total / count:.2f})")
    print(f"  perfect (1.0):      {sum(1 for _, s, _ in scored if s == 1.0)}")
    print(f"  zero:               {sum(1 for _, s, _ in scored if s == 0.0)}")
    print(f"  state_hash mismatch:{hash_mismatch}   (each one scores 0 whatever the calls are)")
    print(f"  empty gold lists:   {empty_gold}")
    if args.show:
        print(f"\nLowest {min(args.show, len(scored))}:")
        for sample_id, score, note in sorted(scored, key=lambda row: row[1])[:args.show]:
            print(f"  {sample_id}  {score:.3f}  {note}")


if __name__ == "__main__":
    main()
