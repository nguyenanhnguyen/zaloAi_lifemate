"""Tách lỗi của một bài nộp thành bucket hành vi, và so hai bài nộp với nhau.

    python scripts/diag/action_buckets.py run.jsonl
    python scripts/diag/action_buckets.py sau.jsonl --baseline truoc.jsonl

Vì sao cần cái này cạnh `score.py`: điểm tổng nói ta tốt hơn hay tệ hơn, nó **không** nói vì
sao. Mà toàn bộ lý do có fine-tune là một bucket cụ thể -- "model không hành động": `local_v4`
đo được 48% mẫu gold cần ghi mà ta không ghi, và phép so hai bài nộp cho 181/634 mẫu bản 22.52
ghi mà bản 13.41 không ghi. Script này trả lời đúng câu đó: adapter có biến "không ghi" thành
"ghi" hay không, và khi ghi thì có ghi đúng không.

Chỉ chạy trên split `train`, vì chỉ `train` có nhãn. Không dùng `state_hash` -- hash gold phải
dựng bằng cách chạy lại gold trong ToolEnv, nên ở đây bucket dựa trên khớp lệnh gọi theo đúng
luật của `score.py` (tên trùng, tham số non-fuzzy trùng, tham số fuzzy token-F1 > 0.6).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from score import calls_match, fuzzy_params, max_matching  # noqa: E402

# Bucket, theo thứ tự ưu tiên khi phân loại. Tên ngắn để in thành bảng.
BUCKETS = ["đúng hẳn", "ghi thiếu", "ghi sai", "KHÔNG ghi", "ghi thừa", "dừng đúng"]


def kinds(data_root: Path, split: str) -> dict[str, str]:
    """tên tool -> kind. Tool nằm lồng trong apps[].tools[], không ở catalog["tools"]."""
    catalog = json.loads((data_root / split / f"{split}_tools_catalog.json")
                         .read_text(encoding="utf-8"))
    return {tool["name"]: tool.get("kind", "write")
            for app in catalog.get("apps", []) for tool in app.get("tools", [])}


def classify(submitted: list[dict], gold: list[dict], fuzzy: set[str],
             kind: dict[str, str]) -> str:
    writes = lambda calls: [c for c in calls if kind.get(c.get("name", ""), "write") == "write"]
    sub_w, gold_w = writes(submitted), writes(gold)

    if not gold_w:
        # Gold không cần ghi gì. Đúng khi ta cũng không ghi -- 111 mẫu train thuộc loại này và
        # model phải học nói DONE ngay, nếu không nó bịa thêm lệnh và mất precision.
        return "dừng đúng" if not sub_w else "ghi thừa"
    if not sub_w:
        return "KHÔNG ghi"
    matched = max_matching(sub_w, gold_w, fuzzy)
    if matched == len(gold_w) == len(sub_w):
        return "đúng hẳn"
    if matched == 0:
        return "ghi sai"          # có ghi, nhưng không lệnh nào khớp gold
    return "ghi thiếu"            # khớp một phần, hoặc khớp đủ nhưng ghi thêm lệnh lạ


def load(path: Path) -> dict[str, list[dict]]:
    out = {}
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        out[row["id"]] = row.get("tool_calls") or []
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--baseline", type=Path, help="bài nộp trước, để so chuyển dịch bucket")
    parser.add_argument("--split", default="train")
    parser.add_argument("--data-root", type=Path,
                        default=ROOT / "data/train+public_test/train+public_test")
    parser.add_argument("--show", type=int, default=8,
                        help="in ra mấy mẫu của bucket KHÔNG ghi")
    args = parser.parse_args()

    gold_all, fuzzy = {}, None
    for line in (args.data_root / args.split / f"{args.split}.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        gold_all[row["id"]] = (row.get("label") or {}).get("tool_calls") or []
    catalog = json.loads((args.data_root / args.split / f"{args.split}_tools_catalog.json")
                         .read_text(encoding="utf-8"))
    fuzzy = fuzzy_params(catalog)
    kind = kinds(args.data_root, args.split)

    runs = {"sau": load(args.submission)}
    if args.baseline:
        runs["trước"] = load(args.baseline)
    ids = sorted(set.intersection(*(set(r) for r in runs.values())) & set(gold_all))
    if not ids:
        raise SystemExit("không có mẫu nào vừa trong bài nộp vừa có nhãn "
                         f"(split {args.split}; bài nộp có {len(runs['sau'])} mẫu)")
    print(f"{len(ids)} mẫu có nhãn và có trong mọi bài nộp\n")

    labels = {}
    for name, run in runs.items():
        labels[name] = {i: classify(run[i], gold_all[i], fuzzy, kind) for i in ids}
        counts = Counter(labels[name].values())
        print(f"### {name} — {args.submission.name if name == 'sau' else args.baseline.name}")
        for bucket in BUCKETS:
            n = counts.get(bucket, 0)
            print(f"  {bucket:<10} {n:4d}  {n / len(ids) * 100:5.1f}%  {'#' * round(n / len(ids) * 40)}")
        print()

    if args.baseline:
        moved = Counter((labels["trước"][i], labels["sau"][i]) for i in ids
                        if labels["trước"][i] != labels["sau"][i])
        print(f"### chuyển dịch ({sum(moved.values())}/{len(ids)} mẫu đổi bucket)")
        for (before, after), n in moved.most_common():
            arrow = "  ✔" if after in ("đúng hẳn", "dừng đúng") else (
                "  ✘" if before in ("đúng hẳn", "dừng đúng") else "   ")
            print(f"  {before:<10} -> {after:<10} {n:4d}{arrow}")
        fixed = sum(n for (b, a), n in moved.items()
                    if b == "KHÔNG ghi" and a != "KHÔNG ghi")
        broke = sum(n for (b, a), n in moved.items()
                    if a == "KHÔNG ghi" and b != "KHÔNG ghi")
        print(f"\n  thoát khỏi 'KHÔNG ghi': {fixed}   rơi vào 'KHÔNG ghi': {broke}")
        print("  Đây là con số phán xét fine-tune. Điểm tổng chỉ nói tốt/tệ, không nói vì sao.")
        return 0

    stuck = [i for i in ids if labels["sau"][i] == "KHÔNG ghi"]
    print(f"### {min(args.show, len(stuck))} mẫu 'KHÔNG ghi' (gold cần ghi, ta không ghi)")
    for sample in stuck[:args.show]:
        names = ", ".join(c["name"] for c in gold_all[sample]
                          if kind.get(c.get("name", ""), "write") == "write")
        print(f"  {sample}  gold ghi: {names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
