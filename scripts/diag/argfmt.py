"""Bucket F nghi có lỗi DẠNG hệ thống: gold dùng ISO datetime đầy đủ, ta dùng giờ treo tường.
Và ta thêm đối số gold không có. Đo trên toàn bộ nhãn + toàn bộ dự đoán."""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path("D:/z_DaiHoc/a_zaloAI")
SPLIT = ROOT / "data/train+public_test/train+public_test/train"
ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")
WALL = re.compile(r"^\d{1,2}:\d{2}$")

catalog = json.loads((SPLIT / "train_tools_catalog.json").read_text(encoding="utf-8"))
kind = {t["name"]: t.get("kind") for a in catalog["apps"] for t in a.get("tools", [])}

# 1) Gold: dạng của mọi đối số thời gian, trên TOÀN BỘ 1545 mẫu
gold_fmt = Counter()
for line in (SPLIT / "train.jsonl").open(encoding="utf-8"):
    item = json.loads(line)
    for call in (item.get("label", {}).get("tool_calls") or []):
        for key, value in (call.get("arguments") or {}).items():
            if not isinstance(value, str):
                continue
            if ISO.match(value):
                gold_fmt["ISO đầy đủ (2026-11-11T05:30:00+07:00)"] += 1
            elif WALL.match(value):
                gold_fmt["giờ treo tường (05:30)"] += 1
print("=== GOLD, toàn bộ 1545 mẫu: dạng đối số thời gian ===")
for k, v in gold_fmt.most_common():
    print(f"  {v:5d}  {k}")

# 2) Dự đoán của ta
preds = [json.loads(l) for l in (ROOT / "results/local_v4/train_check.jsonl").open(encoding="utf-8")]
our_fmt = Counter()
for row in preds:
    for call in (row.get("tool_calls") or []):
        for key, value in (call.get("arguments") or {}).items():
            if not isinstance(value, str):
                continue
            if ISO.match(value):
                our_fmt["ISO đầy đủ"] += 1
            elif WALL.match(value):
                our_fmt["giờ treo tường"] += 1
print("\n=== TA, 100 mẫu local_v4 ===")
for k, v in our_fmt.most_common():
    print(f"  {v:5d}  {k}")

# 3) Ta thêm đối số gold không có, ở những call CÙNG TÊN
gold_by = {}
for line in (SPLIT / "train.jsonl").open(encoding="utf-8"):
    item = json.loads(line)
    gold_by[item["id"]] = item["label"]["tool_calls"] or []
extra = Counter()
missing = Counter()
pairs = 0
for row in preds:
    g = {c["name"]: c.get("arguments") or {} for c in gold_by.get(row["id"], [])}
    for call in (row.get("tool_calls") or []):
        if call["name"] not in g:
            continue
        pairs += 1
        ours, theirs = set(call.get("arguments") or {}), set(g[call["name"]])
        for k in ours - theirs:
            extra[f'{call["name"]}.{k}'] += 1
        for k in theirs - ours:
            missing[f'{call["name"]}.{k}'] += 1
print(f"\n=== {pairs} call trùng tên với gold: khoá đối số ta THÊM mà gold không có ===")
for k, v in extra.most_common(12):
    print(f"  {v:4d}  {k}")
print("\n=== khoá gold CÓ mà ta thiếu ===")
for k, v in missing.most_common(12):
    print(f"  {v:4d}  {k}")
