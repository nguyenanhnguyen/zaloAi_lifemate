"""Bucket C = 48% mẫu gold cần ghi mà ta không ghi. KHÔNG THỂ hay KHÔNG CHỊU?"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("D:/z_DaiHoc/a_zaloAI")
SPLIT = ROOT / "data/train+public_test/train+public_test/train"
sys.path.insert(0, str(ROOT))
from src.pipeline import _available_tools  # noqa: E402

preds = {r["id"]: r for r in map(json.loads, (ROOT / "results/local_v4/train_check.jsonl").open(encoding="utf-8"))}
metrics = {r["id"]: r for r in map(json.loads, (ROOT / "results/local_v4/train_check.metrics.jsonl").open(encoding="utf-8"))}
catalog = json.loads((SPLIT / "train_tools_catalog.json").read_text(encoding="utf-8"))
kind = {t["name"]: t.get("kind") for a in catalog["apps"] for t in a.get("tools", [])}
items = {}
for line in (SPLIT / "train.jsonl").open(encoding="utf-8"):
    item = json.loads(line)
    if item["id"] in preds:
        items[item["id"]] = item

stop = Counter()
installed_missing = 0
pred_called_app = Counter()
examples = []
n = 0
for sid, item in items.items():
    gw = [c for c in (item["label"]["tool_calls"] or []) if kind.get(c["name"]) == "write"]
    pw = [c for c in (preds[sid].get("tool_calls") or []) if kind.get(c["name"]) == "write"]
    if not (gw and not pw):
        continue
    n += 1
    stop[metrics[sid]["stop_reason"]] += 1
    # was the gold write tool even OFFERED to the model (app installed on this sample)?
    allowed = {t["name"] for t in _available_tools(catalog, item.get("context", {}).get("installed_apps", {}))}
    if any(c["name"] not in allowed for c in gw):
        installed_missing += 1
    for c in preds[sid].get("tool_calls") or []:
        pred_called_app[c["name"]] += 1
    if len(examples) < 4:
        examples.append((sid, metrics[sid]["stop_reason"],
                         [c["name"] for c in gw],
                         [c["name"] for c in preds[sid].get("tool_calls") or []]))

print(f"=== Bucket C: {n} mẫu ===\n")
print("stop_reason:")
for k, v in stop.most_common():
    print(f"  {v:4d}  {v/n:5.1%}  {k}")
print(f"\ngold write tool KHÔNG nằm trong app đã cài (tức là ta không được phép gọi): {installed_missing}/{n}")
print("\nTa đã gọi những tool nào trong các mẫu đó:")
for k, v in pred_called_app.most_common(12):
    print(f"  {v:4d}  {k}  [{kind.get(k)}]")
print("\nví dụ:")
for sid, reason, g, p in examples:
    print(f"\n-- {sid}  stop={reason}")
    print(f"   gold write : {g}")
    print(f"   ta gọi     : {p}")
