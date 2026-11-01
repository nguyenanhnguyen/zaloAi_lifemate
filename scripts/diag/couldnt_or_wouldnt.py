"""Quyết định: ở bucket C, model có NHÌN THẤY tool ghi của gold trong prompt không?

Route là theo APP. Nên nếu model đã gọi được một tool READ của ĐÚNG APP mà gold cần ghi, thì app đó
nằm trong prompt, tức tool ghe cũng nằm trong prompt -> model KHÔNG CHỊU làm, chứ không phải không
thấy. Đây là bằng chứng không cần dựng lại router.
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path("D:/z_DaiHoc/a_zaloAI")
SPLIT = ROOT / "data/train+public_test/train+public_test/train"

preds = {r["id"]: r for r in map(json.loads, (ROOT / "results/local_v4/train_check.jsonl").open(encoding="utf-8"))}
metrics = {r["id"]: r for r in map(json.loads, (ROOT / "results/local_v4/train_check.metrics.jsonl").open(encoding="utf-8"))}
catalog = json.loads((SPLIT / "train_tools_catalog.json").read_text(encoding="utf-8"))
kind = {t["name"]: t.get("kind") for a in catalog["apps"] for t in a.get("tools", [])}

saw = Counter()
witness = []
for line in (SPLIT / "train.jsonl").open(encoding="utf-8"):
    item = json.loads(line)
    sid = item["id"]
    if sid not in preds:
        continue
    gw = [c for c in (item["label"]["tool_calls"] or []) if kind.get(c["name"]) == "write"]
    pcalls = preds[sid].get("tool_calls") or []
    pw = [c for c in pcalls if kind.get(c["name"]) == "write"]
    if not (gw and not pw):
        continue
    gold_apps = {c["name"].split(".")[0] for c in gw}
    touched_apps = {c["name"].split(".")[0] for c in pcalls if kind.get(c["name"]) in {"read", "write"}}
    overlap = gold_apps & touched_apps
    if overlap:
        saw["ĐÃ chạm đúng app của gold -> thấy tool ghi mà KHÔNG CHỊU gọi"] += 1
        if len(witness) < 5:
            witness.append((sid, metrics[sid]["stop_reason"], sorted(gold_apps),
                            [c["name"] for c in pcalls]))
    elif touched_apps:
        saw["gọi app KHÁC hẳn -> route sai app, hoặc chọn sai app"] += 1
    else:
        saw["không gọi tool read/write nào -> không bằng chứng"] += 1

total = sum(saw.values())
print(f"=== Bucket C: {total} mẫu ===\n")
for k, v in saw.most_common():
    print(f"  {v:4d}  {v/total:5.1%}  {k}")
print("\nbằng chứng — model đọc đúng app rồi dừng:")
for sid, reason, apps, calls in witness:
    print(f"\n-- {sid}  stop={reason}")
    print(f"   app gold cần ghi : {apps}")
    print(f"   ta gọi           : {calls}")
