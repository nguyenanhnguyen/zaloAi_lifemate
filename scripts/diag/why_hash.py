"""Decompose WHY state_hash mismatches, on the local_v4 run. Zero GPU."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("D:/z_DaiHoc/a_zaloAI")
SPLIT = ROOT / "data/train+public_test/train+public_test/train"

preds = {r["id"]: r for r in map(json.loads, (ROOT / "results/local_v4/train_check.jsonl").open(encoding="utf-8"))}
gold = {}
catalog = json.loads((SPLIT / "train_tools_catalog.json").read_text(encoding="utf-8"))
for line in (SPLIT / "train.jsonl").open(encoding="utf-8"):
    item = json.loads(line)
    if item["id"] in preds:
        gold[item["id"]] = item

# which tools mutate state: only those can change state_hash
kind = {tool["name"]: tool.get("kind", "?")
        for app in catalog["apps"] for tool in app.get("tools", [])}
print(f"catalog: {len(kind)} tools, kinds={Counter(kind.values())}", file=sys.stderr)

buckets = Counter()
detail = []
for sid, pred in preds.items():
    g = gold[sid]["label"]["tool_calls"] or []
    p = pred.get("tool_calls") or []
    gw = [c for c in g if kind.get(c["name"]) == "write"]
    pw = [c for c in p if kind.get(c["name"]) == "write"]
    gwn, pwn = Counter(c["name"] for c in gw), Counter(c["name"] for c in pw)

    if not gw and not pw:
        buckets["A. không bên nào ghi -> hash phải khớp"] += 1
        tag = "A"
    elif not gw and pw:
        buckets["B. gold KHÔNG ghi, ta GHI -> bịa thao tác"] += 1
        tag = "B"
    elif gw and not pw:
        buckets["C. gold ghi, ta KHÔNG ghi -> bỏ việc"] += 1
        tag = "C"
    elif gwn != pwn:
        buckets["D. ghi sai TOOL (tên khác nhau)"] += 1
        tag = "D"
    else:
        same = sorted((c["name"], json.dumps(c.get("arguments", {}), sort_keys=True, ensure_ascii=False)) for c in gw) == \
               sorted((c["name"], json.dumps(c.get("arguments", {}), sort_keys=True, ensure_ascii=False)) for c in pw)
        if same:
            buckets["E. ghi đúng tool VÀ đúng đối số -> hash nên khớp"] += 1
            tag = "E"
        else:
            buckets["F. đúng TOOL, sai ĐỐI SỐ"] += 1
            tag = "F"
    detail.append((tag, sid, gw, pw))

total = len(preds)
print(f"\n=== {total} mẫu local_v4, phân tách theo CALL GHI (write) — chỉ write đổi state_hash ===\n")
for name, n in sorted(buckets.items()):
    print(f"{n:4d}  {n/total:5.1%}  {name}")

# Which argument keys differ, in bucket F
print("\n=== Bucket F: khoá đối số nào sai ===")
keys = Counter()
for tag, sid, gw, pw in detail:
    if tag != "F":
        continue
    for a, b in zip(sorted(gw, key=lambda c: c["name"]), sorted(pw, key=lambda c: c["name"])):
        for k in set(a.get("arguments", {})) | set(b.get("arguments", {})):
            if a.get("arguments", {}).get(k) != b.get("arguments", {}).get(k):
                keys[f'{a["name"]}.{k}'] += 1
for k, n in keys.most_common(15):
    print(f"  {n:3d}  {k}")

print("\n=== Bucket B: ta bịa thao tác ghi gì ===")
inv = Counter()
for tag, sid, gw, pw in detail:
    if tag == "B":
        for c in pw:
            inv[c["name"]] += 1
for k, n in inv.most_common(10):
    print(f"  {n:3d}  {k}")

print("\n=== Bucket C: gold cần ghi gì mà ta bỏ ===")
miss = Counter()
for tag, sid, gw, pw in detail:
    if tag == "C":
        for c in gw:
            miss[c["name"]] += 1
for k, n in miss.most_common(10):
    print(f"  {n:3d}  {k}")

print("\n=== ví dụ bucket F (đúng tool, sai đối số), 3 mẫu ===")
shown = 0
for tag, sid, gw, pw in detail:
    if tag != "F" or shown >= 3:
        continue
    print(f"\n-- {sid}")
    print("   gold:", json.dumps(gw, ensure_ascii=False)[:400])
    print("   ta  :", json.dumps(pw, ensure_ascii=False)[:400])
    shown += 1
