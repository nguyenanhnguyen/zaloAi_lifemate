"""Do xem bo doi so loc tuy chon tren cac tool ma gold goi TRONG thi an duoc bao nhieu call.

Vi sao ton tai: `match()` trong phep cham doi `set(arguments)` khop CHINH XAC. Nen them MOT
doi so loc tuy chon khong lam call sai mot phan -- no lam call chet han. `scripts/diag/argfmt.py`
dem duoc ta them `smart_home.get_status.type` x15, `bill_payment.list_bills.type` x9 tren 100
mau, trong khi gold goi hai tool do trong 86% va 100%.

Chong ro ri, hai chieu:
  - allowlist dung CHI tu mau KHONG held-out (sha256(id)[:8]/0xFFFFFFFF >= 0.15),
  - diem bao cao CHI tren mau held-out.
Nen con so o day la con so tren du lieu adapter chua tung thay, khong phai con so lac quan.

PHAI doc kem: recall KHONG phai rang buoc dang chan. `state_hash` lech o 47/50 mau local_v6, va
lech hash thi 0 diem bat ke call nao dung. Gate nay chi an diem o mau DA khop state.

    python scripts/diag/bare_args.py
    python scripts/diag/bare_args.py results/local_v6/xref_gate.jsonl --min-bare 0.95
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

GOLD = Path("data/train+public_test/train+public_test/train/train.jsonl")
DEFAULT_RUNS = ["results/local_v6/xref_gate.jsonl", "results/local_v7/base.jsonl",
                "results/local_v7/lora.jsonl", "results/local_v4/train_check.jsonl"]


def is_holdout(sample_id: str, fraction: float = 0.15) -> bool:
    """Cung ham chia ma scripts/build_sft.py:holdout() dung. Thuan theo id, nen tai lap duoc."""
    return int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < fraction


def key(call: dict) -> tuple:
    return (call.get("name"),
            json.dumps(call.get("arguments") or {}, sort_keys=True, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", default=DEFAULT_RUNS)
    parser.add_argument("--gold", default=str(GOLD))
    parser.add_argument("--min-calls", type=int, default=20,
                        help="tool phai xuat hien it nhat bao nhieu lan trong gold moi duoc xet")
    parser.add_argument("--min-bare", type=float, default=0.80,
                        help="ti le gold call KHONG co doi so toi thieu. 0.95 cho ban 'safe'")
    args = parser.parse_args()

    gold: dict[str, list[dict]] = {}
    for line in Path(args.gold).open(encoding="utf-8"):
        if line.strip():
            item = json.loads(line)
            gold[item["id"]] = (item.get("label") or {}).get("tool_calls") or []
    if not gold:
        sys.exit(f"{args.gold} khong co mau nao")

    train_ids = [i for i in gold if not is_holdout(i)]
    bare = defaultdict(Counter)
    for sample_id in train_ids:
        for call in gold[sample_id]:
            bare[call["name"]][not (call.get("arguments") or {})] += 1
    allow = {name: (sum(c.values()), c[True] / sum(c.values()))
             for name, c in bare.items()
             if sum(c.values()) >= args.min_calls and c[True] / sum(c.values()) >= args.min_bare}

    print(f"Allowlist tu {len(train_ids)} mau KHONG held-out, nguong >={args.min_calls} call "
          f"va >={args.min_bare:.0%} trong -> {len(allow)} tool")
    for name, (n, ratio) in sorted(allow.items(), key=lambda x: -x[1][0]):
        print(f"   {name:34} gold {n:4d} call, trong {ratio:6.1%}")

    # Thiet hai tiem tang, tren CA tap: gold call thuoc allowlist ma gold THUC SU co doi so.
    total = Counter()
    harm = Counter()
    for sample_id in gold:
        for call in gold[sample_id]:
            if call["name"] in allow:
                total[call["name"]] += 1
                if call.get("arguments"):
                    harm[call["name"]] += 1
    print(f"\nTren ca {len(gold)} mau: {sum(total.values())} gold call thuoc allowlist, "
          f"{sum(harm.values())} trong do CO doi so -> gate se lam sai "
          f"({sum(harm.values())/max(1,sum(total.values())):.1%})")
    for name, count in harm.most_common():
        print(f"   {name:34} {count:3d}/{total[name]}")

    for path in args.runs:
        try:
            rows = [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]
        except FileNotFoundError:
            print(f"\n{path}: khong co file")
            continue
        by_id = {r["id"]: r.get("tool_calls") or [] for r in rows if r.get("id") in gold}
        held = [i for i in by_id if is_holdout(i)]
        if not held:
            print(f"\n{path}: {len(by_id)} mau nhung KHONG mau nao held-out -- bo qua, "
                  "cham tren mau da dung de dung allowlist la tu lua")
            continue
        scores = []
        for strip in (False, True):
            hit = total_gold = 0
            for sample_id in held:
                want = Counter(key(c) for c in gold[sample_id])
                got = Counter(key({"name": c["name"], "arguments": {}}
                                  if strip and c["name"] in allow else c)
                              for c in by_id[sample_id])
                hit += sum((want & got).values())
                total_gold += sum(want.values())
            scores.append(hit)
        print(f"\n{path}  ({len(held)}/{len(by_id)} mau held-out)")
        print(f"   exact {scores[0]}/{total_gold} -> {scores[1]}/{total_gold}  "
              f"(+{scores[1]-scores[0]})")


if __name__ == "__main__":
    main()
