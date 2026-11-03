"""So hai bài nộp trên cùng tập mẫu, không cần nhãn và không cần GPU.

    python scripts/diag/diff_submissions.py A.jsonl B.jsonl
    python scripts/diag/diff_submissions.py A.jsonl B.jsonl --catalog public_test --worst 15

Vì sao đo được mà không có nhãn: `public_test` không phát hành gold, nên không chấm lại được
bài nộp nào. Nhưng khi hai bài nộp chạy trên **đúng cùng 634 mẫu** mà một bản 22.52 điểm và
một bản 13.41, thì chỗ chúng khác nhau là nơi 9 điểm đó nằm. Đây là cách duy nhất hiện có để
khai thác lượt 22.52 mà không chạy lại nó.

Chỉ tool `kind == "write"` đổi `state_hash`, và `state_hash` sai là mẫu ăn 0. Nên phép đếm
quan trọng nhất ở đây là **số lệnh ghi**, không phải tổng số lệnh gọi.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/train+public_test/train+public_test"


def load(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        rows[row["id"]] = row
    return rows


def kinds(split: str) -> dict[str, str]:
    """tên tool -> "read" | "write" | "meta".

    Tool nằm LỒNG trong apps[].tools[], không phải ở catalog["tools"] -- đọc sai chỗ thì hàm
    trả rỗng, mọi tool bị coi là write, và phép đếm "lệnh ghi" thành vô nghĩa. Chỉ `write`
    đổi state_hash; `meta` (sys.ask_missing_info) không chạy trên state.
    """
    path = DATA / split / f"{split}_tools_catalog.json"
    if not path.is_file():
        return {}
    catalog = json.loads(path.read_text(encoding="utf-8"))
    return {tool["name"]: tool.get("kind", "write")
            for app in catalog.get("apps", []) for tool in app.get("tools", [])}


def calls(row: dict) -> list[dict]:
    return row.get("tool_calls") or []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", type=Path, help="bài nộp A (bản tham chiếu, điểm cao hơn)")
    parser.add_argument("b", type=Path, help="bài nộp B")
    parser.add_argument("--catalog", default="public_test", help="split để đọc kind của tool")
    parser.add_argument("--worst", type=int, default=10, help="in ra mấy mẫu lệch nhiều nhất")
    args = parser.parse_args()

    A, B = load(args.a), load(args.b)
    shared = sorted(set(A) & set(B))
    if not shared:
        raise SystemExit("hai bài nộp không có mẫu nào chung")
    print(f"A = {args.a.name}  ({len(A)} mẫu)")
    print(f"B = {args.b.name}  ({len(B)} mẫu)")
    print(f"chung: {len(shared)} mẫu" + ("" if len(shared) == len(A) == len(B)
                                         else "  <-- KHÁC TẬP, mọi so sánh dưới chỉ trên phần chung"))

    kind = kinds(args.catalog)
    if not kind:
        print(f"(không đọc được catalog {args.catalog}; coi mọi tool là write)")

    def writes(row):
        return [c for c in calls(row) if kind.get(c["name"], "write") == "write"]

    same_hash = sum(1 for i in shared if A[i].get("state_hash") == B[i].get("state_hash"))
    print(f"\n### state_hash\n  giống nhau : {same_hash}/{len(shared)} "
          f"({same_hash / len(shared) * 100:.1f}%)")
    print(f"  khác nhau  : {len(shared) - same_hash}  <-- chỉ trong đây mới có chênh điểm")

    print("\n### số lệnh gọi")
    for name, D in (("A", A), ("B", B)):
        total = sum(len(calls(D[i])) for i in shared)
        wr = sum(len(writes(D[i])) for i in shared)
        zero = sum(1 for i in shared if not calls(D[i]))
        nowrite = sum(1 for i in shared if not writes(D[i]))
        print(f"  {name}: {total:5d} lệnh ({total / len(shared):.2f}/mẫu) | "
              f"{wr:5d} GHI ({wr / len(shared):.2f}/mẫu) | "
              f"{zero:3d} mẫu 0 lệnh | {nowrite:3d} mẫu 0 lệnh ghi")

    only_a = sum(1 for i in shared if writes(A[i]) and not writes(B[i]))
    only_b = sum(1 for i in shared if writes(B[i]) and not writes(A[i]))
    print(f"\n  A ghi mà B KHÔNG ghi: {only_a} mẫu  <-- B không hành động")
    print(f"  B ghi mà A không ghi: {only_b} mẫu")

    ca = Counter(c["name"] for i in shared for c in calls(A[i]))
    cb = Counter(c["name"] for i in shared for c in calls(B[i]))
    gap = sorted(((ca[t] - cb[t], t) for t in set(ca) | set(cb)), reverse=True)
    print(f"\n### tool A gọi nhiều hơn B nhất (dấu hiệu B bỏ sót)")
    for delta, tool in gap[:10]:
        if delta <= 0:
            break
        print(f"  +{delta:4d}  {tool:38s} (A {ca[tool]:4d} / B {cb[tool]:4d})"
              f"  [{kind.get(tool, 'write')}]")
    print(f"\n### tool B gọi nhiều hơn A nhất (dấu hiệu B gọi thừa)")
    for delta, tool in gap[::-1][:10]:
        if delta >= 0:
            break
        print(f"  {delta:5d}  {tool:38s} (A {ca[tool]:4d} / B {cb[tool]:4d})"
              f"  [{kind.get(tool, 'write')}]")

    print(f"\n### {args.worst} mẫu A ghi nhiều nhất mà B không ghi gì")
    ranked = sorted(((len(writes(A[i])), i) for i in shared
                     if writes(A[i]) and not writes(B[i])), reverse=True)
    for n, sample in ranked[:args.worst]:
        names = ", ".join(c["name"] for c in writes(A[sample]))
        print(f"  {sample}  A ghi {n}: {names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
