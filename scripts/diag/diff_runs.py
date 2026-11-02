"""So hai run trên CÙNG tập mẫu, từng mẫu một.

    python scripts/diag/diff_runs.py temp/v16 temp/v17

Chỉ cần metrics.jsonl của hai bên, nên dùng được cả khi events.jsonl tải về cụt.
Dùng cho phép so quan trọng nhất còn lại: baseline no-adapter vs có adapter trên cùng
50 mẫu held-out. Điều kiện để phép so có nghĩa là hai run khác nhau ĐÚNG MỘT thứ —
kịch bản V16→V17 đã gộp hai thay đổi (pushback + dòng SYSTEM_PROMPT) và vì thế không
quy được +0.60 cho cái nào.
"""
import argparse
import json
from pathlib import Path


def load(directory):
    path = Path(directory) / "metrics.jsonl"
    if not path.is_file():
        raise SystemExit(f"Không thấy {path}")
    rows = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            print(f"⚠️  {path}: bỏ dòng {number} không đọc được")
            continue
        rows[row["id"]] = row
    return rows


def score(rows):
    return 100 * sum(r["local_score"] for r in rows.values()) / len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("before", help="thư mục run cũ (ví dụ temp/v16)")
    parser.add_argument("after", help="thư mục run mới (ví dụ temp/v17)")
    args = parser.parse_args()

    a, b = load(args.before), load(args.after)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise SystemExit("Hai run không có mẫu nào chung — khác tập mẫu thì không so được.")
    if set(a) != set(b):
        # Đây là cái bẫy của bước 4 RUNBOOK: nếu một bên lấy mẫu từ cả split và bên kia
        # từ held-out, hai điểm trông so được mà thực ra không.
        print(f"⚠️  Tập mẫu KHÁC nhau: {len(a)} vs {len(b)}, chung {len(shared)}. "
              f"Hiệu số dưới đây chỉ tính trên phần chung, và điểm tổng hai bên KHÔNG so được.")

    print(f"{args.before}  ->  {args.after}")
    print(f"{len(shared)} mẫu chung\n")
    print(f"điểm      : {score(a):.2f}  ->  {score(b):.2f}   ({score(b) - score(a):+.2f})")
    print(f"hash khớp : {sum(a[i]['state_match'] for i in shared)}  ->  "
          f"{sum(b[i]['state_match'] for i in shared)}  / {len(shared)}")
    print(f"lượt LLM  : {sum(a[i]['llm_turns'] for i in shared) / len(shared):.1f}  ->  "
          f"{sum(b[i]['llm_turns'] for i in shared) / len(shared):.1f}  (trung bình/mẫu)")
    print(f"giây/mẫu  : {sum(a[i]['seconds'] for i in shared) / len(shared):.1f}  ->  "
          f"{sum(b[i]['seconds'] for i in shared) / len(shared):.1f}  (trung bình)")

    def line(sample_id):
        x, y = a[sample_id], b[sample_id]
        return (f"  {sample_id}  {x['local_score']:.2f} -> {y['local_score']:.2f}"
                f" | hash {str(x['state_match'])[0]}->{str(y['state_match'])[0]}"
                f" | khớp {x['matched_calls']}/{x['gold_calls']} -> {y['matched_calls']}/{y['gold_calls']}"
                f" | lượt {x['llm_turns']}->{y['llm_turns']}"
                f" | {x['stop_reason']}->{y['stop_reason']}")

    for title, picked in (
            ("ĐIỂM LÊN", [i for i in shared if b[i]["local_score"] > a[i]["local_score"]]),
            ("ĐIỂM XUỐNG", [i for i in shared if b[i]["local_score"] < a[i]["local_score"]]),
            # Lật hash mà điểm không đổi = tới đúng trạng thái bằng đường khác gold, nên F1
            # vẫn 0. Nhìn riêng nhóm này mới thấy, vì điểm tổng không phản ánh.
            ("LẬT HASH MÀ ĐIỂM KHÔNG ĐỔI",
             [i for i in shared if a[i]["state_match"] != b[i]["state_match"]
              and a[i]["local_score"] == b[i]["local_score"]]),
    ):
        print(f"\n{title}: {len(picked)} mẫu")
        for sample_id in picked:
            print(line(sample_id))


if __name__ == "__main__":
    main()
