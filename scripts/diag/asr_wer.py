"""Đo WER của một cache transcript trên split `train`, bằng CPU, không cần audio lẫn GPU.

    python scripts/diag/asr_wer.py
    python scripts/diag/asr_wer.py --transcripts train_check.transcripts.jsonl --worst 10

Vì sao đo được mà không cần gì: split `train` có `audio_text` của BTC, và các file
`*.transcripts.jsonl` ở gốc repo là kết quả ASR đã chạy rồi. Ghép hai thứ đó lại là ra WER
từng mẫu. WER là **mốc chặn trên của toàn pipeline** -- `results/local_v4` đã cho thấy mẫu
nào transcript hỏng thì tầng dưới không cứu được.

Chuẩn hoá trước khi so: bỏ dấu câu, về chữ thường, giữ nguyên dấu tiếng Việt. Không bỏ dấu,
vì mất dấu là mất nghĩa và model phía sau cũng nhận đúng chuỗi có dấu đó.
"""
import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "data/train+public_test/train+public_test/train/train.jsonl"


def words(text: str) -> list[str]:
    return re.findall(r"[0-9a-zà-ỹđ]+", unicodedata.normalize("NFC", text.lower()))


def wer(reference: str, hypothesis: str) -> float | None:
    """Levenshtein theo từ, chia cho độ dài tham chiếu. None khi tham chiếu rỗng."""
    ref, hyp = words(reference), words(hypothesis)
    if not ref:
        return None
    row = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        previous, row[0] = row[0], i
        for j in range(1, len(hyp) + 1):
            current = row[j]
            row[j] = min(row[j] + 1, row[j - 1] + 1, previous + (ref[i - 1] != hyp[j - 1]))
            previous = current
    return row[-1] / len(ref)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transcripts", default="train_all.transcripts.jsonl",
                        help="cache {audio, text} do pha ASR sinh ra")
    parser.add_argument("--worst", type=int, default=5, help="in ra mấy mẫu tệ nhất")
    args = parser.parse_args()

    # Khoá theo ĐƯỜNG DẪN audio, không theo id mẫu: cache còn chứa audio của các lượt
    # history nằm cùng thư mục, và khớp theo thư mục sẽ so transcript lượt phụ với
    # audio_text của lượt chính -- đo ra WER 79% trong khi thực tế là 29%.
    gold, name = {}, {}
    for line in TRAIN.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("audio_text") and row.get("audio"):
            gold[row["audio"]] = row["audio_text"]
            name[row["audio"]] = row["id"]

    path = Path(args.transcripts)
    if not path.is_absolute():
        path = ROOT / path
    scored = []
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        key = row["audio"]
        if key not in gold:
            continue
        value = wer(gold[key], row["text"])
        if value is not None:
            scored.append((value, key, row["text"]))
    if not scored:
        raise SystemExit(f"không khớp được mẫu nào giữa {path.name} và train.jsonl")

    values = sorted(value for value, _, _ in scored)
    show = lambda p: values[min(int(len(values) * p / 100), len(values) - 1)]
    print(f"{path.name}: {len(scored)} mẫu khớp nhãn")
    print(f"  WER trung bình {sum(values) / len(values) * 100:.1f}%  "
          f"trung vị {show(50) * 100:.1f}%")
    print("  " + "  ".join(f"p{p}={show(p) * 100:.1f}%" for p in (10, 25, 75, 90, 99)))
    for limit in (0.0, 0.3, 0.5):
        n = sum(1 for v in values if (v == 0 if limit == 0 else v > limit))
        label = "hoàn hảo" if limit == 0 else f"trên {limit:.0%}"
        print(f"  {label:>10}: {n}/{len(values)} ({n / len(values) * 100:.1f}%)")

    print(f"\n{args.worst} mẫu tệ nhất:")
    for value, key, text in sorted(scored, reverse=True)[:args.worst]:
        print(f"  {value * 100:5.0f}%  {name[key]}")
        print(f"         gold: {gold[key]}")
        print(f"          asr: {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
