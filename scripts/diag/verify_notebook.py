"""Kiểm notebook sau vá: cú pháp, sự có mặt, và normalizer trong notebook phải cho ĐÚNG kết quả
như src/text_norm.py trên dữ liệu thật."""
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path("D:/z_DaiHoc/a_zaloAI")
nb = json.loads((ROOT / "notebook/lifemate_kaggle_v2.ipynb").read_text(encoding="utf-8"))
code = [c for c in nb["cells"] if c["cell_type"] == "code"]

fail = 0
# 1) cú pháp
for i, c in enumerate(code):
    try:
        ast.parse("".join(c["source"]))
    except SyntaxError as e:
        print(f"FAIL cell {i} cú pháp: {e}")
        fail += 1
print(f"OK   {len(code)} cell code, cú pháp sạch")

# 2) sự có mặt
whole = "".join("".join(c["source"]) for c in code)
need = {
    "ADAPTER_DIR = None": "config adapter",
    "NORMALISE_NUMBERS = True": "config normalizer",
    "'peft>=0.14'": "cài peft khi có adapter",
    "def spoken_numbers_to_digits": "hàm normalizer",
    "if NORMALISE_NUMBERS:": "áp dụng normalizer trong ASR",
    "adapter=None": "ToolModel nhận adapter",
    "merge_and_unload": "merge LoRA",
    "LLM_MAX_MEMORY, ADAPTER_DIR)": "truyền adapter vào ToolModel",
    "'normalise_numbers':NORMALISE_NUMBERS": "fingerprint ghi normalizer",
    "'adapter':str(ADAPTER_DIR)": "fingerprint ghi adapter",
}
for token, what in need.items():
    if token in whole:
        print(f"OK   {what}")
    else:
        print(f"FAIL thiếu: {what}  ({token})")
        fail += 1

# 3) thứ tự: normalizer phải định nghĩa TRƯỚC khi SpeechModel dùng
cell9 = "".join(code[4]["source"]) if "class SpeechModel" in "".join(code[4]["source"]) else next(
    "".join(c["source"]) for c in code if "class SpeechModel" in "".join(c["source"]))
if cell9.index("def spoken_numbers_to_digits") < cell9.index("class SpeechModel"):
    print("OK   normalizer định nghĩa trước SpeechModel")
else:
    print("FAIL normalizer định nghĩa SAU SpeechModel -> NameError")
    fail += 1

# 4) normalizer trong notebook phải KHỚP src/text_norm.py trên transcript thật
ns = {}
start = cell9.index("UNITS = {")
end = cell9.index("class SpeechModel")
exec(compile("import re, unicodedata\n" + cell9[start:end], "<nb>", "exec"), ns)
sys.path.insert(0, str(ROOT))
from src.text_norm import spoken_numbers_to_digits as reference  # noqa: E402

lines = (ROOT / "public_test.transcripts.jsonl").open(encoding="utf-8").readlines()
mismatch = 0
changed = 0
for line in lines:
    text = json.loads(line)["text"]
    a, b = ns["spoken_numbers_to_digits"](text), reference(text)
    if a != b:
        mismatch += 1
        if mismatch <= 3:
            print(f"FAIL lệch: {text[:60]!r}\n     nb={a[:60]!r}\n     src={b[:60]!r}")
    if b != text:
        changed += 1
if mismatch:
    fail += 1
else:
    print(f"OK   normalizer notebook == src/text_norm.py trên {len(lines)} transcript thật "
          f"({changed} dòng bị đổi, {changed/len(lines):.1%})")

# 5) tên dùng trước khi gán — lớp lỗi chỉ lộ ra sau khi đã tải 13 GiB model trên Kaggle
import subprocess  # noqa: E402

order = subprocess.run([sys.executable, "-X", "utf8",
                        str(ROOT / "scripts/diag/check_notebook_order.py"),
                        str(ROOT / "notebook/lifemate_kaggle_v2.ipynb")],
                       capture_output=True, text=True, encoding="utf-8")
if order.returncode:
    print("FAIL thứ tự định nghĩa:\n" + (order.stdout or order.stderr).rstrip())
    fail += 1
else:
    print("OK   không có tên nào dùng trước khi gán")

print("\n" + ("TẤT CẢ ĐẠT" if not fail else f"{fail} HẠNG MỤC LỖI"))
sys.exit(1 if fail else 0)
