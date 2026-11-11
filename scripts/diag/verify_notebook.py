"""Kiểm notebook sau vá: cú pháp, sự có mặt, và normalizer trong notebook phải cho ĐÚNG kết quả
như src/text_norm.py trên dữ liệu thật."""
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path("D:/z_DaiHoc/a_zaloAI")
# Duong dan ghim cung tung o day va argv bi BO QUA hoan toan: moi lan chay
#     python scripts/diag/verify_notebook.py notebook/lifemate_kaggle_v3.ipynb
# la mot lan kiem v2 roi bao "TAT CA DAT" cho v3. Mot cell v3 KHONG compile duoc da di
# qua phep kiem nay. Phep kiem khong doc doi tuong duoc giao thi te hon la khong co.
NOTEBOOK = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "notebook/lifemate_kaggle_v2.ipynb"
if not NOTEBOOK.is_absolute():
    NOTEBOOK = (Path.cwd() / NOTEBOOK) if (Path.cwd() / NOTEBOOK).exists() else ROOT / NOTEBOOK
print(f"kiem: {NOTEBOOK}")
nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
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
    # Moc cu la literal `ADAPTER_DIR = None`, nhung v2/v3 do adapter bang glob nen dong do
    # khong con ton tai -- phep kiem bao FAIL tren CA HAI notebook dang dung duoc. Mot phep
    # kiem luon do la mot phep kiem bi bo qua, nen no ghim vao cai thuc su phai co: co bat/tat
    # va nhanh gan None khi tat.
    # Ghim literal `USE_ADAPTER = True` la lap lai dung cai benh da sua o dong 10: v4 dat
    # False (lora_V4 khong nap duoc len Qwen3.5) va phep kiem bao FAIL cho mot notebook DUNG.
    # Nen kiem CAI CONG TAC co ton tai, khong kiem GIA TRI cua no.
    "USE_ADAPTER = ": "co bat/tat adapter",
    "if (USE_ADAPTER and _ads) else None": "adapter ve None khi tat hoac khong tim thay",
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

# 4b) allowlist BARE_READ_TOOLS trong notebook phải KHỚP thứ scripts/diag/bare_args.py tính
# được từ gold. Hằng số chép tay thì trôi khỏi phép đo trong im lặng, và một gate bỏ đối số
# của tool SAI làm hỏng call thay vì cứu nó.
if "BARE_READ_GATE" not in whole:
    print("SKIP không có BARE_READ_GATE (notebook trước v3)")
else:
    import hashlib as _h
    from collections import Counter as _C, defaultdict as _dd
    _gold = ROOT / "data/train+public_test/train+public_test/train/train.jsonl"
    if not _gold.exists():
        print("SKIP không có train.jsonl để đối chiếu allowlist")
    else:
        _bare = _dd(_C)
        for _line in _gold.open(encoding="utf-8"):
            if not _line.strip():
                continue
            _it = json.loads(_line)
            if int(_h.sha256(_it["id"].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < 0.15:
                continue          # mẫu held-out không được dùng để dựng allowlist
            for _c in (_it.get("label") or {}).get("tool_calls") or []:
                _bare[_c["name"]][not (_c.get("arguments") or {})] += 1
        def _allow(min_bare):
            return {n for n, c in _bare.items()
                    if sum(c.values()) >= 20 and c[True] / sum(c.values()) >= min_bare}
        _ns = {}
        _start = whole.index("_BARE_SAFE = {")
        _end = whole.index("print('BARE_READ_GATE:'")
        exec(compile("BARE_READ_GATE = 'all'\n" + whole[_start:_end], "<nb>", "exec"), _ns)
        _want, _got = _allow(0.80), set(_ns["BARE_READ_TOOLS"])
        if _want == _got:
            print(f"OK   BARE_READ_TOOLS ({len(_got)} tool) == gold ngưỡng >=20 call, >=80% trống")
        else:
            print(f"FAIL BARE_READ_TOOLS lệch: thiếu {sorted(_want - _got)}, "
                  f"thừa {sorted(_got - _want)}")
            fail += 1
        _safe_ns = dict(_ns)
        exec(compile("BARE_READ_GATE = 'safe'\n" + whole[_start:_end], "<nb>", "exec"), _safe_ns)
        if set(_safe_ns["BARE_READ_TOOLS"]) == _allow(0.95):
            print("OK   mức 'safe' == gold ngưỡng >=95% trống")
        else:
            print(f"FAIL mức 'safe' lệch so với ngưỡng >=95%: "
                  f"{sorted(set(_safe_ns['BARE_READ_TOOLS']) ^ _allow(0.95))}")
            fail += 1

# 4c) rieng cho notebook dung model tron attention (Qwen3.5): phan crop KV-cache chi dung voi
# full attention, nen neu doi sang qwen3_5 thi tai dung cache BAT BUOC phai tat.
if "Qwen/Qwen3.5" in whole:
    for token, what in {
        "REUSE_KV_CACHE = False": "qwen3_5: tai dung KV cache phai TAT",
        "USE_ADAPTER = False": "qwen3_5: adapter Instruct khong nap duoc, phai TAT",
        "if REUSE_KV_CACHE and self.cache is not None:": "doc cache co gan co",
        "output.past_key_values if REUSE_KV_CACHE else None": "ghi cache co gan co",
        "AutoModelForImageTextToText": "co lop nap cho kien truc ConditionalGeneration",
        "ENABLE_THINKING = True": "qwen3_5: bat suy luan ngan (thanh phan cua ban 22.52)",
    }.items():
        if token in whole:
            print(f"OK   {what}")
        else:
            print(f"FAIL thiếu: {what}  ({token})")
            fail += 1
    # tran tham so phai la ASSERT, khong phai log: vuot 6B la bi loai
    if "< 5_000_000_000" in whole and "< 6_000_000_000" in whole:
        print("OK   assert tran tham so 5B/6B con nguyen")
    else:
        print("FAIL mat assert tran tham so 5B/6B")
        fail += 1
    # Phai ghim dong GAN, khong phai chuoi "PhoWhisper-large" o bat ky dau: chinh phan comment
    # giai thich phep tinh 6.203B co chua ten do, va bat vao comment la bao FAIL cho ban DUNG.
    if "ASR_ID = 'vinai/PhoWhisper-large'" in whole:
        print("FAIL Qwen3.5 + PhoWhisper-large = 6.203B > 6B (phai dung medium hoac nho hon)")
        fail += 1
    else:
        print("OK   khong dung PhoWhisper-large cung Qwen3.5 (ngan sach 6B)")

# 5) tên dùng trước khi gán — lớp lỗi chỉ lộ ra sau khi đã tải 13 GiB model trên Kaggle
import subprocess  # noqa: E402

order = subprocess.run([sys.executable, "-X", "utf8",
                        str(ROOT / "scripts/diag/check_notebook_order.py"),
                        str(NOTEBOOK)],
                       capture_output=True, text=True, encoding="utf-8")
if order.returncode:
    print("FAIL thứ tự định nghĩa:\n" + (order.stdout or order.stderr).rstrip())
    fail += 1
else:
    print("OK   không có tên nào dùng trước khi gán")

print("\n" + ("TẤT CẢ ĐẠT" if not fail else f"{fail} HẠNG MỤC LỖI"))
sys.exit(1 if fail else 0)
