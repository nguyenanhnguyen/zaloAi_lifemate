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
        # Model card Qwen3.5 canh bao endless repetition va de xuat presence_penalty=1.5 +
        # temperature=1.0, ca hai ta KHONG dung duoc (luat bat buoc greedy; transformers 5.13
        # khong co presence_penalty). Nen hai lop bao ve duoi day la BAT BUOC khi thinking bat.
        "NO_REPEAT_NGRAM = ": "co chan vong lap nguyen van",
        "'no_repeat_ngram_size': NO_REPEAT_NGRAM": "no_repeat_ngram thuc su vao generate()",
        "THINK_FALLBACK = ": "co duong suy bien khi thinking het token",
        "def _decide_once(": "decide() tach lam wrapper + mot lan thu",
        "enable_thinking=think": "moi lan thu dung CO thinking rieng, khong phai co toan cuc",
        # Ngan sach suy luan TACH RIENG. Dung chung mot con so cho suy luan va cau tra loi
        # la cach V15 chet: suy luan dai an het ngan sach truoc khi kip viet <tool_call>,
        # va mau do nop 0 call. Ban ky luc 22.52 cho reasoning 512 token RIENG.
        "THINK_BUDGET = ": "co tran rieng cho khoi <think>",
        "ANSWER_BUDGET = ": "co tran rieng cho cau tra loi",
        "MAX_NEW_TOKENS = THINK_BUDGET + ANSWER_BUDGET": "tong ngan sach la phep cong hai phan",
        "max_new_tokens=THINK_BUDGET": "pha A thuc su bi gioi han boi THINK_BUDGET",
        "max_new_tokens=ANSWER_BUDGET": "pha B thuc su bi gioi han boi ANSWER_BUDGET",
        "stop_strings=['</think>', '</tool_call>']": "pha A dung som khi het suy luan",
        "seq = ids[:prefix] + first.sequences[0].tolist()": "pha B prefill CA doan cache",
        # Ngan sach THOI GIAN. 634 mau x 512 token khong tai dung cache = 14-44 gio, tran
        # session Kaggle la 12. Deadline co dinh nghia la het gio giua duong va moi mau
        # chua chay an 0 -- lo lon hon moi gate tung do duoc.
        "SESSION_BUDGET_HOURS = ": "co ngan sach thoi gian cho ca session",
        "DEADLINE_FLOOR_SECONDS = ": "co san duoi cho deadline moi mau",
        "NOTEBOOK_T0 = time.monotonic()": "co moc do thoi gian da tieu",
        "deadline_plan.json": "ke hoach thoi gian duoc ghi ra file de doc lai",
        # Chot cache phai dieu kien theo CO, khong theo TEN model. Ban dau ghim 'qwen3_5' va
        # Kaggle tra ve 'qwen3_5_text' -> chan dung cai no phai cho qua, lust chet o giay 648
        # vi chinh phep kiem. Cung benh voi phep-kiem-khong-doc-doi-tuong.
        "if REUSE_KV_CACHE and self.model.config.model_type != 'qwen3':":
            "chot cache dieu kien theo co, khong liet ke ten model_type",
        # Qwen3.5 goi tool bang XML (<function=...><parameter=...>), KHONG bang JSON.
        # chat_template.jinja dan dinh dang do o moi luot. parse_decision chi doc JSON la
        # that bai o MOI buoc -> moi mau nop 0 call -> v4 an ~0 diem.
        "def parse_xml_call(": "co phep doc tool call dinh dang XML goc cua Qwen3.5",
        "def arg_types(": "co bang kieu tham so lay tu catalog",
        "def coerce_arg(": "co phep ep kieu theo schema",
        "xml = parse_xml_call(candidate, types)": "duong XML duoc goi TRUOC duong JSON",
        "types=arg_types(tools)": "kieu tham so lay tu DUNG danh sach tool model da thay",
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
    # Khong duoc liet ke ten model_type o nhanh chan: moi bien the ten moi la mot lan chet
    # sau 11 phut tai model.
    if "model_type == 'qwen3_5'" in whole:
        print("FAIL chot cache con ghim ten 'qwen3_5' (Kaggle tra ve 'qwen3_5_text')")
        fail += 1
    else:
        print("OK   chot cache khong ghim bien the ten model_type")
    if "ASR_ID = 'vinai/PhoWhisper-large'" in whole:
        print("FAIL Qwen3.5 + PhoWhisper-large = 6.203B > 6B (phai dung medium hoac nho hon)")
        fail += 1
    else:
        print("OK   khong dung PhoWhisper-large cung Qwen3.5 (ngan sach 6B)")

    # So hoc: deadline nhan so mau phai NAM TRONG ngan sach session, neu khong thi bai nop
    # se thieu mau va moi mau thieu an 0. Doc tu chinh cac hang so cua notebook.
    # _safe_ns chi chay mot DOAN cua cell cau hinh (vung BARE_READ), nen cac hang so khac
    # khong co trong do. Doc truc tiep bang ast: khong chay code nao, va khong bi lech khi
    # cell cau hinh doi thu tu.
    def _lit(name):
        for _line in whole.splitlines():
            _m = re.match(r"\s*" + re.escape(name) + r"\s*=\s*([0-9.]+)\s*(?:#.*)?$", _line)
            if _m:
                return float(_m.group(1))
        raise KeyError(name)
    try:
        _tb = _lit("THINK_BUDGET"); _ab = _lit("ANSWER_BUDGET")
        _sb = _lit("SESSION_BUDGET_HOURS"); _fl = _lit("DEADLINE_FLOOR_SECONDS")
        _per = _sb * 3600 / 634
        print(f"OK   ngan sach: {_sb} gio / 634 mau = {_per:.0f} s/mau "
              f"(san {_fl:.0f} s, token {_tb:.0f}+{_ab:.0f}={_tb + _ab:.0f})")
        if _per < _fl:
            print(f"FAIL {_sb} gio chia 634 mau = {_per:.0f} s/mau, duoi san {_fl:.0f} s")
            fail += 1
    except KeyError as _exc:
        print(f"FAIL khong doc duoc hang so ngan sach: {_exc}")
        fail += 1

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
