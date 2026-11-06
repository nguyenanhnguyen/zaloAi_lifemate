#!/usr/bin/env python3
"""Điểm vào của image nộp bài: dò layout của /data rồi gọi scripts/infer.py.

Vì sao cần một lớp ở giữa thay vì gọi thẳng infer.py: `infer.py` nhận `--data-root` và
`--split` rồi ghép `data_root/split`, còn luật chỉ nói BTC "mount the private test folder
(same structure as public_test) into the container at /data". Hai cách đọc đều hợp lệ:

    /data/private_test/private_test.jsonl        (mount thư mục CHA)
    /data/private_test.jsonl                     (mount CHÍNH thư mục split)

Đoán sai là cả lượt chấm ra rỗng, và ta chỉ được chấm một lần. Nên dò, không đoán: tìm file
`*_tools_catalog.json`, lấy tên split từ chính tên file, và nếu catalog nằm ngay trong /data
thì dựng một symlink để `data_root/split` trỏ đúng chỗ.
"""
import os
import subprocess
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent
# Trong image, entrypoint.py nằm cạnh scripts/ ở /app. Ngoài image nó nằm trong docker/, nên
# scripts/ ở thư mục cha. Dò thay vì ghim: nhờ vậy chuỗi này KIỂM ĐƯỢC trên máy dev mà không
# cần Docker Engine, và một lỗi đường dẫn không phải chờ tới lượt chấm mới lộ.
SCRIPTS = APP / "scripts" if (APP / "scripts" / "infer.py").exists() else APP.parent / "scripts"
DATA = Path(os.environ.get("DATA_DIR", "/data"))
RESULT = Path(os.environ.get("RESULT_DIR", "/app/result"))


def find_split() -> tuple[Path, str]:
    """Trả về (data_root, split) sao cho data_root/split là thư mục chứa dữ liệu."""
    if not DATA.exists():
        sys.exit(f"Không thấy {DATA}. BTC mount dữ liệu vào đó; đặt DATA_DIR để đổi.")
    catalogs = sorted(DATA.glob("*_tools_catalog.json")) \
        + sorted(DATA.glob("*/*_tools_catalog.json")) \
        + sorted(DATA.glob("*/*/*_tools_catalog.json"))
    if not catalogs:
        listing = "\n  ".join(sorted(str(p.relative_to(DATA)) for p in DATA.glob("**/*"))[:40])
        sys.exit(f"Không thấy *_tools_catalog.json dưới {DATA}. Đang có:\n  {listing}")
    catalog = catalogs[0]
    split = catalog.name[: -len("_tools_catalog.json")]
    folder = catalog.parent
    if folder.name == split:
        return folder.parent, split
    # Catalog nằm ngay trong thư mục được mount: dựng data_root ảo để data_root/split khớp.
    root = Path("/tmp/zaic_root")
    root.mkdir(parents=True, exist_ok=True)
    link = root / split
    if not link.exists():
        link.symlink_to(folder, target_is_directory=True)
    print(f"[entrypoint] catalog nằm ngay trong {folder}; dựng {link} -> {folder}", flush=True)
    return root, split


def main() -> int:
    data_root, split = find_split()
    folder = data_root / split
    print(f"[entrypoint] data_root={data_root} split={split} scripts={SCRIPTS}", flush=True)
    for required in (f"{split}.jsonl", f"{split}_tools_catalog.json"):
        if not (folder / required).exists():
            print(f"[entrypoint] CẢNH BÁO: thiếu {folder / required}", flush=True)
    # zaic_tools là THƯ MỤC package trong mỗi split, không phải một file .py -- `infer.py` nạp
    # nó bằng `sys.path.insert(0, split_dir)`. Image không nhúng nó: BTC giao bản riêng cho
    # từng split và luật bắt mọi lệnh phải đi qua đúng bản đó.
    if not (folder / "zaic_tools").is_dir() and not (folder / "zaic_tools.py").exists():
        print(f"[entrypoint] CẢNH BÁO: không thấy zaic_tools trong {folder} -- "
              "`from zaic_tools import ToolEnv` sẽ chết", flush=True)

    RESULT.mkdir(parents=True, exist_ok=True)
    out = RESULT / "submission.jsonl"

    command = [
        sys.executable, str(SCRIPTS / "infer.py"),
        "--data-root", str(data_root),
        "--split", split,
        "--output", str(out),
        "--metrics-output", str(RESULT / "metrics.jsonl"),
        # Luật: mỗi mẫu trong 30 giây, quá thì mẫu đó 0 điểm. Đây là trần CỨNG, không phải
        # cấu hình chất lượng -- infer.py cảnh báo nếu ai đặt lớn hơn.
        "--max-seconds", os.environ.get("MAX_SECONDS", "30"),
        # --resume để lượt chạy lại sau khi BTC dừng giữa đường không mất phần đã xong.
        "--resume",
    ]
    if os.environ.get("USE_ADAPTER", "1") != "0" and (Path("/opt/adapter") / "adapter_config.json").exists():
        command += ["--adapter", "/opt/adapter"]
        print("[entrypoint] dùng adapter /opt/adapter", flush=True)
    else:
        print("[entrypoint] KHÔNG dùng adapter (USE_ADAPTER=0 hoặc không có file)", flush=True)
    extra = os.environ.get("INFER_ARGS", "").split()
    command += extra
    if extra:
        print(f"[entrypoint] thêm cờ từ INFER_ARGS: {extra}", flush=True)

    print("[entrypoint] " + " ".join(command), flush=True)
    code = subprocess.run(command, cwd=str(SCRIPTS.parent)).returncode
    lines = sum(1 for _ in out.open(encoding="utf-8")) if out.exists() else 0
    print(f"[entrypoint] xong, mã {code}; {out} có {lines} dòng", flush=True)
    # infer.py trả về khác 0 nếu có mẫu lỗi, nhưng bài nộp vẫn đủ dòng và vẫn chấm được:
    # một mẫu lỗi thì 0 điểm mẫu đó, còn container chết thì mất cả lượt.
    return 0 if lines else (code or 1)


if __name__ == "__main__":
    raise SystemExit(main())
