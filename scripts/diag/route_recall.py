"""Đo recall của truy hồi app/tool: từ vựng so với embedding bge-m3.

    python scripts/diag/route_recall.py                      # cả hai, trên GPU nếu có
    python scripts/diag/route_recall.py --method lexical     # chỉ từ vựng, không cần GPU
    python scripts/diag/route_recall.py --limit 300          # chạy nhanh

Vì sao phép đo này quyết định ngân sách GPU: 78% context là catalogue tool (trung vị 32 tool,
12.899 ký tự), mà trung vị chỉ **2 tool** được dùng thật. Cắt catalogue xuống thì cùng số giờ
Kaggle huấn luyện được nhiều mẫu hơn -- 320 mẫu hiện tại là do thời gian, không do dữ liệu.

Nhưng cắt sai là mất trắng: tool cần mà không có trong prompt thì mẫu đó ăn 0, không cứu
được ở tầng nào. `results/local_v4` đã cắt prompt về 1800 token và **điểm rơi từ 22.52 xuống
5.50**. Nên ngưỡng ở đây không phải "khá tốt" mà là gần 100%.

Truy vấn là transcript ASR (WER 28.9%), không phải audio_text -- đúng thứ model nhận lúc thi.
Chỉ xếp hạng trong số app ĐÃ CÀI của mẫu, vì `_available_tools` vốn đã lọc theo installed_apps.
"""
import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "data/train+public_test/train+public_test/train/train_tools_catalog.json"
SFT = ROOT / "sft.train.jsonl"
BGE = "BAAI/bge-m3"
BGE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
KS = (1, 2, 3, 4, 5, 6, 8, 10)


def words(text: str) -> list[str]:
    return re.findall(r"[0-9a-zà-ỹđ]+", unicodedata.normalize("NFC", text.lower()))


def load_catalog() -> tuple[dict[str, str], dict[str, str]]:
    """(app id -> văn bản mô tả app, tên tool -> văn bản mô tả tool).

    Tool nằm LỒNG trong apps[].tools[]. Đọc ở catalog["tools"] sẽ ra rỗng và mọi phép đo
    phía sau bị làm yếu mà không báo lỗi -- đã từng xảy ra một lần.
    """
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    apps, tools = {}, {}
    for app in catalog.get("apps", []):
        aid = app.get("id")
        if not aid:
            continue
        head = f"{app.get('name_vi', '')} {app.get('name', '')} {app.get('description', '')}"
        parts = [head]
        for tool in app.get("tools", []):
            text = f"{tool.get('name', '')} {tool.get('description') or tool.get('desc', '')}"
            tools[tool["name"]] = text
            parts.append(text)
        apps[aid] = " ".join(parts)
    if not tools:
        raise SystemExit("catalogue không có tool nào -- đọc sai chỗ, xem docstring load_catalog")
    return apps, tools


def load_samples(limit: int | None) -> list[dict]:
    call = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
    out = []
    for line in SFT.open(encoding="utf-8"):
        row = json.loads(line)
        gold = set()
        for message in row["messages"]:
            if message["role"] == "assistant":
                for blob in call.findall(message["content"]):
                    try:
                        gold.add(json.loads(blob)["name"])
                    except (ValueError, KeyError):
                        pass
        if not gold:
            continue
        out.append({"id": row["id"], "idx": len(out),
                    "query": json.loads(row["messages"][1]["content"])["latest_request"],
                    "gold_tools": gold,
                    "gold_apps": {name.split(".")[0] for name in gold},
                    "offered": [t["function"]["name"] for t in row["tools"]]})
        if limit and len(out) >= limit:
            break
    return out


def embed(texts: list[str], device: str, batch: int = 8):
    """CLS pooling + chuẩn hoá L2, đúng cách bge-m3 lấy vector dense."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BGE, revision=BGE_REVISION,
                                              cache_dir=str(ROOT / "models/hf-cache"))
    model = AutoModel.from_pretrained(BGE, revision=BGE_REVISION,
                                      cache_dir=str(ROOT / "models/hf-cache"),
                                      dtype=torch.float16 if device == "cuda" else torch.float32)
    model.to(device).eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch):
            part = tokenizer(texts[start:start + batch], padding=True, truncation=True,
                             max_length=512, return_tensors="pt").to(device)
            hidden = model(**part).last_hidden_state[:, 0]
            chunks.append(torch.nn.functional.normalize(hidden.float(), dim=-1).cpu())
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(chunks)


def report(name: str, samples: list[dict], rank, level: str) -> None:
    """rank(sample) -> danh sách ứng viên đã xếp hạng, ở mức app hoặc tool."""
    key = "gold_apps" if level == "app" else "gold_tools"
    hit = dict.fromkeys(KS, 0)
    for sample in samples:
        order = rank(sample)
        for k in KS:
            if sample[key] <= set(order[:k]):
                hit[k] += 1
    n = len(samples)
    print(f"\n{name} — recall {level} trên {n} mẫu")
    for k in KS:
        bar = "#" * round(hit[k] / n * 40)
        print(f"  @{k:<3}{hit[k] / n * 100:5.1f}%  {bar}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=["lexical", "embedding", "both"], default="both")
    parser.add_argument("--limit", type=int, help="chỉ lấy N mẫu đầu, để chạy nhanh")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    apps, tools = load_catalog()
    samples = load_samples(args.limit)
    print(f"catalogue: {len(apps)} app, {len(tools)} tool")
    print(f"mẫu có gold: {len(samples)}")
    offered = [len(s["offered"]) for s in samples]
    print(f"tool được cung cấp/mẫu: trung bình {sum(offered) / len(offered):.1f}")
    need = [len(s["gold_tools"]) for s in samples]
    print(f"tool cần thật/mẫu     : trung bình {sum(need) / len(need):.1f}")

    if args.method in ("lexical", "both"):
        bags = {aid: set(words(text)) for aid, text in apps.items()}
        tool_bags = {name: set(words(text)) for name, text in tools.items()}

        def lex_app(sample):
            query = set(words(sample["query"]))
            pool = sorted({name.split(".")[0] for name in sample["offered"]})
            return sorted(pool, key=lambda a: -len(query & bags.get(a, set())))

        def lex_tool(sample):
            query = set(words(sample["query"]))
            return sorted(sample["offered"],
                          key=lambda t: -len(query & tool_bags.get(t, set())))

        report("TỪ VỰNG", samples, lex_app, "app")
        report("TỪ VỰNG", samples, lex_tool, "tool")

    if args.method in ("embedding", "both"):
        import torch
        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nnạp {BGE} lên {device} ...", flush=True)
        app_ids = sorted(apps)
        tool_ids = sorted(tools)
        vectors = embed([apps[a] for a in app_ids] + [tools[t] for t in tool_ids]
                        + [s["query"] for s in samples], device)
        app_vec = vectors[:len(app_ids)]
        tool_vec = vectors[len(app_ids):len(app_ids) + len(tool_ids)]
        query_vec = vectors[len(app_ids) + len(tool_ids):]
        app_at = {a: i for i, a in enumerate(app_ids)}
        tool_at = {t: i for i, t in enumerate(tool_ids)}

        def emb_app(sample):
            index = sample["idx"]
            pool = sorted({name.split(".")[0] for name in sample["offered"]})
            rows = [app_at[a] for a in pool if a in app_at]
            if not rows:
                return pool
            score = app_vec[rows] @ query_vec[index]
            return [pool[i] for i in score.argsort(descending=True).tolist()]

        def emb_tool(sample):
            index = sample["idx"]
            pool = [t for t in sample["offered"] if t in tool_at]
            if not pool:
                return sample["offered"]
            score = tool_vec[[tool_at[t] for t in pool]] @ query_vec[index]
            return [pool[i] for i in score.argsort(descending=True).tolist()]

        report("EMBEDDING bge-m3", samples, emb_app, "app")
        report("EMBEDDING bge-m3", samples, emb_tool, "tool")

    print("\nNgưỡng dùng được: recall phải gần 100%. Mỗi 1% thiếu là 1% mẫu ăn 0 điểm,")
    print("không tầng nào cứu được. local_v4 cắt prompt và điểm rơi 22.52 -> 5.50.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
