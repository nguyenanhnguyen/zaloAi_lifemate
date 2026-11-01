"""Đọc một thư mục kết quả Kaggle/local và in ra đúng những con số dùng để ra quyết định.

    python scripts/diag/read_run.py <thư mục chứa summary.json / metrics.jsonl>

Thay cho việc mở tay từng file. Chịu được thiếu file: có gì đọc nấy, nói rõ cái gì thiếu.
Mọi ngưỡng ở đây khớp với RUNBOOK.md; sửa một chỗ thì sửa cả hai.
"""
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

# A30 nhanh hơn T4 bao nhiêu lần. Suy từ băng thông (933/320), CHƯA ĐO.
# V15 đo ~9 token/s trên T4, thấp hơn ~4 lần trần băng thông của chính T4 -> nút cổ chai có lẽ là
# overhead mỗi token, và nếu vậy tỉ lệ này lạc quan quá. Giữ nguyên nhưng in kèm cảnh báo.
A30_SPEEDUP = 2.9
DOCKER_LIMIT_SECONDS = 30.0


def read_jsonl(path):
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def numbers(rows, key):
    return [r[key] for r in rows if isinstance(r.get(key), (int, float))]


def describe(values, unit="s"):
    if not values:
        return "—"
    values = sorted(values)
    median = statistics.median(values)
    p95 = values[min(len(values) - 1, int(0.95 * len(values)))]
    return f"trung vị {median:.1f}{unit}  p95 {p95:.1f}{unit}  max {max(values):.1f}{unit}"


def section(title):
    print(f"\n{title}\n" + "─" * len(title))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path,
                        help="thư mục có summary.json / metrics.jsonl (ví dụ .../kaggle_v2/preflight)")
    args = parser.parse_args()
    directory = args.directory
    if not directory.is_dir():
        raise SystemExit(f"Không thấy thư mục: {directory}")

    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    metrics = read_jsonl(directory / "metrics.jsonl")
    events = read_jsonl(directory / "events.jsonl")
    if not metrics:
        raise SystemExit(f"Không có metrics.jsonl trong {directory} — chưa tải về đủ file?")

    print(f"Thư mục : {directory}")
    print(f"Mẫu     : {len(metrics)} bản ghi"
          + (f", chọn {summary['selected']}" if "selected" in summary else "")
          + ("" if summary.get("complete_selection", True) else "  ⚠️ CHƯA ĐỦ (run bị dừng giữa)"))

    # ---------------------------------------------------------------- điểm
    section("ĐIỂM")
    scored = [r for r in metrics if "local_score" in r]
    if not scored:
        print("Không có nhãn (split != train) nên không có điểm. Chỉ đọc được thời gian và stop_reason.")
    else:
        score = summary.get("local_train_score")
        if score is None:
            score = 100 * sum(r["local_score"] for r in scored) / len(scored)
        print(f"local_train_score = {score:.2f}   (local_v4 = 5.50, local_v1 trên leaderboard = 22.52)")
        zero = [r for r in scored if not r["local_score"]]
        print(f"Mẫu được 0 điểm: {len(zero)}/{len(scored)} ({100*len(zero)/len(scored):.0f}%)")
        if zero:
            print("  vì sao:")
            for reason, count in Counter(
                    "quá 30s" if not r.get("within_30_seconds")
                    else "failed, stop_reason=" + str(r.get("stop_reason")) if r.get("failed")
                    else "state_hash lệch" if not r.get("state_match")
                    else "khớp 0 call" for r in zero).most_common():
                print(f"    {count:4d}  {reason}")

        # Trên T4 notebook đặt DEADLINE_SECONDS = 600 (chẩn đoán), nên điểm ở trên KHÔNG áp mốc 30s
        # của BTC: nó là điểm CHẤT LƯỢNG thuần. Vòng Docker áp mốc đó, và mẫu quá hạn về 0.
        # Hai con số dưới là điểm thật sẽ nhận nếu mốc được thực thi.
        def score_under_limit(threshold):
            kept = sum(r["local_score"] for r in scored
                       if isinstance(r.get("seconds"), (int, float)) and r["seconds"] < threshold)
            return 100 * kept / len(scored)

        print(f"\nĐiểm ở trên đo với ngân sách 600 s/mẫu (chẩn đoán T4), KHÔNG áp mốc 30 s của BTC.")
        print(f"  nếu áp mốc 30 s ngay trên T4            : {score_under_limit(30):.2f}")
        print(f"  nếu áp mốc 30 s sau quy đổi A30 ({A30_SPEEDUP}x) : "
              f"{score_under_limit(30*A30_SPEEDUP):.2f}   <- ước lượng điểm vòng Docker")
        print("  (vòng public test ta tự chạy, tự nộp JSONL -> không bị ép 30 s. Dòng đầu là điểm")
        print("   dùng để so với 22.52; hai dòng sau là rủi ro vòng Docker 16/11.)")
        matched = [r for r in scored if r.get("state_match")]
        print(f"state_hash khớp: {len(matched)}/{len(scored)} ({100*len(matched)/len(scored):.0f}%)"
              "   <- chỉ số quyết định: lệch hash là 0 điểm")

        # Hướng lệch ở mức call: V2 trước đây over-specify (thêm arg), không bao giờ under-specify.
        gold = sum(numbers(scored, "gold_calls"))
        predicted = sum(numbers(scored, "predicted_calls"))
        empty_gold = sum(1 for r in scored if r.get("gold_calls") == 0)
        no_call = sum(1 for r in scored if r.get("predicted_calls") == 0)
        print(f"Số call : gold {gold}, ta nộp {predicted}"
              + (f"  ({predicted/gold:.2f}x)" if gold else ""))
        print(f"          {no_call} mẫu ta nộp 0 call, {empty_gold} mẫu gold vốn rỗng")
        wer = numbers(scored, "asr_wer")
        if wer:
            print(f"ASR WER : {100*statistics.median(wer):.1f}% (trung vị)")

    # ------------------------------------------------------- stop_reason
    section("KẾT CỤC TỪNG MẪU")
    reasons = Counter(r.get("stop_reason") for r in metrics)
    good = {"done", "asked_missing"}
    for reason, count in reasons.most_common():
        flag = "   ok" if reason in good else "  <- tính là failed"
        print(f"  {count:4d}  {str(reason):<22}{flag}")
    turns = numbers(metrics, "llm_turns")
    rejected = sum(numbers(metrics, "rejected_calls"))
    if turns:
        print(f"\nSố lượt LLM/mẫu: {describe(turns, '')}   (MAX_STEPS = 8)")
    print(f"Call bị từ chối trước khi thực thi: {rejected}")

    # ------------------------------------------------------------ thời gian
    section("THỜI GIAN  —  ngân sách vòng Docker là 30 s/mẫu trên A30")
    total = numbers(metrics, "seconds")
    print(f"tổng/mẫu : {describe(total)}")
    for key, label in (("asr_seconds", "ASR"), ("llm_seconds", "LLM"), ("tool_seconds", "tool")):
        values = numbers(metrics, key)
        if values:
            share = 100 * sum(values) / sum(total) if total else 0
            print(f"{label:<9}: {describe(values)}   ({share:.0f}% tổng thời gian)")

    if total:
        median = statistics.median(total)
        p95 = sorted(total)[min(len(total) - 1, int(0.95 * len(total)))]
        print(f"\nQuy đổi T4 → A30 (chia {A30_SPEEDUP}):"
              f"  trung vị {median/A30_SPEEDUP:.1f}s   p95 {p95/A30_SPEEDUP:.1f}s"
              f"   / giới hạn {DOCKER_LIMIT_SECONDS:.0f}s")
        headroom = DOCKER_LIMIT_SECONDS * A30_SPEEDUP
        over = sum(1 for t in total if t > headroom)
        verdict = ("ĐẠT" if over == 0 else
                   f"{over}/{len(total)} mẫu ({100*over/len(total):.0f}%) sẽ timeout")
        print(f"Ngưỡng tương đương trên T4: {headroom:.0f}s  ->  {verdict}")
        print("⚠️  Tỉ lệ 2.9x suy từ băng thông, CHƯA ĐO trên A30. V15 cho ~9 token/s trên T4,")
        print("    thấp hơn ~4 lần trần băng thông của chính T4 -> nếu nghẽn là overhead mỗi token")
        print("    thì A30 nhanh hơn ÍT hơn 2.9x và dòng trên đang lạc quan. Đừng nộp Docker dựa")
        print("    vào riêng con số này.")

    # ------------------------------------------------- thông lượng token
    generations = [e for e in events if e.get("event") == "llm_end"]
    if generations:
        section("THÔNG LƯỢNG SINH TOKEN  (từ events.jsonl)")
        seconds = sum(e["seconds"] for e in generations if isinstance(e.get("seconds"), (int, float)))
        produced = sum(e["output_tokens"] for e in generations
                       if isinstance(e.get("output_tokens"), (int, float)))
        if seconds > 0 and produced:
            print(f"{produced} token / {seconds:.0f}s  =  {produced/seconds:.1f} token/s"
                  f"   (V15 đo 9.0 token/s với bản Thinking)")
        print(f"token vào  : {describe(numbers(generations, 'input_tokens'), '')}   (MAX_CONTEXT 12288)")
        print(f"token ra   : {describe(numbers(generations, 'output_tokens'), '')}   (MAX_NEW_TOKENS 256)")
        capped = [e for e in generations if e.get("output_tokens") == 256]
        if capped:
            print(f"⚠️  {len(capped)}/{len(generations)} lượt chạm trần 256 token -> đầu ra bị cắt."
                  "\n    Xem 'response' trong events.jsonl: model đang giải thích thay vì ra tool call.")
        cached = numbers(generations, "cached_tokens")
        if cached:
            print(f"KV cache tái dùng: {describe(cached, '')} token/lượt")
    elif events:
        section("THÔNG LƯỢNG SINH TOKEN")
        print("events.jsonl có nhưng không có bản ghi 'llm_end' — tải thiếu file?")
    else:
        section("THÔNG LƯỢNG SINH TOKEN")
        print("Không có events.jsonl. Tải nó về để đo token/s thật và xem lượt nào bị cắt.")

    # ----------------------------------------------------------- bộ nhớ
    peaks = {}
    for row in metrics:
        for device, value in (row.get("gpu_peak_mb") or {}).items():
            peaks.setdefault(device, []).append(value)
    if peaks:
        section("ĐỈNH BỘ NHỚ GPU")
        for device, values in sorted(peaks.items()):
            print(f"  {device}: max {max(values)} MiB  (T4 = 14912 MiB)")

    # ---------------------------------------------------------- kết luận
    section("BƯỚC TIẾP THEO")
    if not scored:
        print("Đây là run không nhãn. Nếu là public_test và đã đủ mẫu, nộp submission.jsonl.")
    else:
        score = summary.get("local_train_score") or 0
        if score > 15:
            print("Điểm > 15: bản vá luồng có tác dụng. Đi tiếp bước 3 RUNBOOK (fine-tune).")
        elif score >= 8:
            print("Điểm 8–15: khá hơn local_v4 (5.50) nhưng chưa đủ. Vẫn đi tiếp bước 3 — fine-tune")
            print("là đòn còn lại.")
        else:
            print("Điểm < 8: xem bảng KẾT CỤC TỪNG MẪU và state_hash ở trên để biết hỏng ở đâu")
            print("TRƯỚC khi đổi bất cứ cấu hình nào.")
        print("\nDù điểm thế nào, hai câu hỏi dưới vẫn phải trả lời trước vòng Docker 16/11:")
        print("  1. Thời gian/mẫu có xuống dưới ngưỡng tương đương trên T4 không (xem trên)?")
        print("  2. Tỉ lệ T4→A30 thật là bao nhiêu — chưa ai đo.")


if __name__ == "__main__":
    main()
