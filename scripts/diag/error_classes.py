"""Phan loai LOI theo lop, so tool_calls da nop voi gold `label.tool_calls` cua split train.

Vi sao ton tai: moi luot chay tra ve DUNG MOT so (`local_train_score`). Mot con so khong noi
duoc nen sua gi. Luot Kaggle ton 30 phut den 4 gio, nen hoc duoc mot con so moi luot la qua dat.
Script nay bien mot file submission da co thanh bang phan lop, KHONG chay lai model, khong GPU.

Bon lop, cong lai bang toan bo gold call:
  exact        ten + arguments khop het            -> an diem
  args_wrong   dung ten tool, SAI arguments        -> thuong la tu dien ID thay vi tra cuu
  missing      gold co, ta khong goi               -> bo sot buoc
  extra        ta goi, gold khong co               -> goi lang phi (xem rieng ask_missing_info)

Dung:
  python scripts/diag/error_classes.py results/local_v9/lora_ask1.jsonl [...]
  python scripts/diag/error_classes.py --verbose results/local_v9/lora_ask1.jsonl
"""
import argparse, json, statistics as st
from collections import Counter
from pathlib import Path

GOLD = Path("data/train+public_test/train+public_test/train/train.jsonl")


def load_gold(path=GOLD):
    if not path.exists():
        raise SystemExit(f"Khong thay {path}. Script nay chi chay duoc tren split train (co gold).")
    out = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            d = json.loads(line)
            out[d["id"]] = d
    return out


def key(call):
    """Khoa so khop: ten + arguments da chuan hoa thu tu khoa."""
    return (call.get("name"),
            json.dumps(call.get("arguments") or {}, sort_keys=True, ensure_ascii=False))


def classify(pred_path, gold):
    rows = [json.loads(l) for l in Path(pred_path).open(encoding="utf-8") if l.strip()]
    agg = Counter()
    miss, extra, wrongargs = Counter(), Counter(), Counter()
    gold_n, pred_n, perfect, zero = [], [], 0, 0
    prefix_stop = 0
    for r in rows:
        g = gold.get(r["id"])
        if g is None:
            continue
        G = [key(c) for c in g["label"]["tool_calls"]]
        P = [key(c) for c in (r.get("tool_calls") or [])]
        gold_n.append(len(G)); pred_n.append(len(P))
        Gs, Ps = Counter(G), Counter(P)
        gn, pn = Counter(n for n, _ in G), Counter(n for n, _ in P)
        exact = sum((Gs & Ps).values())
        name_ok = sum((gn & pn).values())
        agg["exact"] += exact
        agg["args_wrong"] += max(0, name_ok - exact)
        agg["missing"] += sum((gn - pn).values())
        agg["extra"] += sum((pn - gn).values())
        for n, k in (gn - pn).items(): miss[n] += k
        for n, k in (pn - gn).items(): extra[n] += k
        for n in (gn & pn):
            d = sum((Counter(a for nm, a in G if nm == n) -
                     Counter(a for nm, a in P if nm == n)).values())
            if d: wrongargs[n] += d
        if not P: zero += 1
        if exact == len(G) and len(P) == len(G): perfect += 1
        # pred la tien to DUNG cua gold roi dung lai -> that su la "dung som"
        gnames, pnames = [n for n, _ in G], [n for n, _ in P]
        if pnames and len(pnames) < len(gnames) and pnames == gnames[:len(pnames)]:
            prefix_stop += 1
    return dict(rows=len(rows), agg=agg, miss=miss, extra=extra, wrongargs=wrongargs,
                gold_total=sum(gold_n), pred_total=sum(pred_n), perfect=perfect, zero=zero,
                prefix_stop=prefix_stop, gold_mean=st.mean(gold_n) if gold_n else 0,
                pred_mean=st.mean(pred_n) if pred_n else 0)


def report(path, r, gold, verbose=False):
    g, p, a = r["gold_total"], r["pred_total"], r["agg"]
    pct = lambda x, d: f"{x / d * 100:5.1f}%" if d else "    -"
    print(f"=== {path}  ({r['rows']} mau)")
    print(f"  gold {g} call ({r['gold_mean']:.2f}/mau) | ta nop {p} call ({r['pred_mean']:.2f}/mau)"
          f" = {p / g * 100:.0f}% khoi luong gold" if g else "")
    print(f"  exact      {a['exact']:5d}  {pct(a['exact'], g)} cua gold   <- RECALL thuc")
    print(f"  args_wrong {a['args_wrong']:5d}  {pct(a['args_wrong'], g)}")
    print(f"  missing    {a['missing']:5d}  {pct(a['missing'], g)}")
    print(f"  extra      {a['extra']:5d}  {pct(a['extra'], p)} cua pred")
    print(f"  mau hoan hao {r['perfect']}/{r['rows']} | 0 call {r['zero']} | "
          f"dung-som-dung-cho {r['prefix_stop']}")
    # Mat do gold cua CA tap, de biet tap do co lech phan phoi hay khong
    allg = [len(v["label"]["tool_calls"]) for v in gold.values()]
    if r["gold_mean"]:
        print(f"  CANH BAO phan phoi: tap do {r['gold_mean']:.2f} call/mau vs ca train "
              f"{st.mean(allg):.2f} -> nang hon {r['gold_mean'] / st.mean(allg):.2f}x"
              if abs(r["gold_mean"] - st.mean(allg)) > 0.3 else
              f"  phan phoi tap do khop ca train ({st.mean(allg):.2f} call/mau)")
    if verbose:
        for title, c in (("BO SOT nhieu nhat", r["miss"]),
                         ("goi THUA nhieu nhat", r["extra"]),
                         ("SAI ARGS nhieu nhat", r["wrongargs"])):
            if c:
                print(f"  {title}:")
                for n, k in c.most_common(10):
                    print(f"    {k:4d}  {n}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submissions", nargs="+")
    ap.add_argument("--verbose", action="store_true", help="liet ke tool bo sot / thua / sai args")
    ap.add_argument("--gold", type=Path, default=GOLD)
    args = ap.parse_args()
    gold = load_gold(args.gold)
    for s in args.submissions:
        report(s, classify(s, gold), gold, args.verbose)


if __name__ == "__main__":
    main()
