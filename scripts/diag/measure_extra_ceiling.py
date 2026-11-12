"""Do TRAN TREN cua don bay 'goi thua': neu bo duoc extra call thi an them bao nhieu diem?

Khong xay gi truoc khi biet con so nay. Neu tran tren nho thi don bay nay vo nghia.
"""
import json, sys
from pathlib import Path
R = Path('D:/z_DaiHoc/a_zaloAI/data/train+public_test/train+public_test')

gold = {}
for line in open(R/'train/train.jsonl', encoding='utf-8'):
    if line.strip():
        d = json.loads(line)
        gold[d['id']] = ((d.get('label') or {}).get('tool_calls')) or []

cat = json.load(open(R/'train/train_tools_catalog.json', encoding='utf-8'))
TOOLS = {t['name']: t for app in cat['apps'] for t in app['tools']}

def match(a, b):
    if a['name'] != b['name']: return False
    aa, bb = a.get('arguments') or {}, b.get('arguments') or {}
    if set(aa) != set(bb): return False
    return all(aa[k] == bb[k] for k in aa)

def score(pred, g):
    if not g: return float(not pred)
    rem = list(g); m = 0
    for p in pred:
        for i, q in enumerate(rem):
            if match(p, q): m += 1; rem.pop(i); break
    return 2*m/(len(pred)+len(g))

# phan loai tool doc/ghi tu TEN (catalog khong danh dau) -- in ra de kiem bang mat
READ_PREF = ('get_','list_','search','check_','find','read_','lookup','query','_info')
def is_read(name):
    tail = name.split('.',1)[-1]
    return any(p in tail for p in READ_PREF) or tail.startswith(('get','list','search','check'))

for f in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f, encoding='utf-8') if l.strip()]
    rows = [r for r in rows if r['id'] in gold]
    base = sum(score(r.get('tool_calls') or [], gold[r['id']]) for r in rows)
    # ORACLE: bo moi call khong match gold
    orc = 0.0; orc_read = 0.0
    n_extra = n_extra_read = 0
    for r in rows:
        g = gold[r['id']]; p = r.get('tool_calls') or []
        rem = list(g); keep = []
        for c in p:
            hit = next((i for i,q in enumerate(rem) if match(c,q)), None)
            if hit is None: n_extra += 1; n_extra_read += is_read(c['name'])
            else: rem.pop(hit); keep.append(c)
        orc += score(keep, g)
        # oracle chi bo extra la READ-ONLY (an toan voi state_hash)
        rem2 = list(g); keep2 = []
        for c in p:
            hit = next((i for i,q in enumerate(rem2) if match(c,q)), None)
            if hit is None and is_read(c['name']): continue
            if hit is not None: rem2.pop(hit)
            keep2.append(c)
        orc_read += score(keep2, g)
    n = len(rows)
    print('%-42s n=%d' % (Path(f).name, n))
    print('   hien tai            : %.2f  (x100 = %.2f)' % (base/n, 100*base/n))
    print('   oracle bo MOI extra  : %.2f  (x100 = %.2f)  +%.2f diem' % (orc/n, 100*orc/n, 100*(orc-base)/n))
    print('   oracle bo extra DOC  : %.2f  (x100 = %.2f)  +%.2f diem' % (orc_read/n, 100*orc_read/n, 100*(orc_read-base)/n))
    print('   extra: %d, trong do read-only: %d (%.0f%%)' % (n_extra, n_extra_read, 100*n_extra_read/max(n_extra,1)))
