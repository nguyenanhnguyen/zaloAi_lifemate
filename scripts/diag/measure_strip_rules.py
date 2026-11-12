"""Do cac luat CAI DUOC (khong nhin gold). Oracle o buoc truoc chi noi tran tren."""
import json, sys
from pathlib import Path
R = Path('D:/z_DaiHoc/a_zaloAI/data/train+public_test/train+public_test')
gold = {}
for line in open(R/'train/train.jsonl', encoding='utf-8'):
    if line.strip():
        d = json.loads(line); gold[d['id']] = ((d.get('label') or {}).get('tool_calls')) or []
cat = json.load(open(R/'train/train_tools_catalog.json', encoding='utf-8'))
TOOLS = {t['name']: t for app in cat['apps'] for t in app['tools']}
READ_PREF = ('get_','list_','search','check_','find','read_','lookup','query','_info')
def is_read(n):
    t = n.split('.',1)[-1]
    return any(p in t for p in READ_PREF) or t.startswith(('get','list','search','check'))
def match(a,b):
    if a['name']!=b['name']: return False
    aa,bb=a.get('arguments') or {},b.get('arguments') or {}
    return set(aa)==set(bb) and all(aa[k]==bb[k] for k in aa)
def score(pred,g):
    if not g: return float(not pred)
    rem=list(g); m=0
    for p in pred:
        for i,q in enumerate(rem):
            if match(p,q): m+=1; rem.pop(i); break
    return 2*m/(len(pred)+len(g))

def key(c): return (c['name'], json.dumps(c.get('arguments') or {}, sort_keys=True, ensure_ascii=False))

def R0(p): return list(p)
def R1(p): return [c for c in p if not is_read(c['name'])]                 # bo MOI read
def R2(p):                                                                 # bo TRUNG LAP
    seen=set(); out=[]
    for c in p:
        k=key(c)
        if k in seen: continue
        seen.add(k); out.append(c)
    return out
def R3(p):                                                                 # bo read DUOI write cuoi
    last=max([i for i,c in enumerate(p) if not is_read(c['name'])], default=-1)
    return [c for i,c in enumerate(p) if i<=last or not is_read(c['name'])]
def R4(p): return R2(R3(p))                                                # trung lap + read duoi
def R5(p):                                                                 # bo read lap lai CUNG TEN (giu lan dau)
    seen=set(); out=[]
    for c in p:
        if is_read(c['name']):
            if c['name'] in seen: continue
            seen.add(c['name'])
        out.append(c)
    return out
def R6(p): return R3(R5(R2(p)))

RULES=[('R0 nguyen ban',R0),('R1 bo MOI read',R1),('R2 bo trung lap y het',R2),
       ('R3 bo read sau write cuoi',R3),('R4 = R2+R3',R4),
       ('R5 bo read trung TEN',R5),('R6 = R2+R5+R3',R6)]

for f in sys.argv[1:]:
    rows=[json.loads(l) for l in open(f,encoding='utf-8') if l.strip()]
    rows=[r for r in rows if r['id'] in gold]
    n=len(rows); print('\n=== %s  (n=%d) ===' % (Path(f).name,n))
    base=None
    for nm,fn in RULES:
        s=sum(score(fn(r.get('tool_calls') or []), gold[r['id']]) for r in rows)/n
        if base is None: base=s
        print('   %-28s %6.2f   %+.2f' % (nm, 100*s, 100*(s-base)))
