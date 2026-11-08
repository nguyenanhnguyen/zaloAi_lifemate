"""Chay THAT `train_score` cua notebook tren mau thuc, de kiem bang quy cong gate co dem duoc.

Vi sao ton tai: luot v3 chay MOT lan thay vi bon, nen cong cua tung gate phai quy duoc trong
chinh luot do. Co che la `wanted_calls` -- danh sach call NHU MODEL MUON, truoc khi
BARE_READ_GATE sua -- roi cham ca hai danh sach voi gold. Neu phep do do sai thi ca luot 4.7
gio tra ve mot ket luan sai ma khong bao gi, nen no phai duoc kiem trong vai giay o day.

Script nay KHONG tu viet lai phep cham: no `exec` dung doan `train_score` trong notebook, nen
kiem chinh ma se chay tren Kaggle. Doi chieu voi `scripts/diag/bare_args.py`, mot ban hien thuc
hoan toan doc lap: ca hai phai ra +17 tren results/local_v6/xref_gate.jsonl.

    python scripts/diag/check_gate_attribution.py
"""
import json, io, hashlib, copy, re, unicodedata
from collections import Counter
from pathlib import Path

nb = json.load(io.open('notebook/lifemate_kaggle_v3.ipynb', encoding='utf-8'))
cell = ''.join(nb['cells'][16]['source'])
start = cell.index('def fuzzy_f1')
end = cell.index('def sample_failed')
ns = {'Counter': Counter, 'copy': copy, 're': re, 'unicodedata': unicodedata, 'json': json}
exec(compile(cell[start:end], '<nb16>', 'exec'), ns)
train_score = ns['train_score']
print('nap duoc train_score tu notebook')

root = Path('data/train+public_test/train+public_test/train')
gold = {}
for line in (root / 'train.jsonl').open(encoding='utf-8'):
    if line.strip():
        it = json.loads(line); gold[it['id']] = it
catalog = json.loads((root / 'train_tools_catalog.json').read_text(encoding='utf-8'))

BARE = {'banking.get_balance','banking.list_beneficiaries','bill_payment.list_bills',
        'chat_ott.list_chats','door_lock.list_locks','ewallet.get_balance',
        'food_delivery.list_restaurants','home_security.list_alerts','learning.list_courses',
        'calendar.list_events','smart_home.get_status'}

rows = [json.loads(l) for l in open('results/local_v6/xref_gate.jsonl', encoding='utf-8') if l.strip()]
tot_gain = tot_exact = tot_unstripped = n = 0
for r in rows:
    item = gold.get(r['id'])
    if not item:
        continue
    wanted = r.get('tool_calls') or []
    stripped = [{'name': c['name'], 'arguments': {}} if c['name'] in BARE else c for c in wanted]
    sub = {'id': r['id'], 'tool_calls': stripped, 'state_hash': 'x' * 64}
    out = train_score(item, {}, sub, catalog, lambda s: 'y' * 64, elapsed=1.0, failed=False,
                      wanted_calls=wanted)
    tot_exact += out['exact']; tot_unstripped += out['exact_unstripped']
    tot_gain += out['bare_read_gain']; n += 1
print(f'{n} mau: exact {tot_exact}, exact_unstripped {tot_unstripped}, '
      f'bare_read_gain {tot_gain}')
assert tot_gain == tot_exact - tot_unstripped
print('bare_read_gain nhat quan voi hai con so kia')
