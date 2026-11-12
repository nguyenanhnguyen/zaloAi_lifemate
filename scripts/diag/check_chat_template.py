"""Render hoi thoai NHIEU BUOC bang CHINH chat_template.jinja cua Qwen/Qwen3.5-4B.

    python -X utf8 scripts/diag/check_chat_template.py [notebook.ipynb]

Day la phep kiem le ra phai co tu dau: no bat duoc CA HAI loi da lam chet hai lust Kaggle
(lust 25 chot model_type, lust 26 arguments la chuoi JSON) ma khong can GPU, khong can 10
phut tai model, khong can mot session Kaggle.

Dung dung cau truc messages ma cell infer_sample dung, va doi chieu vong kin:
  dict args -> template render XML -> parse_decision doc lai -> phai bang dict ban dau.
"""
import hashlib
import io
import json
import re
import sys
from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT = Path(__file__).resolve().parents[2]
FIX = Path(__file__).parent / 'fixtures'
TPL_PATH = FIX / 'qwen35_chat_template.jinja'
PROV = json.loads((FIX / 'PROVENANCE.json').read_text(encoding='utf-8'))
if hashlib.sha256(TPL_PATH.read_bytes()).hexdigest() != PROV['sha256']:
    sys.exit('fixture doi noi dung: %s khong khop sha256 trong PROVENANCE.json' % TPL_PATH)
TPL = TPL_PATH.read_text(encoding='utf-8')
NB = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'notebook/lifemate_kaggle_v4.ipynb'
R = ROOT / 'data/train+public_test/train+public_test'
print('kiem   : %s' % NB)
print('template: %s @ %s' % (PROV['repo'], PROV['revision'][:12]))

# ---- nap parse_decision THAT tu notebook -------------------------------------------------
cell11 = ''.join(json.load(io.open(NB, encoding='utf-8'))['cells'][11]['source'])
start = cell11.index('# ---- Qwen3.5 goi tool bang XML')
_tail = cell11[start:]
block = _tail[:_tail.index('\nclass ')] if '\nclass ' in _tail else _tail


class InvalidDecision(Exception):
    pass


ns = {'re': re, 'json': json, 'InvalidDecision': InvalidDecision,
      'monitor': type('M', (), {'event': lambda *a, **k: None})()}
exec(block, ns)
parse_decision, arg_types = ns['parse_decision'], ns['arg_types']


# ---- moi truong Jinja giong transformers -------------------------------------------------
def raise_exception(message):
    raise ValueError(message)


env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                    extensions=['jinja2.ext.loopcontrols'])
env.filters['tojson'] = lambda v, **kw: json.dumps(v, ensure_ascii=False)
env.globals['raise_exception'] = raise_exception
template = env.from_string(TPL)

cat = json.load(open(R / 'public_test/public_test_tools_catalog.json', encoding='utf-8'))
TOOLS = [t for app in cat['apps'] for t in app['tools']]
TYPES = arg_types(TOOLS)

FAIL = 0


def check(name, got, want=True):
    global FAIL
    if got == want:
        print('  DAT   %s' % name)
    else:
        FAIL += 1
        print('  TRUOT %s\n        duoc: %r\n        muon: %r' % (name, got, want))


def render(messages, tools, think):
    return template.render(messages=messages, tools=tools, add_generation_prompt=True,
                           enable_thinking=think)


SYS = 'Ban la LifeMate. Chi tool call hoac DONE.'
CTX = json.dumps({'latest_request': 'chuyen 50 nghin cho me', 'current_time': '2026-11-12T08:00:00'},
                 ensure_ascii=False)

ARGS = {'from_account_id': 'acc_01', 'beneficiary_id': 'ben_03', 'amount': 50000}

print('=== 1. mot buoc: system + user, thinking BAT ===')
out = render([{'role': 'system', 'content': SYS}, {'role': 'user', 'content': CTX}], TOOLS, True)
check('prompt ket thuc bang <think> DANG MO', out.endswith('<think>\n'))
check('co chi dan dinh dang XML', '<parameter=example_parameter_1>' in out)
check('khong co chi dan JSON kieu cu', '"arguments":' not in out)

print('\n=== 2. thinking TAT: khoi <think> da dong san ===')
out = render([{'role': 'system', 'content': SYS}, {'role': 'user', 'content': CTX}], TOOLS, False)
check('prefill khoi dong', out.endswith('<think>\n\n</think>\n\n'))

print('\n=== 3. HAI BUOC -- day la cho lust 26 chet ===')
messages = [
    {'role': 'system', 'content': SYS},
    {'role': 'user', 'content': CTX},
    # DUNG cau truc cell infer_sample append (sau khi sua: arguments la DICT)
    {'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': 'call_0', 'type': 'function',
         'function': {'name': 'banking.transfer', 'arguments': ARGS}}]},
    {'role': 'tool', 'tool_call_id': 'call_0',
     'content': json.dumps({'ok': True, 'transaction_id': 'txn_9'}, ensure_ascii=False)},
]
try:
    out = render(messages, TOOLS, True)
    print('  DAT   render KHONG nem loi (truoc khi sua: TypeError o dong 120)')
except Exception as exc:                                   # noqa: BLE001
    FAIL += 1
    print('  TRUOT render nem %s: %s' % (type(exc).__name__, exc))
    out = ''

if out:
    check('call cu hien ra dang XML', '<function=banking.transfer>' in out)
    check('tham so dang <parameter=>', '<parameter=amount>' in out)
    check('ket qua tool trong <tool_response>', '<tool_response>' in out)

    print('\n=== 4. VONG KIN: dict -> template -> parse_decision -> dict ===')
    # rindex, khong index: chuoi chi dan dinh dang trong system prompt CHUA SAN mot
    # <tool_call> vi du (example_function_name). Lay cai dau tien la lay vi du do.
    piece = out[out.rindex('<tool_call>'):out.rindex('</tool_call>') + len('</tool_call>')]
    back = parse_decision(piece, types=TYPES)
    check('ten tool', back['name'], 'banking.transfer')
    check('doi so giong HOAN TOAN ban dau', back['arguments'], ARGS)

print('\n=== 5. arguments la CHUOI JSON (bug cua lust 26) phai bi phat hien ===')
bad = [m if m.get('role') != 'assistant' else
       {**m, 'tool_calls': [{'id': 'call_0', 'type': 'function',
                             'function': {'name': 'banking.transfer',
                                          'arguments': json.dumps(ARGS)}}]}
       for m in messages]
try:
    render(bad, TOOLS, True)
    check('phai nem loi voi arguments dang chuoi', 'khong nem', 'nem')
except Exception as exc:                                   # noqa: BLE001
    print('  DAT   arguments dang chuoi -> %s (dung loi da thay tren Kaggle)'
          % type(exc).__name__)

print('\n=== 6. vong kin tren 300 gold call that cua train, qua template THAT ===')
tcat = json.load(open(R / 'train/train_tools_catalog.json', encoding='utf-8'))
ttools = [t for app in tcat['apps'] for t in app['tools']]
TTYPES = arg_types(ttools)
names = {t['name'] for t in ttools}
total = ok = 0
bad_ex = []
for line in open(R / 'train/train.jsonl', encoding='utf-8'):
    if not line.strip() or total >= 300:
        continue
    for call in (((json.loads(line).get('label') or {}).get('tool_calls')) or []):
        if call['name'] not in names or total >= 300:
            continue
        total += 1
        msgs = [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': CTX},
                {'role': 'assistant', 'content': '', 'tool_calls': [
                    {'id': 'c0', 'type': 'function',
                     'function': {'name': call['name'],
                                  'arguments': call.get('arguments') or {}}}]},
                {'role': 'tool', 'tool_call_id': 'c0', 'content': '{"ok":true}'}]
        text = render(msgs, ttools, True)
        piece = text[text.rindex('<tool_call>'):text.rindex('</tool_call>') + 12]
        back = parse_decision(piece, types=TTYPES)
        if back['name'] == call['name'] and back['arguments'] == (call.get('arguments') or {}):
            ok += 1
        elif len(bad_ex) < 5:
            bad_ex.append((call, back))
print('  %d/%d gold call di qua template THAT roi doc lai KHONG doi' % (ok, total))
for call, back in bad_ex:
    print('    gold : %s' % json.dumps(call, ensure_ascii=False)[:150])
    print('    doc  : %s' % json.dumps(back, ensure_ascii=False)[:150])
check('khong mat call nao', ok, total)

print('\n=== 7. duong TU CHOI/THU LAI: chen mot message user vao giua ===')
# cell infer_sample append {'role':'user'} khi call bi tu choi (schema sai, ID bia, output bi
# cat). Day la mot user turn SAU assistant+tool, va template co logic ns.last_query_index phu
# thuoc vi tri user cuoi -- duong nay chua tung duoc render qua template nay.
retry = messages + [
    {'role': 'user', 'content': 'Dau ra chua hop le hoac bi cat. Goi lai dung schema.'},
    {'role': 'assistant', 'content': 'Can doc so du truoc.', 'tool_calls': [
        {'id': 'call_1', 'type': 'function',
         'function': {'name': 'banking.get_balance', 'arguments': {}}}]},
    {'role': 'tool', 'tool_call_id': 'call_1', 'content': '{"ok":true,"balance":120000}'},
]
try:
    out7 = render(retry, TOOLS, True)
    print('  DAT   render KHONG nem loi')
    check('call khong tham so ra dung', '<function=banking.get_balance>' in out7)
    check('assistant co CA content lan tool_call', 'Can doc so du truoc.' in out7)
    check('hai tool_response rieng biet', out7.count('<tool_response>'), 2)
    check('ket thuc bang <think> dang mo', out7.endswith('<think>\n'))
    piece7 = out7[out7.rindex('<tool_call>'):out7.rindex('</tool_call>') + 12]
    check('doc lai call khong tham so', parse_decision(piece7, types=TYPES),
          {'name': 'banking.get_balance', 'arguments': {}})
except Exception as exc:                                   # noqa: BLE001
    FAIL += 1
    print('  TRUOT render nem %s: %s' % (type(exc).__name__, exc))

print('\n' + ('TAT CA DAT' if not FAIL else '%d PHEP KIEM TRUOT' % FAIL))
sys.exit(1 if FAIL else 0)
