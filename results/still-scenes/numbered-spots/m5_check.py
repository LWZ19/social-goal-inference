"""Zero-call check (an idea, before any freeze): among the fresh unmarked plan answers (M2's A0, trials
3-5, five models), how many wrong targets name a person while the episode's target is a place or object, and how
many of those name someone waiting in a line (end of the line, queue, last, customer). m5_check.py"""
import collections, json, re
from pathlib import Path
rows_by = {}
PERSON = re.compile(r'\b(person|people|woman|man|guy|lady|customer|staff|barista|attendant|clerk|member|someone|girl|boy|individual)\b', re.I)
LINE = re.compile(r'\b(end of the (line|queue)|back of the (line|queue)|last (person|customer|one)|in (the )?(line|queue)|queue|line)\b', re.I)
STAFF = re.compile(r'\b(staff|barista|attendant|clerk|cashier|server|employee|worker)\b', re.I)
tot = collections.Counter()
for m in ('qwen38_low', 'glm53', 'dsv41', 'x3', 'gemini'):
    run = Path(__file__).resolve().parent / 'runs' / f'm2_{m}'
    rows = {(r['scene'], r['episode'], r['trial']): r for r in map(json.loads, (run / 'm2_rows.jsonl').read_text().splitlines())}
    calls = {(c['scene_id'], c['episode'], c['trial']): c for c in map(json.loads, (run / 'calls.jsonl').read_text().splitlines()) if c['arm'] == 'A0'}
    c = collections.Counter()
    for k, r in rows.items():
        a0 = calls.get(k)
        if not a0 or a0['status'] != 'ok' or r.get('A0_right'):
            continue
        nav = next((s for s in a0['subtasks'] if s['type'] == 'navigate' and s.get('point')), None)
        txt = (nav or {}).get('target') or ''
        person_scene = 'person' in k[0] or 'conversation' in k[0]
        c['wrong'] += 1
        if PERSON.search(txt) and not person_scene:
            c['names a person (place/object scene)'] += 1
            if LINE.search(txt) and not STAFF.search(txt):
                c['  ...a person waiting in line'] += 1
            elif STAFF.search(txt):
                c['  ...staff'] += 1
    print(m, dict(c))
    tot.update(c)
print('all', dict(tot))
