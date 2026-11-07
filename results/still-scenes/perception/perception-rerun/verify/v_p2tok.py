"""P2 with the recorded output_tokens as denominator; numerator = word/punct proxy tokens of the people lists."""
import sys, re
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
OUT = __file__.replace('.py', '_out.txt')
L = []
def P(*a):
    s = ' '.join(map(str, a)); print(s); L.append(s)
REAL = ('real', 'staffed', 'plain')
LIST = re.compile(r'"people"\s*:\s*(\[[^\]]*\])')
tok = lambda s: len(re.findall(r"\w+|[^\w\s]", s))
for stage, model in (('s19', 'glm53'), ('s20', 'dsv41'), ('s23', 'glm53'), ('s23', 'dsv41')):
    seen = {}
    for r in H.runs(stage=stage, model=model, method='con'):
        if r.arm not in REAL: continue
        for rec in H.records(r):
            if rec.people is None: continue
            for c in rec.result.get('calls') or []:
                if c['stage'] == 'constraints' and c.get('raw_text'):
                    u = c.get('usage') or {}
                    seen[c['raw_text']] = (u.get('output_tokens') or 0, u.get('reasoning_tokens') or 0)
    OT = sum(v[0] for v in seen.values()); RT = sum(v[1] for v in seen.values())
    C = sum(len(t) for t in seen)
    vb = sum(tok(m.group(1)) for t in seen for m in LIST.finditer(t))           # values incl brackets
    vk = sum(tok(m.group(0)) for t in seen for m in LIST.finditer(t))           # with key
    nlists = sum(1 for t in seen for m in LIST.finditer(t))
    P(f'{stage} {model}: answers {len(seen)}, output_tokens {OT} (reasoning {RT}), chars/token {C/OT:.2f}, lists {nlists} ({nlists/len(seen):.1f}/answer)')
    P(f'   list values incl brackets, proxy tokens {vb} = {100*vb/OT:.2f}% of output_tokens; with key {vk} = {100*vk/OT:.2f}%')
open(OUT, 'w').write('\n'.join(L) + '\n')
