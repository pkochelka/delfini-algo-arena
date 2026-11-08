"""Summarise results.jsonl per (tag, file, set): round3 pnl + synth mean with calm/vol/shock parts.
uv run python marek/research/round3/strategy/summ2.py TAG_PREFIX [...] [--base TAG]   (base = first row of that tag)"""
import json, sys, pathlib, collections
rows = [json.loads(l) for l in open(pathlib.Path(__file__).parent / 'results.jsonl') if l.strip()]
args = [a for a in sys.argv[1:] if not a.startswith('--')]
base = sys.argv[sys.argv.index('--base') + 1] if '--base' in sys.argv else None
by = collections.OrderedDict()
for r in rows:
    if 'pnl' not in r: continue
    key = (r['tag'], pathlib.Path(r['file']).stem, json.dumps(r['set'], sort_keys=True))
    by.setdefault(key, {})[r['tape']] = r
def syn(d):
    v = [r for t, r in d.items() if 'synth' in t]
    if not v: return None
    return sum(r['pnl'] for r in v) / len(v), len(v), {k: sum(r['parts'][k] for r in v) / len(v) for k in ('calm', 'volatile', 'shock')}
bk = next((k for k in by if k[0] == base), None) if base else None
b = by.get(bk)
print(f"{'tag/file/set':60s} {'r3':>7s} {'calm':>6s} {'vol':>6s} {'shk':>6s} | {'synth':>7s} n {'calm':>6s} {'vol':>6s} {'shk':>6s} | d_r3  d_syn")
for key, d in by.items():
    if args and not any(key[0].startswith(a) for a in args): continue
    r3 = d.get('data/round3.csv'); s = syn(d)
    name = f"{key[0]}/{key[1]} {key[2] if key[2] != '{}' else ''}"
    line = f"{name[:60]:60s} "
    line += f"{r3['pnl']/1e3:7.1f} {r3['parts']['calm']/1e3:6.1f} {r3['parts']['volatile']/1e3:6.1f} {r3['parts']['shock']/1e3:6.1f} | " if r3 else ' ' * 29 + '| '
    line += f"{s[0]/1e3:7.1f} {s[1]} {s[2]['calm']/1e3:6.1f} {s[2]['volatile']/1e3:6.1f} {s[2]['shock']/1e3:6.1f} | " if s else ' ' * 33 + '| '
    if b:
        if r3 and 'data/round3.csv' in b: line += f"{(r3['pnl'] - b['data/round3.csv']['pnl'])/1e3:+6.1f} "
        bs = syn({t: r for t, r in b.items() if t in d})
        s2 = syn({t: r for t, r in d.items() if t in b})
        if s2 and bs: line += f"{(s2[0] - bs[0])/1e3:+6.1f}"
    print(line)
