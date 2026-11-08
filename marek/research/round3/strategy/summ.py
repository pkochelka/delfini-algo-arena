"""Summarise results.jsonl: per tag, round3 pnl and synth mean, with calm/vol/shock parts.
uv run python marek/research/round3/strategy/summ.py [TAG_PREFIX ...] [--base TAG]"""
import json, sys, pathlib, collections
rows = [json.loads(l) for l in open(pathlib.Path(__file__).parent / 'results.jsonl') if l.strip()]
args = [a for a in sys.argv[1:] if not a.startswith('--')]
base = sys.argv[sys.argv.index('--base') + 1] if '--base' in sys.argv else None
by = collections.OrderedDict()
for r in rows:
    if 'pnl' not in r: continue
    by.setdefault(r['tag'], {})[r['tape']] = r
def syn(d):
    v = [r for t, r in d.items() if 'synth' in t]
    return (sum(r['pnl'] for r in v) / len(v), len(v), {k: sum(r['parts'][k] for r in v) / len(v) for k in ('calm', 'volatile', 'shock')}) if v else None
b = by.get(base)
print(f"{'tag':28s} {'r3':>8s} {'calm':>7s} {'vol':>7s} {'shock':>7s} | {'synth':>8s} n {'calm':>7s} {'vol':>7s} {'shock':>7s} | d_r3 d_syn")
for tag, d in by.items():
    if args and not any(tag.startswith(a) for a in args): continue
    r3 = d.get('data/round3.csv'); s = syn(d)
    line = f"{tag:28s} "
    line += f"{r3['pnl']/1e3:8.1f} {r3['parts']['calm']/1e3:7.1f} {r3['parts']['volatile']/1e3:7.1f} {r3['parts']['shock']/1e3:7.1f} | " if r3 else ' ' * 33 + '| '
    line += f"{s[0]/1e3:8.1f} {s[1]} {s[2]['calm']/1e3:7.1f} {s[2]['volatile']/1e3:7.1f} {s[2]['shock']/1e3:7.1f} | " if s else ' ' * 37 + '| '
    if b:
        if r3 and 'data/round3.csv' in b: line += f"{(r3['pnl'] - b['data/round3.csv']['pnl'])/1e3:+6.1f} "
        bs = syn(b)
        if s and bs: line += f"{(s[0] - bs[0])/1e3:+6.1f}"
    print(line)
