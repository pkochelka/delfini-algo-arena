"""dev_summ.py TAGPREFIX... [--base TAG]: per tag r3 / v1 mean (n) / v2 mean (n) / r2, parts, and deltas vs base on the
tapes both have (develop angle). Base defaults to the r3_best rows (fin_best_r3, fin_best_v1, fin_v2)."""
import json, sys, pathlib, collections
rows = [json.loads(l) for l in open(pathlib.Path(__file__).parents[1] / 'strategy/results.jsonl') if l.strip()]
args = [a for a in sys.argv[1:] if not a.startswith('--')]
base = sys.argv[sys.argv.index('--base') + 1] if '--base' in sys.argv else None
by = collections.OrderedDict()
for r in rows:
    if 'pnl' not in r: continue
    by.setdefault(r['tag'], {})[r['tape'].split('/')[-1].replace('.csv', '')] = r
B = {}
for t in ('fin_best_r3', 'fin_best_v1', 'fin_v2'):
    B.update(by.get(t, {}))
if base: B = dict(B, **by.get(base, {}))
def fam(d, f):
    return {k: v for k, v in d.items() if (f == 'r3' and k == 'round3') or (f == 'r2' and k == 'round2') or (f in ('v1', 'v2') and k.startswith('r3_' + f))}
print(f"{'tag':30s} {'r3':>7s} {'dr3':>6s} {'dcalm':>6s} {'dvol':>6s} {'dshk':>6s} | {'v1':>7s} n {'dv1':>6s} {'dcalm':>6s} {'dvol':>6s} {'dshk':>6s} | {'v2':>6s} n {'dv2':>6s} | {'r2':>6s} {'dr2':>6s}")
for tag, d in by.items():
    if args and not any(tag.startswith(a) for a in args): continue
    out = f'{tag:30s} '
    for f in ('r3', 'v1', 'v2', 'r2'):
        x = fam(d, f)
        if not x:
            out += {'r3': ' ' * 35 + '| ', 'v1': ' ' * 38 + '| ', 'v2': ' ' * 17 + '| ', 'r2': ''}[f]; continue
        m = sum(r['pnl'] for r in x.values()) / len(x)
        bk = [k for k in x if k in B]
        dm = sum(x[k]['pnl'] - B[k]['pnl'] for k in bk) / len(bk) / 1e3 if bk else float('nan')
        dp = {p: sum(x[k]['parts'][p] - B[k]['parts'][p] for k in bk) / len(bk) / 1e3 if bk else float('nan') for p in ('calm', 'volatile', 'shock')}
        if f == 'r3': out += f"{m/1e3:7.1f} {dm:+6.1f} {dp['calm']:+6.1f} {dp['volatile']:+6.1f} {dp['shock']:+6.1f} | "
        elif f == 'v1': out += f"{m/1e3:7.1f} {len(x)} {dm:+6.1f} {dp['calm']:+6.1f} {dp['volatile']:+6.1f} {dp['shock']:+6.1f} | "
        elif f == 'v2': out += f"{m/1e3:6.1f} {len(x)} {dm:+6.1f} | "
        else: out += f"{m/1e3:6.1f} {dm:+6.1f}"
    print(out)
