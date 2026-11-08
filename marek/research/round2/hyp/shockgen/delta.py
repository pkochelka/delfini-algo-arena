"""Per-shock episode pnl delta (window [onset+2, onset+10], net of breach penalty) between two probe sets.
python delta.py BASEPREFIX CANDPREFIX"""
import pickle, json, csv, bisect, sys, pathlib
import numpy as np
D = pathlib.Path(__file__).parent; ROOT = D.parents[4]
bp, cp = sys.argv[1], sys.argv[2]
def load(pre, t):
    X = pickle.load(open(D/f'{pre}_{t}.pkl', 'rb'))
    e = np.array(X['eq']); te = e[:, 0]; eqv = e[:, 1] + e[:, 2]*e[:, 3]
    def ep(a, b):
        f = lambda x: eqv[min(bisect.bisect_left(te, x), len(te)-1)]
        pen = sum(ms*10 for (t0, ms) in X['breaches'] if a <= t0 < b)
        return f(b) - f(a) - pen
    return X['res']['pnl'], ep
allr = []
for t in ['r2'] + [f's{i}' for i in range(1, 8)]:
    if t == 'r2':
        ons = [float(r['onset']) for r in csv.DictReader(open(ROOT/'marek/research/round2/flow/shock_events.csv'))]
        Js = [None]*len(ons)
    else:
        tr = json.load(open(ROOT/f'data/synth/r2_v2_seed{t[1:]}.truth.json')); ons, Js = tr['onsets'], tr['jumps']
    P0, e0 = load(bp, t); P1, e1 = load(cp, t)
    ds = []
    seen = set()
    for o, J in zip(ons, Js):
        if round(o, 2) in seen: continue
        seen.add(round(o, 2))
        d = e1(o+2, o+10) - e0(o+2, o+10); ds.append((d, o, J))
    tot = P1 - P0; sh = sum(d for d, _, _ in ds)
    big = max(ds, key=lambda x: abs(x[0]))
    up = sum(1 for d, _, _ in ds if d > 100); dn = sum(1 for d, _, _ in ds if d < -100)
    ex5398 = tot - sum(d for d, o, _ in ds if t == 'r2' and abs(o - 5395.8) < 1)
    print(f"{t}: total {tot:8.0f} shocks {sh:8.0f} nonshock {tot-sh:8.0f} up {up:2d} dn {dn:2d} biggest {big[0]:7.0f} @{big[1]:.1f} J={big[2] if big[2] is None else round(big[2],1)} | total-biggest {tot-big[0]:8.0f}" + (f" ex5398 {ex5398:.0f}" if t=='r2' else ''))
    allr.append((t, tot, big[0], up, dn, ds))
    if "-v" in sys.argv:
        for d, o, J in sorted(ds):
            if abs(d) > 1500: print("   ", t, round(o, 2), J if J is None else round(J, 1), round(d))
syn = [x for x in allr if x[0] != 'r2']
print('synth mean delta %.0f | minus biggest shock per tape %.0f | minus s1 4391 %.0f' % (
    np.mean([x[1] for x in syn]), np.mean([x[1]-x[2] for x in syn]),
    np.mean([x[1] - (sum(d for d, o, _ in x[5] if abs(o-4391.06) < 0.1) if x[0]=='s1' else 0) for x in syn])))
print('shocks improved %d worsened %d (|d|>100), all 8 tapes' % (sum(x[3] for x in allr), sum(x[4] for x in allr)))
