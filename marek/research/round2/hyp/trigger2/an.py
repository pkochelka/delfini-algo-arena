import json, sys, pathlib
D = pathlib.Path(__file__).parent; SG = D.parent/'shockgen'
tapes = ['r2','s1','s2','s3']
data = {t: json.load(open(D/f'ev_{t}.json')) for t in tapes}
import csv
def onsets(t):
    if t=='r2': return [float(r['onset']) for r in csv.DictReader(open('/Users/marek/Documents/delfini-algo-arena/marek/research/round2/flow/shock_events.csv'))]
    return json.load(open(f'/Users/marek/Documents/delfini-algo-arena/data/synth/r2_v2_seed{t[1:]}.truth.json'))['onsets']
V5 = {}
for t in tapes:
    m = {round(s['onset'],3): s['LV']['5'][1] for s in json.load(open(SG/f'stale_{t}.json'))}
    V5[t] = [m.get(round(o,3), 0) for o in onsets(t)]
def det(ev, k, m, w, g0, g1, inv=None):
    cs = [(x, e) for x, e in ev if e in ('ca', 'cb')]
    iv = [x for x, e in ev if e == 'ii']
    for j, (x, e) in enumerate(cs):
        if not (g0 <= x <= g1): continue
        if inv is not None and not any(x - inv <= y < x for y in iv): continue
        na = sum(1 for y, f in cs[:j+1] if y > x - w and f == 'ca'); nb = sum(1 for y, f in cs[:j+1] if y > x - w and f == 'cb')
        if na >= k and nb <= m: return x, 1
        if nb >= k and na <= m: return x, -1
    return None
def clip(ev):
    for x, e in ev:
        if 0 <= x <= 0.1 and e in ('clipU', 'clipD'): return x, (1 if e == 'clipU' else -1)
    return None
def run(k, m, w, g0, g1, inv=None, show=False):
    R = dict(n=0, hit=0, ok=0, early=0, dl=[], nc=0, nc_hit=0, nc_ok=0, ncV=0, ncVok=0, ctl=0, nctl=0, rnd=0, nrnd=0, wrongV=0, okV=0, totV=0, pre=0)
    for t in tapes:
        for w_ in data[t]:
            ev = w_['ev']
            r = det(ev, k, m, w, g0, g1, inv)
            if w_['kind'] == 'shock':
                J = w_['J']
                if J is None or abs(J) >= 100 or J == 0: continue
                v = V5[t][w_['i']]; d = 1 if J > 0 else -1
                R['n'] += 1; R['totV'] += v
                c = clip(ev); nc = c is None
                if nc: R['nc'] += 1; R['ncV'] += v
                if r:
                    R['hit'] += 1
                    if r[0] < 0: R['pre'] += 1
                    if r[1] == d: R['ok'] += 1; R['dl'].append(r[0]*1e3); R['okV'] += v
                    else: R['wrongV'] += v
                    if nc:
                        R['nc_hit'] += 1
                        if r[1] == d: R['nc_ok'] += 1; R['ncVok'] += v
            else:
                kk = 'ctl' if w_['kind'] == 'ctl' else 'rnd'
                R['n'+kk] += 1
                # control: gate relative to ref (same width)
                if r: R[kk] += 1
    dl = sorted(R['dl']); med = dl[len(dl)//2] if dl else -1
    print(f"k={k} m={m} w={w*1e3:.0f} g=[{g0*1e3:.0f},{g1*1e3:.0f}] inv={inv} | n={R['n']} hit={R['hit']} ok={R['ok']} ({R['ok']/max(R['hit'],1):.0%}) pre<T={R['pre']} med={med:.1f}ms p75={dl[3*len(dl)//4] if dl else -1:.1f} | V5 ok={R['okV']/1e3:.0f}k wrong={R['wrongV']/1e3:.0f}k of {R['totV']/1e3:.0f}k | noclip {R['nc_ok']}/{R['nc_hit']}/{R['nc']} V {R['ncVok']/1e3:.0f}k/{R['ncV']/1e3:.0f}k | FP ctl {R['ctl']}/{R['nctl']} rnd {R['rnd']}/{R['nrnd']}")
if __name__ == '__main__':
    for g0, g1 in [(-0.05, 0.15), (0.0, 0.15)]:
        for k in (2, 3, 4):
            for m in (0, 1):
                for w in (0.002, 0.003, 0.005):
                    run(k, m, w, g0, g1)
    for inv in (0.01, 0.02, 0.05):
        for k in (1, 2, 3):
            run(k, 0, 0.005, -0.05, 0.15, inv)
