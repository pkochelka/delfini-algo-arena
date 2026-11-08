"""Non-shock stale-book events: stale value beyond the first CAP lots at +5/+10 ms after each S move."""
import sys, json, bisect, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[5]
tape = sys.argv[1]; CAP = 100
if tape == 'r2':
    csvp = ROOT/'data/round2.csv'
    import pandas as pd
    d = pd.read_parquet(ROOT/'marek/research/round2/s_dynamics/s_path.parquet')
    St = d.t_us.values*1e-6; Sv = d.s_hat.values
    ons = [float(l.split(',')[0]) for l in open(ROOT/'marek/research/round2/flow/shock_events.csv').read().split('\n')[1:] if l]
else:
    k = tape[1:]; csvp = ROOT/f'data/synth/r2_v2_seed{k}.csv'
    a = np.load(ROOT/f'marek/research/round2/hyp/filter/S_seed{k}.npy'); St, Sv = a[:,0], a[:,1]
    ons = json.load(open(ROOT/f'data/synth/r2_v2_seed{k}.truth.json'))['onsets']
ons = np.array(sorted(ons))
def S_at(t):
    i = bisect.bisect_right(St, t) - 1; return Sv[max(i,0)]
def in_shock(t):
    i = np.searchsorted(ons, t) - 1
    return i >= 0 and t - ons[i] < 4.5
# event times: S moves >= 1 tick between consecutive S points
dS = np.diff(Sv); ev = St[1:][np.abs(dS) >= 1.0]
ev = [t for t in ev if not in_shock(t)]
oid = {}; lv = {'B': {}, 'A': {}}; px = {'B': [], 'A': []}
def add(s, p, q):
    L = lv[s]
    if p not in L: L[p] = 0; bisect.insort(px[s], p)
    L[p] += q
def rem(s, p, q):
    L = lv[s]; L[p] -= q
    if L[p] <= 0: del L[p]; i = bisect.bisect_left(px[s], p); px[s].pop(i)
# for taker fills we need per-order queue; approximate with level aggregate (fine for stale value)
fifo = {'B': {}, 'A': {}}
def measure(S, side):
    """stale value on `side` vs S, sorted best-first (largest edge first); return lots, val, beyond-cap lots/val, net@3/4."""
    out = []
    if side == 'A':
        for p in px['A']:
            if p >= S - 1: break
            out.append((S - p, lv['A'][p]))
    else:
        for p in reversed(px['B']):
            if p <= S + 1: break
            out.append((p - S, lv['B'][p]))
    L = V = Lb = Vb = n3 = n4 = 0; used = 0
    for e, q in out:
        L += q; V += q*e
        b = max(0, min(q, used + q - CAP)); used += q
        Lb += b; Vb += b*e; n3 += b*max(0, e-3); n4 += b*max(0, e-4)
    return L, V, Lb, Vb, n3, n4
def depth(S, side, k):  # exit depth: for long excess, bids >= S-k
    if side == 'B':
        return sum(lv['B'][p] for p in px['B'] if p >= S - k)
    return sum(lv['A'][p] for p in px['A'] if p <= S + k)
checks = []
for t in ev:
    checks += [(t+0.005, t, 5), (t+0.010, t, 10), (t+0.050, t, 50)]
checks.sort(); ci = 0
res = {}
with open(csvp) as fh:
    next(fh)
    for line in fh:
        a = line.rstrip('\n').split(','); t = float(a[0])
        while ci < len(checks) and checks[ci][0] <= t:
            tc, te, x = checks[ci]; ci += 1
            S = S_at(tc)
            r = res.setdefault(te, {})
            if x < 50:
                for sd in ('A', 'B'):
                    m = measure(S, sd)
                    if m[0]: r[(x, sd)] = m
            else:
                r['S50'] = S
                r['dep'] = {sd: (depth(S, sd, 2), depth(S, sd, 3), depth(S, sd, 4)) for sd in ('A','B')}
        op, o, side = a[1], a[2], a[3]
        if op == 'Cancel':
            if o in oid:
                s, p, q = oid.pop(o); rem(s, p, q)
            continue
        q = int(a[4]); p = int(a[5])
        if a[6] == 'G':
            oid[o] = [side, p, q]; add(side, p, q); continue
        # taker
        opp = 'A' if side == 'B' else 'B'
        while q > 0 and px[opp]:
            bp = px[opp][0] if opp == 'A' else px[opp][-1]
            if (opp == 'A' and bp > p) or (opp == 'B' and bp < p): break
            take = min(q, lv[opp][bp]); q -= take
            # consume orders at level (oldest first)
            left = take
            for k2 in [k3 for k3, v in oid.items() if v[0] == opp and v[1] == bp] if take else []:
                v = oid[k2]; z = min(left, v[2]); v[2] -= z; left -= z
                if v[2] == 0: del oid[k2]
                if left == 0: break
            rem(opp, bp, take)
import pickle
pickle.dump(res, open(ROOT/f'marek/research/round2/hyp/bigjump/scan_{tape}.pkl', 'wb'))
# summary
for x in (5, 10):
    rows = []
    for te, r in res.items():
        for sd in ('A','B'):
            if (x, sd) in r:
                rows.append((te, sd) + r[(x, sd)])
    Vb = np.array([z[5] for z in rows]); n3 = np.array([z[6] for z in rows]); n4 = np.array([z[7] for z in rows]); Lb=np.array([z[4] for z in rows])
    print(f"{tape} +{x}ms events_with_stale={len(rows)} beyond-cap: n(Vb>0)={int((Vb>0).sum())} sumVb={Vb.sum():.0f} "
          f"n(Vb>=3k)={int((Vb>=3000).sum())} sum={Vb[Vb>=3000].sum():.0f} n(Vb>=10k)={int((Vb>=10000).sum())} sum={Vb[Vb>=10000].sum():.0f} "
          f"net3={n3.sum():.0f} net4={n4.sum():.0f} n(net3>=1k)={int((n3>=1000).sum())} sumLb={Lb.sum()}")
    big = sorted(rows, key=lambda z: -z[5])[:12]
    for z in big:
        r = res[z[0]]
        print(f"   t={z[0]:9.3f} side={z[1]} L={z[2]} V={z[3]:.0f} Lb={z[4]} Vb={z[5]:.0f} net3={z[6]:.0f} dep50={r.get('dep',{}).get('B' if z[1]=='A' else 'A')}")
