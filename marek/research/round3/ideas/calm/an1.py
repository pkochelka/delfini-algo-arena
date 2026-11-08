"""Calm-regime decomposition of a logged run (strategies/marek/round3/ideas/calm_log.py).
uv run --with pyarrow python an1.py LOG.pkl"""
import sys, pickle, csv, pathlib, numpy as np, pandas as pd
R3 = pathlib.Path(__file__).resolve().parents[2]
L = pickle.load(open(sys.argv[1], 'rb'))
sp = pd.read_parquet(R3 / 's_dynamics/s_path.parquet', columns=['t_us', 's_hat'])
ST, SS = sp.t_us.values, sp.s_hat.values
def S(t):  # S at time t_us (last insert at or before)
    i = np.searchsorted(ST, t, side='right') - 1
    return SS[np.clip(i, 0, len(SS) - 1)]
vs = [(float(r['start']), float(r['end'])) for r in csv.DictReader(open(R3 / 's_dynamics/vol_segments.csv'))]
on = [float(r['onset']) for r in csv.DictReader(open(R3 / 'flow/shock_events.csv'))]
def lab(t):  # t seconds
    t = np.asarray(t, float); g = np.zeros(t.shape, int)  # 0 calm 1 vol 2 shock
    for a, b in vs: g[(t >= a) & (t < b)] = 1
    for o in on: g[(t >= o - 1) & (t < o + 20)] = 2
    return g
sent = pd.DataFrame(L['sent'], columns='t oid side px q ioc F sd turb bb ba pos'.split())
fill = pd.DataFrame(L['fill'], columns='t oid side px q F sd pos_after'.split())
seen = pd.DataFrame([(k, *v) for k, v in L['seen'].items()], columns='oid t_seen ahead lvl'.split())
fill = fill.merge(sent[['oid', 't', 'ioc', 'F', 'px', 'pos', 'bb', 'ba']].rename(columns={'t': 't_sent', 'F': 'F_sent', 'px': 'px_sent', 'pos': 'pos_sent'}), on='oid', how='left')
fill = fill.merge(seen, on='oid', how='left')
fill['g'] = lab(fill.t * 1e-6)
fill['S0'] = S(fill.t.values)
for h in (1, 5, 30, 60):
    fill[f'm{h}'] = fill.side * (S(fill.t.values + h * 1_000_000) - fill.px)
fill['e0'] = fill.side * (fill.S0 - fill.px)
fill['eF'] = fill.side * (fill.F - fill.px)
pos_before = fill.pos_after - fill.side * fill.q
fill['red'] = np.sign(pos_before) == -fill.side
fill.to_pickle(sys.argv[1] + '.fills.pkl')
names = ['calm', 'vol', 'shock']
print('=== fills per regime: lots, edge vs exact S at fill, markout vs S after h (per lot) ===')
for g in range(3):
    for io in (False, True):
        d = fill[(fill.g == g) & (fill.ioc == io)]
        if not len(d): continue
        w = d.q
        print(f"{names[g]:6s} {'IOC' if io else 'LIM'} n={len(d):6d} lots={w.sum():8d} "
              + ' '.join(f"{c}={np.average(d[c], weights=w):+.3f}" for c in ['eF', 'e0', 'm1', 'm5', 'm30', 'm60'])
              + f"  e0*lots={np.sum(d.e0 * w):9.0f} m60*lots={np.sum(d.m60 * w):9.0f}")
# calm passive: by reducing / not, by queue-ahead
c = fill[(fill.g == 0) & (~fill.ioc)]
print('\n=== calm passive fills by reduce vs increase ===')
for r in (True, False):
    d = c[c.red == r]; w = d.q
    print(f"reduce={r} lots={w.sum()} " + ' '.join(f"{k}={np.average(d[k], weights=w):+.3f}" for k in ['e0', 'm1', 'm5', 'm30', 'm60']))
print('\n=== calm passive fills by market qty ahead at first sight ===')
c = c.assign(ab=pd.cut(c.ahead.fillna(-1), [-2, -0.5, 0.5, 100, 300, 1000, 3000, 1e9]))
for k, d in c.groupby('ab', observed=True):
    w = d.q
    print(f"{str(k):18s} lots={w.sum():7d} " + ' '.join(f"{x}={np.average(d[x], weights=w):+.3f}" for x in ['e0', 'm1', 'm5', 'm30', 'm60']))
print('\n=== calm passive fills by fill price vs touch at send (0 = joined touch) ===')
c = c.assign(rel=np.where(c.side > 0, c.bb - c.px_sent, c.px_sent - c.ba))
for k, d in c.groupby('rel'):
    w = d.q
    if w.sum() < 200: continue
    print(f"rel={k:+d} lots={w.sum():7d} " + ' '.join(f"{x}={np.average(d[x], weights=w):+.3f}" for x in ['e0', 'm1', 'm5', 'm30', 'm60']))
# state samples
sm = pd.DataFrame(L['samp'], columns='t pos turb F sd bb ba bq aq obp obq oba oap oaq oaa'.split())
sm['g'] = lab(sm.t * 1e-6)
sm['S'] = S(sm.t.values)
cs = sm[sm.g == 0]
print(f"\n=== calm samples: {len(cs)} x 50 ms = {len(cs) * 0.05:.0f} s ===")
ap = cs.pos.abs()
for th in (0, 25, 50, 75, 90, 100):
    print(f"|pos|>={th}: {np.mean(ap >= th):.3f}", end='  ')
print()
print('turb flag in calm:', round(cs.turb.mean(), 3), ' F-S sd', round((cs.F - cs.S).std(), 3), 'mean |F-S|', round((cs.F - cs.S).abs().mean(), 3))
hb = cs.obq > 0; ha = cs.oaq > 0
print('have bid', round(hb.mean(), 3), 'have ask', round(ha.mean(), 3), 'bid at touch', round((cs.obp == cs.bb).mean(), 3),
      'ask at touch', round((cs.oap == cs.ba).mean(), 3))
print('our bid qty mean (when present)', round(cs.obq[hb].mean(), 1), 'ask', round(cs.oaq[ha].mean(), 1))
print('spread dist calm', (cs.ba - cs.bb).value_counts(normalize=True).sort_index().round(3).to_dict())
# PnL decomposition per regime: sum over 50ms samples of pos * dS (inventory) vs fill edge
sm['dS'] = sm.S.shift(-1) - sm.S
inv = sm.groupby('g').apply(lambda d: float((d.pos * d.dS).sum()))
ed = fill.assign(v=fill.e0 * fill.q).groupby('g').v.sum()
print('\n=== PnL vs exact S: edge at fill + inventory (pos*dS on 50 ms grid) ===')
for g in range(3):
    print(f"{names[g]:6s} edge {ed.get(g, 0):9.0f} inventory {inv.get(g, 0):9.0f} total {ed.get(g, 0) + inv.get(g, 0):9.0f}")
