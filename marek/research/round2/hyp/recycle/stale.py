"""Stale-lot volume per S jump on round2 (replay of the tape book + exact S path).
uv run --with pyarrow python marek/research/round2/hyp/recycle/stale.py"""
import csv, numpy as np, pandas as pd
from collections import defaultdict

S = pd.read_parquet('marek/research/round2/s_path.parquet' if False else 'marek/research/round2/s_dynamics/s_path.parquet')
s_hat = S.s_hat.values
orders = {}            # id -> [side, price, qty, seq]
lv = {'B': defaultdict(dict), 'A': defaultdict(dict)}   # price -> {id: qty} (insertion ordered)
ev = []                # active trackers
done = []
ins = -1
seq = 0
def best(side):
    d = lv[side]
    return (max(d) if side == 'B' else min(d)) if d else None
def rm(oid, q=None):
    o = orders[oid]
    if q is None or q >= o[2]:
        del lv[o[0]][o[1]][oid]
        if not lv[o[0]][o[1]]:
            del lv[o[0]][o[1]]
        del orders[oid]
    else:
        o[2] -= q; lv[o[0]][o[1]][oid] = o[2]
with open('data/round2.csv') as fh:
    r = csv.reader(fh); next(r)
    for row in r:
        t = float(row[0]); op = row[1]; oid = row[2]
        if op == 'Cancel':
            if oid in orders:
                o = orders[oid]
                for e in ev:
                    if oid in e['stale']:
                        e['canc'] += o[2]; e['stale'].discard(oid)
                rm(oid)
            continue
        ins += 1
        side, q, px, life = row[3], int(row[4]), int(row[5]), row[6]
        s1 = s_hat[ins]; s0 = s_hat[ins-1] if ins else s1
        if life == 'F':
            opp = 'A' if side == 'B' else 'B'
            while q > 0:
                b = best(opp)
                if b is None or (side == 'B' and b > px) or (side == 'A' and b < px):
                    break
                for mid, mq in list(lv[opp][b].items()):
                    f = min(q, mq)
                    for e in ev:
                        if mid in e['stale']:
                            e['taken'] += f
                            e['tk_edge'] += f * abs(e['S'] - b)
                            if orders[mid][2] == f: e['stale'].discard(mid)
                    rm(mid, f); q -= f
                    if q == 0: break
        else:
            orders[oid] = [side, px, q]; lv[side][px][oid] = q
        # jump detection
        if abs(s1 - s0) > 5:
            up = s1 > s0
            st = set(); V = 0; E = 0.0; edges = []
            for i, o in orders.items():
                if up and o[0] == 'A' and o[1] < s1 - 1:
                    st.add(i); V += o[2]; edges += [s1 - o[1]] * o[2]
                if not up and o[0] == 'B' and o[1] > s1 + 1:
                    st.add(i); V += o[2]; edges += [o[1] - s1] * o[2]
            edges.sort(reverse=True)
            ev.append(dict(t=t, S=s1, J=s1 - s0, up=up, V=V, E=sum(edges), E100=sum(edges[:100]), E200=sum(edges[:200]),
                           stale=st, taken=0, tk_edge=0.0, canc=0, clear_t=None, shock=bool(S.shock.values[ins])))
        for e in list(ev):
            if e['clear_t'] is None and not e['stale']:
                e['clear_t'] = t - e['t']
            if t - e['t'] > 2.0:
                ev.remove(e); e['left'] = sum(orders[i][2] for i in e['stale'] if i in orders); del e['stale']; done.append(e)
d = pd.DataFrame(done)
pd.set_option('display.width', 200)
d.to_csv('marek/research/round2/hyp/recycle/stale_events.csv', index=False)
print(len(d), 'jumps')
print(d[['J','V','E','E100','E200','taken','canc','left','clear_t']].describe().round(1))
print('sum E', d.E.sum(), 'E100', d.E100.sum(), 'E200', d.E200.sum())
print('by shock', d.groupby('shock')[['V','E','E100','E200']].sum())
sg = np.sign(d.J.values); print('sign autocorr', np.corrcoef(sg[:-1], sg[1:])[0,1], 'n', len(sg))
dt = np.diff(d.t.values); print('gap between jumps: q10/50/90', np.percentile(dt, [10,50,90]))
