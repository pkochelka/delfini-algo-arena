"""Ceiling angle, step 4: passive ceiling. For every market taker, what a resting quote of ours at
P = max(ceil(S + x), bid + 1) (sells; buys mirrored) would have filled, ahead of market orders at P or behind them,
and its edge vs exact S after the taker's kick (and at +1 s / +10 s). Optional position clip at +-100 (no unwind)."""
import sys, pathlib, math
import numpy as np, pandas as pd
H = pathlib.Path(__file__).parent
sys.path.insert(0, str(H.parents[2] / 'strategy'))
import eval3
ROOT = H.parents[5]
on, reg = eval3.load_ctx('data/round3.csv')
z = np.load(ROOT / 'data/.cache/round3_S_oracle.npz'); TI, SA = z['t_us'], z['S']
tk = pd.read_parquet(H / '_cache/ceil_takers.parquet')
tk['reg'] = [eval3.label(t * 1e-6, on, reg) for t in tk.t_us]
def S_at(t):
    j = np.searchsorted(TI, np.asarray(t), side='right') - 1
    return SA[np.maximum(j, 0)]
tk['S1'] = S_at(tk.t_us.values + 1e6); tk['S10'] = S_at(tk.t_us.values + 10e6)
fl = [[(int(a), int(b)) for a, b in (x.split(':') for x in f.split(';'))] if f else [] for f in tk.fills]
rows = []
for x in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
    for ahead in [True, False]:
        res = {r: [0, 0.0, 0.0, 0.0] for r in ['calm', 'volatile', 'shock']}
        pos = 0; resc = {r: [0, 0.0] for r in res}
        for i, (side, lim, q, Sb, Sa, bb, ba, rg, s1, s10) in enumerate(zip(tk.side, tk.lim, tk.q, tk.S_before, tk.S_after,
                                                                             tk.bb, tk.ba, tk.reg, tk.S1, tk.S10)):
            if side == 'B':  # taker buys: our ask
                if bb is None or np.isnan(bb):
                    continue
                P = max(math.ceil(Sb + x), int(bb) + 1)
                if lim < P:
                    continue
                below = sum(b for a, b in fl[i] if a < P) + (0 if ahead else sum(b for a, b in fl[i] if a == P))
                f = max(0, min(100, q - below))
                e, e1, e10 = P - Sa, P - s1, P - s10
                fc = max(0, min(f, 100 + pos)); pos -= fc
            else:
                if ba is None or np.isnan(ba):
                    continue
                P = min(math.floor(Sb - x), int(ba) - 1)
                if lim > P:
                    continue
                below = sum(b for a, b in fl[i] if a > P) + (0 if ahead else sum(b for a, b in fl[i] if a == P))
                f = max(0, min(100, q - below))
                e, e1, e10 = Sa - P, s1 - P, s10 - P
                fc = max(0, min(f, 100 - pos)); pos += fc
            r = res[rg]; r[0] += f; r[1] += f * e; r[2] += f * e1; r[3] += f * e10
            resc[rg][0] += fc; resc[rg][1] += fc * e
        for rg in res:
            rows.append((x, 'ahead' if ahead else 'behind', rg, res[rg][0], round(res[rg][1]), round(res[rg][2]),
                         round(res[rg][3]), resc[rg][0], round(resc[rg][1])))
df = pd.DataFrame(rows, columns=['x', 'queue', 'reg', 'lots', 'edge_kick', 'edge_1s', 'edge_10s', 'lots_clip', 'edge_clip'])
pd.set_option('display.width', 200)
print(df.to_string())
