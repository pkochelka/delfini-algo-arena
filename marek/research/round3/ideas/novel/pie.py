"""Where r3_best's trades sit vs the tape's whole taker flow, per regime.
usage: uv run --with pyarrow python pie.py NULL_LOGDIR RUN_LOGDIR
edge = signed (S_hat - price) per lot from the side that is us (or from the maker side for the tape's taker flow);
h-edges use S_hat at t+h (markout).
"""
import sys, bisect, csv, numpy as np, pandas as pd
nd, rd = sys.argv[1], sys.argv[2]
cols = ['ts', 'price', 'tside', 'q', 'tid', 'mid', 'town', 'mown']
S = pd.read_parquet('marek/research/round3/s_dynamics/s_path.parquet', columns=['t_us', 's_hat'])
st = S.t_us.values; sh = S.s_hat.values
def s_at(t):
    return sh[np.clip(np.searchsorted(st, t, side='right') - 1, 0, len(sh) - 1)]
R3 = 'marek/research/round3/'
on = [float(r['onset']) for r in csv.DictReader(open(R3 + 'flow/shock_events.csv'))]
reg = [(float(r['start']), float(r['end']), r['regime']) for r in csv.DictReader(open(R3 + 'flow/regimes_1s.csv'))]
ra = np.array([a for a, b, g in reg]); rg = [g for a, b, g in reg]
def lab(ts):
    t = ts / 1e6; out = []
    for x in t:
        i = bisect.bisect_right(on, x + 1.0) - 1
        if i >= 0 and on[i] - 1 <= x < on[i] + 20: out.append('shock'); continue
        j = bisect.bisect_right(ra, x) - 1; out.append(rg[j] if j >= 0 else 'calm')
    return np.array(out)
def summarize(df, who):
    df = df.copy(); df['g'] = lab(df.ts.values)
    for h in (0, 1_000_000, 10_000_000, 30_000_000):
        df[f'e{h//1000000}'] = df.sign * (s_at(df.ts.values + h) - df.price) * df.q
    g = df.groupby('g').agg(n=('q', 'size'), vol=('q', 'sum'), e0=('e0', 'sum'), e1=('e1', 'sum'), e10=('e10', 'sum'), e30=('e30', 'sum'))
    for c in ('e0', 'e1', 'e10', 'e30'): g[c + '/lot'] = (g[c] / g.vol).round(3)
    print(who); print(g.to_string(float_format=lambda x: f'{x:.2f}' if abs(x) < 100 else f'{x:.0f}')); print()
nt = pd.read_csv(f'{nd}/trades.csv', names=cols, header=0)
rt = pd.read_csv(f'{rd}/trades.csv', names=cols, header=0)
tk = nt[nt.town == 'market'].copy()   # tape taker flow, as seen by the maker side (incl. crossing G inserts)
tk['sign'] = np.where(tk.tside == 'A', 1, -1)  # maker buys when taker sells
summarize(tk, 'TAPE taker flow, maker-side edge (null replay)')
u = rt[(rt.town == 'user') | (rt.mown == 'user')].copy()
u['sign'] = np.where(((u.town == 'user') & (u.tside == 'B')) | ((u.mown == 'user') & (u.tside == 'A')), 1, -1)
summarize(u[u.mown == 'user'], 'US passive')
summarize(u[u.town == 'user'], 'US IOC/taking')
