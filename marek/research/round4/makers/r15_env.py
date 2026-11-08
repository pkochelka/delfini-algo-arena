"""Market environment inside 15-lot agent windows vs other calm time: taker rate and side, maker rate, ladder rate."""
import numpy as np, pandas as pd
o = pd.read_pickle('cache/orders.pkl')
ch = pd.read_pickle('cache/chains.pkl')
S = pd.read_parquet('../s_dynamics/s_path.parquet')
tw = pd.read_pickle('cache/tw.pkl')
a = pd.read_pickle('cache/r15_v2.pkl')
g = a.groupby('chain').agg(t0=('t_us', 'min'), t1=('end_t_us', 'max'), sg=('sg', 'first'))
t = o.t_us.to_numpy()
inag = np.zeros(len(o), bool); agsg = np.zeros(len(o))
for r in g.itertuples():
    m = (t >= r.t0) & (t <= r.t1); inag |= m; agsg[m] = r.sg
calm = ~S.vol.to_numpy() & ~S.shock.to_numpy()
twc = np.zeros(len(o), bool); twc[tw.index[tw.tw_L >= 2]] = True
isag = np.zeros(len(o), bool); isag[a.index] = True
lad = (ch.L >= 2).to_numpy()
T_ag = sum((r.t1 - r.t0) for r in g.itertuples()) / 1e6  # overlaps ignored
# approximate calm time outside agent: count by 0.1 s grid
grid = np.arange(0, 6000e6, 1e5).astype(np.int64)
gi = np.clip(np.searchsorted(t, grid, 'right') - 1, 0, len(t) - 1)
gag = inag[gi] & (np.abs(t[gi + 1 if True else gi] - grid) < 1e9)
gcalm = calm[gi]
Tag = (gcalm & inag[gi]).sum() * 0.1; Tcal = (gcalm & ~inag[gi]).sum() * 0.1
print(f'calm time in agent windows {Tag:.0f} s, outside {Tcal:.0f} s')
for lab, m, T in [('agent windows', calm & inag, Tag), ('other calm', calm & ~inag, Tcal)]:
    tk = o.ioc.to_numpy() & m & ~twc
    mk = ~o.ioc.to_numpy() & m & ~lad & ~isag
    print(f'{lab}: regular takers {tk.sum() / T:.2f}/s, TWAP children {(twc & m).sum() / T:.2f}/s, base makers {mk.sum() / T:.1f}/s, '
          f'ladder {(lad & m).sum() / T:.2f}/s, agent orders {(isag & m).sum() / T:.2f}/s')
m = calm & inag & o.ioc.to_numpy() & ~twc
s_rel = o.sg.to_numpy()[m] * agsg[m]
print('regular takers in agent windows: share on agent side (same side as agent order)', (s_rel > 0).mean().round(3), 'n', m.sum())
fv = o.filled.to_numpy()[m]; vv = o.volume.to_numpy()[m]
for side in [1, -1]:
    k = s_rel == side
    print(' side', side, 'n', k.sum(), 'median vol', np.median(vv[k]), 'fill share', (fv[k] > 0).mean().round(3))
