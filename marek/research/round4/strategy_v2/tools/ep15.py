"""PnL / carry / filter bias of a logged run (logf/j15log npz) inside the 15-lot agent's episodes (makers/r15_episodes.csv).
uv run --with pyarrow python ep15.py NPZ [NPZ2]"""
import sys, numpy as np, pandas as pd, pathlib
R4 = pathlib.Path(__file__).resolve().parents[2]
ep = pd.read_csv(R4 / 'makers/r15_episodes.csv')
sp = pd.read_parquet(R4 / 's_dynamics/s_path.parquet', columns=['t_us', 's_hat'])
st = sp.t_us.values * 1e-6; sh = sp.s_hat.values
def S_at(tt): return sh[np.clip(np.searchsorted(st, tt, side='right') - 1, 0, len(sh) - 1)]
for fn in sys.argv[1:]:
    z = np.load(fn); x = z['x']; cv = z['curve']
    t, F, sd, pv, turb, pos, bb, ba = x.T[:8]
    ct, cp = cv[:, 0], cv[:, 1]
    def P_at(tt): return cp[np.clip(np.searchsorted(ct, tt, side='right') - 1, 0, len(cp) - 1)]
    S = S_at(t); dS = np.r_[np.diff(S), 0]
    tot = dict(pnl=0, pnl_after=0, carry=0, carry_after=0)
    print(fn)
    print(' ep  sg   dur    dS  pnl_ep pnl+5s carry  pos0  pos_mid  pos_end  bias(F-S)*sg  turb')
    for i, r in ep.iterrows():
        a, b, g = r.t0 * 1e-6, r.t1 * 1e-6, r.sg
        m = (t >= a) & (t < b); m2 = (t >= b) & (t < b + 5)
        pe = P_at(b) - P_at(a); pa = P_at(b + 5) - P_at(b)
        ca = np.sum(pos[m] * dS[m]); ca2 = np.sum(pos[m2] * dS[m2])
        tot['pnl'] += pe; tot['pnl_after'] += pa; tot['carry'] += ca; tot['carry_after'] += ca2
        pm = pos[m]
        print(f'{i:3d} {g:+d} {b-a:5.1f} {S_at(b)-S_at(a):+6.1f} {pe:7.0f} {pa:6.0f} {ca:6.0f} {pm[0]*g:+5.0f} {np.mean(pm)*g:+7.0f} {pm[-1]*g:+6.0f} {np.mean((F[m]-S[m]))*g:+7.2f} {turb[m].mean():.2f}')
    print({k: round(v) for k, v in tot.items()})
