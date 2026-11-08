"""Burst takers: does side follow sign(S - mid_before) (trade toward S against the stale book)? By time after jump,
vs |S - mid|. Control: same stat outside bursts."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
X = pd.read_csv(D / 'shock_events.csv'); r = pd.read_pickle(C / 'rows.pkl'); ins = r[r.op == 0].reset_index(drop=True)
S = pd.read_parquet(D.parent / 's_dynamics/s_path.parquet')
ti = ins.t.to_numpy(); s = S.s_hat.to_numpy(); idx = np.flatnonzero(ins.ioc.to_numpy())
sg = np.where(ins.side.to_numpy()[idx] == 0, 1, -1); tk = ti[idx]
mid = ((ins.bid_before + ins.ask_before) / 2).to_numpy()[idx]
gap = s[idx - 1] - mid
dtj = np.full(len(tk), np.inf)
for tj in X.t_jump: m = (tk > tj) & (tk < tj + 10); dtj[m] = tk[m] - tj
for nm, sel in [('burst 0-0.25s', dtj < 0.25), ('burst 0.25-1s', (dtj >= 0.25) & (dtj < 1)), ('burst 1-3.5s', (dtj >= 1) & (dtj < 3.5)), ('outside', ~np.isfinite(dtj))]:
    g = gap[sel]; z = sg[sel]; ok = np.isfinite(g)
    t = pd.DataFrame({'agree': (np.sign(g[ok]) == z[ok]), 'absgap': np.abs(g[ok])})
    print(nm, 'n', ok.sum(), 'P(side = sign(S-mid))', t.agree.mean().round(3))
    print('   by |S-mid|:', t.groupby(pd.cut(t.absgap, [0, 0.5, 1, 2, 4, 8, 200])).agree.agg(['mean', 'size']).round(3).T.to_string().replace('\n', '\n   '))
