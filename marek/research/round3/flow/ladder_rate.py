"""Ladder-start process and non-ladder maker rate by regime; dispersion; rate vs spread."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); ins = r[r.op == 0].reset_index(drop=True)
L = pd.read_pickle(C / 'ladders.pkl'); mem = np.load(C / 'ladder_member.npy')
reg = (np.load(C / 'hmm_joint_2.npy')[:, 1] > .5); mask = np.load(C / 'bins1s.npz')['mask']
clean = L.steps.apply(lambda x: all(s == 1 for s in x)).to_numpy()
mk = ~ins.ioc.to_numpy(); t = ins.t.to_numpy()
nonlad = mk & (mem[:, 0] < 0)
c_nl = np.bincount(t[nonlad].astype(int), minlength=6000)[:6000]
c_ld = np.bincount(L.t0.to_numpy().astype(int), minlength=6000)[:6000]
c_li = np.bincount(t[mem[:, 0] >= 0].astype(int), minlength=6000)[:6000]
inner = np.array([reg[max(0, i - 3):i + 4].all() or (~reg[max(0, i - 3):i + 4]).all() for i in range(6000)]) & mask
for nm, s in [('calm', ~reg & inner), ('vol', reg & inner)]:
    print(f'{nm}: nonladder makers {c_nl[s].mean():.2f}/s DI {c_nl[s].var() / c_nl[s].mean():.2f}; ladders {c_ld[s].mean():.3f}/s DI {c_ld[s].var() / c_ld[s].mean():.2f}; ladder inserts {c_li[s].mean():.2f}/s')
    for w in [10, 30]:
        # within long segments, 10 s windows fully inside
        cw = c_ld[:6000 // w * w].reshape(-1, w).sum(1); sw = s[:6000 // w * w].reshape(-1, w).all(1)
        print(f'   ladders per {w}s window: mean {cw[sw].mean():.2f} DI {cw[sw].var() / cw[sw].mean():.2f} n {sw.sum()}')
# per-segment ladder rate in calm segments (are there calm sub-episodes with ladders?)
seg = pd.read_csv(D / 'regimes_1s.csv')
seg['lad_rate'] = [c_ld[a:b].mean() for a, b in zip(seg.start, seg.end)]
seg['tk_rate'] = [np.load(C / 'bins1s.npz')['tk'][a:b].mean() for a, b in zip(seg.start, seg.end)]
pd.set_option('display.width', 200); print(seg[seg.dur >= 20].round(2).to_string())
