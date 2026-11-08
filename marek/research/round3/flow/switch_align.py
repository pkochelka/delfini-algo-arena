"""Event-study around S-based regime switch times (s_dynamics vol_segments): ladder start rate, ladder rung size
(median log), taker rate, non-ladder maker rate, in 0.5 s bins from -10 to +10 s. Tests whether ladders/sizes switch on
the same clock as the price regime (one chain) or lag/lead (separate coupled process)."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
V = pd.read_csv(D.parent / 's_dynamics/vol_segments.csv'); L = pd.read_pickle(C / 'ladders.pkl')
r = pd.read_pickle(C / 'rows.pkl'); ins = r[r.op == 0].reset_index(drop=True); mem = np.load(C / 'ladder_member.npy')
ti = ins.t.to_numpy(); ioc = ins.ioc.to_numpy(); v = ins.vol.to_numpy()
lt = L.t0.to_numpy(); lsz = np.array([np.median(x) for x in L.sizes])
tk = ti[ioc]; mk = ti[~ioc & (mem[:, 0] < 0)]
edges = np.arange(-10, 10.01, 1.0)
for nm, sw in [('calm->vol', V.start.to_numpy()), ('vol->calm', V.end.to_numpy())]:
    sw = sw[(sw > 15) & (sw < 5985)]
    print(f'==== {nm} (n {len(sw)})')
    print('  bin     ladders/s  ladder log-size  takers/s  base makers/s')
    for a, b in zip(edges[:-1], edges[1:]):
        nl = 0; ls = []; nt = 0; nm_ = 0
        for x in sw:
            m = (lt >= x + a) & (lt < x + b); nl += m.sum(); ls += list(np.log(lsz[m]))
            nt += ((tk >= x + a) & (tk < x + b)).sum(); nm_ += ((mk >= x + a) & (mk < x + b)).sum()
        k = len(sw) * (b - a)
        print(f'  {a:+5.1f} {nl / k:8.2f} {np.mean(ls) if ls else np.nan:10.2f} {nt / k:9.2f} {nm_ / k:9.1f}')
