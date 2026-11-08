"""Is the post-jump taker burst directional? Share of takers on the jump side, by time after the jump, pooled over shocks.
Same test on round 2 (jump sign from round-2 flow shock_event_moves.csv / mid)."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
D = Path(__file__).parent; C = D / 'cache'
X = pd.read_csv(D / 'shock_events.csv'); r = pd.read_pickle(C / 'rows.pkl'); k = r[(r.op == 0) & r.ioc]
tk = k.t.to_numpy(); sg = np.where(k.side.to_numpy() == 0, 1, -1)
win = [(-3, 0), (0, 0.25), (0.25, 0.5), (0.5, 1), (1, 2), (2, 3.5), (3.5, 10)]
def table(tj, J, tk, sg, label):
    print(label)
    for a, b in win:
        n = 0; same = 0
        for t0, j in zip(tj, J):
            m = (tk > t0 + a) & (tk <= t0 + b); n += m.sum(); same += (sg[m] == np.sign(j)).sum()
        print(f'  [{a:+5.2f},{b:+5.2f}) n {n:5d} rate/shock/s {n / len(tj) / (b - a):6.1f} share in jump dir {same / n:.3f} ± {0.5 / np.sqrt(n):.3f}  binom p {stats.binomtest(same, n).pvalue:.4f}')
big = np.abs(X.jump) > 10
table(X.t_jump.to_numpy(), X.jump.to_numpy(), tk, sg, 'round 3, all shocks')
table(X.t_jump[big].to_numpy(), X.jump[big].to_numpy(), tk, sg, 'round 3, |J|>10')
# round 2
R2 = Path(D.parents[1] / 'round2/flow'); e2 = pd.read_csv(R2 / 'shock_events.csv')
r2 = pd.read_pickle(R2 / 'cache/rows.pkl'); k2 = r2[(r2.op == 0) & r2.ioc]
S2 = pd.read_parquet(D.parents[1] / 'round2/s_dynamics/s_path.parquet'); i2 = r2[r2.op == 0]
t2 = i2.t.to_numpy(); s2 = S2.s_hat.to_numpy()
def S2at(x): return s2[np.clip(np.searchsorted(t2, x, 'right') - 1, 0, None)]
J2 = np.array([S2at(o + 3.4) - S2at(o + 2.8) for o in e2.onset])
table((e2.onset + 3.0).to_numpy(), J2, k2.t.to_numpy(), np.where(k2.side.to_numpy() == 0, 1, -1), 'round 2, all shocks')
