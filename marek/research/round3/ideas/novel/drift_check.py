"""Check the post-taker drift in volatile with an S-free(ish) probe: later takers' limit prices (round(S +- 2.55)).
For each volatile taker k (the 'event'), implied S of later takers j: lim_j -+ 2.55. Signed by event side,
relative to the implied S of takers in the 3 s before the event. Medians to kill the aggressive tail.
usage: uv run --with pyarrow python drift_check.py
"""
import numpy as np, pandas as pd
S = pd.read_parquet('marek/research/round3/s_dynamics/s_path.parquet', columns=['vol', 'shock', 's_hat'])
tape = pd.read_csv('data/round3.csv', usecols=['Time', 'Operation', 'Side', 'Volume', 'Price', 'Lifespan'])
ins = tape[tape.Operation == 'Insert'].reset_index(drop=True)
m = (ins.Lifespan == 'F').values & S.vol.values & ~S.shock.values
t = ins.Time.values[m]; sd = np.where(ins.Side.values[m] == 'B', 1, -1); q = ins.Volume.values[m]
imp = ins.Price.values[m] - sd * 2.55          # implied S
res = imp - S.s_hat.values[m]
ok = np.abs(res) < 1.5                         # drop tail takers
t, sd, q, imp = t[ok], sd[ok], q[ok], imp[ok]
wins = [(-3, 0), (0.005, 0.2), (0.2, 0.5), (0.5, 1), (1, 2), (2, 5), (5, 10), (10, 30)]
for lo_q, hi_q in ((200, 600), (600, 1e9)):
    ev = np.where((q >= lo_q) & (q < hi_q))[0]
    acc = {w: [] for w in wins}
    for k in ev:
        a = np.searchsorted(t, t[k] - 3); b = np.searchsorted(t, t[k] + 30)
        pre = imp[a:k]
        if len(pre) < 3: continue
        base = np.median(pre)
        for w in wins[1:]:
            j0 = np.searchsorted(t, t[k] + w[0]); j1 = np.searchsorted(t, t[k] + w[1])
            if j1 > j0: acc[w].append(sd[k] * (np.median(imp[j0:j1]) - base))
    print(f'q[{lo_q},{hi_q}) n={len(ev)}', ' '.join(f'{w[0]}-{w[1]}s:{np.mean(acc[w]):.2f}±{np.std(acc[w])/np.sqrt(max(len(acc[w]),1)):.2f}' for w in wins[1:]))
