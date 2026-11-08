"""How much of the Y drift does r3_best already hold? corr(pos, -Y), model drift captured = sum pos * (-Y/T) dt."""
import sys, pickle, pathlib, numpy as np, pandas as pd
R3 = pathlib.Path(__file__).resolve().parents[2]
L = pickle.load(open(sys.argv[1], 'rb'))
sm = pd.DataFrame(L['samp'], columns='t pos turb F sd bb ba bq aq obp obq oba oap oaq oaa'.split())
G = np.load(R3 / 'ideas/calm/Y_grid.npy')
i = np.clip(np.searchsorted(G[:, 0], sm.t.values * 1e-6) - 1, 0, len(G) - 1)
sm['Y'] = G[i, 1]; sm['g'] = G[i, 2]
c = sm[sm.g == 0]
print('corr(pos, -Y) calm', np.corrcoef(c.pos, -c.Y)[0, 1])
cap = np.sum(c.pos * (-c.Y / 16.7) * 0.05)
ideal = np.sum(100 * np.abs(c.Y) / 16.7 * 0.05)
print(f'model drift captured by pos: {cap:.0f}   ideal +-100: {ideal:.0f}')
for b in [(-1e9, -1), (-1, -0.3), (-0.3, 0.3), (0.3, 1), (1, 1e9)]:
    m = (c.Y > b[0]) & (c.Y <= b[1])
    print(b, f'share {m.mean():.2f} mean pos {c.pos[m].mean():+.1f}')
