"""How does the taker-high regime relate to price behaviour? Per-segment drift/vol; 10 s bins taker rate vs realised vol."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
z = np.load(C / 'bins1s.npz'); S = pd.read_csv(D / 'taker_regimes.csv'); mask = z['mask']
dm = np.nan_to_num(z['dm']); tk = z['tk']
for _, s in S.iterrows():
    a, b = int(s.start), int(s.end); x = dm[a:b]; m = mask[a:b]
    print(f"{'HI' if s.high else 'lo'} {a:5d}-{b:5d} dur {b-a:4d} net {x.sum():+7.1f} sd1s {x[m].std():6.2f} taker/s {tk[a:b][m].mean():6.2f}")
# 10 s bins
k = 10; n = len(dm) // k
M = mask[:n * k].reshape(n, k).all(1)
T10 = tk[:n * k].reshape(n, k).sum(1)[M]; V10 = np.sqrt((dm[:n * k].reshape(n, k) ** 2).sum(1))[M]
q = pd.qcut(V10, 10)
print(pd.DataFrame({'rv10': V10, 'tk10': T10}).groupby(q, observed=True).agg(['mean', 'count']).round(2).to_string())
