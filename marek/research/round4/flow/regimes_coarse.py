"""Coarse (1 s) regime segments from joint taker+maker Poisson HMM -> regimes_1s.csv."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
p = np.load(C / 'hmm_joint_2.npy')[:, 1]
x = (p > .5).astype(int); ch = np.flatnonzero(np.diff(x)) + 1
st = np.r_[0, ch]; en = np.r_[ch, len(x)]
df = pd.DataFrame({'start': st, 'end': en, 'regime': np.where(x[st] == 1, 'volatile', 'calm')})
df['dur'] = df.end - df.start
df.to_csv(D / 'regimes_1s.csv', index=False)
print(df.groupby('regime').dur.describe())
