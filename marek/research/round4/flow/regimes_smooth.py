"""Smooth the 1 s joint HMM posterior (5 s median filter on p>0.5) -> regimes_1s.csv (start,end,regime,dur)."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.ndimage import median_filter
D = Path(__file__).parent; C = D / 'cache'
p = np.load(C / 'hmm_joint_2.npy')[:, 1]
x = median_filter((p > .5).astype(int), size=7, mode='nearest')
ch = np.flatnonzero(np.diff(x)) + 1
st = np.r_[0, ch]; en = np.r_[ch, len(x)]
df = pd.DataFrame({'start': st, 'end': en, 'regime': np.where(x[st] == 1, 'volatile', 'calm')})
df['dur'] = df.end - df.start
df.to_csv(D / 'regimes_1s.csv', index=False)
np.save(C / 'vol1s.npy', x)
print(df.groupby('regime').dur.describe()); print('vol occ', x.mean())
print(df[df.regime == 'volatile'][['start', 'end']].to_numpy().tolist())
