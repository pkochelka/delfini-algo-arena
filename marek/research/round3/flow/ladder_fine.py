import numpy as np, pandas as pd
from pathlib import Path
C = Path(__file__).parent / 'cache'
a = np.load(C / 'ladder_next.npy'); dt = a[:, 1] * 1e3
e = np.arange(0.5, 3.6, 0.1); h = np.histogram(dt, e)[0]
for x, y in zip(e[:-1], h): print(f'{x:4.1f} {y:6d}', '#' * int(y / 300))
