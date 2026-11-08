import numpy as np, pandas as pd
from prep import load
o = load(); m = o[~o.ioc.astype(bool)].copy()
g = np.load('cache/regime_sz.npz')['post']
m['vol'] = g[(m.t_us//1_000_000).to_numpy(), 1] > .5
m['life'] = (m.cancel_us - m.t_us)/1e3
m['rel'] = np.where(m.sg>0, m.price - m.bid, m.ask - m.price)  # >0 improves touch, 0 at touch, <0 behind
w = m[(m.t_us > 30e6) & (m.t_us < 36e6)]
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
print(w[w.volume.between(90, 320)][['row','t_us','sg','price','volume','bid','ask','rel','life','filled']].to_string())
