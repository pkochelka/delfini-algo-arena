import numpy as np, pandas as pd
o = pd.read_pickle('cache/chains.pkl')
q = pd.read_pickle('cache/quotes.pkl')
lad = o[(o.L >= 2) & (o.L <= 6) & ~o.ioc.astype(bool)].copy()
lad['end_t'] = lad.end_t_us.fillna(6e9)
starts = lad[(lad.k == 0) & (lad.t_us > 700e6)].head(12)
for _, r in starts.iterrows():
    t0 = r.t_us; tb = r.bid if r.sg > 0 else r.ask
    live = lad[(lad.t_us < t0) & (lad.end_t > t0) & (lad.sg == r.sg)]
    mine = lad[lad.chain == r.chain]
    print(f'--- burst t={t0/1e6:.4f} side {r.sg} touch {tb}  new prices', mine.price.tolist(), 'sizes', mine.volume.tolist())
    print('   live:', [(int(p), int(v), round((t0-t)/1e3)) for p, v, t in zip(live.price, live.volume, live.t_us)])
    qq = q[(q.t_us > t0 - 300e3) & (q.t_us <= t0)]
    col = 'bid' if r.sg > 0 else 'ask'
    ch = qq[qq[col].diff() != 0]
    print('   touch history (ms before, value):', [(round((t0-t)/1e3), int(v)) for t, v in zip(ch.t_us, ch[col])][-6:])
