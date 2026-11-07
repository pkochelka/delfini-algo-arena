"""Taker behaviour: price rule vs touch/mid, side predictability of future mid, sign clustering, size-sign."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); ev = pd.read_csv(D / 'shock_events.csv')
g2 = np.load(C / 'hmm_taker_gamma2.npy'); hi = g2[:, 1] > 0.5
tt = r.t.to_numpy(); mid = ((r.bid_after + r.ask_after) / 2).ffill().to_numpy()
idx = np.where((r.op == 0) & r.ioc)[0]
k = r.iloc[idx].copy()
k['sg'] = np.where(k.side == 0, 1, -1)
k['mid_pre'] = ((k.bid_before + k.ask_before) / 2).to_numpy()
k['mid_post'] = mid[idx]
k['touch'] = np.where(k.side == 0, k.ask_before, k.bid_before)
k['off_touch'] = k.sg * (k.px - k.touch)
k['off_mid'] = k.sg * (k.px - k.mid_pre)
k['hi'] = hi[k.t.astype(int).clip(0, 5999)]
shock = np.zeros(len(k), bool)
for o in ev.onset: shock |= ((k.t > o - 0.5) & (k.t < o + 15)).to_numpy()
k['shock'] = shock
k['spread'] = k.ask_before - k.bid_before
pd.set_option('display.width', 200)
print('n takers', len(k), 'in shock windows', shock.sum())
print('price - touch (signed, ticks):'); print(k.off_touch.value_counts().sort_index().head(15).to_string())
print('price - mid (signed) by spread:'); print(pd.crosstab(k.spread.clip(upper=4), k.off_mid.clip(-3, 4)).to_string())
print('volume>=? price offsets by regime:'); print(pd.crosstab([k.shock, k.hi], k.off_touch.clip(-2, 4), normalize='index').round(3).to_string())
def mid_at(x): return mid[np.clip(np.searchsorted(tt, x, 'right') - 1, 0, None)]
K = k[~k.shock]
for h in [0.1, 1, 5, 30, 120]:
    fut = mid_at(K.t.to_numpy() + h)
    d_pre = K.sg * (fut - K.mid_pre); d_post = K.sg * (fut - K.mid_post)
    line = f'h={h:6.1f}s  E[sg*(mid_h - mid_pre)] {d_pre.mean():+.3f} (se {d_pre.std()/np.sqrt(len(K)):.3f})  E[sg*(mid_h - mid_post)] {d_post.mean():+.3f} (se {d_post.std()/np.sqrt(len(K)):.3f})'
    for hh in [False, True]:
        m = (K.hi == hh).to_numpy(); line += f' | hi={hh}: post {d_post[m].mean():+.3f}±{d_post[m].std()/np.sqrt(m.sum()):.3f}'
    print(line)
# sign autocorrelation
s = K.sg.to_numpy(); print('sign acf lags1-5', [round(np.corrcoef(s[:-l], s[l:])[0, 1], 4) for l in range(1, 6)])
gap = np.diff(K.t.to_numpy()); same = s[1:] == s[:-1]
print('P(same side | gap bin):'); print(pd.Series(same).groupby(pd.cut(gap, [0, 0.001, 0.01, 0.05, 0.2, 1, 10])).agg(['mean', 'count']).round(3).to_string())
# size-sign and size vs future
K = K.assign(lv=np.log(K.vol))
print('log size by side', K.groupby('side').lv.agg(['mean', 'std', 'count']).round(4).to_string())
print('log size by regime', K.groupby('hi').lv.agg(['mean', 'std', 'count']).round(4).to_string())
fut = mid_at(K.t.to_numpy() + 5); K['d5post'] = K.sg * (fut - K.mid_post)
print('future 5s move (post) by size quintile'); print(K.groupby(pd.qcut(K.vol, 5, duplicates='drop')).d5post.agg(['mean', 'count']).round(3).to_string())
# buy fraction in volatile segments vs segment drift
S = pd.read_csv(D / 'taker_regimes.csv')
print('volatile segment: net move vs taker buy share')
for _, sgm in S[S.high].iterrows():
    x = k[(k.t >= sgm.start) & (k.t < sgm.end) & ~k.shock]
    print(f'  {sgm.start:5d}-{sgm.end:5d} net {mid_at(sgm.end) - mid_at(sgm.start):+7.1f} buyshare {(x.side == 0).mean():.3f} n {len(x)} signedvol {(x.sg * x.vol).sum():+8.0f}')
k.to_pickle(C / 'takers.pkl')
