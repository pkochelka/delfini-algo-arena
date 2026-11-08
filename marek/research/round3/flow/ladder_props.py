import numpy as np, pandas as pd
from pathlib import Path
C = Path(__file__).parent / 'cache'
L = pd.read_pickle(C / 'ladders.pkl')
L['clean'] = L.steps.apply(lambda x: all(s == 1 for s in x))
print('clean (all steps +1):', L.clean.mean())
for vol in [False, True]:
    S = L[(L.vol == vol) & L.clean]
    print('==== vol', vol, 'n', len(S))
    for n in range(2, 8):
        Z = np.array([x for x in S[S.n == n].sizes])
        if len(Z) == 0: continue
        print(f' n={n} cnt {len(Z)} median size per rung', np.median(Z, 0), 'mean log', np.round(np.log(Z).mean(0), 2))
    # first rung price relative to touch
    tp = np.where(S.side == 0, S.bid_before - S.p0, S.p0 - S.ask_before)
    print(' p0 depth behind same-side touch (neg = improves):', pd.Series(tp).value_counts().sort_index().head(15).to_dict())
    allz = np.concatenate([np.array(x) for x in S.sizes])
    print(' rung size quantiles', np.quantile(allz, [.01, .1, .25, .5, .75, .9, .99]))
    # do sizes within ladder equal? ratio
    Z = [np.array(x) for x in S.sizes]
    print(' frac ladders with all equal sizes', np.mean([len(set(z)) == 1 for z in Z]), 'monotone increasing', np.mean([np.all(np.diff(z) >= 0) for z in Z]))
