import numpy as np, pandas as pd
d = pd.read_csv('data/round4.csv')
d['tu'] = np.round(d.Time.values * 1e6).astype(np.int64)
op = d.Operation.values; ls = d.Lifespan.fillna('-').values
t = d.tu.values
# lookup for cancels: order info
ins = d[d.Operation == 'Insert'].set_index('OrderId')
idx = np.where(np.diff(t) == 1)[0]
# IF > IG
sel = [i for i in idx if op[i] == 'Insert' and ls[i] == 'F' and op[i+1] == 'Insert' and ls[i+1] == 'G']
a = d.iloc[sel]; b = d.iloc[[i+1 for i in sel]]
print('IF>IG n', len(sel))
print('same side frac', (a.Side.values == b.Side.values).mean())
print('price diff b-a (taker limit):', pd.Series(b.Price.values - a.Price.values).value_counts().head(10).to_dict())
print('maker vol q', b.Volume.quantile([.1,.5,.9]).tolist(), 'taker vol q', a.Volume.quantile([.1,.5,.9]).tolist())
print('maker vol == taker vol', (a.Volume.values == b.Volume.values).mean())
print(pd.DataFrame({'t':a.Time.values,'tside':a.Side.values,'tpx':a.Price.values,'tv':a.Volume.values,'mside':b.Side.values,'mpx':b.Price.values,'mv':b.Volume.values}).head(25).to_string())
# C > IG
sel = [i for i in idx if op[i] == 'Cancel' and op[i+1] == 'Insert' and ls[i+1] == 'G']
a = d.iloc[sel]; b = d.iloc[[i+1 for i in sel]]
ai = ins.reindex(a.OrderId.values)
print('\nC>IG n', len(sel))
print('same side', (ai.Side.values == b.Side.values).mean(), 'same vol', (ai.Volume.values == b.Volume.values).mean())
print('px diff new-old', pd.Series(b.Price.values - ai.Price.values).value_counts().head(10).to_dict())
print('age of cancelled', (a.Time.values - ai.Time.values).round(3)[:20])
