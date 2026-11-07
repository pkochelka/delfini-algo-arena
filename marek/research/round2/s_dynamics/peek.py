import numpy as np, sys
import prep
d = prep.build(); tag = sys.argv[1]; a = float(sys.argv[2]); b = float(sys.argv[3])
r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']
ii = np.where((t >= a) & (t < b))[0]
K = ['bid', 'ask', 'bidC', 'askC', 'BUY', 'SELL']
for i in ii:
    print('%.4f %-4s %6d %5d/%-5d bb%6.0f ba%6.0f | S %.2f sd %.2f [%.2f %.2f]' % (t[i], K[d['kind'][i]], d['price'][i], d['filled'][i], d['volume'][i], d['bb'][i], d['ba'][i], r['mean'][i], r['sd'][i], r['lo'][i], r['hi'][i]))
