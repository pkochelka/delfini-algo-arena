import numpy as np
from pathlib import Path
C = Path(__file__).parent / 'cache'
z = np.load(C / 'bins1s.npz'); tk, mk, mask = z['tk'], z['mk'], z['mask']
a = np.load(C / 'hmm_tk_2.npy')[:, 1] > .5; b = np.load(C / 'hmm_mk_2.npy')[:, 1] > .5; j = np.load(C / 'hmm_joint_2.npy')[:, 1] > .5
print('agree tk vs mk', (a == b)[mask].mean(), 'tk vs joint', (a == j)[mask].mean(), 'mk vs joint', (b == j)[mask].mean())
for nm, s in [('calm', ~j & mask), ('vol', j & mask)]:
    print(nm, 'n', s.sum(), 'mk mean %.2f var %.2f DI %.2f' % (mk[s].mean(), mk[s].var(), mk[s].var() / mk[s].mean()),
          'tk mean %.2f DI %.2f' % (tk[s].mean(), tk[s].var() / tk[s].mean()))
# segments of joint
x = j.astype(int); ch = np.flatnonzero(np.diff(x)) + 1
st = np.r_[0, ch]; en = np.r_[ch, len(x)]
segs = [(s, e, x[s]) for s, e in zip(st, en)]
print('n segs', len(segs))
for s, e, v in segs: print(v, s, e, e - s)
