exec(open('an1.py').read().split('samp = np')[0])
sf = np.load('sfilt.npz')
samp = np.array([(x[0], x[4], x[5], x[6], x[7], x[1]) for x in L['samp']])
t, F, sd, bb, ba, pos = samp.T
inv = np.zeros(len(t), bool)
for a, b in segs: inv |= (t >= a + 1) & (t < b - 1)
calm = ~inv
for o in on:
    m = (t >= o - 1) & (t < o + 25); inv &= ~m; calm &= ~m
Sh = S(t); Sf = sf['f'][np.clip(np.searchsorted(sf['t'], t, side='right') - 1, 0, len(sf['t']) - 1)]
mid = (bb + ba) / 2
for nm, m in (('vol', inv), ('calm', calm)):
    e1 = F[m] - Sh[m]; e2 = Sf[m] - Sh[m]; e3 = mid[m] - Sh[m]
    print(nm, 'F-S: bias %.3f sd %.3f MAE %.3f | s_filt-S: sd %.3f MAE %.3f | mid-S sd %.3f | our sd median %.3f' % (e1.mean(), e1.std(), abs(e1).mean(), e2.std(), abs(e2).mean(), e3.std(), np.median(sd[m])))
m = inv
e1 = np.abs(F[m] - Sh[m]); e2 = np.abs(Sf[m] - Sh[m])
print('vol |F-S| quantiles 50/90/99/99.9', np.round(np.percentile(e1, [50, 90, 99, 99.9]), 2), ' s_filt', np.round(np.percentile(e2, [50, 90, 99, 99.9]), 2))
print('share of vol time |F-S|>1: %.3f, >2: %.3f' % ((e1 > 1).mean(), (e1 > 2).mean()))
# errors vs our sd: z
z = (F[m] - Sh[m]) / sd[m]
print('z quantiles', np.round(np.percentile(np.abs(z), [50, 90, 99]), 2))
