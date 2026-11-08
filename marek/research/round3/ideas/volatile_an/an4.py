exec(open('an1.py').read().split('samp = np')[0])
samp = np.array([(x[0], x[1], x[2], x[3], x[9]) for x in L['samp']])
t, pos, turb, pv, shk = samp.T
pnl = np.array([[float(v) for v in l.split(',')] for l in open(sys.argv[2]).read().split('\n')[1:] if l])
pt = pnl[:, 0] * 1e-6 if pnl[0, 0] > 1e5 else pnl[:, 0]; pp = pnl[:, 1]
def P(x): return np.interp(x, pt, pp)
ent, ex = [], []
for a, b in segs:
    # entry lag: first sample >= a with turb and pv>0.7
    i = np.searchsorted(t, a)
    j = i
    while j < len(t) and not (pv[j] > 0.7): j += 1
    ent.append(t[j] - a if j < len(t) else np.nan)
    k = np.searchsorted(t, b)
    while k < len(t) and pv[k] > 0.7: k += 1
    ex.append(t[k] - b if k < len(t) else np.nan)
ent, ex = np.array(ent), np.array(ex)
print('entry lag s: median %.2f  quantiles' % np.nanmedian(ent), np.round(np.nanpercentile(ent, [10, 25, 75, 90]), 2))
print('exit lag s: median %.2f  quantiles' % np.nanmedian(ex), np.round(np.nanpercentile(ex, [10, 25, 75, 90]), 2))
# false turb in calm: fraction of calm time with pv>0.7 (excluding shocks)
inv = np.zeros(len(t), bool)
for a, b in segs: inv |= (t >= a) & (t < b)
m = ~inv & (shk == 0)
print('calm time with pv>0.7: %.3f ; vol time with pv<=0.7: %.3f' % ((pv[m] > 0.7).mean(), (pv[inv] <= 0.7).mean()))
# pnl in windows around entry/exit
for nm, xs in (('entry', [a for a, b in segs]), ('exit', [b for a, b in segs])):
    for w0, w1 in ((-5, 0), (0, 2), (2, 5), (5, 10), (10, 20)):
        print(nm, w0, w1, 'pnl sum %.0f' % sum(P(x + w1) - P(x + w0) for x in xs))
