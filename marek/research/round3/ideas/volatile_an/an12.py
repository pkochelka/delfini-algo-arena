exec(open('an8.py').read().split("m = inv\n")[0])
e = F - Sh; ef = Sf - Sh
bad = inv & (np.abs(e) > 1.5)
# runs
runs = []
i = 0
while i < len(t):
    if not bad[i]: i += 1; continue
    j = i
    while j + 1 < len(t) and bad[j + 1] and t[j + 1] - t[j] < 0.1: j += 1
    runs.append((t[i], t[j] - t[i] + 0.02, np.max(np.abs(e[i:j + 1])), np.max(np.abs(ef[i:j + 1])), np.mean(sd[i:j+1])))
    i = j + 1
runs = np.array(runs)
print('bad runs (|F-S|>1.5 in vol):', len(runs), 'total s %.1f, median dur %.2f s, p90 %.2f' % (runs[:, 1].sum(), np.median(runs[:, 1]), np.percentile(runs[:, 1], 90)))
print('in these runs, s_filt max err median %.2f (ours %.2f); our sd mean %.2f' % (np.median(runs[:, 3]), np.median(runs[:, 2]), np.mean(runs[:, 4])))
big = runs[np.argsort(-runs[:, 1])][:12]
for r in big: print('  t %.2f dur %.2f maxerr %.2f sfilt %.2f sd %.2f' % tuple(r))
# fills during bad runs
send = {s[1]: s for s in L['send']}
ft = np.array([f[0] for f in L['fill']]); fq = np.array([f[4] for f in L['fill']]); fs = np.array([f[2] for f in L['fill']]); fp = np.array([f[3] for f in L['fill']], float)
inr = np.zeros(len(ft), bool)
for r in runs: inr |= (ft >= r[0] - 0.05) & (ft <= r[0] + r[1] + 0.05)
ed = fs * (S(ft + 0.02) - fp)
print('fills in bad runs: qty %d, S-edge sum %.0f (per lot %.2f)' % (fq[inr].sum(), (ed * fq)[inr].sum(), (ed * fq)[inr].sum() / max(1, fq[inr].sum())))
