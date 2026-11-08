exec(open('an1.py').read().split('samp = np')[0])
samp = np.array([(x[0], x[1], x[2]) for x in L['samp']])
t, pos, turb = samp.T
inv = np.zeros(len(t), bool)
for a, b in segs: inv |= (t >= a) & (t < b)
sg = np.sign(pos)
# run lengths of constant sign (±) inside vol
runs = []
i = 0
while i < len(t):
    if not inv[i] or sg[i] == 0: i += 1; continue
    j = i
    while j + 1 < len(t) and inv[j + 1] and sg[j + 1] == sg[i]: j += 1
    runs.append(t[j] - t[i] + 0.02)
    i = j + 1
runs = np.array(runs)
print('vol same-sign holding runs: n %d, mean %.2f s, median %.2f, p25 %.2f p75 %.2f p90 %.2f; time-weighted mean %.2f' % (len(runs), runs.mean(), np.median(runs), *np.percentile(runs, [25, 75, 90]), (runs**2).sum() / runs.sum()))
