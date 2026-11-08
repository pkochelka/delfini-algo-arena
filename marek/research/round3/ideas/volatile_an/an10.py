exec(open('an8.py').read().split("m = inv\n")[0])
g = F - mid
for nm, m in (('vol', inv), ('calm', calm)):
    for h in (0.05, 0.2, 0.5, 1, 2, 5, 15):
        fut = S(t[m] + h) - F[m]  # S at horizon minus our current estimate
        fm = (S(t[m] + h) + 0) - S(t[m] + 0.02)
        b = np.polyfit(g[m], fut, 1); c = np.corrcoef(g[m], fut)[0, 1]
        print(nm, 'h %5.2f: S(t+h)-F on g=F-mid: beta %+.3f corr %+.3f | sd(g) %.2f' % (h, b[0], c, g[m].std()))
