import sys, numpy as np
sys.argv += []
exec(open('marek/research/round2/hyp/jumpfilter/decomp.py').read().split("print('vol S changes")[0])
bad = vv & (np.abs(e) > 1.0)
idx = np.where(bad)[0]
eps = []
for i in idx:
    if eps and t[i] - eps[-1][1] < 0.5: eps[-1][1] = t[i]; eps[-1][2] = max(eps[-1][2], abs(e[i]))
    else: eps.append([t[i], t[i], abs(e[i]), i])
print(len(eps), 'episodes')
for a, b, mx, i in eps[:25]:
    w = (tS > a - 1.0) & (tS <= b)
    big = [(round(x - a, 3), round(y, 2), 'F' if n else 'J') for x, y, n in zip(tS[w], dS[w], near[w]) if abs(y) > 0.5]
    print(f'{a:8.2f} dur {b-a:5.2f} max|e| {mx:4.2f} e0 {e[i]:+.2f} sd {sd[i]:.2f}', big[:8])
