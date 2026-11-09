"""For big moves (W s, J ticks): how long until the strategy's position is aligned (>= A lots), and how much of the
move happened before that. Also lead of true S / filter F over mid at onset."""
import numpy as np
g = np.load('vol_signature/cache/grid.npz'); tg = g['g']; M = g['M']; S = g['S']; on = g['on']
x = np.load('jumps/cache/r4_volsig_round4.npz')['x']
lt, F, pos, kv = x[:, 0], x[:, 1], x[:, 5], x[:, 9]
def at(arr, t): return arr[max(0, np.searchsorted(lt, t) - 1)]
shockmask = np.zeros_like(tg, bool)
for o in on: shockmask |= (tg >= o - 2) & (tg < o + 25)
W, J, A = 5.0, 40, 60
k = int(W / 0.05); d = np.r_[M[k:] - M[:-k], np.zeros(k)]
ev = []; i = 0
while i < len(tg) - k:
    if abs(d[i]) >= J and not shockmask[i:i + k].any(): ev.append(i); i += k
    else: i += 1
rows = []
for i in ev:
    t0 = tg[i]; sg = np.sign(d[i])
    # time to aligned
    j = np.searchsorted(lt, t0); jj = j
    while jj < len(lt) and sg * pos[jj] < A and lt[jj] < t0 + W: jj += 1
    ta = lt[min(jj, len(lt) - 1)] - t0
    mv_before = sg * (np.interp(t0 + ta, tg, M) - M[i])
    rows.append((sg * pos[j], ta, mv_before, sg * d[i], sg * (S[i] - M[i]), sg * (at(F, t0) - M[i]), sg * at(kv, t0)))
r = np.array(rows)
wrong = r[:, 0] < A
print(f'n={len(r)}, not aligned at onset {wrong.sum()}')
for name, sel in [('not aligned', wrong), ('aligned', ~wrong)]:
    q = r[sel]
    print(name, 'time to align: median %.2f s, mean %.2f; move before aligned mean %.1f of %.1f; S-mid %.2f F-mid %.2f kv %.2f'
          % (np.median(q[:, 1]), q[:, 1].mean(), q[:, 2].mean(), q[:, 3].mean(), q[:, 4].mean(), q[:, 5].mean(), q[:, 6].mean()))
np.save('jumps/cache/ev5_40.npy', np.array(ev))
