"""Can causal features pick profitable 900-lot burst windows? Linear model of each direction's round-trip value
(per lot, tau s), fit on one half, tested on the other; report net of the top-N picks (non-overlapping) out of sample."""
import numpy as np
z = np.load('jumps/cache/book_r4_volsig.npz'); x, db, da = z['x'], z['db'], z['da']
t, F, sd, kv, Y, turb, pos, bb, ba = x.T
g = np.load('vol_signature/cache/grid.npz'); on = g['on']
shock = np.zeros(len(t), bool)
for o in on: shock |= (t >= o - 2) & (t < o + 25)
mid = (bb + ba) / 2
def sw(D, E):
    c = np.cumsum(D, 1); k = np.minimum((c < E).sum(1), 19)
    prev = np.where(k > 0, np.take_along_axis(c, np.maximum(k - 1, 0)[:, None], 1)[:, 0], 0)
    lv = np.arange(20)[None, :]
    out = ((D * lv * (lv < k[:, None])).sum(1) + (E - prev) * k) / E
    out[c[:, -1] < E] = np.nan; return out
E = 900; sa, sb = sw(da, E), sw(db, E)
def lag(a, k): return np.r_[np.full(k, np.nan), a[:-k]]
feat = np.c_[kv, F - mid, Y, mid - lag(mid, 5), mid - lag(mid, 10), mid - lag(mid, 20), ba - bb,
             np.log1p(da[:, :5].sum(1)) - np.log1p(db[:, :5].sum(1)), sa, sb, kv * abs(kv), (F - mid) * abs(F - mid)]
for tau in (0.5, 1.0, 2.0):
    k = int(round(tau / 0.05)); n = len(t) - k
    up = np.r_[(bb[k:] - sb[k:]) - (ba[:n] + sa[:n]), np.full(k, np.nan)]
    dn = np.r_[(bb[:n] - sb[:n]) - (ba[k:] + sa[k:]), np.full(k, np.nan)]
    ok = (turb > 0) & ~shock & np.isfinite(up) & np.isfinite(dn) & np.isfinite(feat).all(1)
    ok &= np.r_[~shock[k:], np.zeros(k, bool)]
    pen = 10000 * tau / E
    for tr_half in (0, 1):
        tr = ok & ((t < 3000) == (tr_half == 0)); te = ok & ~((t < 3000) == (tr_half == 0))
        A = np.c_[feat, np.ones(len(t))]
        bu, *_ = np.linalg.lstsq(A[tr], up[tr], rcond=None); bd, *_ = np.linalg.lstsq(A[tr], dn[tr], rcond=None)
        pu, pdn = A @ bu, A @ bd
        best = np.where(pu > pdn, pu, pdn); real = np.where(pu > pdn, up, dn)
        idx = np.where(te)[0]; idx = idx[np.argsort(-best[idx])]
        used = np.zeros(len(t), bool); picks = []
        for i in idx:
            if used[i:i + k + 1].any(): continue
            used[i:i + k + 1] = True; picks.append(i)
            if len(picks) >= 200: break
        picks = np.array(picks); netl = real[picks] - pen
        cs = np.cumsum(netl * E) / 1e3
        print(f'tau={tau} test half {1 - tr_half}: pred/lot top {best[picks[0]]:.1f}; realized gross/lot top20 {real[picks[:20]].mean():.1f} '
              f'(need {pen:.1f}); cum net top 10/20/50/100: {cs[9]:.0f} {cs[19]:.0f} {cs[49]:.0f} {cs[99]:.0f}k')
