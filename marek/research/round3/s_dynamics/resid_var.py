"""Variogram of the residual E (S minus deterministic impact path, shocks removed) over calm-only / vol-only windows.
Fit V(h) = 2*sY2*(1-exp(-h/T)) + sF2*h (OU transient part + permanent random walk). Usage: resid_var.py TAG params.json"""
import sys, os, json, numpy as np, pandas as pd
from scipy import optimize
import prep, smooth_seg as SS
tag, pf = sys.argv[1], sys.argv[2]
P = json.load(open(prep.common.HERE / pf))
d = SS.d; r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']; vol = r['vol'].astype(bool); sh = r['shock'].astype(bool)
def arr(k, dflt): return np.where(vol, P['vol'].get(k, dflt), P['calm'].get(k, dflt)).astype(float)
shift = SS.impact(slice(0, len(t)), arr('c', 0), arr('e', 0.6), arr('pperm', 1.0), arr('tdec', 15.0))
E = m - np.cumsum(np.r_[0, shift[:-1]])
g = np.arange(0, 6000, 0.5); i = np.clip(np.searchsorted(t, g, 'right') - 1, 0, len(t) - 1)
Eg = E[i]; vg = vol[i]
SH = pd.read_csv(os.environ.get('SHOCKS', prep.common.HERE / '../flow/shock_events.csv')).onset.to_numpy()
seg = np.searchsorted(SH + 2.5, g)
badg = np.zeros(len(g), bool)
for a in SH: badg |= (g >= a - 0.5) & (g < a + 6)
if os.environ.get('SWX', '0') == '1':      # also drop +-5 s around regime switches and 1 s around big unexplained maker-step moves
    VS = pd.read_csv(prep.common.HERE / 'vol_segments.csv') if 'VSEG' not in os.environ else pd.read_csv(os.environ['VSEG'])
    for a in np.r_[VS.start, VS.end]: badg |= (g >= a - 5) & (g < a + 5)
    dEm = np.diff(E); mk = ~d['ioc'][:-1] & (np.abs(dEm) > 0.2)
    for a in t[np.where(mk)[0]]: badg |= (g >= a - 1) & (g < a + 1)
# run id: contiguous same regime, same shock interval, not bad
key = seg * 4 + vg * 2 + badg
run = np.r_[0, np.cumsum(key[1:] != key[:-1])]
hs = np.array([0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 90, 120])
for nm, want in (('calm', 0), ('vol', 1)):
    V = []; N = []
    for h in hs:
        k = int(h / 0.5)
        ok = (run[k:] == run[:-k]) & (vg[:-k] == want) & ~badg[:-k]
        x = Eg[k:][ok] - Eg[:-k][ok]
        V.append(np.mean(x * x)); N.append(ok.sum())
    V = np.array(V); N = np.array(N)
    f = lambda th: (2 * th[0] * (1 - np.exp(-hs / th[1])) + th[2] * hs - V) / (V / np.sqrt(np.maximum(N * 0.5 / hs, 1)))
    rr = optimize.least_squares(f, [0.5, 15, 0.01], bounds=([0, 0.5, 0], [100, 500, 10]))
    print(nm, 'V(h):', ' '.join('%g:%.3f' % (h, v) for h, v in zip(hs, V)))
    print('   V/h:', ' '.join('%g:%.4f' % (h, v / h) for h, v in zip(hs, V)))
    print('   fit sY2 %.3f T %.1f s sF2 %.4f /s  => transient innovation rate 2sY2/T = %.4f /s' % (rr.x[0], rr.x[1], rr.x[2], 2 * rr.x[0] / rr.x[1]))
    def Vk(th):
        q, pp, T, m2 = th
        a = pp * pp * hs + 2 * pp * (1 - pp) * T * (1 - np.exp(-hs / T)) + (1 - pp) ** 2 * T / 2 * (1 - np.exp(-2 * hs / T))
        b = (1 - pp) ** 2 * (1 - np.exp(-hs / T)) ** 2 * T / 2
        return q * (a + b) + 2 * m2
    f2 = lambda th: (Vk(th) - V) / (V / np.sqrt(np.maximum(N * 0.5 / hs, 1)))
    r2 = optimize.least_squares(f2, [V[0] / 0.5, 0.5, 15, 0.002], bounds=([0, 0, 0.5, 0], [100, 1, 500, 1]))
    print('   kernel model (kick xi with K = p + (1-p)exp(-t/T)): q %.4f /s  p_xi %.3f  T %.1f s  meas var %.4f  ; rms rel err %.3f' % (*r2.x, np.sqrt(np.mean(((Vk(r2.x) - V) / V) ** 2))))
