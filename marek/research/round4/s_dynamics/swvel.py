"""Velocity continuity across regime switches: residual drift (E slope) in W s before vs after each switch."""
import sys, json, numpy as np, pandas as pd
import prep, smooth_seg as SS
tag, pf = sys.argv[1], sys.argv[2]; W = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
P = json.load(open(prep.common.HERE / pf))
d = SS.d; r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']; vol = r['vol'].astype(bool)
def arr(k, dflt): return np.where(vol, P['vol'].get(k, dflt), P['calm'].get(k, dflt)).astype(float)
d['vshift'][:] = 0.0
shift = SS.impact(slice(0, len(t)), arr('c', 0), arr('e', 0.6), arr('pperm', 1.0), arr('tdec', 15.0))
E = m - np.cumsum(np.r_[0, shift[:-1]])
def at(x): return E[np.clip(np.searchsorted(t, x, 'right') - 1, 0, len(t) - 1)]
V = pd.read_csv(prep.common.HERE / 'vol_segments.csv')
SH = pd.read_csv(prep.common.HERE / '../flow/shock_events.csv').onset.to_numpy()
for nm, xs in (('calm->vol', V.start), ('vol->calm', V.end[V.end < 5999])):
    b, a = [], []
    for x in xs:
        if np.any(np.abs(SH + 3 - x) < W + 4): continue
        G = float(__import__("os").environ.get("GAP", "0")); b.append((at(x - G) - at(x - G - W)) / W); a.append((at(x + G + W) - at(x + G)) / W)
    b, a = np.array(b), np.array(a)
    print('%s n %d: sd before %.2f after %.2f corr %.3f slope(after on before) %.2f' % (nm, len(b), b.std(), a.std(), np.corrcoef(b, a)[0, 1], np.polyfit(b, a, 1)[0]))
# profile of |velocity| (E slope over 0.5 s bins) around switches, and of |impact| increments
I = np.cumsum(np.r_[0, shift[:-1]])
def atI(x): return I[np.clip(np.searchsorted(t, x, 'right') - 1, 0, len(t) - 1)]
for nm, xs in (('calm->vol', V.start), ('vol->calm', V.end[V.end < 5999])):
    xs = [x for x in xs if not np.any(np.abs(SH + 3 - x) < 10)]
    bins = np.arange(-4, 4.01, 0.5)
    rmsE = [np.sqrt(np.mean([((at(x + b0 + 0.5) - at(x + b0)) / 0.5) ** 2 for x in xs])) for b0 in bins]
    rmsI = [np.sqrt(np.mean([((atI(x + b0 + 0.5) - atI(x + b0)) / 0.5) ** 2 for x in xs])) for b0 in bins]
    print(nm, 'rms E-slope by 0.5 s bin start:', ' '.join('%+.1f:%.2f' % (b, v) for b, v in zip(bins, rmsE)))
    print(nm, 'rms impact rate             :', ' '.join('%+.1f:%.2f' % (b, v) for b, v in zip(bins, rmsI)))
print('aligned (by sign of E(x+3)-E(x+0.5)) mean E slope around calm->vol, 0.25 s bins; and around vol->calm (sign of E(x-0.5)-E(x-3))')
for nm, xs, ref in (('calm->vol', V.start, 1), ('vol->calm', V.end[V.end < 5999], -1)):
    xs = [x for x in xs if not np.any(np.abs(SH + 3 - x) < 10)]
    bins = np.arange(-3, 2.01, 0.25)
    sg = [np.sign(at(x + 3) - at(x + 0.5)) if ref > 0 else np.sign(at(x - 0.5) - at(x - 3)) for x in xs]
    al = [np.mean([s * (at(x + b0 + 0.25) - at(x + b0)) / 0.25 for x, s in zip(xs, sg)]) for b0 in bins]
    print(nm, ' '.join('%+.2f:%.2f' % (b, v) for b, v in zip(bins, al)))
