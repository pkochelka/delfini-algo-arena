import sys, numpy as np
sys.path.insert(0, 'strategies/peter'); sys.path.insert(0, '.')
from analysis.cache import load_tape
from fv_mm import FV, FV_PARAMS
csv = sys.argv[1]; t1 = float(sys.argv[2])
tp = load_tape(csv, verbose=False); ev, o = tp.events, tp.orders
filled = np.zeros(len(ev), dtype=np.int64); oo = o[o.ioc]; filled[oo.row.to_numpy()] = oo.filled.to_numpy()
bid = ev.bid.to_numpy(); ask = ev.ask.to_numpy(); T = ev.t_us.to_numpy() / 1e6
ins = (ev.op == 'I').to_numpy(); G = (ev.lifespan == 'G').to_numpy(); B = (ev.side == 'B').to_numpy()
px = np.nan_to_num(ev.price.to_numpy()).astype(np.int64); pb = np.r_[0, bid[:-1]]; pa = np.r_[0, ask[:-1]]
f = FV(FV_PARAMS, (pb[1] + pa[1]) / 2)
rec = []
for i in range(1, len(T)):
    t = T[i]
    if t > t1: break
    if not ins[i]: continue
    if G[i]:
        opp = pa[i] if B[i] else pb[i]
        cl = bool(opp) and (px[i] == opp - 1 if B[i] else px[i] == opp + 1)
        if t > 20 and not cl and f.pv < 0.05 and f.sd < 0.25:
            rec.append((f.mean - px[i] if B[i] else px[i] - f.mean, f.sd))
        f.maker(t, bool(B[i]), int(px[i]), int(opp) if opp else None)
    elif filled[i]:
        f.taker(t, bool(B[i]), None, int(filled[i]))
r = np.array(rec)
x = r[:, 0]
print(csv, len(x), 'mean', x.mean().round(3), 'sd', x.std().round(3))
qs = [.01, .05, .1, .25, .5, .75, .9, .95, .99, .999]
print(' '.join(f'{q}:{np.quantile(x, q):.2f}' for q in qs))
print('frac x>5', (x > 5).mean(), 'x>10', (x > 10).mean(), 'x<0', (x < 0).mean())
