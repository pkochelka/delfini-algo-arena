"""Sub-second regime switch times. Per-event log-likelihood ratio (vol vs calm) from empirical laws learned in deep
segment interiors: (a) non-ladder maker insert price relative to the opposite touch (clipped), (b) taker arrivals
(rate 7.9 vs 12.0 /s), (c) ladder starts (1.1 vs 11.3 /s), (d) non-ladder maker rate (186 vs 188.4). Then a single
change point in ±10 s around each coarse boundary from the 1 s joint HMM. -> regime_switches.csv, regimes.csv."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); ins = r[r.op == 0].reset_index(drop=True)
mem = np.load(C / 'ladder_member.npy'); L = pd.read_pickle(C / 'ladders.pkl')
seg = pd.read_csv(D / 'regimes_1s.csv'); ev = pd.read_csv(D / 'shock_events.csv')
reg = np.load(C / 'hmm_joint_2.npy')[:, 1] > 0.5
t = ins.t.to_numpy(); ioc = ins.ioc.to_numpy(); s = ins.side.to_numpy(); p = ins.px.to_numpy()
opp = np.where(s == 0, ins.ask_before.to_numpy() - p, p - ins.bid_before.to_numpy())  # maker: distance to opposite touch
own = np.where(s == 0, ins.bid_before.to_numpy() - p, p - ins.ask_before.to_numpy())  # maker: behind own touch (neg = improves)
feat = np.clip(np.nan_to_num(opp, nan=9), -1, 9).astype(int) * 12 + np.clip(np.nan_to_num(own, nan=9), -2, 9).astype(int) + 2
inner = np.array([reg[max(0, i - 5):i + 6].all() or (~reg[max(0, i - 5):i + 6]).all() for i in range(6000)])
for o in ev.onset: inner[max(0, int(o) - 2):int(o) + 20] = False
sec = t.astype(int)
base = ~ioc & (mem[:, 0] < 0)
nb = 12 * 11 + 20
hc = np.bincount(feat[base & inner[sec] & ~reg[sec]], minlength=nb) + 0.5
hv = np.bincount(feat[base & inner[sec] & reg[sec]], minlength=nb) + 0.5
llr_m = np.log(hv / hv.sum()) - np.log(hc / hc.sum())
Tc = (inner & ~reg).sum(); Tv = (inner & reg).sum()
rate = {}
streams = {'tk': t[ioc], 'lad': L.t0.to_numpy(), 'mk': t[base]}
for k, x in streams.items():
    xs = x.astype(int); rate[k] = ((inner[xs] & ~reg[xs]).sum() / Tc, (inner[xs] & reg[xs]).sum() / Tv)
print('rates calm/vol', {k: tuple(round(a, 3) for a in v) for k, v in rate.items()})
print('maker-price info per insert (KL vol||calm) %.4f nats' % ((hv / hv.sum()) * llr_m).sum())
# event list with llr
ev_t = np.r_[t[base], streams['tk'], streams['lad']]
ev_l = np.r_[llr_m[feat[base]], np.full(len(streams['tk']), np.log(rate['tk'][1] / rate['tk'][0])), np.full(len(streams['lad']), np.log(rate['lad'][1] / rate['lad'][0]))]
o = np.argsort(ev_t); ev_t = ev_t[o]; ev_l = ev_l[o]; cum = np.r_[0, np.cumsum(ev_l)]
drift = sum(rate[k][1] - rate[k][0] for k in rate) - (rate['mk'][1] - rate['mk'][0])  # compensator per s (vol-calm), maker-price part has none
drift = sum(rate[k][1] - rate[k][0] for k in rate)
def LL(a, b):  # log LR vol vs calm over [a,b)
    i, j = np.searchsorted(ev_t, [a, b]); return cum[j] - cum[i] - drift * (b - a)
rows = []
for i in range(1, len(seg)):
    b0 = seg.start[i]; to_vol = seg.regime[i] == 'volatile'
    lo = max(b0 - 10, seg.start[i - 1] + 0.5); hi = min(b0 + 10, seg.end[i] - 0.5)
    grid = np.arange(lo, hi, 0.005)
    ii = np.searchsorted(ev_t, grid); c = cum[ii] - drift * grid  # running LLR at grid
    # change point: maximize LL(after) for vol side: sign
    sc = (c[-1] - c) if to_vol else (c - c[0])
    k = np.argmax(sc); tb = grid[k]
    # 1-LL support interval
    ok = grid[sc >= sc[k] - 1.0]
    rows.append(dict(t_switch=round(tb, 3), to=('volatile' if to_vol else 'calm'), ci_lo=round(ok.min(), 3), ci_hi=round(ok.max(), 3), coarse=b0))
S = pd.DataFrame(rows)
# merge out tiny segments (<1.5 s coarse) — report raw too
S.to_csv(C / 'regime_switches_raw.csv', index=False)
S['ci_w'] = S.ci_hi - S.ci_lo
pd.set_option('display.width', 200); print(S.to_string())
