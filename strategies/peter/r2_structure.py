"""Structure audit for any tape (round 2 onward): anomalies, jumps & regimes, periodicity, trends,
per-split stability and a predictive-model comparison.

Usage (repo root):
    uv run python strategies/peter/r2_structure.py data/round2_public.csv --folds data/splits_r2/wf/folds.json --hold-parts 4

Unlike r1_structure.py it does not assume the round-1 generator (an exact S from taker fills); on
round 2 that model fails (535k maker orders violate its bounds). The structural price is instead the
maker-implied price P: the median over each 100 ms bin of (bid insert + 0.55) / (ask insert - 0.55),
forward-filled. P[k] uses only inserts in bin k, so as a feature at the end of bin k it is causal.
A "jump" is |P change| > JUMP ticks in one 100 ms bin; "turbulent" is 10 s realised vol of P > TURB.
Splits are the walk-forward eval windows of --folds plus the held-out tail cut into --hold-parts.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np, pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.cache import load_tape  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('csv'); ap.add_argument('--folds', required=True); ap.add_argument('--hold-parts', type=int, default=4)
ap.add_argument('--jump', type=float, default=5.0); ap.add_argument('--turb', type=float, default=2.0)
args = ap.parse_args()
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40)

t = load_tape(args.csv, verbose=False)
ev, o = t.events, t.orders
T = ev.t_us.to_numpy(); dur = T[-1] / 1e6
bid, ask = ev.bid.to_numpy().astype(float), ev.ask.to_numpy().astype(float)
bq, aq = ev.bid_qty.to_numpy().astype(float), ev.ask_qty.to_numpy().astype(float)
two = (bid > 0) & (ask > 0)
MID = pd.Series(np.where(two, (bid + ask) / 2, np.nan)).ffill().bfill().to_numpy()
f, gm = o[o.ioc], o[~o.ioc]
ft = f.t_us.to_numpy(); fsgn = np.where(f.side == 'B', 1.0, -1.0)

# ---------- maker-implied price on a 100 ms grid
DT = 100_000; nb01 = int(T[-1] // DT) + 1
imp_o = np.where(gm.side == 'B', gm.price + 0.55, gm.price - 0.55)
P = pd.Series(imp_o).groupby((gm.t_us.to_numpy() // DT).astype(int)).median().reindex(range(nb01)).ffill().bfill().to_numpy()
g01 = (np.arange(nb01) + 1) * DT                      # grid point = end of bin k
i01 = np.searchsorted(T, g01, 'right') - 1
m01 = MID[i01]; gap01 = m01 - P
dP = np.r_[0.0, np.diff(P)]
rv10 = np.sqrt(pd.Series(dP ** 2).rolling(100, min_periods=10).mean().to_numpy() * 10)   # trailing, ticks/sqrt(s)
turb = rv10 > args.turb
jump = np.abs(dP) > args.jump

# ---------- 1 s grid
nb = int(dur)
g1 = (np.arange(nb) + 1) * 1_000_000
i1 = np.searchsorted(T, g1, 'right') - 1
mid1 = MID[i1]; r1 = np.diff(np.r_[MID[0], mid1])
P1 = P[np.minimum(np.arange(nb) * 10 + 9, nb01 - 1)]; dP1 = np.diff(np.r_[P[0], P1])
sec = lambda x: (np.asarray(x) // 1_000_000).astype(int)
cnt = {'maker_ins': np.bincount(sec(gm.t_us), minlength=nb)[:nb],
       'taker': np.bincount(sec(ft), minlength=nb)[:nb],
       'taker_vol': np.bincount(sec(ft), weights=f.volume.to_numpy(), minlength=nb)[:nb],
       'cancel': np.bincount(sec(T[ev.op.to_numpy() == 'C']), minlength=nb)[:nb],
       'sq_ret': r1 ** 2,
       'jumps': np.bincount(np.flatnonzero(jump) // 10, minlength=nb)[:nb]}

print(f'tape {args.csv}: {dur:.0f} s, {len(ev):,} events, {len(o):,} orders')
print('\n===== A. ANOMALIES')
one = ~two
st = np.flatnonzero(one & ~np.r_[False, one[:-1]]); en = np.flatnonzero(one & ~np.r_[one[1:], False])
ep_ms = (T[np.minimum(en + 1, len(T) - 1)] - T[st]) / 1e3; late = T[st] > 5e6
print(f'one-sided book: {len(st)} episodes ({late.sum()} after 5 s); duration median {np.median(ep_ms[late]):.1f} ms, '
      f'max {ep_ms[late].max():.1f} ms; emptied by IOC {(ev.lifespan.to_numpy()[st][late] == "F").sum()}, by cancel {(ev.op.to_numpy()[st][late] == "C").sum()}')
gaps = np.diff(T) / 1e6
print(f'max inter-event gap {gaps.max()*1e3:.2f} ms at t={T[np.argmax(gaps)]/1e6:.1f}s; Poisson expectation ~{np.log(len(T))/(len(T)/dur)*1e3:.1f} ms')
print('duplicate timestamps:', int((gaps == 0).sum()), ' order-id gaps:', int((np.diff(o.order_id.astype(np.int64).to_numpy()) != 1).sum()))
for name, x in [('events/100ms', np.bincount((T // 100_000).astype(int))), ('takers/1s', cnt['taker'])]:
    lam = x.mean()
    print(f'{name}: mean {lam:.1f} max {x.max()} min {x.min()}  P(max>=obs | Poisson) = {min(1, len(x)*stats.poisson.sf(x.max()-1, lam)):.3g}')
for nm, x in [('maker', cnt['maker_ins']), ('taker', cnt['taker']), ('cancel', cnt['cancel'])]:
    di = [x[: len(x) // h * h].reshape(-1, h).sum(1) for h in (1, 10, 60)]
    print(f'{nm:6s} dispersion index 1/10/60 s: ' + ' / '.join(f'{y.var()/y.mean():.2f}' for y in di) + '  (Poisson = 1)')
for nm, sub in [('taker', f), ('maker', gm)]:
    lv = np.log(sub.volume.to_numpy()); mu, sd = lv.mean(), lv.std()
    pmax = 1 - stats.norm.cdf((np.log(sub.volume.max() + 0.5) - mu) / sd) ** len(lv)
    print(f'{nm} size: max {sub.volume.max()} (p99.99 {np.quantile(sub.volume, .9999):.0f}); log-size mean {mu:.3f} sd {sd:.3f}; P(lognormal max>=obs) {pmax:.3f}')
print(f'mid - P: sd {np.std(gap01):.3f}, min {gap01.min():.1f}, max {gap01.max():.1f}')
sp = (ask - bid)[two]
print('spread (event-weighted, top 6):', {int(k): round(v, 4) for k, v in pd.Series(sp).value_counts(normalize=True).sort_index().head(6).items()},
      f' max {sp.max():.0f}')
print(f'1 s mid return: sd {r1.std():.3f}, kurtosis {stats.kurtosis(r1):.1f}, max |r| {np.abs(r1).max():.1f} at t={np.argmax(np.abs(r1))+1}s')
calm1 = ~turb[np.arange(nb) * 10 + 9]
print(f'1 s mid return sd in calm seconds {r1[calm1].std():.3f}, turbulent {r1[~calm1].std():.3f}')
print(f'IOCs with zero fill {(f.filled == 0).mean():.4f}')

print('\n===== A2. JUMPS & REGIMES')
jt = np.flatnonzero(jump); js = dP[jt]
print(f'jumps (|dP|>{args.jump:g} in 100 ms): {len(jt)}  ({len(jt)/dur*60:.2f}/min), up share {(js>0).mean():.3f}, |size| median {np.median(np.abs(js)):.1f} p90 {np.quantile(np.abs(js), .9):.1f} max {np.abs(js).max():.1f}')
print(f'sum of jump sizes {js.sum():+.1f}  vs total P move {P[-1]-P[0]:+.1f}')
ia = np.diff(jt) / 10
print(f'inter-jump gap mean {ia.mean():.1f}s CV {ia.std()/ia.mean():.2f} (Poisson 1); share of gaps < 5 s {(ia<5).mean():.2f}')
print('jump size sign ACF lag1..3:', [round(np.corrcoef(np.sign(js[:-k]), np.sign(js[k:]))[0, 1], 3) for k in (1, 2, 3)])
# reversal after jumps: P and mid move over the next 1/5/30 s, signed by jump
for h in (10, 50, 300):
    ok = jt + h < nb01
    print(f'  after jump, signed P move next {h/10:g}s: {np.mean((P[jt[ok]+h]-P[jt[ok]])*np.sign(js[ok])):+.2f} ticks '
          f'(mid {np.mean((m01[jt[ok]+h]-m01[jt[ok]])*np.sign(js[ok])):+.2f}); mid lag at the jump bin {np.mean((P[jt]-m01[jt])*np.sign(js)):+.2f}')
ts_, te_ = np.flatnonzero(turb & ~np.r_[False, turb[:-1]]), np.flatnonzero(turb & ~np.r_[turb[1:], False])
L = (te_ - ts_ + 1) / 10
print(f'turbulent (trailing 10 s rv > {args.turb:g}): {turb.mean():.3f} of time, {len(ts_)} episodes, length median {np.median(L):.1f}s max {L.max():.0f}s')
dPc = dP[~turb]; print(f'P vol (ticks/sqrt s) calm {dPc.std()*np.sqrt(10):.3f}, turbulent {dP[turb].std()*np.sqrt(10):.3f}')
# do jumps start turbulence? P(jump in next 10 s | jump now)
nxt = np.array([(jump[k + 1:k + 101]).any() for k in jt if k + 101 < nb01])
print(f'P(another jump within 10 s | jump) = {nxt.mean():.3f}  vs unconditional {1-np.exp(-len(jt)/dur*10):.3f}')
# taker behaviour by regime
tk01 = np.minimum((ft // DT).astype(int), nb01 - 1)
for nm, m in [('calm', ~turb[tk01]), ('turbulent', turb[tk01])]:
    tsec = (~turb).sum() / 10 if nm == 'calm' else turb.sum() / 10
    print(f'  takers {nm:9s}: {m.sum()/tsec:.2f}/s, mean size {f.volume.to_numpy()[m].mean():.0f}, fill share {f.filled.to_numpy()[m].sum()/f.volume.to_numpy()[m].sum():.3f}')
mk01 = np.minimum((gm.t_us.to_numpy() // DT).astype(int), nb01 - 1)
for nm, m in [('calm', ~turb[mk01]), ('turbulent', turb[mk01])]:
    tsec = (~turb).sum() / 10 if nm == 'calm' else turb.sum() / 10
    print(f'  makers {nm:9s}: {m.sum()/tsec:.1f}/s;  touch qty {((bq+aq)/2)[i01][(~turb) if nm=="calm" else turb].mean():.0f};  spread {(ask-bid)[i01][((~turb) if nm=="calm" else turb) & two[i01]].mean():.2f}')

print('\n===== B. PERIODICITY')
def fisher_g(x):
    x = x - x.mean(); Pw = np.abs(np.fft.rfft(x))[1:] ** 2; m = len(Pw); g = Pw.max() / Pw.sum(); k = np.argmax(Pw) + 1
    return len(x) / k, min(1.0, m * (1 - g) ** (m - 1))
for nm, x in cnt.items():
    per, p = fisher_g(x.astype(float)); print(f'{nm:10s} strongest period {per:8.1f}s  Fisher g p {p:.3g}')
per, p = fisher_g(turb.astype(float)); print(f'turbulence indicator (100 ms) strongest period {per/10:.1f}s p {p:.3g}')
for mod in [60_000_000, 10_000_000, 1_000_000, 100_000, 10_000, 1000]:
    ph = (T % mod) / mod; print(f'timestamp phase mod {mod/1e6:g}s: KS p = {stats.kstest(ph[::11], "uniform").pvalue:.3f}')
ph = (jt / 10 % 60) / 60; print(f'jump time phase mod 60 s: KS p = {stats.kstest(ph, "uniform").pvalue:.3f};  mod 10 s: {stats.kstest((jt/10 % 10)/10, "uniform").pvalue:.3f}')
def acf(x, L):
    x = x - x.mean(); v = (x * x).mean(); return np.array([(x[:-l] * x[l:]).mean() / v for l in L])
Ls = [1, 2, 3, 5, 10, 30, 60, 120, 300]
print('lags', Ls, ' 2/sqrt(n)=', round(2 / np.sqrt(nb), 3))
print('ACF 1s mid returns ', np.round(acf(r1, Ls), 3))
print('ACF 1s P returns   ', np.round(acf(dP1, Ls), 3))
print('ACF |1s P returns| ', np.round(acf(np.abs(dP1), Ls), 3), ' (vol clustering)')
print('ACF taker count/1s ', np.round(acf(cnt['taker'].astype(float), Ls), 3))
print('ACF taker sign (per order)', np.round(acf(fsgn, [1, 2, 3, 5, 10, 50]), 4), ' 2/sqrt(n)=', round(2 / np.sqrt(len(fsgn)), 4))
def lb_p(x, h):
    n = len(x); r = acf(x, range(1, h + 1)); q = n * (n + 2) * (r ** 2 / (n - np.arange(1, h + 1))).sum(); return stats.chi2.sf(q, h)
for nm, x in [('mid r1', r1), ('P r1', dP1), ('|P r1|', np.abs(dP1))]:
    print(f'Ljung-Box {nm:7s} lag 20 p={lb_p(x, 20):.3g}  lag 300 p={lb_p(x, 300):.3g}')
for h in [1, 5, 10, 30, 60, 300]:
    print(f'variance ratio h={h:4d}s  P {np.var(np.diff(P1[::h]))/(h*np.var(dP1)):.3f}  mid {np.var(np.diff(mid1[::h]))/(h*np.var(r1)):.3f}')

print('\n===== C. TRENDS')
xs = np.arange(nb)
for nm in ['maker_ins', 'taker', 'taker_vol', 'cancel', 'sq_ret', 'jumps']:
    sl, ic, rr, p, se = stats.linregress(xs, cnt[nm]); print(f'{nm:10s} trend {sl*nb/max(ic,1e-9)*100:+.1f}% over tape, p={p:.3f}')
drift = P1[-1] - P[0]
print(f'P drift {drift:+.1f} ticks; z vs martingale {drift/(dP1.std()*np.sqrt(nb)):.2f};  without jumps {drift-js.sum():+.1f}, z {(drift-js.sum())/(dPc.std()*np.sqrt(len(dPc))):.2f}')
print(f'taker buy share {(fsgn>0).mean():.4f} z={((fsgn>0).mean()-.5)/np.sqrt(.25/len(fsgn)):.2f}')
step = int(np.ceil(nb / 12 / 100) * 100)
for a in range(0, nb, step):
    b = min(a + step, nb); m = (ft >= a * 1e6) & (ft < b * 1e6)
    print(f'  [{a:5d},{b:5d}) P {P1[b-1]-(P1[a-1] if a else P[0]):+7.1f}  jumps {cnt["jumps"][a:b].sum():3d}  turb {turb[a*10:b*10].mean():.2f}  takers {m.sum()}  buy {np.mean(fsgn[m]>0):.3f}')

print('\n===== D. PER-SPLIT STABILITY')
FJ = json.load(open(args.folds)); folds = FJ['folds']
win = [(fd['name'], fd['eval_start_s'], fd['end_s']) for fd in folds]
h0 = FJ['held_out_from_s']; hl = (dur - h0) / args.hold_parts
win += [(f'hold{i+1}', h0 + hl * i, h0 + hl * (i + 1)) for i in range(args.hold_parts)]
rows = []
for nm, a, b in win:
    sel = (g01 > a * 1e6) & (g01 <= b * 1e6); gp = gap01[sel]
    phi = np.corrcoef(gp[:-1], gp[1:])[0, 1]
    oi = o[(o.t_us > a * 1e6) & (o.t_us <= b * 1e6)]; fi, gi = oi[oi.ioc], oi[~oi.ioc]
    lt = gi.end_t_us - gi.t_us
    # per-split impact: signed P move 1 s after each calm taker, slope per sqrt(filled)
    k = np.minimum((fi.t_us.to_numpy() // DT).astype(int), nb01 - 11); s = np.where(fi.side == 'B', 1, -1)
    q = fi.filled.to_numpy(); okk = (q > 0) & ~turb[k] & (k > 0)
    c_hat = np.polyfit(np.sqrt(q[okk]), (P[k[okk] + 10] - P[k[okk] - 1]) * s[okk], 1)[0]
    rs = r1[int(a):int(b)]; jsel = (jt >= a * 10) & (jt < b * 10)
    rows.append(dict(split=nm, maker_rate=len(gi)/(b-a), taker_rate=len(fi)/(b-a), buy=(fi.side == 'B').mean(),
                     taker_mu=np.log(fi.volume).mean(), maker_mu=np.log(gi.volume).mean(),
                     fill_share=fi.filled.sum()/fi.volume.sum(), life_med=np.nanmedian(lt)/1e6,
                     spread1=((ask - bid)[i01][sel] == 1).mean(), touch=((bq + aq)[i01][sel]).mean()/2,
                     sd_gap=gp.std(), gap_hl_ms=-100*np.log(2)/np.log(max(phi, 1e-3)), vol1s=rs.std(),
                     vol_calm=rs[calm1[int(a):int(b)]].std(), jumps=jsel.sum(), jump_abs=np.abs(dP[jt[jsel]]).sum(),
                     turb=turb[sel].mean(), c_hat=c_hat, drift=P[min(int(b*10), nb01-1)]-P[int(a*10)]))
D = pd.DataFrame(rows).set_index('split')
print(D.round(3).to_string())
print('\ncoefficient of variation across splits:'); print((D.std() / D.mean().abs()).round(3).to_string())
ln = np.array([b - a for _, a, b in win])
tk = np.array([((ft > a * 1e6) & (ft <= b * 1e6)).sum() for _, a, b in win])
print(f'\ntaker count homogeneity chi2 p = {stats.chisquare(tk, tk.sum()*ln/ln.sum()).pvalue:.3g}')
mk = D.maker_rate.to_numpy() * ln; print(f'maker count homogeneity chi2 p = {stats.chisquare(mk, mk.sum()*ln/ln.sum()).pvalue:.3g}')
jc = D.jumps.to_numpy(); print(f'jump count homogeneity chi2 p = {stats.chisquare(jc, jc.sum()*ln/ln.sum()).pvalue:.3g}  (counts {jc.tolist()})')
print(f'taker buy share across splits chi2 p = {stats.chi2_contingency(np.c_[D.buy*tk, (1-D.buy)*tk]).pvalue:.3f}')
print('drift z per split:', np.round(D.drift / (dP1.std() * np.sqrt(ln)), 2).tolist())
print('Spearman corr across splits (turb vs ...):', {c: round(stats.spearmanr(D.turb, D[c])[0], 2) for c in ['taker_rate', 'spread1', 'touch', 'vol1s', 'sd_gap', 'fill_share']})

print('\n===== E. MODEL COMPARISON: mid change over next H on the 100 ms grid, leave-one-split-out OLS R^2')
fold_id = np.full(nb01, -1)
for j, (_, a, b) in enumerate(win):
    fold_id[(g01 > a * 1e6) & (g01 <= b * 1e6)] = j
imb01 = np.nan_to_num(bq[i01] / np.maximum(bq[i01] + aq[i01], 1) - 0.5)
mic = np.where(two[i01], (bid[i01] * aq[i01] + ask[i01] * bq[i01]) / np.maximum(bq[i01] + aq[i01], 1) - m01, 0.0)
pr1 = np.r_[np.zeros(10), m01[10:] - m01[:-10]]; pr5 = np.r_[np.zeros(50), m01[50:] - m01[:-50]]
kk = np.searchsorted(T, gm.t_us.to_numpy(), 'left') - 1
agg = np.bincount((gm.t_us.to_numpy() // DT).astype(int), weights=np.clip(gm.price.to_numpy() - MID[np.maximum(kk, 0)], -5, 5), minlength=nb01)[:nb01]
tflow = np.bincount(np.minimum((ft // DT).astype(int), nb01 - 1), weights=fsgn * np.sqrt(f.filled.to_numpy()), minlength=nb01)
gapc = np.clip(-gap01, -20, 20)
feats = {'RW (zero)': [], 'AR past returns': [pr1, pr5], 'imbalance': [imb01], 'microprice': [mic],
         'insert aggressiveness': [agg], 'taker flow 100ms': [tflow], 'public (all above)': [imb01, mic, pr1, pr5, agg, tflow],
         'P - mid': [gapc], 'P - mid + public': [gapc, imb01, mic, pr1, agg, tflow]}
for H in [5, 10, 50]:
    y = np.r_[m01[H:] - m01[:-H], np.full(H, np.nan)]
    ok = (fold_id >= 0) & ~np.isnan(y)
    print(f'-- horizon {H/10:g}s: out-of-sample R^2 mean [min, max] over splits; calm-only mean')
    for nm, F in feats.items():
        r2, r2c = [], []
        for j in range(len(win)):
            te = ok & (fold_id == j); trn = ok & (fold_id != j)
            if F:
                X = np.c_[np.ones(nb01), np.column_stack(F)]; beta = np.linalg.lstsq(X[trn], y[trn], rcond=None)[0]; pred = X @ beta
            else:
                pred = np.zeros(nb01)
            for msk, out in [(te, r2), (te & ~turb, r2c)]:
                out.append(1 - ((y[msk] - pred[msk]) ** 2).sum() / ((y[msk] - y[msk].mean()) ** 2).sum())
        r2 = np.array(r2); print(f'   {nm:24s} {r2.mean():+.4f}  [{r2.min():+.4f}, {r2.max():+.4f}]   calm {np.mean(r2c):+.4f}')
y = np.r_[m01[10:] - m01[:-10], np.full(10, np.nan)]; ok = ~np.isnan(y) & ~turb
print(f'\ngap AR(1) at 100 ms (calm): phi={np.corrcoef(gap01[:-1][~turb[:-1]], gap01[1:][~turb[:-1]])[0,1]:.3f}; '
      f'slope of 1 s mid change on (P - mid), calm: {np.polyfit(gapc[ok], y[ok], 1)[0]:.3f}')
