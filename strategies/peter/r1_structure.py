"""Round-1 structure audit: anomalies, periodicity, trends, per-split stability, model comparison.

Usage: uv run python strategies/peter/r1_structure.py   (run from the repo root; ~2 min)
Splits are the walk-forward eval windows (data/splits_r1/wf/folds.json) plus the held-out 9000-12000 s in 4 parts.
Model comparison: leave-one-split-out OLS R^2 for the mid change over 0.5/1/5 s on a 100 ms grid.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.cache import load_tape

C, S0 = 0.026, 10000.0
t = load_tape('data/round1.csv', verbose=False)
ev, o, tr = t.events, t.orders, t.trades
T = ev.t_us.to_numpy()
dur = 12000.0

# ---------- S path
f = o[o.ioc]
ft = f.t_us.to_numpy()
fsgn = np.where(f.side == 'B', 1.0, -1.0)
cu = np.cumsum(fsgn * np.sqrt(f.filled.to_numpy()))
def S_at(tus):  # S after all takers with t <= tus
    k = np.searchsorted(ft, tus, 'right') - 1
    return S0 + C * np.where(k >= 0, cu[np.maximum(k, 0)], 0.0)

bid, ask = ev.bid.to_numpy().astype(float), ev.ask.to_numpy().astype(float)
bq, aq = ev.bid_qty.to_numpy().astype(float), ev.ask_qty.to_numpy().astype(float)
two = (bid > 0) & (ask > 0)
_m = np.where(two, (bid + ask) / 2, np.nan)
MID = pd.Series(_m).ffill().bfill().to_numpy()   # last two-sided mid
one = ~two
st = np.flatnonzero(one & ~np.r_[False, one[:-1]]); en = np.flatnonzero(one & ~np.r_[one[1:], False])
ep_ms = (T[np.minimum(en + 1, len(T) - 1)] - T[st]) / 1e3
late = T[st] > 5e6
print(f'one-sided book: {len(st)} episodes ({late.sum()} after t=5s, last at {T[st].max()/1e6:.0f}s); '
      f'duration median {np.median(ep_ms[late]):.1f} ms, max {ep_ms[late].max():.1f} ms; '
      f'emptied by IOC {(ev.lifespan.to_numpy()[st][late] == "F").sum()}, by cancel {(ev.op.to_numpy()[st][late] == "C").sum()}')

# ---------- grids
def grid(dt):
    g = np.arange(int(dt * 1e6), int(dur * 1e6) + 1, int(dt * 1e6))
    i = np.searchsorted(T, g, 'right') - 1
    return g, i, MID[i]
g1, i1, mid1 = grid(1.0)
S1 = S_at(g1)
r1 = np.diff(mid1)
sec = lambda x: (x // 1_000_000).astype(int)
nb = int(dur)
cnt = {
    'maker_ins': np.bincount(sec(o.t_us[~o.ioc].to_numpy()), minlength=nb)[:nb],
    'taker': np.bincount(sec(ft), minlength=nb)[:nb],
    'taker_vol': np.bincount(sec(ft), weights=f.volume.to_numpy(), minlength=nb)[:nb],
    'cancel': np.bincount(sec(T[ev.op.to_numpy() == 'C']), minlength=nb)[:nb],
}
cnt['sq_ret'] = np.r_[0, r1 ** 2][:nb]

print('\n===== A. ANOMALIES')
gaps = np.diff(T) / 1e6
print(f'max inter-event gap {gaps.max()*1e3:.2f} ms at t={T[np.argmax(gaps)]/1e6:.1f}; expected max for Poisson(368/s) over 4.4M ~ {np.log(len(T))/368*1e3:.1f} ms')
print('duplicate timestamps:', (gaps == 0).sum(), ' order-id gaps:', (np.diff(o.order_id.astype(np.int64).to_numpy()) != 1).sum())
for name, x in [('events/100ms', np.bincount((T // 100_000).astype(int))), ('takers/1s', cnt['taker'])]:
    lam = x.mean(); z = (x.max() - lam) / np.sqrt(lam)
    print(f'{name}: mean {lam:.1f} max {x.max()} min {x.min()}  P(max>=obs | Poisson, n={len(x)}) = {min(1, len(x)*stats.poisson.sf(x.max()-1, lam)):.3g}')
for nm, sub, mu, sd in [('taker', f, 4.106, 1.396), ('maker', o[~o.ioc], 2.658, 1.799)]:
    v = sub.volume.to_numpy(); n = len(v)
    pmax = 1 - stats.norm.cdf((np.log(v.max() + 0.5) - mu) / sd) ** n
    print(f'{nm} size max {v.max()} (p99.99 {np.quantile(v, .9999):.0f}); P(lognormal max >= obs) = {pmax:.3f}')
S_ev = S_at(T)
gap_ms = (bid + ask) / 2 - S_ev
gap_ms[~two] = np.nan
print(f'mid-S: sd {np.nanstd(gap_ms):.3f}, min {np.nanmin(gap_ms):.2f}, max {np.nanmax(gap_ms):.2f}')
sp = (ask - bid)[two]
print('spread distribution (event-weighted):', {int(k): round(v, 5) for k, v in pd.Series(sp).value_counts(normalize=True).sort_index().items()})
gm = o[~o.ioc]
d = np.where(gm.side == 'B', S_at(gm.t_us.to_numpy() - 1) - gm.price, gm.price - S_at(gm.t_us.to_numpy() - 1))
print(f'maker distance from S: max {d.max():.2f} ticks; >5 ticks: {(d > 5).sum()} ; >8: {(d > 8).sum()}  (Exp(0.45) predicts P(>5)={np.exp(-(5-0.05)/0.45)*len(d):.1f} orders)')
k1 = stats.kurtosis(r1); print(f'1s mid return kurtosis {k1:.2f}, max |r| {np.abs(r1).max()} ticks at t={g1[1:][np.argmax(np.abs(r1))]/1e6:.0f}s, sd {r1.std():.3f}')
dS1 = np.diff(S1)
print(f'1s S return: sd {dS1.std():.3f} kurt {stats.kurtosis(dS1):.2f} max |dS| {np.abs(dS1).max():.2f}')
# biggest S moves over 10 s compared with the null (i.i.d. signs): z
g10 = g1[9::10]; S10 = S_at(g10); dS10 = np.diff(S10)
print(f'10s S move max |z| {np.abs(dS10/dS10.std()).max():.2f} (Gaussian max over {len(dS10)} ~ {stats.norm.isf(0.25/len(dS10)):.2f})')
nofill = (f.filled == 0).mean(); print(f'IOCs with zero fill {nofill:.4f}; cancel rows for filled orders handled as no-ops')
print('first 5 s: events', (T < 5e6).sum(), 'touch qty at t=5s', bq[np.searchsorted(T, 5e6)], aq[np.searchsorted(T, 5e6)])

print('\n===== B. PERIODICITY')
def fisher_g(x):
    x = x - x.mean(); P = np.abs(np.fft.rfft(x))[1:] ** 2; m = len(P)
    gstat = P.max() / P.sum(); k = np.argmax(P) + 1
    p = min(1.0, m * (1 - gstat) ** (m - 1))
    return len(x) / k, gstat, p
for nm, x in cnt.items():
    per, gs, p = fisher_g(x.astype(float))
    print(f'{nm:10s} strongest period {per:8.1f}s  Fisher g p-value {p:.3g}')
for nm, x in [('maker_ins 10ms bins', np.bincount((o.t_us[~o.ioc].to_numpy() // 10_000).astype(int)))]:
    per, gs, p = fisher_g(x[: 1_200_000].astype(float)); print(f'{nm}: period {per*0.01:.3f}s p {p:.3g}')
for mod in [1_000_000, 100_000, 10_000, 1000]:
    ph = (T % mod) / mod
    print(f'timestamp phase mod {mod/1e6:g}s: KS vs uniform p = {stats.kstest(ph[::11], "uniform").pvalue:.3f}')
def acf(x, L):
    x = x - x.mean(); v = (x * x).mean(); return np.array([(x[:-l] * x[l:]).mean() / v for l in L])
L = [1, 2, 3, 5, 10, 30, 60, 120, 300]
print('ACF 1s mid returns  ', np.round(acf(r1, L), 3), ' 2/sqrt(n)=', round(2 / np.sqrt(len(r1)), 3))
print('ACF 1s S returns    ', np.round(acf(dS1, L), 3))
print('ACF |1s S returns|  ', np.round(acf(np.abs(dS1), L), 3), ' (vol clustering)')
print('ACF taker count/1s  ', np.round(acf(cnt['taker'].astype(float), L), 3))
print('ACF taker sign (per order)', np.round(acf(fsgn, [1, 2, 3, 5, 10, 50]), 4), ' 2/sqrt(n)=', round(2 / np.sqrt(len(fsgn)), 4))
def lb_p(x, h):
    n = len(x); r = acf(x, range(1, h + 1)); q = n * (n + 2) * (r ** 2 / (n - np.arange(1, h + 1))).sum()
    return stats.chi2.sf(q, h)
for nm, x in [('mid r1', r1), ('S r1', dS1), ('|S r1|', np.abs(dS1)), ('S r1^2', dS1 ** 2)]:
    print(f'Ljung-Box {nm:7s} lag 20 p={lb_p(x, 20):.3f}  lag 300 p={lb_p(x, 300):.3f}')
for h in [1, 5, 10, 30, 60, 300, 600]:
    a = S1[::h]; vr = np.var(np.diff(a)) / (h * np.var(dS1))
    b = mid1[::h]; vrm = np.var(np.diff(b)) / (h * np.var(r1))
    print(f'variance ratio h={h:4d}s  S {vr:.3f}  mid {vrm:.3f}')

print('\n===== C. TRENDS')
xs = np.arange(nb)
for nm in ['maker_ins', 'taker', 'taker_vol', 'cancel']:
    sl, ic, rr, p, se = stats.linregress(xs, cnt[nm])
    print(f'{nm:10s} trend {sl*12000/ic*100:+.2f}% over tape, p={p:.3f}')
drift = S1[-1] - S0
print(f'S drift total {drift:+.2f} ticks; z vs martingale = {drift/(dS1.std()*np.sqrt(len(dS1))):.2f}')
buy_share = (fsgn > 0).mean(); print(f'taker buy share {buy_share:.4f} z={(buy_share-.5)/np.sqrt(.25/len(fsgn)):.2f}')
# rolling 1000s drift windows
for a in range(0, 12000, 1000):
    print(f'  [{a:5d},{a+1000:5d}) S {S1[min(a+999, nb-1)] - (S1[a-1] if a else S0):+7.2f}  takers {cnt["taker"][a:a+1000].sum()}  buy {np.mean(fsgn[(ft>=a*1e6)&(ft<(a+1000)*1e6)]>0):.3f}')

print('\n===== D. PER-SPLIT STABILITY')
folds = json.load(open('data/splits_r1/wf/folds.json'))['folds']
win = [(fd['name'], fd['eval_start_s'], fd['end_s']) for fd in folds]
win += [(f'hold{i+1}', 9000 + 750 * i, 9000 + 750 * (i + 1)) for i in range(4)]
# 100ms grid for gap dynamics + model features
g01, i01, m01 = grid(0.1)
S01 = S_at(g01); gap01 = m01 - S01
imb01 = np.nan_to_num(bq[i01] / np.maximum(bq[i01] + aq[i01], 1) - 0.5)
sp01 = ask[i01] - bid[i01]
rows = []
for nm, a, b in win:
    sel = (g01 > a * 1e6) & (g01 <= b * 1e6)
    gp = gap01[sel]; phi = np.corrcoef(gp[:-1], gp[1:])[0, 1]
    oi = o[(o.t_us > a * 1e6) & (o.t_us <= b * 1e6)]
    fi = oi[oi.ioc]; gi = oi[~oi.ioc]
    lt = gi.end_t_us - gi.t_us
    # per-fold impact coefficient: regress 10s mid change on sum sign*sqrt(filled)
    a0 = int(a) + 1
    s10 = np.arange(a0, int(b) - 10, 10)
    dm = mid1[s10 + 9] - mid1[s10 - 1]
    tt = fi.t_us.to_numpy(); u = np.where(fi.side == 'B', 1, -1) * np.sqrt(fi.filled.to_numpy())
    kb = ((tt / 1e6 - a0) // 10).astype(int); okb = (kb >= 0) & (kb < len(s10))
    uu = np.bincount(kb[okb], weights=u[okb], minlength=len(s10))
    cfit = np.polyfit(uu, dm, 1)[0]
    dSf = np.diff(S1[int(a):int(b)])
    rows.append(dict(split=nm, maker_rate=len(gi) / (b - a), taker_rate=len(fi) / (b - a), buy=(fi.side == 'B').mean(),
                     taker_mu=np.log(fi.volume).mean(), maker_mu=np.log(gi.volume).mean(),
                     fill_share=fi.filled.sum() / fi.volume.sum(), life_med=np.nanmedian(lt) / 1e6,
                     spread1=(sp01[sel] == 1).mean(), touch=(bq[i01][sel] + aq[i01][sel]).mean() / 2,
                     sd_gap=gp.std(), gap_hl_ms=-100 * np.log(2) / np.log(phi), vol1s=np.diff(mid1[int(a):int(b)]).std(),
                     c_hat=cfit, S_drift=S1[int(b) - 1] - S1[int(a) - 1], acf1_dS=np.corrcoef(dSf[:-1], dSf[1:])[0, 1]))
D = pd.DataFrame(rows).set_index('split')
pd.set_option('display.width', 250)
print(D.round(4).to_string())
print('\ncoefficient of variation across splits:'); print((D.std() / D.mean().abs()).round(4).to_string())
# homogeneity tests on counts
tk = np.array([((ft > a * 1e6) & (ft <= b * 1e6)).sum() for _, a, b in win]); ln = np.array([b - a for _, a, b in win])
exp = tk.sum() * ln / ln.sum(); print(f'\ntaker count homogeneity chi2 p = {stats.chisquare(tk, exp).pvalue:.3f}')
print(f'maker rate homogeneity chi2 p = {stats.chisquare(D.maker_rate * ln, (D.maker_rate * ln).sum() * ln / ln.sum()).pvalue:.3f}')
print(f'taker buy share across splits: chi2 p = {stats.chi2_contingency(np.c_[D.buy * D.taker_rate * ln, (1 - D.buy) * D.taker_rate * ln]).pvalue:.3f}')
print(f'S drift per split / sd(sqrt(750)*vol): z = {np.round(D.S_drift / (dS1.std() * np.sqrt(ln)), 2).tolist()}')

print('\n===== E. MODEL COMPARISON: predict mid change over next H (100ms grid), leave-one-split-out')
fold_id = np.full(len(g01), -1)
for j, (_, a, b) in enumerate(win):
    fold_id[(g01 > a * 1e6) & (g01 <= b * 1e6)] = j
# micro-price & past returns
mic = np.where(two[i01], (bid[i01] * aq[i01] + ask[i01] * bq[i01]) / np.maximum(bq[i01] + aq[i01], 1) - m01, 0.0)
pr1 = np.r_[np.zeros(10), m01[10:] - m01[:-10]]          # past 1 s mid return
pr5 = np.r_[np.zeros(50), m01[50:] - m01[:-50]]
# recent maker insert aggressiveness: signed (price-mid) over last 100ms
gi = o[~o.ioc]; gt = gi.t_us.to_numpy(); kk = np.searchsorted(T, gt, 'left') - 1
mid_before = MID[np.maximum(kk, 0)]
agg = np.bincount((gt // 100_000).astype(int), weights=np.clip(gi.price.to_numpy() - mid_before, -5, 5), minlength=len(g01) + 2)[1:len(g01) + 1]
feats = {'RW (zero)': [], 'AR past returns': [pr1, pr5], 'imbalance': [imb01], 'microprice': [mic],
         'insert aggressiveness': [agg], 'imb+micro+AR+aggr (public)': [imb01, mic, pr1, pr5, agg],
         'S - mid (structural)': [-gap01], 'S - mid + public': [-gap01, imb01, mic, pr1, agg]}
for H in [5, 10, 50]:  # 0.5s, 1s, 5s
    y = np.r_[m01[H:] - m01[:-H], np.full(H, np.nan)]
    ok = (fold_id >= 0) & ~np.isnan(y) & np.isfinite(mic)
    print(f'-- horizon {H/10:g}s: out-of-sample RÂ² per model (mean over splits, min, max)')
    for nm, F in feats.items():
        r2 = []
        for j in range(len(win)):
            te = ok & (fold_id == j); trn = ok & (fold_id != j)
            if not F:
                pred = np.zeros(te.sum())
            else:
                X = np.c_[np.ones(len(g01)), np.column_stack(F)]
                beta = np.linalg.lstsq(X[trn], y[trn], rcond=None)[0]; pred = X[te] @ beta
            r2.append(1 - ((y[te] - pred) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum())
        r2 = np.array(r2); print(f'   {nm:28s} {r2.mean():+.4f}  [{r2.min():+.4f}, {r2.max():+.4f}]')
# OU fit of gap: does mid revert to S? regress dGap on gap at 100ms
dg = np.diff(gap01); print(f'\ngap AR(1) at 100ms: phi={np.corrcoef(gap01[:-1], gap01[1:])[0,1]:.3f}; regress mid change(1s) on (S-mid): slope ', end='')
y = np.r_[m01[10:] - m01[:-10], np.full(10, np.nan)]; ok = ~np.isnan(y)
print(round(np.polyfit(-gap01[ok], y[ok], 1)[0], 3))
