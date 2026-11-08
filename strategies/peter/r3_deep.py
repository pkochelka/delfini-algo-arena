"""Round-3 deep dive: the checks that separate round 3 from round 2 (run next to r2_structure.py on any tape).

Usage (repo root):
    uv run python strategies/peter/r3_deep.py data/round3_public.csv --folds data/splits_r3/wf/folds.json
    uv run python strategies/peter/r3_deep.py data/round2_public.csv --folds data/splits_r2/wf/folds.json   # reference

Sections (P = maker-implied price, the 100 ms median of bid insert + 0.55 / ask insert - 0.55, as in r2_structure):
    A  shocks: maker droughts (rate < half the median for >= 1 s), change-point onset, jump size and timing
    B  maker placement by offset from P and regime: rate, size, lifetime (round 3's deep ladder)
    C  impact kernel R(h): mid response per unit of sign * filled^0.62, shock windows excluded
    D  jump reversion (shock vs other jumps) and P - EMA(20 s) mean reversion, calm / turbulent
    E  taker PnL per lot vs fill at 0.1..60 s, volume weighted
    F  the same statistics per split (walk-forward eval windows + the holdout in --hold-parts parts)
    G  timestamp granularity
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
ap.add_argument('--turb', type=float, default=2.0); ap.add_argument('--jump', type=float, default=5.0)
args = ap.parse_args()
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30)

t = load_tape(args.csv, verbose=False)
ev, o, tr = t.events, t.orders, t.trades
T = ev.t_us.to_numpy(); dur = T[-1] / 1e6
gm, f = o[~o.ioc].copy(), o[o.ioc]
gt = gm.t_us.to_numpy()
bid, ask = ev.bid.to_numpy().astype(float), ev.ask.to_numpy().astype(float)
two = (bid > 0) & (ask > 0)
MID = pd.Series(np.where(two, (bid + ask) / 2, np.nan)).ffill().bfill().to_numpy()
DT = 100_000; nb = int(T[-1] // DT) + 1
kg = (gt // DT).astype(int)
P = pd.Series(np.where(gm.side == 'B', gm.price + 0.55, gm.price - 0.55)).groupby(kg).median().reindex(range(nb)).ffill().bfill().to_numpy()
m = MID[np.searchsorted(T, (np.arange(nb) + 1) * DT, 'right') - 1]   # mid at the end of each bin
mprev = np.r_[m[0], m[:-1]]
dP = np.r_[0.0, np.diff(P)]
rv10 = np.sqrt(pd.Series(dP ** 2).rolling(100, min_periods=10).mean().to_numpy() * 10)
turb = rv10 > args.turb
print(f'tape {args.csv}: {dur:.0f} s, {len(ev):,} events, turbulent share {turb.mean():.3f}')

# ---------------------------------------------------------------- A. shocks
print('\n===== A. SHOCKS (maker droughts)')
mk = np.bincount(kg, minlength=nb).astype(float)
rate = pd.Series(mk).rolling(10, center=True, min_periods=5).mean().to_numpy() * 10
med_rate = np.median(rate)
low = rate < 0.5 * med_rate
st = np.flatnonzero(low & ~np.r_[False, low[:-1]]); en = np.flatnonzero(low & ~np.r_[low[1:], False])
st = st[(en - st + 1) >= 10]
rows = []
for s in st:
    lo, hi = int((s / 10 - 1.5) * 1e6), int((s / 10 + 1.5) * 1e6)
    tt = gt[(gt >= lo) & (gt < hi)]
    n1 = np.arange(len(tt)); n2 = len(tt) - n1
    L1 = (tt - lo) / 1e6 + 1e-9; L2 = (hi - tt) / 1e6 + 1e-9
    ll = n1 * np.log(n1 / L1 + 1e-9) - n1 + n2 * np.log(n2 / L2 + 1e-9) - n2   # rate change-point MLE
    on = tt[np.argmax(ll)] / 1e6
    k0 = int((on + 1) * 10); back = k0 + np.argmax(rate[k0:] > 0.7 * med_rate)
    a = int(on * 10); seg = np.diff(P[a:a + 60]); kk = int(np.argmax(np.abs(seg)))
    inside = mk[int(on * 10) + 3:int(on * 10) + 27].mean() * 10
    rows.append(dict(onset=on, drought_s=back / 10 - on, inside_rate=inside, jump_at=(a + kk + 1) / 10 - on,
                     jump=P[min(nb - 1, a + 40)] - P[a + 25]))
sh = pd.DataFrame(rows)
gaps = np.diff(sh.onset)
print(f'{len(sh)} droughts; maker rate {med_rate:.0f}/s -> {sh.inside_rate.mean():.0f}/s for {sh.drought_s.mean():.2f} s '
      f'(sd {sh.drought_s.std():.2f}); gaps mean {gaps.mean():.0f} s, range {gaps.min():.0f}-{gaps.max():.0f}, CV {gaps.std() / gaps.mean():.2f}')
big = sh[np.abs(sh.jump) >= 8]
print(f'jump (P at onset+4 s minus onset+2.5 s): sd {sh.jump.std():.1f}, mean |J| {np.abs(sh.jump).mean():.1f}, '
      f'|J|>=8 in {len(big)}/{len(sh)}; largest-move time after onset, those shocks: median {big.jump_at.median():.2f} s, '
      f'IQR {big.jump_at.quantile(.25):.2f}-{big.jump_at.quantile(.75):.2f}')
print('jumps:', ' '.join(f'{j:+.0f}' for j in sh.jump))
shock_bad = np.zeros(nb, bool)      # [onset - 1 s, onset + 15 s): excluded from the kernels below
shock_jump = np.zeros(nb, bool)     # bins where the shock jump lands
for on in sh.onset:
    shock_bad[max(0, int(on * 10) - 10):int(on * 10) + 150] = True
    shock_jump[int(on * 10) + 20:int(on * 10) + 45] = True

# ---------------------------------------------------------------- B. placement
print('\n===== B. MAKER PLACEMENT by offset from P (rounded ticks behind fair; inserts per second of regime time)')
Pp = P[np.maximum(kg - 1, 0)]
gm['off'] = np.round(np.where(gm.side == 'B', Pp - gm.price, gm.price - Pp)).clip(-1, 10)
gm['reg'] = np.where(turb[np.maximum(kg - 1, 0)], 'turb', 'calm')
gm['life'] = (gm.end_t_us - gm.t_us) / 1e6
secs = {'calm': (~turb).mean() * dur, 'turb': turb.mean() * dur}
g = gm.groupby(['reg', 'off'])
n = g.size()
tab = pd.DataFrame({'rate': n / np.array([secs[r] for r, _ in n.index]), 'med_size': g.volume.median(),
                    'med_life': g.life.median(), 'fill_any': g.filled.apply(lambda x: (x > 0).mean())})
print(tab.round(2).unstack(0).to_string())
deep = gm[(gm.reg == 'calm') & (gm.off >= 4) & (gm.off <= 8)]
print(f'calm deep ladder (offset 4-8): {len(deep) / secs["calm"]:.2f}/s, median size {deep.volume.median():.0f}, '
      f'log-size {np.log(deep.volume).mean():.2f} +- {np.log(deep.volume).std():.2f}, median life {deep.life.median():.2f} s, '
      f'bid share {(deep.side == "B").mean():.3f}')

# ---------------------------------------------------------------- C. impact kernel
print('\n===== C. IMPACT KERNEL: mid response per unit of sign*filled^0.62 (shock windows excluded)')
ft = f.t_us.to_numpy(); sg = np.where(f.side == 'B', 1.0, -1.0)
x = np.bincount((ft // DT).astype(int), weights=sg * f.filled.to_numpy() ** 0.62, minlength=nb)[:nb]
xs = np.where(shock_bad, 0.0, x)


def kernel(h, sel):
    y = m[np.minimum(np.arange(nb) + h - 1, nb - 1)] - mprev
    return (xs[sel] * y[sel]).sum() / max((xs[sel] ** 2).sum(), 1e-9)


H = [1, 2, 5, 10, 20, 50, 100, 200, 300, 600, 1200]
ok = ~shock_bad
for nm, sel in [('all', ok), ('calm', ok & ~turb), ('turb', ok & turb)]:
    print(f'{nm:5}', ' '.join(f'{h / 10:g}s={kernel(h, sel):.4f}' for h in H))

# ---------------------------------------------------------------- D. jump reversion, mean reversion
print('\n===== D. JUMP REVERSION (signed P move after the jump / mean |jump|) and P - EMA mean reversion')
J = np.flatnonzero(np.abs(dP) > args.jump)
for nm, sel in [('shock jumps', shock_jump[J]), ('other jumps', ~shock_jump[J])]:
    jj = J[sel]; jj = jj[(jj > 10) & (jj < nb - 1200)]; jj = jj[np.r_[True, np.diff(jj) > 10]]
    if not len(jj):
        continue
    s = np.sign(dP[jj]); a = P[jj - 1]; sz = np.abs(P[jj + 5] - a).mean()
    print(f'{nm} n={len(jj)} mean |J| {sz:.1f}:  P', ' '.join(f'+{h / 10:g}s {np.mean(s * (P[jj + h] - a)) / sz:.2f}' for h in (5, 10, 50, 100, 300, 600, 1200)))
    print(f'{"":{len(nm)}}{"":>17}mid', ' '.join(f'+{h / 10:g}s {np.mean(s * (m[jj + h] - m[jj - 1])) / sz:.2f}' for h in (5, 10, 50, 100, 300, 600, 1200)))
e = np.empty(nb); e[0] = P[0]; al = 1 / 200
for i in range(1, nb):
    e[i] = e[i - 1] + al * (P[i] - e[i - 1])
dev = P - e; h = 200
xv, yv = dev[:-h], P[h:] - P[:-h]; okh = ~shock_bad[:-h]
for nm, sel in [('calm', okh & ~turb[:-h]), ('turb', okh & turb[:-h])]:
    print(f'{nm}: slope of P(+20 s) - P on P - EMA20: {np.polyfit(xv[sel], yv[sel], 1)[0]:+.3f}, corr {np.corrcoef(xv[sel], yv[sel])[0, 1]:+.3f}')

# ---------------------------------------------------------------- E. taker PnL
print('\n===== E. TAKER PnL per lot vs fill price (volume weighted; + = takers win, makers lose)')
trt = tr.t_us.to_numpy(); tsg = np.where(tr.aggressor == 'B', 1, -1); tq = tr.qty.to_numpy(); tp = tr.price.to_numpy()


def tpnl(hs, sel=slice(None)):
    mm = MID[np.searchsorted(T, trt[sel] + int(hs * 1e6), 'right') - 1]
    return np.average(tsg[sel] * (mm - tp[sel]), weights=tq[sel])


print(' '.join(f'{hs:g}s {tpnl(hs):+.3f}' for hs in (0.1, 0.5, 1, 2, 5, 10, 30, 60)))

# ---------------------------------------------------------------- F. per split
print('\n===== F. PER SPLIT')
folds = json.loads(Path(args.folds).read_text())['folds']
spl = [(fd['name'], fd['eval_start_s'], fd['end_s']) for fd in folds]
h0 = folds[-1]['end_s']; Lh = (dur - h0) / args.hold_parts
spl += [(f'hold{i + 1}', h0 + i * Lh, h0 + (i + 1) * Lh) for i in range(args.hold_parts)]
offv = gm.off.to_numpy(); regv = gm.reg.to_numpy(); volv = gm.volume.to_numpy()
rows = []
for nm, a0, b0 in spl:
    sel = np.zeros(nb, bool); sel[int(a0 * 10):max(int(a0 * 10), int(b0 * 10) - 600)] = True; sel &= ok
    gs = (gt >= a0 * 1e6) & (gt < b0 * 1e6)
    calm_s = (~turb[int(a0 * 10):int(b0 * 10)]).sum() / 10
    dsel = gs & (regv == 'calm') & (offv >= 4) & (offv <= 8)
    ts = (trt >= a0 * 1e6) & (trt < (b0 - 60) * 1e6)
    r2, r30 = kernel(20, sel), kernel(300, sel)
    rows.append(dict(split=nm, turb=turb[int(a0 * 10):int(b0 * 10)].mean(), R2s=r2, R30s=r30, transient=1 - r30 / r2,
                     deep_rate_calm=dsel.sum() / max(calm_s, 1), deep_med_size=np.median(volv[dsel]) if dsel.any() else np.nan,
                     taker_pnl_1s=tpnl(1, ts), taker_pnl_60s=tpnl(60, ts),
                     shocks=int(((sh.onset >= a0) & (sh.onset < b0)).sum())))
d = pd.DataFrame(rows).set_index('split')
print(d.round(3).to_string())
print('CV across splits:', (d.std() / d.mean().abs()).round(2).to_dict())
print(f'transient impact share > 0 in {(d.transient > 0).sum()}/{len(d)} splits (sign test p {stats.binomtest(int((d.transient > 0).sum()), len(d)).pvalue:.2g}); '
      f'Spearman(turb, deep_rate_calm) {stats.spearmanr(d.turb, d.deep_rate_calm)[0]:+.2f}, (turb, taker_pnl_1s) {stats.spearmanr(d.turb, d.taker_pnl_1s)[0]:+.2f}')

# ---------------------------------------------------------------- G. timestamps
print('\n===== G. TIMESTAMPS')
print(f't_us mod 1000: chi2 p (100 bins) {stats.chisquare(np.bincount((T % 1000) // 10, minlength=100)).pvalue:.3f}, '
      f'(1000 bins) {stats.chisquare(np.bincount(T % 1000, minlength=1000)).pvalue:.3f};  t_us mod 10 counts {np.bincount(T % 10).tolist()}')
