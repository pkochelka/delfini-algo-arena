"""Shock details with exact S: jump time/size, drought taker rate, burst profile/size and symmetry, first taker in jump
direction, pre-shock tells (maker/ladder/taker rate, spread, S drift in the 10 s before onset), post-shock regime.
Writes extended shock_events.csv."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); ins = r[r.op == 0].reset_index(drop=True)
S = pd.read_parquet(D.parent / 's_dynamics/s_path.parquet'); L = pd.read_pickle(C / 'ladders.pkl')
ev = pd.read_csv(D / 'shock_events.csv')[['onset', 'end', 'dur', 'maker_rate', 'gap']]; R = pd.read_csv(D / 'regimes.csv')
ti = ins.t.to_numpy(); s = S.s_hat.to_numpy(); ioc = ins.ioc.to_numpy(); side = ins.side.to_numpy()
def S_at(x): return s[np.clip(np.searchsorted(ti, x, 'right') - 1, 0, None)]
tk = ti[ioc]; tks = side[ioc]; mk = ti[~ioc]
isvol = lambda x: ((R.regime == 'volatile') & (R.start <= x) & (R.end > x)).any()
rows = []
for _, e in ev.iterrows():
    o = e.onset
    j0 = np.searchsorted(ti, o + 2.8); j1 = np.searchsorted(ti, o + 3.4); k = np.argmax(np.abs(np.diff(s[j0:j1])))
    tj = ti[j0 + k + 1]; J = S_at(o + 3.4) - S_at(o + 2.8)
    sgn = np.sign(J)
    tk_d = tk[(tk > o) & (tk < o + 3)]; tk_b = (tk > tj) & (tk < tj + 3.5)
    first_dir = tk[(tk > tj) & (np.where(tks == 0, 1, -1) == sgn)]
    pre = lambda x, a, b: ((x >= o + a) & (x < o + b)).sum() / (b - a)
    lad = L.t0.to_numpy()
    sp = (r.ask_after - r.bid_after).to_numpy(); rt = r.t.to_numpy()
    rows.append(dict(onset=round(o, 3), drought_end=round(e.end, 3), drought_dur=round(e.dur, 3), drought_maker_rate=round(e.maker_rate, 1),
                     t_jump=round(tj, 4), jump_lag=round(tj - o, 3), jump_lag_from_end=round(tj - e.end, 3), jump=round(J, 2),
                     regime_at_jump='volatile' if isvol(tj) else 'calm',
                     drought_takers_per_s=round(len(tk_d) / 3, 2), burst_takers_3p5s=int(tk_b.sum()),
                     burst_buy_share=round((tks[tk_b] == 0).mean(), 3), first_dir_taker_ms=round((first_dir[0] - tj) * 1e3, 1) if len(first_dir) else np.nan,
                     pre10_maker_rate=round(pre(mk, -10, 0), 1), pre10_ladder_rate=round(pre(lad, -10, 0), 2), pre10_taker_rate=round(pre(tk, -10, 0), 2),
                     pre1_maker_rate=round(pre(mk, -1, 0), 1),
                     pre10_spread=round(np.mean(sp[(rt > o - 10) & (rt < o)]), 3), pre10_dS=round(S_at(o) - S_at(o - 10), 2), pre60_dS=round(S_at(o) - S_at(o - 60), 2),
                     post60_dS=round(S_at(tj + 60) - S_at(tj + 0.3), 2)))
X = pd.DataFrame(rows); X.insert(1, 'gap', ev.gap.round(3).to_numpy()); pd.set_option('display.width', 250)
print(X.to_string())
print('jump lag from onset: mean %.3f sd %.3f; from drought end: mean %.3f sd %.3f' % (X.jump_lag.mean(), X.jump_lag.std(), X.jump_lag_from_end.mean(), X.jump_lag_from_end.std()))
J = X.jump.to_numpy()
J3 = pd.read_csv(D.parents[1] / 'round3/flow/shock_events.csv').jump.to_numpy()
print('KS r4 vs r3 jumps p=%.3f; r3 sd %.1f; F-test var ratio %.2f' % (stats.ks_2samp(J, J3).pvalue, J3.std(), J.var() / J3.var()))
g = ev.gap.dropna().to_numpy(); g3 = pd.read_csv(D.parents[1] / 'round3/flow/shock_events.csv').onset.diff().dropna().to_numpy()
print('gaps: min %.1f max %.1f mean %.1f; KS U(145,300) p=%.3f, U(150,300) p=%.3f; vs r3 gaps KS p=%.3f' % (g.min(), g.max(), g.mean(),
      stats.kstest(g, 'uniform', (145, 155)).pvalue, stats.kstest(g, 'uniform', (150, 150)).pvalue, stats.ks_2samp(g, g3).pvalue))
print('jump: mean %.1f sd %.1f mean|J| %.1f kurt %.2f; KS N(0,sd) p=%.3f; KS Laplace p=%.3f; r2 sd 43.5' % (J.mean(), J.std(), np.abs(J).mean(), stats.kurtosis(J),
      stats.kstest(J / J.std(), 'norm').pvalue, stats.kstest(J / np.abs(J).mean(), 'laplace').pvalue))
print('drought taker rate %.2f /s (calm baseline 7.9)' % X.drought_takers_per_s.mean())
base = np.where(X.regime_at_jump == 'volatile', 12.3, 8.2) * 3.5
X['vel_pre2'] = [S_at(o) - S_at(o - 2) for o in X.onset]
print('corr(J, S move over 2 s before onset) %.2f; |J| by regime at jump:' % np.corrcoef(X.vel_pre2, J)[0, 1], X.assign(aJ=np.abs(J)).groupby('regime_at_jump').aJ.agg(['mean', 'size']).round(1).to_dict())
print('burst extra takers in 3.5 s: mean %.1f sd %.1f; corr with |J| %.2f; buy share overall %.3f; corr(buyshare-0.5, sign J) %.2f' % (
    (X.burst_takers_3p5s - base).mean(), (X.burst_takers_3p5s - base).std(), np.corrcoef(np.abs(J), X.burst_takers_3p5s - base)[0, 1],
    X.burst_buy_share.mean(), np.corrcoef(X.burst_buy_share - 0.5, np.sign(J))[0, 1]))
print('first taker in jump direction after jump (ms): median %.1f' % X.first_dir_taker_ms.median())
for c in ['pre10_maker_rate', 'pre10_ladder_rate', 'pre10_taker_rate', 'pre1_maker_rate', 'pre10_spread', 'pre10_dS', 'pre60_dS', 'post60_dS']:
    print(f'  corr({c}, J) {np.corrcoef(X[c], J)[0, 1]:+.2f}; corr with |J| {np.corrcoef(X[c], np.abs(J))[0, 1]:+.2f}; mean {X[c].mean():.2f}')
X.to_csv(D / 'shock_events.csv', index=False)
