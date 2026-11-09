"""Momentum as an increment over the filter's own state. Target S(t+h) - F(t) (S = s_path s_hat), sampled every 0.25 s,
half-split OOS R^2. Base regressors: kv, F - mid, Y. Added: tape-only momentum tells (signed taker flow over 0.25 / 0.5 /
1 / 2 s, in lots and in impact units q^0.64, trade-count imbalance), past mid and F moves (0.5 / 1 / 2 / 5 s).
Also: how the position sign lines up with the next move, and whether the tells predict the move when the position
is against it (that is where flip timing could gain).
uv run --with pyarrow python marek/research/round4/edge/tools/mom.py cache/r4_volsig_round4.npz"""
import sys, pathlib, numpy as np, pandas as pd
R4 = pathlib.Path(__file__).resolve().parents[2]
z = np.load(sys.argv[1])
x, tk = z['x'], z['tk']
t, F, sd, pv, turb, pos, bb, ba, Y, kv, sgw = x.T
g = np.arange(30, 5970, 0.01)
i = np.clip(np.searchsorted(t, g, 'right') - 1, 0, len(t) - 1)
F, Y, kv, turb, pos, mid, sgw = F[i], Y[i], kv[i], turb[i].astype(bool), pos[i], 0.5 * (bb[i] + ba[i]), sgw[i]
sp = pd.read_parquet(R4 / 's_dynamics/s_path.parquet', columns=['t_us', 's_hat'])
st = sp.t_us.values * 1e-6
S = sp.s_hat.values[np.clip(np.searchsorted(st, g, 'right') - 1, 0, len(st) - 1)]
on = pd.read_csv(R4 / 'flow/shock_events.csv').onset.values
sh = np.zeros(len(g), bool)
for o in on:
    sh |= (g >= o - 5) & (g < o + 25)
# taker flow on the grid: cumulative signed lots / impact units / counts
tt, tsg, tq = tk[:, 0], tk[:, 1], tk[:, 2]
j = np.clip(np.searchsorted(g, tt, 'right') - 1, 0, len(g) - 1)
def cum(v):
    c = np.zeros(len(g)); np.add.at(c, j, v); return np.cumsum(c)
C_q, C_i, C_n = cum(tsg * tq), cum(tsg * tq ** 0.64), cum(tsg)
idx = np.arange(2000, len(g) - 2000, 25)
def win(C, h):
    return C[idx] - C[idx - int(h * 100)]
def lag(X, h):
    return X[idx] - X[idx - int(h * 100)]
half = idx < len(g) // 2
def fit(X, y, m):
    X = np.column_stack([np.ones(len(y)), X]); out = []
    for tr in (half, ~half):
        a, b = m & tr, m & ~tr
        be = np.linalg.lstsq(X[a], y[a], rcond=None)[0]; r = y[b] - X[b] @ be; out.append(1 - r.var() / y[b].var())
    return np.mean(out), np.linalg.lstsq(X[m], y[m], rcond=None)[0][1:]
G = F[idx] - mid[idx]
base = [kv[idx], G, Y[idx]]
flow_q = [win(C_q, h) for h in (0.25, 0.5, 1, 2)]
flow_i = [win(C_i, h) for h in (0.25, 0.5, 1, 2)]
flow_n = [win(C_n, h) for h in (0.25, 0.5, 1, 2)]
pmid = [lag(mid, h) for h in (0.5, 1, 2, 5)]
pF = [lag(F, h) for h in (0.5, 1, 2, 5)]
for nm, rm in (('turb', turb[idx]), ('calm', ~turb[idx])):
    m = rm & ~sh[idx] & (sgw[idx] == 0)
    print(f'\n== {nm} n={m.sum()}  target S(t+h)-F(t)')
    for h in (0.5, 1, 2, 5):
        y = S[idx + int(h * 100)] - F[idx]
        r0, b0 = fit(np.column_stack(base), y, m)
        rq, _ = fit(np.column_stack(base + flow_q), y, m)
        ri, bi = fit(np.column_stack(base + flow_i), y, m)
        rn, _ = fit(np.column_stack(base + flow_n), y, m)
        rp, bp = fit(np.column_stack(base + pmid), y, m)
        rf, _ = fit(np.column_stack(base + pF), y, m)
        ra, _ = fit(np.column_stack(base + flow_i + pmid), y, m)
        rfo, _ = fit(np.column_stack(flow_i + pmid), y, m)
        print(f'h={h:>3} base {r0:.3f} [kv {b0[0]:.2f} F-mid {b0[1]:.2f} Y {b0[2]:.3f}] +flowq {rq:.3f} +flowi {ri:.3f} '
              f'+flown {rn:.3f} +pmid {rp:.3f} +pF {rf:.3f} +flowi+pmid {ra:.3f} | flow+pmid only {rfo:.3f} | '
              f'flowi b ' + ' '.join(f'{v:.3f}' for v in bi[3:]) + ' pmid b ' + ' '.join(f'{v:.2f}' for v in bp[3:]))
    # position alignment with the next move
    for h in (1, 2, 5):
        dS = S[idx + int(h * 100)] - S[idx]
        mm = m & (np.abs(pos[idx]) >= 50)
        al = np.sign(pos[idx][mm]) * dS[mm]
        print(f'  |pos|>=50 share {mm.sum()/max(m.sum(),1):.2f}; h={h}: mean carry/lot {al.mean():+.3f}, aligned '
              f'{(al > 0).mean():.2f}; E|dS| {np.abs(dS[mm]).mean():.2f}; perfect-sign capture ratio {al.mean()/np.abs(dS[mm]).mean():.2f}')
    # when the position is against kv / against flow: what happens next
    dS1 = S[idx + 100] - S[idx]
    for nm2, sig in (('kv', kv[idx]), ('flow_i 0.5', flow_i[1]), ('pmid 1', pmid[1]), ('F-mid', G)):
        mm = m & (np.abs(pos[idx]) >= 50)
        ag = mm & (np.sign(pos[idx]) * sig < 0)
        wi = mm & (np.sign(pos[idx]) * sig > 0)
        print(f'  pos against {nm2:10s}: share {ag.sum()/max(mm.sum(),1):.2f}, next-1s carry/lot {np.mean(np.sign(pos[idx][ag]) * dS1[ag]):+.3f}; '
              f'with: {np.mean(np.sign(pos[idx][wi]) * dS1[wi]):+.3f}')
