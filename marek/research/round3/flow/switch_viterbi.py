"""Fine 2-state regime path: Viterbi on 5 ms bins with per-bin log-LR = sum of per-event LLRs (maker price vs touch,
taker arrival, ladder start) minus compensator. Shock windows (onset-0.5 .. onset+8 s) carry no evidence.
Iterates: learn emission laws from the current path interior, re-decode (3 passes).
-> cache/regimes_T<TEMPER>.csv (start, end, regime, dur, llr), cache/regime_fine.npy (5 ms state)."""
import numpy as np, pandas as pd
from pathlib import Path
from numba import njit
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); ins = r[r.op == 0].reset_index(drop=True)
mem = np.load(C / 'ladder_member.npy'); L = pd.read_pickle(C / 'ladders.pkl'); ev = pd.read_csv(D / 'shock_events.csv')
t = ins.t.to_numpy(); ioc = ins.ioc.to_numpy(); s = ins.side.to_numpy(); p = ins.px.to_numpy()
bb = ins.bid_before.to_numpy(); ab = ins.ask_before.to_numpy()
opp = np.where(s == 0, ab - p, p - bb); own = np.where(s == 0, bb - p, p - ab)
feat = (np.clip(np.nan_to_num(opp, nan=9), -1, 9).astype(int) + 1) * 12 + np.clip(np.nan_to_num(own, nan=9), -2, 9).astype(int) + 2
base = ~ioc & (mem[:, 0] < 0)
import sys
TEMPER = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
W = 0.005; NB = int(6000 / W)
shock = np.zeros(NB, bool)
for o in ev.onset: shock[int((o - 0.5) / W):int((o + 8) / W)] = True
state = np.repeat(np.load(C / 'hmm_joint_2.npy')[:, 1] > 0.5, int(1 / W))

@njit(cache=True)
def viterbi(e, lp_stay, lp_sw):
    n = len(e); dc = 0.0; dv = -5.0; bp = np.zeros((n, 2), np.int8)
    for i in range(n):
        # calm
        a = dc + lp_stay[0]; b = dv + lp_sw[1]
        if a >= b: nc = a; bp[i, 0] = 0
        else: nc = b; bp[i, 0] = 1
        a = dv + lp_stay[1]; b = dc + lp_sw[0]
        if a >= b: nv = a; bp[i, 1] = 1
        else: nv = b; bp[i, 1] = 0
        dc = nc; dv = nv + e[i]
    out = np.zeros(n, np.int8); out[-1] = 1 if dv > dc else 0
    for i in range(n - 1, 0, -1): out[i - 1] = bp[i, out[i]]
    return out

tk_t = t[ioc]; lad_t = L.t0.to_numpy(); mk_t = t[base]
for it in range(3):
    sec_ok = ~shock
    # interior: bins whose ±2 s neighbourhood is in one state
    k = int(2 / W); cs = np.r_[0, np.cumsum(state)]
    frac = (cs[np.minimum(np.arange(NB) + k, NB)] - cs[np.maximum(np.arange(NB) - k, 0)]) / (np.minimum(np.arange(NB) + k, NB) - np.maximum(np.arange(NB) - k, 0))
    inC = (frac == 0) & sec_ok; inV = (frac == 1) & sec_ok
    bi = (t / W).astype(int)
    hc = np.bincount(feat[base & inC[bi]], minlength=150) + 0.5; hv = np.bincount(feat[base & inV[bi]], minlength=150) + 0.5
    llr = np.log(hv / hv.sum()) - np.log(hc / hc.sum())
    Tc = inC.sum() * W; Tv = inV.sum() * W
    rates = {}
    for nm, x in [('tk', tk_t), ('lad', lad_t), ('mk', mk_t)]:
        b = (x / W).astype(int); rates[nm] = (inC[b].sum() / Tc, inV[b].sum() / Tv)
    e = np.zeros(NB)
    np.add.at(e, (mk_t / W).astype(int), TEMPER * llr[feat[base]] + np.log(rates['mk'][1] / rates['mk'][0]))
    np.add.at(e, (tk_t / W).astype(int), np.log(rates['tk'][1] / rates['tk'][0]))
    np.add.at(e, (lad_t / W).astype(int), np.log(rates['lad'][1] / rates['lad'][0]))
    e -= W * sum(v[1] - v[0] for v in rates.values())
    e[shock] = 0.0
    # transition probs per bin from current path
    sw = np.flatnonzero(np.diff(state.astype(int)))
    nCV = (np.diff(state.astype(int)) == 1).sum(); nVC = (np.diff(state.astype(int)) == -1).sum()
    pcv = W / 110.0; pvc = W / 45.0  # fixed from the 1 s HMM dwell (re-estimating them at 5 ms runs away)
    lp_stay = np.log(np.array([1 - pcv, 1 - pvc])); lp_sw = np.log(np.array([pcv, pvc]))
    state = viterbi(e, lp_stay, lp_sw).astype(bool)
    print(f'iter {it}: rates {dict((k, tuple(np.round(v, 3))) for k, v in rates.items())} switches {np.abs(np.diff(state.astype(int))).sum()} vol occ {state.mean():.3f}')
np.save(C / 'regime_fine.npy', state)
x = state.astype(int); ch = np.flatnonzero(np.diff(x)) + 1; st = np.r_[0, ch]; en = np.r_[ch, NB]
R = pd.DataFrame({'start': st * W, 'end': en * W, 'regime': np.where(x[st] == 1, 'volatile', 'calm')}); R['dur'] = R.end - R.start
# evidence strength of each segment: total LLR
cum = np.r_[0, np.cumsum(e)]; R['llr'] = [cum[b] - cum[a] for a, b in zip(st, en)]
R['in_shock'] = [shock[a:b].mean() for a, b in zip(st, en)]
R.to_csv(C / f'regimes_T{TEMPER}.csv', index=False)
pd.set_option('display.width', 200); print(R[R.regime == 'volatile'].round(2).to_string())
print(R.groupby('regime').dur.describe())
