"""Carry decomposition: sum pos(t) * dS over the 10 ms grid by regime, and pinned share; our F log + s_path."""
import sys, numpy as np, pandas as pd, pathlib
R4 = pathlib.Path(__file__).resolve().parents[2]
x = np.load(sys.argv[1])['x']; tape = sys.argv[2] if len(sys.argv) > 2 else 'round4'
t, F, sd, pv, turb, pos, bb, ba = x.T
R = R4 if tape == 'round4' else R4.parent / 'round3'
sp = pd.read_parquet(R / 's_dynamics/s_path.parquet', columns=['t_us', 's_hat'])
st = sp.t_us.values * 1e-6; sh = sp.s_hat.values
S = sh[np.clip(np.searchsorted(st, t, side='right') - 1, 0, len(sh) - 1)]
rel = t - t[0]
reg = pd.read_csv(R / 'flow/regimes_1s.csv'); shk = pd.read_csv(R / 'flow/shock_events.csv')
vol = np.zeros(len(t), bool)
for a, b, g in reg[['start', 'end', 'regime']].values:
    if g == 'volatile': vol |= (rel >= a) & (rel < b)
sh_ = np.zeros(len(t), bool)
for o in shk.onset.values: sh_ |= (rel >= o - 1) & (rel < o + 20)
dS = np.r_[np.diff(S), 0]
for nm, m in [('calm', ~vol & ~sh_), ('vol', vol & ~sh_), ('shock', sh_)]:
    c = np.sum(pos[m] * dS[m])
    print(f'{nm:6s} carry(pos*dS) {c/1e3:8.1f}k  pinned |pos|>=95 {np.mean(np.abs(pos[m])>=95):.2f}  mean|pos| {np.mean(np.abs(pos[m])):.0f}')
# carry by horizon after a sign change of position in vol
for h in (0.5, 1, 2, 5, 10, 20):
    k = int(h / 0.01)
    fut = np.r_[S[k:] - S[:-k], np.zeros(k)]
    m = vol & ~sh_ & (np.abs(pos) >= 50)
    print(f'vol |pos|>=50: E[sign(pos) * dS(+{h}s)] = {np.mean(np.sign(pos[m]) * fut[m]):+.3f}')
# forward carry by age of the current position sign (time since sign(pos) last changed), vol, |pos|>=50
sg = np.sign(pos)
chg = np.r_[True, sg[1:] != sg[:-1]]
last = np.maximum.accumulate(np.where(chg, np.arange(len(t)), 0))
age = rel - rel[last]
for hh in (1, 5):
    k = int(hh / 0.01); fut = np.r_[S[k:] - S[:-k], np.zeros(k)]
    m0 = vol & ~sh_ & (np.abs(pos) >= 50)
    out = []
    for a0, a1 in [(0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 12), (12, 20), (20, 40), (40, 1e9)]:
        m = m0 & (age >= a0) & (age < a1)
        out.append(f'{a0}-{a1}:{np.mean(sg[m]*fut[m]):+.2f}({m.mean()*100:.0f}%)')
    print(f'fwd {hh}s by age:', ' '.join(out))
# same with "age" = time since last change of sign of (F - EMA_1s): momentum age
