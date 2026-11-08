"""Event study: exact S (s_hat) around tape takers, per unit q^0.61, signed by taker side.
Is there a predictable post-trade drift (delayed impact) a 5 ms-latency trader can ride?
usage: uv run --with pyarrow python impact_profile.py
"""
import numpy as np, pandas as pd
S = pd.read_parquet('marek/research/round3/s_dynamics/s_path.parquet', columns=['t_us', 's_hat', 'vol', 'shock'])
tape = pd.read_csv('data/round3.csv', usecols=['Time', 'Operation', 'Side', 'Volume', 'Lifespan'])
ins = tape[tape.Operation == 'Insert'].reset_index(drop=True)
assert len(ins) == len(S)
st = S.t_us.values; sh = S.s_hat.values
tk = np.where(ins.Lifespan.values == 'F')[0]
sgn = np.where(ins.Side.values[tk] == 'B', 1.0, -1.0); q = ins.Volume.values[tk].astype(float)
vol = S.vol.values[tk]; shk = S.shock.values[tk]
t0 = st[tk]
pre = sh[tk - 1]                      # S at the previous insert (before this taker)
def s_at(t):
    return sh[np.clip(np.searchsorted(st, t, side='right') - 1, 0, len(sh) - 1)]
taus = [0, 5e3, 20e3, 50e3, 100e3, 200e3, 500e3, 1e6, 2e6, 5e6, 10e6, 20e6, 30e6, 60e6]
x = q ** 0.61
for name, m in (('calm', (~vol) & (~shk)), ('vol', vol & (~shk))):
    row = []
    for tau in taus:
        d = sgn * (s_at(t0 + tau) - pre) if tau > 0 else sgn * (sh[tk] - pre)
        b = (d[m] * x[m]).sum() / (x[m] ** 2).sum()   # regression on q^0.61 through origin
        row.append(f'{b:.4f}')
    print(name, m.sum(), ' '.join(f'{int(t/1e3)}ms:{r}' for t, r in zip(taus, row)))
# by size bucket, vol: mean signed move in ticks
for name, m in (('calm', (~vol) & (~shk)), ('vol', vol & (~shk))):
    for lo, hi in ((0, 50), (50, 200), (200, 600), (600, 1e9)):
        mm = m & (q >= lo) & (q < hi)
        out = [np.mean(sgn[mm] * (s_at(t0[mm] + tau) - pre[mm])) for tau in (5e3, 200e3, 1e6, 2e6, 10e6, 30e6)]
        print(name, f'q[{lo},{hi})', mm.sum(), ' '.join(f'{v:.2f}' for v in out))
