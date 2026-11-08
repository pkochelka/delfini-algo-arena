"""Do events inside the 3 s drought carry the sign of the coming jump? (maker prices vs S, cancels by side, taker sides)
usage: uv run --with pyarrow python drought_tell.py
"""
import numpy as np, pandas as pd
sh = pd.read_csv('marek/research/round3/s_dynamics/shocks.csv')
S = pd.read_parquet('marek/research/round3/s_dynamics/s_path.parquet', columns=['t_us', 's_hat'])
tape = pd.read_csv('data/round3.csv')
tape['t'] = tape.Time
ins = tape[tape.Operation == 'Insert'].copy(); ins['S'] = S.s_hat.values
idside = ins.set_index('OrderId')[['Side', 'Price']]
can = tape[tape.Operation == 'Cancel'].copy()
can = can.join(idside, on='OrderId', rsuffix='_o')
rows = []
for _, r in sh.iterrows():
    a, b = r.onset, r.t_jump
    w = ins[(ins.t >= a) & (ins.t < b - 0.001)]
    mk = w[w.Lifespan == 'G']; tk = w[w.Lifespan == 'F']
    s0 = ins[ins.t < a].S.iloc[-1]
    bid = mk[mk.Side == 'B']; ask = mk[mk.Side == 'A']
    cw = can[(can.t >= a) & (can.t < b - 0.001)]
    rows.append(dict(J=r.jump, mb=(bid.Price - bid.S).mean(), ma=(ask.Price - ask.S).mean(),
                     mkS=(mk.Price - s0).mean(), nb=len(bid), na=len(ask),
                     cb=(cw.Side == 'B').sum(), ca=(cw.Side == 'A').sum(),
                     tb=(tk.Side == 'B').sum(), ta=(tk.Side == 'A').sum(),
                     tlim=((tk.Price - tk.S) * np.where(tk.Side == 'B', 1, -1)).mean(),
                     dS=w.S.iloc[-1] - s0 if len(w) else 0))
d = pd.DataFrame(rows)
d['cimb'] = (d.cb - d.ca) / (d.cb + d.ca); d['timb'] = (d.tb - d.ta) / (d.tb + d.ta).clip(lower=1)
d['mimb'] = (d.nb - d.na) / (d.nb + d.na)
print(d.round(2).to_string())
for c in ('mb', 'ma', 'mkS', 'cimb', 'timb', 'mimb', 'tlim', 'dS'):
    print(c, 'corr with J: %.2f' % np.corrcoef(d[c].fillna(0), d.J)[0, 1])
