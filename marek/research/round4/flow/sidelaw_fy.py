"""Do takers compare S = F + Y or only the permanent F with the mid? Y proxy = sum of (1-p)*kick*exp(-dt/T) over past
fills (c 0.0168 calm / 0.062 vol, exponent 0.62, p 0.5, T 15 s; r3 law). Logit of side on (S - mid) and Y."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import minimize
C = Path(__file__).parent / 'cache'
tk = pd.read_parquet(C / 'takers_s.parquet')
t = tk.t.to_numpy(); s = tk.sg.to_numpy(); f = tk.filled.to_numpy().astype(float); reg = tk.reg.to_numpy()
kick = s * np.where(reg == 1, 0.062, 0.0168) * f ** 0.62
Y = np.zeros(len(t)); acc = 0.
for i in range(1, len(t)): acc = acc * np.exp(-(t[i] - t[i - 1]) / 15) + 0.5 * kick[i - 1]; Y[i] = acc
x = tk.s_hat.to_numpy() - tk.mid.to_numpy(); y = (s > 0).astype(float)
ok = ((~tk.tw) & ((tk.shock_dt < -1) | (tk.shock_dt > 8)) & (tk.s_sd < 0.5)).to_numpy() & np.isfinite(x) & (t > 60)
print('sd Y', Y[ok].std().round(3), 'sd x', x[ok].std().round(3), 'corr(x,Y)', np.corrcoef(x[ok], Y[ok])[0, 1].round(3))
for R in [0, 1, None]:
    m = ok & ((reg == R) if R is not None else True); A = np.c_[np.ones(m.sum()), x[m], Y[m]]
    for cols in [[0, 1], [0, 1, 2]]:
        B = A[:, cols]; nll = lambda b: np.sum(np.logaddexp(0, B @ b) - y[m] * (B @ b))
        res = minimize(nll, np.zeros(len(cols)), method='BFGS'); print(f'reg {R} cols {cols} LL {-res.fun:.1f} coef {np.round(res.x, 3)}')
