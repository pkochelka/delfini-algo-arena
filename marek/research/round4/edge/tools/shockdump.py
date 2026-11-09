"""Order-by-order dump of a logged run (logx npz) around chosen shocks: what the sniper and the unwind did.
uv run --with pyarrow python marek/research/round4/edge/tools/shockdump.py cache/a.npz 0 6 23 [--w0 -0.1 --w1 1.0]"""
import argparse, pathlib, numpy as np, pandas as pd
R4 = pathlib.Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(); ap.add_argument('npz'); ap.add_argument('ids', nargs='*', type=int)
ap.add_argument('--w0', type=float, default=-0.1); ap.add_argument('--w1', type=float, default=1.0)
ap.add_argument('--max', type=int, default=40)
a = ap.parse_args()
ev = pd.read_csv(R4 / 'flow/shock_events.csv')
z = np.load(a.npz)
od = pd.DataFrame(z['od'][:, :14], columns=['t', 'side', 'price', 'qty', 'ioc', 'pos', 'rr', 'rc', 'turb', 'shock', 'ss', 'stale', 'F', 'sd'])
fl = pd.DataFrame(z['fl'][:, :7], columns=['t', 'side', 'price', 'qty', 'ioc', 'pos', 'F'])
sp = pd.read_parquet(R4 / 's_dynamics/s_path.parquet', columns=['t_us', 's_hat'])
st = sp.t_us.values * 1e-6; S = sp.s_hat.values
ct, cp = z['curve'][:, 0], z['curve'][:, 1]
for i in a.ids:
    r = ev.iloc[i]; tj = r.t_jump
    s_after = S[min(np.searchsorted(st, tj + 1.0), len(S) - 1)]
    print(f'\n=== shock {i}: jump {r.jump:+.0f}, onset {r.onset:.3f}, t_jump {tj:.3f}, S(+1 s) {s_after:.1f}; '
          f'pnl [-1, +20 s] {(cp[min(np.searchsorted(ct, r.onset + 20), len(cp) - 1)] - cp[max(np.searchsorted(ct, r.onset - 1) - 1, 0)]) / 1e3:+.1f}k')
    o = od[(od.t >= tj + a.w0) & (od.t <= tj + a.w1) & (od.ioc == 1)]
    f = fl[(fl.t >= tj + a.w0) & (fl.t <= tj + a.w1) & (fl.ioc == 1)]
    print(f'  IOC orders {len(o)} lots {int(o.qty.sum())}; IOC fills {len(f)} lots {int(f.qty.sum())}; '
          f'max |pos| after fill {int((f.pos + f.side * f.qty).abs().max()) if len(f) else 0}')
    for _, x in o.head(a.max).iterrows():
        fx = f[(f.t >= x.t) & (f.t <= x.t + 0.02) & (f.side == x.side)]
        S_now = S[min(np.searchsorted(st, x.t), len(S) - 1)]
        print(f'  {x.t - tj:+7.3f}s {"B" if x.side > 0 else "S"} lim {int(x.price):6d} q {int(x.qty):4d} pos {int(x.pos):5d} '
              f'ss {int(x.ss):+d} shk {int(x.shock)} stale {int(x.stale):4d} F {x.F:8.1f} S {S_now:8.1f} '
              f'-> filled {int(fx.qty.sum()):4d} @ {np.average(fx.price, weights=fx.qty) if len(fx) else float("nan"):8.1f}')
