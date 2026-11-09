"""PnL rate by |position| bucket and regime (shock windows -5..+25 s removed), time shares, and the take truncation
restricted to real takes (|pos| <= 100 at send, outside shock windows).
uv run --with pyarrow python marek/research/round4/edge/tools/pinned.py cache/r4_volsig_round4.npz"""
import sys, pathlib, numpy as np, pandas as pd
R4 = pathlib.Path(__file__).resolve().parents[2]
z = np.load(sys.argv[1])
x, curve, od = z['x'], z['curve'], z['od']
t, F, sd, pv, turb, pos, bb, ba, Y, kv, sgw = x.T
ct, cp = curve[:, 0], curve[:, 1]
on = pd.read_csv(R4 / 'flow/shock_events.csv').onset.values
dp = np.diff(cp); tc = ct[:-1]
k = np.clip(np.searchsorted(t, tc, 'right') - 1, 0, len(t) - 1)
sh = np.zeros(len(tc), bool)
for o in on:
    sh |= (tc >= o - 5) & (tc < o + 25)
ap = np.abs(pos[k]); tb = turb[k].astype(bool)
print('PnL per second by |pos| bucket (1 s curve increments, shock windows removed)')
for nm, rm in (('calm', ~tb), ('turb', tb)):
    print(f' {nm}:')
    for lo, hi in ((0, 20), (20, 50), (50, 80), (80, 95), (95, 101)):
        m = rm & ~sh & (ap >= lo) & (ap < hi)
        print(f'   |pos| {lo:3d}-{hi:3d}: {m.sum():5d} s ({m.sum()/max((rm & ~sh).sum(),1):.2f}), pnl {dp[m].sum()/1e3:+7.1f}k, per s {dp[m].sum()/max(m.sum(),1):+6.1f}')
# calm: position sign vs S drift sign over the second
sp = pd.read_parquet(R4 / 's_dynamics/s_path.parquet', columns=['t_us', 's_hat'])
st = sp.t_us.values * 1e-6; S = sp.s_hat.values
S0 = S[np.clip(np.searchsorted(st, tc, 'right') - 1, 0, len(S) - 1)]
S1 = S[np.clip(np.searchsorted(st, tc + 1, 'right') - 1, 0, len(S) - 1)]
carry = pos[k] * (S1 - S0)
for nm, rm in (('calm', ~tb), ('turb', tb)):
    m = rm & ~sh
    print(f' {nm}: carry (pos x dS over the next 1 s) sum {carry[m].sum()/1e3:+7.1f}k of pnl {dp[m].sum()/1e3:+7.1f}k; '
          f'pinned (|pos|>=95) carry {carry[m & (ap >= 95)].sum()/1e3:+7.1f}k')
# real takes
o = pd.DataFrame(od[:, :12], columns=['t', 'side', 'price', 'qty', 'ioc', 'pos', 'rr', 'rc', 'turb', 'shock', 'ss', 'stale'])
shk = np.zeros(len(o), bool)
for oo in on:
    shk |= (o.t.values >= oo - 5) & (o.t.values < oo + 25)
o['F'] = F[np.clip(np.searchsorted(t, o.t.values, 'right') - 1, 0, len(F) - 1)]
tk = o[(o.ioc == 1) & (o.ss == 0) & (o.shock == 0) & (o.pos.abs() <= 100) & ~shk].copy()
tk['room'] = 100 - tk.side * tk.pos - tk.rr - tk.rc
tk['room_free'] = 100 - tk.side * tk.pos - tk.rc
tk['extra'] = np.maximum(np.minimum(tk.stale, tk.room_free) - tk.qty, 0)
tk['edgeF'] = (tk.F - tk.price) * tk.side  # at the IOC limit: a lower bound on the fill edge
print('\nreal IOC takes:', len(tk), 'lots', int(tk.qty.sum()))
for nm, m in (('turb', tk.turb == 1), ('calm', tk.turb == 0)):
    s = tk[m]
    print(f'  {nm}: n {len(s)}, lots {int(s.qty.sum())}, mean limit edge vs F {np.average(s.edgeF, weights=s.qty):+.2f}; '
          f'room-limited {(s.qty == s.room).mean():.2f}; resting same side held room in {(s.rr > 0).mean():.2f}; '
          f'extra lots if resting freed {int(s.extra.sum())} (limit-edge-weighted {(s.extra * s.edgeF).sum():+.0f} ticks)')
    s2 = s[s.rr > 0]
    print(f'     takes with resting same-side: n {len(s2)}, lots {int(s2.qty.sum())}, rr mean {s2.rr.mean():.0f}, stale mean {s2.stale.mean():.0f}, qty mean {s2.qty.mean():.0f}')
