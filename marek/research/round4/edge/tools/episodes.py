"""Take episodes: IOC takes on one side within 100 ms of each other. First take's stale qty / room / room_free vs the
episode's total qty: is the truncation by our own resting same-side orders recovered by follow-up takes?
uv run --with pyarrow python marek/research/round4/edge/tools/episodes.py cache/r4_volsig_round4.npz"""
import sys, pathlib, numpy as np, pandas as pd
R4 = pathlib.Path(__file__).resolve().parents[2]
z = np.load(sys.argv[1]); od = z['od']
on = pd.read_csv(R4 / 'flow/shock_events.csv').onset.values
o = pd.DataFrame(od, columns=['t', 'side', 'price', 'qty', 'ioc', 'pos', 'rr', 'rc', 'turb', 'shock', 'ss', 'stale'])
shk = np.zeros(len(o), bool)
for oo in on:
    shk |= (o.t.values >= oo - 5) & (o.t.values < oo + 25)
tk = o[(o.ioc == 1) & (o.ss == 0) & (o.shock == 0) & (o.pos.abs() <= 100) & ~shk].reset_index(drop=True)
tk['room'] = 100 - tk.side * tk.pos - tk.rr - tk.rc
tk['room_free'] = 100 - tk.side * tk.pos - tk.rc
ep = []
i = 0
while i < len(tk):
    j = i
    while j + 1 < len(tk) and tk.side[j + 1] == tk.side[i] and tk.t[j + 1] - tk.t[j] < 0.1:
        j += 1
    s = tk.iloc[i:j + 1]
    ep.append(dict(t=tk.t[i], turb=tk.turb[i], n=j - i + 1, dur=tk.t[j] - tk.t[i], stale0=tk.stale[i], room0=tk.room[i],
                   free0=tk.room_free[i], rr0=tk.rr[i], qty0=tk.qty[i], qty=s.qty.sum(), stale_max=s.stale.max(),
                   lim0=tk.room[i] == tk.qty[i]))
    i = j + 1
e = pd.DataFrame(ep)
e['want0'] = np.minimum(e.stale0, e.free0)
e['short'] = np.maximum(e.want0 - e.qty, 0)
e['short_rr'] = np.minimum(e.short, e.rr0)  # part attributable to the resting order (room held by rr0)
for nm, m in (('turb', e.turb == 1), ('calm', e.turb == 0)):
    s = e[m]; h = s[s.rr0 > 0]
    print(f'{nm}: episodes {len(s)}, lots {int(s.qty.sum())}, single-take {int((s.n == 1).sum())}, mean n {s.n.mean():.2f}, mean dur {s.dur.mean()*1e3:.1f} ms')
    print(f'   first take room-limited {s.lim0.mean():.2f}; episodes with rr0>0: {len(h)} (lots {int(h.qty.sum())}), '
          f'of which first take room-limited {h.lim0.mean():.2f}; want0 (min(stale0, room without rr)) {int(h.want0.sum())}, '
          f'got {int(h.qty.sum())}, shortfall {int(h.short.sum())}, shortfall attributable to rr {int(h.short_rr.sum())}')
    hh = h[h.lim0]
    print(f'   rr0>0 & room-limited first take: n {len(hh)}, qty0 mean {hh.qty0.mean():.0f}, episode qty mean {hh.qty.mean():.0f}, '
          f'stale0 mean {hh.stale0.mean():.0f}, free0 mean {hh.free0.mean():.0f}, rr0 mean {hh.rr0.mean():.0f}, n mean {hh.n.mean():.2f}, '
          f'shortfall_rr {int(hh.short_rr.sum())}')
