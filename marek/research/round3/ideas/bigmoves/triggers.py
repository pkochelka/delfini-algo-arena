"""Per shock: times (ms rel. s_dyn tj) of candidate jump signals, value an IOC (cap 950, limit stale touch+8) would get
if it arrived 5 ms after each, and when r3_best fired. uv run --with pyarrow python triggers.py NAME [SIMLOG]"""
import sys, numpy as np, pandas as pd
from common import *
name = sys.argv[1]; simlog = sys.argv[2] if len(sys.argv) > 2 else None
tp, o, tr = load(name); ev = tp.events
S = s_path(name); sh = shocks(name, S)
evt = ev.t_us.values * 1e-6
mk = o[~o.ioc & (o.resting_quantity > 0)]
fills = tr.copy(); fills['t'] = fills.t_us * 1e-6
fg = fills.groupby('maker_row')
oid = o.set_index('order_id')
fire = None
if simlog:
    e = pd.read_csv(pathlib.Path(simlog) / 'events.csv')
    e = e[(e.source == 'user') & (e.operation == 'add_sent') & (e.quantity >= 300)]
    fire = (e.timestamp_us.values * 1e-6, np.where(e.side.values == 'B', 1, -1))
rows = []
for k, s in enumerate(sh):
    J = s['J']; d = 1 if J > 0 else -1; a = s['onset']; tj = s['tj']
    if 'S_new' in s: Sn, Sp = s['S_new'], s['S_pre']
    else:
        i = np.searchsorted(evt, a + 2.995) - 1; Sp = (ev.bid.iloc[i] + ev.ask.iloc[i]) / 2; Sn = Sp + J
    if abs(J) < 4: continue
    i0 = np.searchsorted(evt, a + 2.85); i1 = np.searchsorted(evt, tj + 0.3)
    sig = {}
    def put(key, t):
        if key not in sig: sig[key] = t
    for i in range(i0, i1):
        r = ev.iloc[i]; t = evt[i]
        bb, ba = ev.bid.iloc[i - 1], ev.ask.iloc[i - 1]
        if r.op == 'C':
            if r.order_id not in oid.index: continue
            q = oid.loc[r.order_id]
            vis = (q.side == 'B' and q.price >= bb - 19) or (q.side == 'A' and q.price <= ba + 19)
            if not vis: put('invisible', t); continue
            st = (q.side == 'A' and d > 0 and q.price < Sn) or (q.side == 'B' and d < 0 and q.price > Sn)
            if st: put('stale_cancel', t)
            else: put('wrong_cancel', t)
            continue
        px = r.price; sd = r.side
        if r.lifespan == 'G':
            vis = (sd == 'B' and px >= bb - 19) or (sd == 'A' and px <= ba + 19)
            if not vis: put('invisible', t); continue
            if ba - bb >= 3 and ((sd == 'B' and px == ba - 1 and d > 0) or (sd == 'A' and px == bb + 1 and d < 0)):
                put('clip', t)
            if ba - bb >= 3 and ((sd == 'B' and px == ba - 1 and d < 0) or (sd == 'A' and px == bb + 1 and d > 0)):
                put('wrong_clip', t)
        else:
            n = tr[tr.taker_row == r.name] if False else None
            # levels swept by this taker
            if (sd == 'B' and d > 0 and px >= ba + 1) or (sd == 'A' and d < 0 and px <= bb - 1):
                put('sweep2', t)
            if (sd == 'B' and px >= Sp + 6 and d > 0) or (sd == 'A' and px <= Sp - 6 and d < 0):
                put('far_taker', t)
    # stale pieces at a+2.85 (fate)
    side = 'A' if d > 0 else 'B'
    live = mk[(mk.side == side) & (mk.t < tj + 0.3) & (mk.end_t >= a + 2.85)]
    pieces = []
    for r in live.itertuples():
        e_ = (Sn - r.price) * d
        if e_ <= 0: continue
        q = r.resting_quantity
        if r.row in fg.groups:
            for g in fg.get_group(r.row).itertuples():
                pieces.append((r.t, g.t, g.qty, e_, r.price)); q -= g.qty
        if q > 0: pieces.append((r.t, r.end_t, q, e_, r.price))
    def val(t):  # IOC arriving at t: cap 950, limit stale touch at decision time + 8
        av = [(e_, q, p) for (t0, t1, q, e_, p) in pieces if t0 < t - 0.005 and t1 > t]
        if not av: return 0
        ref = min(p for _, _, p in av) if d > 0 else max(p for _, _, p in av)
        av = sorted((x for x in av if abs(x[2] - ref) <= 8), reverse=True)
        tot = v = 0
        for e_, q, p in av:
            z = min(q, 950 - tot); tot += z; v += z * e_
            if tot >= 950: break
        return round(v)
    row = dict(k=k, J=round(J, 1))
    for key in ('invisible', 'clip', 'stale_cancel', 'sweep2', 'far_taker', 'wrong_clip', 'wrong_cancel'):
        t = sig.get(key)
        row[key] = round((t - tj) * 1e3, 1) if t is not None else None
        if key in ('invisible', 'clip', 'stale_cancel', 'sweep2'):
            row['v_' + key] = val(t + 0.005) if t is not None else 0
    good = [sig[x] for x in ('clip', 'stale_cancel', 'sweep2', 'far_taker') if x in sig]
    row['v_best'] = max([val(t + 0.005) for t in good] + [0])
    if fire is not None:
        m = (fire[0] > a + 2.5) & (fire[0] < tj + 0.5)
        if m.any():
            t = fire[0][m][0]; row['fire'] = round((t - tj) * 1e3, 1); row['fire_dir_ok'] = int(fire[1][m][0] == d); row['v_fire'] = val(t + 0.005)
    rows.append(row)
X = pd.DataFrame(rows); pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(X.to_string())
print(X[[c for c in X.columns if c.startswith('v_')]].sum().to_string())
X.to_csv(pathlib.Path(__file__).parent / f'triggers_{name}.csv', index=False)
