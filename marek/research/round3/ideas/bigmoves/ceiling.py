"""Shock ceiling: stale value at the jump, how it disappears, what an IOC sniper could take, and what r3_best took.
uv run --with pyarrow python ceiling.py NAME [SIMLOG_DIR]  -> ceiling_NAME.csv"""
import sys, numpy as np, pandas as pd
from common import *
name = sys.argv[1]; simlog = sys.argv[2] if len(sys.argv) > 2 else None
tp, o, tr = load(name); ev = tp.events
S = s_path(name); sh = shocks(name, S)
evt = ev.t_us.values * 1e-6
mk = o[~o.ioc & (o.resting_quantity > 0)]
row2i = pd.Series(np.arange(len(o)), index=o.row.values)
# fills per maker order: (t, qty, taker_row)
trm = tr.copy(); trm['t'] = trm.t_us * 1e-6
tk = o[o.ioc]
tk_by_row = tk.set_index('row')
fills = trm.groupby('maker_row')
mine = None
if simlog:
    st = pd.read_csv(pathlib.Path(simlog) / 'trades.csv')
    st = st[(st.taker_owner == 'user') | (st.maker_owner == 'user')].copy(); st['t'] = st.timestamp_us * 1e-6
    st['sgn'] = np.where(((st.taker_owner == 'user') & (st.taker_side == 'B')) | ((st.maker_owner == 'user') & (st.taker_side == 'A')), 1, -1)
    mine = st
rows = []
for k, s in enumerate(sh):
    J = s['J']; d = 1 if J > 0 else -1
    a = s['onset']; tj = s['tj']
    # S_new: synth = mid at a+2.99 + J; tapes: S path just after
    w = mk[(mk.t <= tj) & (mk.end_t > a + 2.85)]
    if 'S_new' in s:
        Sn, Sp = s['S_new'], s['S_pre']
    else:
        i = np.searchsorted(evt, a + 2.995) - 1
        Sp = (ev.bid.iloc[i] + ev.ask.iloc[i]) / 2; Sn = Sp + J
    # first informative insert after a+2.85 (direction + time)
    ins = o[(o.t > a + 2.85) & (o.t < tj + 0.3)]
    sig_t, wrong = None, 0
    for r in ins.itertuples():
        if r.ioc:
            dd = 1 if (r.side == 'B' and r.price >= Sp + 6) else -1 if (r.side == 'A' and r.price <= Sp - 6) else 0
        else:
            dd = 1 if ((r.side == 'B' and r.price >= Sp + 1.5) or (r.side == 'A' and r.price >= Sp + 10)) else \
                -1 if ((r.side == 'A' and r.price <= Sp - 1.5) or (r.side == 'B' and r.price <= Sp - 10)) else 0
        if dd == d:
            sig_t = r.t; break
        if dd == -d:
            wrong += 1
    tau = sig_t if sig_t is not None else tj
    # stale book just before tau
    side = 'A' if d > 0 else 'B'
    live = mk[(mk.side == side) & (mk.t < tau) & (mk.end_t >= tau)].copy()
    rem = []
    for r in live.itertuples():
        q = r.resting_quantity
        if r.row in fills.groups:
            f = fills.get_group(r.row); q -= f[f.t < tau].qty.sum()
        rem.append(q)
    live['q'] = rem; live = live[live.q > 0]
    live['e'] = (Sn - live.price) * d
    best = live.price.min() if d > 0 else live.price.max()
    live['vis'] = (live.price - best).abs() <= 19
    stl = live[live.e > 0]
    V_all = (stl.q * stl.e).sum(); L_all = stl.q.sum()
    sv = stl[stl.vis]; V_vis = (sv.q * sv.e).sum(); L_vis = sv.q.sum()
    # fate of each stale lot: taken by market taker vs cancelled, and when (rel tau)
    take_v = canc_v = 0.0; rem_at = {x: 0.0 for x in (0, 5, 10, 20, 50, 100, 300)}
    pieces = []  # (t_removed, qty, e, how)
    for r in stl.itertuples():
        q = r.q; e = r.e
        if r.row in fills.groups:
            f = fills.get_group(r.row); f = f[f.t >= tau]
            for g in f.itertuples():
                pieces.append((g.t - tau, g.qty, e, 'take')); q -= g.qty
        if q > 0:
            pieces.append((r.end_t - tau, q, e, 'cancel' if r.end == 'cancelled' else 'alive'))
    for t_, q, e, how in pieces:
        if how == 'take': take_v += q * e
        elif how == 'cancel' and t_ < 1.0: canc_v += q * e
        for x in rem_at:
            if t_ > x * 1e-3: rem_at[x] += q * e
    # ideal IOC arriving at tau + L ms, limit stale_ref + lim (or no limit), cap Q (first lots best edge first)
    def cap(L, Q, lim=None):
        av = [(e, q) for t_, q, e, how in pieces if t_ > L * 1e-3]
        # price-sorted: largest edge first
        av.sort(reverse=True)
        tot = 0; v = 0.0
        ref = best
        for e, q in av:
            px = Sn - e * d
            if lim is not None and (px - ref) * d > lim: continue
            z = min(q, Q - tot)
            if z <= 0: break
            tot += z; v += z * e
        return round(v), tot
    row = dict(k=k, onset=a, tau_rel_tj=round((tau - tj) * 1e3, 1), sig=sig_t is not None, wrong=wrong, J=round(J, 1),
               L_vis=int(L_vis), V_vis=round(V_vis), L_all=int(L_all), V_all=round(V_all),
               take_v=round(take_v), canc_v=round(canc_v), **{f'rem{x}': round(v) for x, v in rem_at.items()})
    for L in (5, 10, 20):
        for Q in (100, 950, 10**6):
            v, n = cap(L, Q, 8); row[f'c{L}_{Q if Q < 10**6 else "inf"}_l8'] = v
    for L in (5, 10):
        v, n = cap(L, 950); row[f'c{L}_950_nolim'] = v; row[f'n{L}_950_nolim'] = n
    if mine is not None:
        m = mine[(mine.t > tau - 0.3) & (mine.t < tau + 0.5)]
        mi = m[m.sgn == d]
        row['my_first_ms'] = round((mi.t.min() - tau) * 1e3, 1) if len(mi) else None
        row['my_q'] = int(mi.quantity.sum()); row['my_v'] = round(((Sn - mi.price) * d * mi.quantity).sum())
        mo = m[m.sgn == -d]
        row['my_q_opp'] = int(mo.quantity.sum()); row['my_v_opp'] = round(((mo.price - Sn) * d * mo.quantity).sum())
    if 'S_new' not in s:  # synth: stale value at the true jump time T and signal latency
        T = a + 3.0; row['lat_ms'] = round((tau - T) * 1e3, 1)
        lv = mk[(mk.side == side) & (mk.t < T) & (mk.end_t >= T)]
        qq = []
        for r in lv.itertuples():
            q = r.resting_quantity
            if r.row in fills.groups:
                f = fills.get_group(r.row); q -= f[f.t < T].qty.sum()
            qq.append(q)
        e = (Sn - lv.price.values) * d; qq = np.array(qq)
        row['V_T'] = round((np.clip(qq, 0, None) * np.clip(e, 0, None)).sum())
    rows.append(row)
X = pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
print(X.to_string())
print(X.drop(columns=['k', 'onset', 'tau_rel_tj', 'J', 'sig', 'my_first_ms'], errors='ignore').sum().round().to_string())
X.to_csv(pathlib.Path(__file__).parent / f'ceiling_{name}.csv', index=False)
