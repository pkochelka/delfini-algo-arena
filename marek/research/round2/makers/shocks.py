"""Shock jumps: exact jump time vs the drought onset, the stale book left behind, how fast it goes and how.

For each shock (flow agent onsets): J = S after - S before; jump time = first maker insert priced at the new level on the
unclipped side (ask > S_before + |J|/2 for an up-jump, bid < S_before - |J|/2 for a down-jump). Stale volume = resting
orders on the wrong side of S_after by > 0.5 tick, tracked order by order from the order table (each order's
insert, price, fills, end time), so we can see whether they leave by cancel or by fill.
"""
import numpy as np, pandas as pd
from pathlib import Path
from s_load import with_s
import states

HERE = Path(__file__).parent
o = with_s()
g = o[~o.ioc]
rows = []
decay = []
for on in states.ONSETS_US:
    w = o[(o.t_us > on + 2_000_000) & (o.t_us < on + 4_000_000)]
    pre = o[(o.t_us > on + 2_500_000) & (o.t_us < on + 2_900_000)]
    post = o[(o.t_us > on + 3_200_000) & (o.t_us < on + 3_600_000)]
    S0, S1 = pre.S.median(), post.S.median()
    J = S1 - S0
    wg = w[~w.ioc]
    if J > 0:
        newlvl = wg[(wg.sg == -1) & (wg.price > S0 + abs(J) / 2)]  # unclipped side: asks at the new level
    else:
        newlvl = wg[(wg.sg == 1) & (wg.price < S0 - abs(J) / 2)]
    tj = newlvl.t_us.min() if len(newlvl) else np.nan
    # stale resting orders at tj: inserted before tj, end after tj, on the wrong side of S1 by > 0.5
    live = g[(g.t_us < tj) & (g.end_t_us.fillna(np.inf) > tj)]
    if J > 0:
        stale = live[(live.sg == -1) & (live.price < S1 - 0.5)]
        edge = S1 - stale.price
    else:
        stale = live[(live.sg == 1) & (live.price > S1 + 0.5)]
        edge = stale.price - S1
    vol = stale.resting_quantity if 'resting_quantity' in stale else stale.volume
    q = stale.volume - 0  # volume at insert; fills before tj negligible (book thin)
    end_rel = (stale.end_t_us - tj) / 1e3
    filled_by = stale.end == 'filled'
    first_taker = o[(o.t_us >= tj) & o.ioc & (o.sg == np.sign(J))].t_us.min()
    rows.append({'onset_s': on / 1e6, 'jump_after_onset_ms': (tj - on) / 1e3, 'J': J, 'n_stale': len(stale),
                 'stale_qty': int(q.sum()), 'stale_edge_qty_wtd': float((edge * q).sum() / max(q.sum(), 1)),
                 'stale_value_ticks_x_lots': float((edge * q).sum()),
                 'stale_life_ms_p50': float(end_rel.median()) if len(stale) else np.nan,
                 'stale_life_ms_max': float(end_rel.max()) if len(stale) else np.nan,
                 'share_qty_filled': float(q[filled_by].sum() / max(q.sum(), 1)),
                 'first_taker_same_dir_ms': (first_taker - tj) / 1e3})
    for _, r in stale.iterrows():
        decay.append({'end_ms': (r.end_t_us - tj) / 1e3, 'qty': r.volume, 'filled': r.end == 'filled'})
df = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(df.round(2).to_string())
df.to_csv(HERE / 'shock_stale.csv', index=False)
d = pd.DataFrame(decay)
print('stale orders total', len(d), 'qty', d.qty.sum(), 'filled share (qty)', d.qty[d.filled].sum() / d.qty.sum())
for ms in [5, 10, 20, 50, 100, 200, 500]:
    print(f'  share of stale qty still resting at +{ms} ms: {d.qty[d.end_ms > ms].sum() / d.qty.sum():.3f}')
print('jump time after onset ms: median %.1f, range %.1f..%.1f' % (df.jump_after_onset_ms.median(), df.jump_after_onset_ms.min(), df.jump_after_onset_ms.max()))
