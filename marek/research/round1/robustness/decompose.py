"""Decompose a full-logs simulator run's PnL into spread capture and inventory carry.

Usage: uv run python research/round1/robustness/decompose.py logs/r1_ladder_mm data/round1.csv

spread   = sum over our fills of side * (mid just before the fill - fill price) * qty
carry    = integral of inventory * d(mid)   (cumulated at each mid change)
  drift  = integral of inventory * mu dt    with mu = (last mid - first mid) / duration (linear trend)
  resid  = carry - drift
end gap  = markout(final inv) - final inv * last mid   (liquidation cost vs mid, small)
Also maker markouts of our fills at several horizons (ticks per lot, positive = good for us),
and PnL by sign of inventory.
"""
import sys

import numpy as np
import pandas as pd

from analysis import features as F
from analysis import load_tape

run, tape_path = sys.argv[1], sys.argv[2]
tape = load_tape(tape_path, verbose=False)
q = tape.quotes
q = q[q['mid'].notna()]
t0 = tape.start_us
tr = pd.read_csv(f'{run}/trades.csv', dtype={'taker_order_id': str, 'maker_order_id': str})
mine = tr[(tr.maker_owner == 'user') | (tr.taker_owner == 'user')].copy()
user_is_maker = mine.maker_owner == 'user'
# taker_side B = buyer aggressor; if we are maker we are the opposite side
taker_buy = mine.taker_side == 'B'
mine['s'] = np.where(user_is_maker, np.where(taker_buy, -1, 1), np.where(taker_buy, 1, -1))
mine['t_s'] = (mine.timestamp_us - t0) / 1e6
mine['mid0'] = F.asof(tape, mine.t_s.to_numpy(), strict=True)
mine['spread_pnl'] = mine.s * (mine.mid0 - mine.price) * mine.quantity
mine['dq'] = mine.s * mine.quantity
print(f'fills {len(mine)}  maker share {user_is_maker.mean():.3f}  volume {mine.quantity.sum()}')
print(f'spread capture vs mid: {mine.spread_pnl.sum():,.0f}  ({mine.spread_pnl.sum()/mine.quantity.sum():.3f} ticks/lot)')
for h in (0.1, 1, 10, 60, 300):
    m = F.asof(tape, mine.t_s.to_numpy() + h)
    mk = mine.s * (m - mine.price) * mine.quantity
    print(f'  markout {h:>5}s: {np.nansum(mk)/mine.quantity.sum():+.3f} ticks/lot  (total {np.nansum(mk):,.0f})')

# inventory path on the quote timeline
qt = q['t_s'].to_numpy()
mid = q['mid'].to_numpy()
# the fill happens at the event that moves the tape quote from idx-1 to idx; mid0 is mid[idx-1], so the
# inventory must already be held over that step (identity: cash + inv*mid_T = spread + carry)
fill_idx = np.searchsorted(qt, mine.t_s.to_numpy(), side='left') - 1
inv_change = np.zeros(len(qt))
np.add.at(inv_change, np.clip(fill_idx, 0, None), mine.dq.to_numpy())
inv = np.cumsum(inv_change)  # inventory held after quote i
dmid = np.diff(mid)
carry_steps = inv[:-1] * dmid
carry = carry_steps.sum()
T = qt[-1] - qt[0]
mu = (mid[-1] - mid[0]) / T
dt = np.diff(qt)
drift = (inv[:-1] * mu * dt).sum()
print(f'carry (inv*dmid): {carry:,.0f}   of which linear drift {drift:,.0f}  residual {carry-drift:,.0f}')
print(f'mid {mid[0]} -> {mid[-1]}, mu={mu*1000:.2f} ticks/1000s; time-avg inv {np.sum(inv[:-1]*dt)/T:+.1f}')
print(f'share of time inv>=90: {np.sum(dt[inv[:-1]>=90])/T:.3f}  <=-90: {np.sum(dt[inv[:-1]<=-90])/T:.3f}')
print(f'spread + carry = {mine.spread_pnl.sum()+carry:,.0f}  (final inv {inv[-1]})')
# carry when long vs short
print(f'carry while long {carry_steps[inv[:-1]>0].sum():,.0f}  while short {carry_steps[inv[:-1]<0].sum():,.0f}')
# per-1000s buckets
b = (qt[:-1] // 1000).astype(int)
sp_b = pd.Series(mine.spread_pnl.to_numpy()).groupby((mine.t_s.to_numpy() // 1000).astype(int)).sum()
cb = pd.Series(carry_steps).groupby(b).sum()
mb = pd.Series(dmid).groupby(b).sum()
print(pd.DataFrame({'spread': sp_b, 'carry': cb, 'dmid': mb}).round(0).to_string())
