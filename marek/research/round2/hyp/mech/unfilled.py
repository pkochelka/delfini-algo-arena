import pandas as pd, numpy as np
tk = pd.read_pickle('marek/research/round2/flow/cache/takers.pkl')
tr = pd.read_pickle('marek/research/round2/flow/cache/trades.pkl')
f = tr.groupby('taker_row').qty.sum()
tk['filled'] = f.reindex(tk.index).fillna(0).values
tk['unf'] = tk.vol - tk.filled
print('takers', len(tk), 'ioc share', tk.ioc.mean(), 'vol', tk.vol.sum(), 'filled', tk.filled.sum(), 'unfilled', tk.unf.sum())
buy = tk.side == 1  # check convention
print(tk[['side','px','bid_before','ask_before','vol','filled']].head(10))
# limit relative to the touch it faces
tk['lim_off'] = np.where(tk.side == 0, tk.px - tk.ask_before, tk.bid_before - tk.px)
tk['lim_mid'] = np.where(tk.side == 0, tk.px - tk.mid_pre, tk.mid_pre - tk.px)
print(tk.groupby(tk.lim_off.clip(-3, 6)).agg(n=('vol','size'), vol=('vol','sum'), filled=('filled','sum'), unf=('unf','sum')))
print('unfilled by hi(volatile?)', tk.groupby('hi').unf.sum(), tk.groupby('shock').unf.sum())
# unfilled remainder with its limit vs mid
u = tk[tk.unf > 0]
print(u.groupby(u.lim_mid.clip(-2, 6).round()).agg(n=('unf','size'), unf=('unf','sum')))
print('per second unfilled', tk.unf.sum()/6000)
# unfilled remainder limit vs the touch BEFORE (how many levels swept)
print(u.groupby(u.lim_off.clip(-1, 8)).agg(n=('unf','size'), unf=('unf','sum')))
