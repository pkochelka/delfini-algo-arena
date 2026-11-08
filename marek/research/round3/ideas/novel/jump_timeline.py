"""First events after each shock jump: who reveals the jump first, and how fast does stale liquidity disappear?
Prints, per shock: J, ms from jump to first stale-side cancel / first crossing trade / first jump-side taker,
stale quantity (null replay book) at the jump, and the share of it removed by cancels vs trades in the first 5/10/20/50 ms.
usage: uv run --with pyarrow python jump_timeline.py NULL_LOGDIR
"""
import sys, numpy as np, pandas as pd
nd = sys.argv[1]
sh = pd.read_csv('marek/research/round3/s_dynamics/shocks.csv')
ev = pd.read_csv(f'{nd}/events.csv')          # market adds/cancels as applied in the null replay
dl = pd.read_csv(f'{nd}/book_deltas.csv')
tr = pd.read_csv(f'{nd}/trades.csv')
tape = pd.read_csv('data/round3.csv', usecols=['Time', 'Operation', 'OrderId', 'Side', 'Price', 'Lifespan'])
tape['t_us'] = (tape.Time * 1e6).round().astype(np.int64)
S = pd.read_parquet('marek/research/round3/s_dynamics/s_path.parquet', columns=['t_us', 's_hat'])
out = []
for _, r in sh.iterrows():
    tj = int(r.t_jump * 1e6); J = r.jump; up = J > 0
    stale_side = 'A' if up else 'B'
    s0 = S.s_hat[S.t_us < tj].iloc[-1]
    s1 = s0 + J
    # book at jump from deltas: quantity per (side, price) of resting orders on the stale side beyond S_new -0.5 (i.e. good to hit)
    d = dl[dl.timestamp_us < tj]
    book = d.groupby(['side', 'price']).qty_diff.sum()
    bs = book[stale_side]; bs = bs[bs > 0]
    good = bs[bs.index < s1 - 0.5] if up else bs[bs.index > s1 + 0.5]
    val = ((s1 - good.index.values) * good.values).sum() if up else ((good.index.values - s1) * good.values).sum()
    w = dl[(dl.timestamp_us >= tj) & (dl.timestamp_us < tj + 200_000) & (dl.side == stale_side)]
    w = w[(w.price < s1 - 0.5) if up else (w.price > s1 + 0.5)]
    w = w[w.qty_diff < 0]
    # classify removals: trade rows at same ts & price vs cancel
    tw = tr[(tr.timestamp_us >= tj) & (tr.timestamp_us < tj + 200_000)]
    trade_keys = set(zip(tw.timestamp_us, tw.price))
    w = w.assign(kind=[('trade' if (a, b) in trade_keys else 'cancel') for a, b in zip(w.timestamp_us, w.price)])
    row = dict(J=round(J, 1), stale_q=int(good.sum()), stale_val=int(val))
    for ms in (5, 10, 20, 50, 200):
        ww = w[w.timestamp_us < tj + ms * 1000]
        row[f'c{ms}'] = int(-ww[ww.kind == 'cancel'].qty_diff.sum()); row[f't{ms}'] = int(-ww[ww.kind == 'trade'].qty_diff.sum())
    fc = w[w.kind == 'cancel'].timestamp_us.min(); ft = w[w.kind == 'trade'].timestamp_us.min()
    row['first_cancel_ms'] = round((fc - tj) / 1e3, 1) if fc == fc else None
    row['first_trade_ms'] = round((ft - tj) / 1e3, 1) if ft == ft else None
    te = tape[(tape.t_us >= tj) & (tape.t_us < tj + 50_000)].head(6)
    row['first_rows'] = ' '.join(f"{(a-tj)/1e3:.1f}{o[0]}{s}{'' if p != p else int(p - s0)}{l if l == l else ''}" for a, o, s, p, l in zip(te.t_us, te.Operation, te.Side.fillna('-'), te.Price, te.Lifespan))
    out.append(row)
d = pd.DataFrame(out)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 80)
print(d.to_string())
print(d[['stale_q', 'stale_val'] + [f'{k}{m}' for m in (5, 10, 20, 50, 200) for k in 'ct']].sum())
