"""Compact per-tape caches for the diff battery: orders with pre-insert touch, cancels, 1s grid of mid."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/Users/marek/Documents/delfini-algo-arena')
from analysis.cache import load_tape
for name in sys.argv[1:]:
    t = load_tape(f'/Users/marek/Documents/delfini-algo-arena/data/{name}.csv')
    ev = t.events; o = t.orders
    t0 = int(ev.t_us.iloc[0])
    bid = ev.bid.to_numpy().astype(float); ask = ev.ask.to_numpy().astype(float)
    bid[bid <= 0] = np.nan; ask[ask <= 0] = np.nan
    bq = ev.bid_qty.to_numpy(); aq = ev.ask_qty.to_numpy()
    r = o.row.to_numpy()
    pb = np.r_[np.nan, bid[:-1]][r]; pa = np.r_[np.nan, ask[:-1]][r]
    pbq = np.r_[0, bq[:-1]][r]; paq = np.r_[0, aq[:-1]][r]
    df = pd.DataFrame(dict(row=r, t=(o.t_us.to_numpy() - t0) / 1e6, oid=o.order_id.astype('int64').to_numpy(),
        side=(o.side == 'A').to_numpy().astype(np.int8), price=o.price.to_numpy(), vol=o.volume.to_numpy(),
        ioc=o.ioc.to_numpy(), filled=o.filled.to_numpy(), end_t=(o.end_t_us.to_numpy() - t0) / 1e6,
        end=o.end.cat.codes.to_numpy(), pbid=pb, pask=pa, pbq=pbq, paq=paq, crossed=o.crossed.to_numpy(), dist=o.dist.to_numpy()))
    df.to_parquet(f'cache/{name}_orders.parquet')
    c = ev[ev.op == 'C']
    cd = pd.DataFrame(dict(row=c.index.to_numpy(), t=(c.t_us.to_numpy() - t0) / 1e6, oid=c.order_id.astype('int64').to_numpy()))
    cd.to_parquet(f'cache/{name}_cancels.parquet')
    # mid on a 0.1s grid (last state)
    tt = (ev.t_us.to_numpy() - t0) / 1e6
    grid = np.arange(0, 6000, 0.1)
    idx = np.searchsorted(tt, grid, side='right') - 1
    mid = (bid[idx] + ask[idx]) / 2
    pd.DataFrame(dict(t=grid, bid=bid[idx], ask=ask[idx], mid=mid, bq=bq[idx], aq=aq[idx])).to_parquet(f'cache/{name}_grid.parquet')
    tr = t.trades
    pd.DataFrame(dict(t=(tr.t_us.to_numpy() - t0) / 1e6, price=tr.price, qty=tr.qty, agg=(tr.aggressor == 'A').astype(np.int8), taker_row=tr.taker_row, maker_row=tr.maker_row, depth=tr.depth)).to_parquet(f'cache/{name}_trades.parquet')
    print(name, len(df), len(cd))
