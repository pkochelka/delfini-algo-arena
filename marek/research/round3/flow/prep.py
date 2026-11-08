"""Build a compact per-insert table for round3 flow analysis -> flow/cache/inserts.parquet."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from analysis.cache import load_tape

OUT = Path(__file__).parent / 'cache'; OUT.mkdir(exist_ok=True)
tape = load_tape('data/round3.csv')
print(tape.events.columns.tolist()); print(tape.orders.columns.tolist()); print(tape.trades.columns.tolist())
raw = pd.read_csv('data/round3.csv')
ev = tape.events
ins = raw['Operation'].eq('Insert').to_numpy()
df = pd.DataFrame({'t': raw['Time'].to_numpy(), 'op': raw['Operation'].map({'Insert': 0, 'Cancel': 1}).to_numpy(),
                   'oid': raw['OrderId'].to_numpy(), 'side': raw['Side'].map({'B': 0, 'A': 1}).to_numpy(),
                   'vol': raw['Volume'].to_numpy(), 'px': raw['Price'].to_numpy(),
                   'ioc': raw['Lifespan'].eq('F').to_numpy()})
# book before each row = after previous row
bid = ev['bid'].to_numpy().astype(float); ask = ev['ask'].to_numpy().astype(float)
bid[bid <= 0] = np.nan; ask[ask <= 0] = np.nan
df['bid_before'] = np.r_[np.nan, bid[:-1]]; df['ask_before'] = np.r_[np.nan, ask[:-1]]
df['bid_after'] = bid; df['ask_after'] = ask
for c in ['bid_qty', 'ask_qty']:
    v = ev[c].to_numpy(); df[c + '_before'] = np.r_[0, v[:-1]]
o = tape.orders
print(o.head())
df.to_pickle(OUT / 'rows.pkl')
o.to_pickle(OUT / 'orders.pkl'); tape.trades.to_pickle(OUT / 'trades.pkl')
