"""Run s_ladder on the full round1 tape with overrides and split its fills by kind and distance from S.

    uv run python marek/research/round1/s_ladder/decompose.py '{"s_pull":0.35}' [out_dir]

Edge per lot is measured against the tape's exact S right after the fill (data/.cache/round1_S.npz,
from build_s_oracle.py). Sum of edges + inventory term = PnL.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'strategies' / 'peter'))
from arena import load_agent  # noqa
from simulator.main import MAX_POSITION, MAX_POTENTIAL_POSITION, ORDER_ACCEPT_DELAY_MS, POSITION_BREACH_PENALTY_PER_SECOND
from simulator.simulator import ExchangeSimulator, SimulationConfig

ov = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/tmp/s_dec')
out.mkdir(parents=True, exist_ok=True)
tape = Path(ov.pop('_tape', str(ROOT / 'data/round1.csv')))
strat = load_agent(ROOT / 'strategies/marek/s_ladder.py', ov, 'dec')
cfg = SimulationConfig(max_position=MAX_POSITION, max_potential_position=MAX_POTENTIAL_POSITION,
                       position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND,
                       order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS, trade_log_path=out / 'trades.csv')
r = ExchangeSimulator(strategy=strat, config=cfg).run(tape)
print('pnl', r.pnl, 'fills', r.fills, 'vol', r.traded_volume, 'breach', r.breach_penalty, 'q50', r.abs_inventory_q50)
z = np.load(ROOT / 'data/.cache/round1_S.npz'); jt, js = z['t'], z['S']
def S(t, side):
    k = np.searchsorted(jt, t, side) - 1
    return np.where(k >= 0, js[np.maximum(k, 0)], 10000.0)
tr = pd.read_csv(out / 'trades.csv', low_memory=False)
me = tr[(tr.maker_owner == 'user') | (tr.taker_owner == 'user')].copy()
me['kind'] = np.where(me.taker_owner == 'user', 'take', 'make')
# our side: maker on bid / taker selling -> we bought?  taker_side is the aggressor side
buy = np.where(me.kind == 'make', me.taker_side == 'A', me.taker_side == 'B')
me['sgn'] = np.where(buy, 1, -1)
t = me.timestamp_us.to_numpy()
Sb, Sa = S(t, 'left'), S(t, 'right')
me['dist'] = me.sgn * (Sb - me.price)  # >0: bought below S / sold above S
me['edge'] = me.sgn * (Sa - me.price) * me.quantity
me['e1'] = me.sgn * (S(t + 1_000_000, 'right') - me.price) * me.quantity
me['b'] = pd.cut(me.dist, [-9, -0.5, 0, 0.25, 0.35, 0.5, 0.75, 1, 1.25, 1.5, 9])
tb = me.groupby(['kind', 'b'], observed=True).agg(lots=('quantity', 'sum'), edge=('edge', 'sum'), e1=('e1', 'sum'))
tb['per_lot'] = tb.edge / tb.lots
pd.set_option('display.width', 200)
print(tb.round(2))
print(me.groupby('kind')[['quantity', 'edge', 'e1']].sum())
