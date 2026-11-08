"""Does the market cancel flow carry S information the filter lacks? Runs r3_man's filter without trading, records F
and each cancel event (no trades, one market order gone) with its side, distance from F and order age, then regresses the
future change of F on the trailing cancel imbalance.

    uv run python strategies/peter/r3_cxsignal.py
"""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))


def main():
    from arena import load_agent
    from simulator.simulator import ExchangeSimulator, SimulationConfig
    s = load_agent(HERE / 'r3_man.py', {'trade_start_us': 10 ** 13}, 'cxs')
    live = {}  # market order id -> (side, price, t insert, F at insert)
    ev = []  # (t, F, sd, turb, kind, side, price, F0, age)  kind 0 = sample, 1 = cancel
    onb0 = s.on_order_book_change
    st = {'next': 0.0}
    def onb(snap, trades):
        onb0(snap, trades)
        f = s.f
        if f is None:
            return
        t = snap.timestamp_us * 1e-6
        F = f.mean
        cur = {}
        for side, views in ((1, snap.bids), (-1, snap.asks)):
            for v in views:
                if v.owner == 'market':
                    cur[v.order_id] = (side, v.price)
        if not trades:
            gone = [oid for oid in live if oid not in cur]
            for oid in gone:
                sd_, px, t0, F0 = live.pop(oid)
                ev.append((t, F, f.sd, f.pv, 1, sd_, px, F0, t - t0))
        else:
            for oid in [oid for oid in live if oid not in cur]:
                live.pop(oid)
        for oid, (sd_, px) in cur.items():
            if oid not in live:
                live[oid] = (sd_, px, t, F)
        if t >= st['next']:
            ev.append((t, F, f.sd, f.pv, 0, 0, 0, 0, 0))
            st['next'] = t + 0.05
    s.on_order_book_change = onb
    ExchangeSimulator(strategy=s, config=SimulationConfig()).run(Path('data/round3_public.csv'))
    E = np.array(ev)
    np.save(HERE.parents[1] / 'logs/r3man/cxsignal.npy', E)
    print('events', len(E), 'cancels', int(E[:, 4].sum()))


if __name__ == '__main__':
    main()
