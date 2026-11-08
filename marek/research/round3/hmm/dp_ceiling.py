"""Path-consistent take-only ceiling: DP over stale events with a +-cap position, edge vs S at arrival, no carry.
Shows how much better a room-aware take policy could be than 'take everything above threshold'."""
import numpy as np, pandas as pd, sys
C = 'marek/research/round3/ideas/ceiling/_cache/'
st = pd.read_parquet(C + 'ceil_stale.parquet'); ev = pd.read_parquet(C + 'ceil_events.parquet')
cap = int(sys.argv[1]) if len(sys.argv) > 1 else 100
st = st[st.edge_arr > 0]
g = st.groupby(['t_us', 'side'])
for reg in ['calm', 'volatile', 'shock', 'all']:
    e = ev if reg == 'all' else ev[ev.reg == reg]
    keys = list(zip(e.t_us, e.side))
    N = 2 * cap + 1
    V = np.zeros(N)  # value-to-go by position index (pos = i - cap), processed backward
    pos = np.arange(N) - cap
    tot_greedy = 0.0
    curves = []
    for k in keys:
        try:
            d = g.get_group(k)
        except KeyError:
            curves.append(None); continue
        ed = np.repeat(d.edge_arr.values, d.rem_arr.values.astype(int))
        ed = np.sort(ed)[::-1][:N]
        cum = np.concatenate([[0], np.cumsum(ed)])  # cum[q] = value of taking q lots
        curves.append((k[1], cum))
    for c in reversed(curves):
        if c is None: continue
        side, cum = c
        nq = len(cum) - 1
        newV = V.copy()
        # buy event (stale asks, side 'A'): pos -> pos + q
        for q in range(1, nq + 1):
            if side == 'A':
                cand = np.full(N, -np.inf); cand[:N - q] = cum[q] + V[q:]
            else:
                cand = np.full(N, -np.inf); cand[q:] = cum[q] + V[:N - q]
            newV = np.maximum(newV, cand)
        V = newV
    print(f'{reg:9s} events {len(keys):5d}  DP ceiling from flat: {V[cap]/1e3:8.1f}k')
