"""Ceiling angle, step 1: replay the round-3 market book (no us) against the exact S and write
   _cache/ceil_stale.parquet  one row per market order that became stale (edge > THR0 vs exact S)
   _cache/ceil_takers.parquet one row per market taker with the book it met (for the passive ceiling)
uv run --with pyarrow --with sortedcontainers python marek/research/round3/ideas/ceiling/ceiling_book.py
"""
import pathlib
import numpy as np
import pandas as pd
from sortedcontainers import SortedDict
from collections import deque

ROOT = pathlib.Path(__file__).resolve().parents[5]
OUT = pathlib.Path(__file__).parent / '_cache'
OUT.mkdir(exist_ok=True)
THR0 = 0.25
LAT = 5000  # us

tp = pd.read_csv(ROOT / 'data/round3.csv')
z = np.load(ROOT / 'data/.cache/round3_S_oracle.npz')
S_ins = z['S']  # S right after each insert (taker: after its kick)
T_us = np.round(tp.Time.values * 1e6).astype(np.int64)
op = tp.Operation.values
oidv = tp.OrderId.values
sidev = tp.Side.values
volv = tp.Volume.values
prv = tp.Price.values
lsv = tp.Lifespan.values

books = {'B': SortedDict(), 'A': SortedDict()}  # price -> deque [oid, qty]
orders = {}  # oid -> [side, price, qty, t_ins]
hist = {}  # oid -> list of (t_us, remaining) after each change
stale_first = {}  # oid -> (t_us, edge, side, price, qty at t, S, ev_index)
takers = []
ins_i = -1
S = float(S_ins[0])
n = len(tp)
for k in range(n):
    t = int(T_us[k])
    o = op[k]
    if o == 'Insert':
        ins_i += 1
        S_before = S
        S = float(S_ins[ins_i])
        side, px, q, oid = sidev[k], int(prv[k]), int(volv[k]), int(oidv[k])
        if lsv[k] == 'F':
            # taker: match against the opposite side up to the limit
            ob = books['A' if side == 'B' else 'B']
            bb = books['B'].peekitem(-1)[0] if books['B'] else None
            ba = books['A'].peekitem(0)[0] if books['A'] else None
            left = q
            fills = []
            while left and ob:
                lp, dq = ob.peekitem(0 if side == 'B' else -1)
                if (side == 'B' and lp > px) or (side == 'A' and lp < px):
                    break
                while left and dq:
                    e = dq[0]
                    f = min(left, e[1])
                    e[1] -= f
                    left -= f
                    fills.append((lp, f))
                    orders[e[0]][2] = e[1]
                    hist.setdefault(e[0], []).append((t, e[1]))
                    if e[1] == 0:
                        dq.popleft()
                        del orders[e[0]]
                if not dq:
                    del ob[lp]
            takers.append((t, ins_i, side, px, q, q - left, S_before, S, bb, ba,
                           ';'.join(f'{a}:{b}' for a, b in fills)))
        else:
            b = books[side]
            if px not in b:
                b[px] = deque()
            b[px].append([oid, q])
            orders[oid] = [side, px, q, t]
            hist[oid] = [(t, q)]
    else:  # cancel
        oid = int(oidv[k])
        e = orders.pop(oid, None)
        if e is not None:
            b = books[e[0]]
            dq = b.get(e[1])
            if dq is not None:
                for j, x in enumerate(dq):
                    if x[0] == oid:
                        del dq[j]
                        break
                if not dq:
                    del b[e[1]]
            hist[oid].append((t, 0))
    # stale scan: asks below S - THR0, bids above S + THR0 (only the next event at a later time matters,
    # but scanning after every row is cheap)
    if k + 1 < n and T_us[k + 1] == t:
        continue
    a = books['A']
    for lp in a.irange(maximum=S - THR0):
        for x in a[lp]:
            if x[0] not in stale_first:
                stale_first[x[0]] = (t, S - lp, 'A', lp, x[1], S, ins_i)
    bk = books['B']
    for lp in bk.irange(minimum=S + THR0, reverse=True):
        for x in bk[lp]:
            if x[0] not in stale_first:
                stale_first[x[0]] = (t, lp - S, 'B', lp, x[1], S, ins_i)

# remaining qty and S at arrival (t + LAT)
t_ins = z['t_us']
rows = []
for oid, (t, e, side, lp, q, s, ii) in stale_first.items():
    a = t + LAT
    h = hist[oid]
    rem = h[0][1]
    for th, r in h:
        if th <= a:
            rem = r
        else:
            break
    j = int(np.searchsorted(t_ins, a, side='right')) - 1
    sa = float(S_ins[j])
    ea = (sa - lp) if side == 'A' else (lp - sa)
    t_end = h[-1][0] if h[-1][1] == 0 else 10**12
    t0 = h[0][0]
    rows.append((oid, t, t0, t_end, side, lp, q, e, s, rem, ea, sa))
st = pd.DataFrame(rows, columns=['oid', 't_us', 't_ins', 't_end', 'side', 'price', 'q', 'edge', 'S', 'rem_arr',
                                 'edge_arr', 'S_arr'])
st.to_parquet(OUT / 'ceil_stale.parquet')
tk = pd.DataFrame(takers, columns=['t_us', 'ins_i', 'side', 'lim', 'q', 'filled', 'S_before', 'S_after', 'bb', 'ba',
                                   'fills'])
tk.to_parquet(OUT / 'ceil_takers.parquet')
print(len(st), len(tk))
