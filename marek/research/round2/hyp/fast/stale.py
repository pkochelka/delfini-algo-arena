import sys, importlib.util, json
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[5]; sys.path.insert(0, str(ROOT))
from simulator.main import (MAX_POSITION, MAX_POTENTIAL_POSITION, ORDER_ACCEPT_DELAY_MS, POSITION_BREACH_PENALTY_PER_SECOND)
from simulator.simulator import ExchangeSimulator, SimulationConfig
spec = importlib.util.spec_from_file_location('m', sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
tape = sys.argv[2]
if 'round2' in tape: ons = list(pd.read_csv(ROOT/'marek/research/round2/flow/shock_events.csv').onset)
else: ons = json.load(open(tape.replace('.csv','.truth.json')))['onsets']
Js = [o+3.0 for o in ons]
class P(m.FVAShock):
    def __init__(s):
        super().__init__(); s.i=0; s.snap=None; s.out=[]
    def on_order_book_change(s, snap, tr):
        super().on_order_book_change(snap, tr)
        t=s.now_t
        if s.i < len(Js):
            J=Js[s.i]
            if s.snap is None and t >= J - 0.0005:
                s.snap = ([(v.price,v.quantity) for v in snap.bids if v.owner=='market'], [(v.price,v.quantity) for v in snap.asks if v.owner=='market'], s.f.mean)
            if s.snap is not None and t >= J + 0.5:
                b,a,F0=s.snap; S=s.f.mean
                up = S > F0
                if up: lots=[(q, S-p) for p,q in a if p < S-1]
                else: lots=[(q, p-S) for p,q in b if p > S+1]
                s.out.append((J, S-F0, sum(q for q,_ in lots), sum(q*e for q,e in lots)))
                s.snap=None; s.i+=1
cfg = SimulationConfig(max_position=MAX_POSITION, max_potential_position=MAX_POTENTIAL_POSITION, position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND, order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS)
st=P(); ExchangeSimulator(strategy=st, config=cfg).run(Path(tape))
tl=tv=0
for J,dS,l,v in st.out:
    print(f'J{J:8.2f} dS{dS:+7.1f} stale_lots {l:6.0f} value {v:9.0f} beyond200 {v*max(0,l-200)/max(l,1):9.0f}'); tl+=l; tv+=v
print('total lots', tl, 'value', tv)
