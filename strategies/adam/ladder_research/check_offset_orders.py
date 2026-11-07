"""Controlled public-API probes of the proposed non-executing offset orders.

Uses Peter's actual LadderMM, with isolated parameter overrides and a cleanup
callback, against the unmodified exchange. No production strategy is changed.
"""
import csv
import hashlib
import json
from collections import defaultdict
from research import HERE, OUT, SOURCE, REPO, load, ExchangeSimulator, SimulationConfig
from simulator.models import OrderRequest, OrderType, Side, OrderStatus

DEST = HERE/'offset_order_experiment'
DEST.mkdir(exist_ok=True)
LadderMM = load().__class__


class Probe(LadderMM):
    def __init__(self, cap, direction, dummy, flatten):
        super().__init__()
        self.p = dict(self.p,cap=cap,size_top=2*cap,trade_start_us=0)
        self.direction = direction
        self.dummy = dummy
        self.flatten = flatten
        self.dummy_id = None
        self.stopped = False
        self.events = []

    def on_private_order_update(self,event):
        self.events.append(event.model_dump(mode='json'))
        super().on_private_order_update(event)

    def on_order_book_change(self,snapshot,trades):
        if self.stopped:
            return
        if snapshot.timestamp_us >= 2_000_000:
            self.stopped = True
            for oid in list(self.orders):
                self.cancel_order(oid)
            if self.dummy_id:
                self.cancel_order(self.dummy_id)
            if self.flatten and self.pos:
                side = Side.SELL if self.pos>0 else Side.BUY
                touch = self._touch(snapshot.bids if self.pos>0 else snapshot.asks)
                # Peter's newer helper returns (price, quantity); the historical helper returned price.
                price = touch[0] if isinstance(touch, tuple) else touch
                self.send_order(OrderRequest(side=side,price=price,quantity=abs(self.pos),order_type=OrderType.IOC))
            return
        if self.dummy and self.dummy_id is None and snapshot.bids and snapshot.asks:
            # An opposite-side order far beyond every market order in this tape.
            side = Side.SELL if self.direction=='long' else Side.BUY
            price = 1_000_000_000 if side==Side.SELL else 1
            self.dummy_id = self.send_order(OrderRequest(side=side,price=price,quantity=self.p['cap'],order_type=OrderType.LIMIT))
        super().on_order_book_change(snapshot,trades)


def tape(direction):
    path = DEST/(direction+'.csv')
    rows = [
        [0,'Insert','bid','B',1000,9999,'G'],
        [.000001,'Insert','ask','A',1000,10001,'G'],
        [.000002,'Insert','deep_bid','B',5000,9998,'G'],
        [.000003,'Insert','deep_ask','A',5000,10002,'G'],
        [1,'Insert','aggressor','A' if direction=='long' else 'B',1200,9999 if direction=='long' else 10001,'F'],
        [2,'Insert','clock2','B',1,9997,'G'],
        [3,'Insert','clock3','B',1,9997,'G'],
        [4,'Insert','clock4','B',1,9997,'G'],
    ]
    with path.open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['Time','Operation','OrderId','Side','Volume','Price','Lifespan'])
        w.writerows(rows)
    return path


def original_distribution():
    start,end = 9060001434,11999999025
    inventory=0
    last=start
    duration=defaultdict(int)
    with (OUT/'full_logs_peter_original'/'trades.csv').open(newline='') as f:
        for r in csv.DictReader(f):
            if r['taker_owner']!='user' and r['maker_owner']!='user':
                continue
            t=int(r['timestamp_us'])
            assert start<=t<=end and t>=last
            duration[inventory]+=t-last
            sign=1 if r['taker_side']=='B' else -1
            inventory+=int(r['quantity'])*sign*(1 if r['taker_owner']=='user' else -1)
            last=t
    duration[inventory]+=end-last
    total=end-start
    return dict(start_us=start,end_us=end,final_inventory=inventory,
                fraction_at_minus100=duration[-100]/total,fraction_at_plus100=duration[100]/total,
                fraction_at_either_limit=(duration[-100]+duration[100])/total,
                fraction_abs_at_least99=sum(v for k,v in duration.items() if abs(k)>=99)/total,
                time_weighted_mean=sum(k*v for k,v in duration.items())/total,
                inventory_duration_us=dict(sorted(duration.items())))


def main():
    before=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    cfg=SimulationConfig(order_accept_delay_ms=5,max_position=100,max_potential_position=1000,
                         position_breach_penalty_per_second=10000)
    cases=[]
    for direction in ('long','short'):
        path=tape(direction)
        for cap,dummy,flatten in ((100,False,False),(100,True,False),(200,False,False),
                                  (200,True,False),(200,True,True),(1100,False,False),(1100,True,False)):
            strategy=Probe(cap,direction,dummy,flatten)
            result=ExchangeSimulator(strategy,cfg).run(path)
            record=dict(direction=direction,cap=cap,dummy=dummy,flatten=flatten,
                        **result.model_dump(mode='json'),events=strategy.events)
            if cap==200:
                assert abs(result.breach_penalty-(10050 if flatten else 30000))<1e-6
                assert result.inventory==(0 if flatten else (200 if direction=='long' else -200))
            elif cap==100:
                assert result.breach_penalty==0 and abs(result.inventory)==100
            else:
                assert result.inventory==0 and result.fills==0
                assert any(e['status']=='rejected' for e in strategy.events)
            if dummy and cap<=200:
                events=[e for e in strategy.events if e['order_id']==strategy.dummy_id]
                assert [e['status'] for e in events]==['accepted','canceled']
                assert all(e['filled_quantity']==0 for e in events)
            cases.append(record)
    for direction in ('long','short'):
        for cap in (100,200,1100):
            pair=[r for r in cases if r['direction']==direction and r['cap']==cap and not r['flatten']]
            keys=['pnl','inventory','cash','breach_penalty','traded_volume']
            assert all(pair[0][k]==pair[1][k] for k in keys)
    unchanged=before==hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert unchanged
    distribution=original_distribution()
    (DEST/'results.json').write_text(json.dumps(dict(source_sha256=before,original_source_unchanged=unchanged,
        cases=cases,original_test_inventory=distribution),indent=2))
    lines=['# Ladder: non-executing offset-order experiment','',
        'Result: the proposed offset does not avoid either exchange check. Peter\'s source and the simulator are unchanged. All 14 controlled runs used the actual LadderMM with isolated cap/size overrides, 5 ms latency, limit 100, potential limit 1000 and penalty 10000 per second.','',
        'The market forces a one-sided fill at 1 second. At 2 seconds the probe cancels all resting strategy orders (effective at 2.005 seconds). The replay ends at 4 seconds. The far opposite order uses price 1 for bids or 1,000,000,000 for asks; it is accepted, remains unfilled, and is canceled successfully.','',
        '| Direction | Cap | Distant offset | Executed hedge at cleanup | Final inventory | Penalty |',
        '|---|---:|---|---|---:|---:|']
    for r in cases:
        lines.append(f"| {r['direction']} | {r['cap']} | {r['dummy']} | {r['flatten']} | {r['inventory']} | {r['breach_penalty']:,.0f} |")
    lines+=['',
        'With cap 200, the penalty is 30,000 with or without the distant order: three seconds outside ±100. Canceling unfilled orders leaves the filled inventory unchanged. A genuine executed hedge removes inventory at 2.005 seconds, but still incurs 10,050 for the earlier 1.005 seconds outside the limit.','',
        'With cap 1100, the original ladder orders and the distant order are rejected independently against the potential-position cap 1000. Opposite-side orders do not net against this check.','',
        'Code: simulator/simulator.py:_exceeds_position_limit uses inventory plus SAME-side resting quantity and the new order; _accrue_breach_penalty uses abs(filled inventory)>100 only. LadderMM._room likewise guards each side separately.','',
        '## Original-default inventory distribution','',
        f"Reconstructed from actual user fills over the same 49-minute trading interval: {distribution['fraction_at_minus100']:.2%} of time at −100; {distribution['fraction_at_plus100']:.2%} at +100; {distribution['fraction_at_either_limit']:.2%} at either exact boundary; {distribution['fraction_abs_at_least99']:.2%} at |inventory|≥99. Mean inventory {distribution['time_weighted_mean']:.3f}.",
        '',
        'The limit is binding. A non-executing order is not a hedge of filled inventory. The available strategy changes are to increase turnover within the cap, reduce inventory through executed trades, or adjust quoting before reaching the boundary. No submission-ready bypass strategy was created because the proposed accounting mechanism fails.',
        '',
        'Reproduce from the repository:',
        '```powershell',
        '.\\.venv\\Scripts\\python.exe -B strategies\\adam\\ladder_research\\check_offset_orders.py',
        '```','']
    (DEST/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('PASS: 14 controlled replays; distant offsets leave PnL/inventory/penalties identical; source unchanged.')
    print(json.dumps({k:v for k,v in distribution.items() if k!='inventory_duration_us'}))


if __name__=='__main__':
    main()
