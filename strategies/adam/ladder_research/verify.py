"""Check identical simulator semantics, warmup gates and worst-case risk accounting."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from research import load,AuditSimulator,REPO
from simulator.simulator import ExchangeSimulator,SimulationConfig
from simulator.models import BookOrderView,OrderBookSnapshot,PrivateOrderEvent,OrderStatus,Side
from simulator.strategy import StrategyContext


def snapshot(t,bid=100,ask=102):
    return OrderBookSnapshot(t,[BookOrderView('b',Side.BUY,bid,100,'market')],
                             [BookOrderView('a',Side.SELL,ask,100,'market')])


def attach(s):
    sent,cancels = {},[]
    def send(request):
        key = f'v{len(sent)}'
        sent[key] = request
        return key
    s._attach_context(StrategyContext(send,cancels.append,lambda *a:None,lambda:0))
    return sent,cancels


def event(s,key,status,filled=0,remaining=0):
    s.on_private_order_update(PrivateOrderEvent(timestamp_us=0,order_id=key,status=status,
                                               filled_quantity=filled,remaining_quantity=remaining,
                                               fill_price=100 if filled else None))


def main():
    s=load({'trade_start_us':60_000_000})
    sent,_=attach(s)
    for t in (0,59_999_999):
        s.on_order_book_change(snapshot(t),[])
    assert not sent
    s.on_order_book_change(snapshot(60_000_000),[])
    assert sent
    for cap,size in ((80,10),(100,200)):
        for filling_side in (Side.BUY,Side.SELL):
            s=load(dict(cap=cap,size_top=size,size_deep=10,levels=2,topup=5))
            sent,_=attach(s)
            for step in range(100):
                s.on_order_book_change(snapshot(step*100_000),[])
                for key,order in list(s.orders.items()):
                    event(s,key,OrderStatus.ACCEPTED,remaining=order[2])
                    if order[0]==filling_side:
                        event(s,key,OrderStatus.FILLED,filled=order[2])
                buys=sum(o[2] for o in s.orders.values() if o[0]==Side.BUY)
                sells=sum(o[2] for o in s.orders.values() if o[0]==Side.SELL)
                assert s.pos+buys<=cap and s.pos-sells>=-cap
            assert abs(s.pos)==cap
    cfg=SimulationConfig(order_accept_delay_ms=5,max_position=100,max_potential_position=1000,
                         position_breach_penalty_per_second=10_000)
    params=dict(cap=80,size_top=10,size_deep=0,trade_start_us=0)
    stock=ExchangeSimulator(load(params),cfg).run(REPO/'data'/'sample.csv')
    audited=AuditSimulator(load(params),cfg).run(REPO/'data'/'sample.csv')
    assert stock.model_dump()==audited.model_dump()
    result=dict(warmup_gate=True,one_sided_risk_cases=4,stock_audited_exact_parity=True,
                sample_result=stock.model_dump(mode='json'))
    (HERE/'verification.json').write_text(json.dumps(result,indent=2))
    print('PASS: absolute warmup gate, four worst-case risk scenarios, exact stock/audited simulator parity.')


if __name__=='__main__':
    main()
