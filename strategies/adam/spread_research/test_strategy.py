"""Risk, adaptation, warmup and original-splitter parity checks."""
import ast
import csv
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from research import load,STRATEGY,REPO
from split_round1 import split
from simulator.models import BookOrderView,OrderBookSnapshot,PrivateOrderEvent,OrderStatus,Side
from simulator.strategy import StrategyContext


def book(t,bid=100,ask=102):
    return OrderBookSnapshot(t,[BookOrderView('b',Side.BUY,bid,100,'market')],
                             [BookOrderView('a',Side.SELL,ask,100,'market')])


def attach(s):
    sent = {}
    canceled = []
    def send(order):
        key = f't{len(sent)}'
        sent[key] = order
        return key
    s._attach_context(StrategyContext(send,canceled.append,lambda *a:None,lambda:0))
    return sent,canceled


def event(s,key,status,qty=0,remaining=0):
    s.on_private_order_update(PrivateOrderEvent(timestamp_us=0,order_id=key,status=status,
                                               filled_quantity=qty,remaining_quantity=remaining,
                                               fill_price=100 if qty else None))


def test_warmup_is_observation_only():
    s = load({'warmup_us':60_000_000,'trade_start_us':0})
    sent,_ = attach(s)
    for t in (0,1_000_000,59_999_999):
        s.on_order_book_change(book(t),[])
    assert not sent
    s.on_order_book_change(book(60_000_000),[])
    assert len(sent)==2


def test_absolute_fold_gate():
    s = load({'warmup_us':0,'trade_start_us':100_000_000})
    sent,_ = attach(s)
    s.on_order_book_change(book(90_000_000),[])
    assert not sent
    s.on_order_book_change(book(100_000_000),[])
    assert len(sent)==2


def test_volatility_and_adverse_selection_widen_quotes():
    widths=[]
    for variance,toxicity in ((.25,0),(25,0),(.25,2)):
        s = load({'warmup_us':0,'spread_mult':2,'tox_weight':1})
        attach(s)
        s.variance,s.toxicity = variance,toxicity
        s.on_order_book_change(book(0),[])
        widths.append(s.last_half_spread)
    assert widths[1]>widths[0] and widths[2]>widths[0]


@pytest.mark.parametrize('side',[Side.BUY,Side.SELL])
def test_worst_case_one_sided_fills_stop_at_cap(side):
    s = load({'warmup_us':0})
    sent,_ = attach(s)
    for step in range(100):
        s.on_order_book_change(book(step*100_000),[])
        for key,order in list(s.orders.items()):
            if not order['accepted']:
                event(s,key,OrderStatus.ACCEPTED,remaining=order['remaining'])
            if order['side']==side:
                event(s,key,OrderStatus.FILLED,qty=order['remaining'])
        buys = sum(o['remaining'] for o in s.orders.values() if o['side']==Side.BUY)
        sells = sum(o['remaining'] for o in s.orders.values() if o['side']==Side.SELL)
        assert s.inventory+buys<=80 and s.inventory-sells>=-80
    assert abs(s.inventory)==80


def test_pending_cancel_and_rejection_do_not_release_live_reservation():
    s = load({'warmup_us':0})
    sent,canceled = attach(s)
    s.on_order_book_change(book(0),[])
    for key in list(s.orders):
        event(s,key,OrderStatus.ACCEPTED,remaining=sent[key].quantity)
    s.on_order_book_change(book(100_000,110,112),[])
    assert len(canceled)==2
    s.on_order_book_change(book(200_000,110,112),[])
    assert len(sent)==2
    key = canceled[0]
    event(s,key,OrderStatus.REJECTED)
    assert key in s.orders
    event(s,key,OrderStatus.CANCELED)
    s.on_order_book_change(book(300_000),[])
    assert len(sent)==3


def test_streaming_split_outputs_equal_peters_original(tmp_path):
    path = tmp_path/'small.csv'
    path.write_text('Time,Operation,OrderId,Side,Volume,Price,Lifespan\n'
                    '0,Insert,b,B,10,100,G\n0,Insert,a,A,10,102,G\n'
                    '2,Insert,t,A,3,100,F\n5,Cancel,b,,,,\n'
                    '6,Insert,b2,B,10,99,G\n10,Insert,a2,A,5,101,G\n'
                    '15,Cancel,a,,,,\n20,Insert,b3,B,3,98,G\n')
    meta = split(path,tmp_path/'streamed',count=2,warmup=1)
    import importlib.util
    spec = importlib.util.spec_from_file_location('peter_reference',REPO/'strategies/peter/split_data.py')
    peter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(peter)
    with path.open() as file:
        reader = csv.reader(file)
        next(reader)
        rows = list(reader)
    times = [float(row[0]) for row in rows]
    starts = [d['start_s'] for d in meta['windows']]
    snaps = peter.snapshots_at(rows,starts)
    for d,snap in zip(meta['windows'],snaps):
        expected = tmp_path/(d['name']+'_expected.csv')
        peter.write_window(expected,rows,times,d['start_s'],d['end_s'],snap)
        assert Path(d['file']).read_bytes()==expected.read_bytes()


def test_submission_has_only_reviewed_imports_and_one_strategy_class():
    tree = ast.parse(STRATEGY.read_text())
    assert len([n for n in tree.body if isinstance(n,ast.ClassDef)])==1
    assert STRATEGY.stat().st_size<1024*1024
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            raise AssertionError('Only reviewed from imports expected')
        if isinstance(node,ast.ImportFrom):
            assert node.module in ('collections','math','simulator.models','simulator.strategy')
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
            assert node.func.id not in ('open','type','getattr','eval','exec','__import__')
