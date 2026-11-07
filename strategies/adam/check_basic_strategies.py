r"""Local verification and comparison; not part of any strategy submission.

Run from the repo: .\.venv\Scripts\python.exe strategies\adam\check_basic_strategies.py
Pass additional CSV paths to compare more tapes. Results stay beside this script.
"""
import ast
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').exists() and (p / 'simulator').is_dir())
sys.path.insert(0, str(REPO))

from simulator.main import load_strategy_from_file
from simulator.models import (
    BookOrderView, OrderBookSnapshot, OrderStatus, PrivateOrderEvent, Side,
)
from simulator.simulator import ExchangeSimulator, SimulationConfig
from simulator.strategy import StrategyContext


class AuditedSimulator(ExchangeSimulator):
    def __init__(self, *args, **kwargs):
        self.max_abs_inventory_seen = 0
        self.rejections = 0
        self.cancel_race_rejections = 0
        super().__init__(*args, **kwargs)

    def _private_notify(self, event):
        self.max_abs_inventory_seen = max(self.max_abs_inventory_seen, abs(self._inventory))
        self.rejections += event.status == OrderStatus.REJECTED
        self.cancel_race_rejections += (
            event.status == OrderStatus.REJECTED
            and event.message == 'cancel rejected: no resting order with this id'
        )
        super()._private_notify(event)


def check_source(path):
    source = path.read_text(encoding='utf-8')
    tree = ast.parse(source, filename=str(path))
    compile(tree, str(path), 'exec')
    assert len([n for n in tree.body if isinstance(n, ast.ClassDef)]) == 1
    assert path.stat().st_size <= 1024 * 1024
    banned_calls = {'open', 'eval', 'exec', 'getattr', 'setattr', 'type', '__import__'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            raise AssertionError('Only reviewed from-imports expected')
        if isinstance(node, ast.ImportFrom):
            assert node.module in {'math', 'simulator.models', 'simulator.strategy'}
            if node.module == 'simulator.strategy':
                assert [n.name for n in node.names] == ['BaseStrategy']
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in banned_calls
        if isinstance(node, ast.Attribute) and node.attr.startswith('__'):
            assert node.attr == '__init__'  # Normal BaseStrategy constructor only.


def check_lifecycle(path):
    s = load_strategy_from_file(path)
    sent = {}
    cancels = []

    def send(request):
        order_id = f'test{len(sent)}'
        sent[order_id] = request
        return order_id

    s._attach_context(StrategyContext(send, cancels.append, lambda *a: None, lambda: 0))

    def book(timestamp, bid=100, ask=102):
        return OrderBookSnapshot(timestamp, [BookOrderView('b', Side.BUY, bid, 50, 'market')],
                                 [BookOrderView('a', Side.SELL, ask, 50, 'market')])

    def notify(order_id, status, filled=0, remaining=0):
        s.on_private_order_update(PrivateOrderEvent(
            timestamp_us=100_000, order_id=order_id, status=status,
            filled_quantity=filled, remaining_quantity=remaining,
            fill_price=100 if filled else None,
        ))

    s.on_order_book_change(book(0), [])
    assert len(sent) == 2
    s.on_order_book_change(book(200_000), [])
    assert len(sent) == 2, 'Pending orders must prevent duplicate submissions'
    buy_id = next(k for k, v in sent.items() if v.side == Side.BUY)
    sell_id = next(k for k, v in sent.items() if v.side == Side.SELL)
    for key in (buy_id, sell_id):
        notify(key, OrderStatus.ACCEPTED, remaining=sent[key].quantity)
    notify(buy_id, OrderStatus.PARTIALLY_FILLED, filled=2, remaining=3)
    assert s.inventory == 2 and s.orders[buy_id]['remaining'] == 3
    s.on_order_book_change(book(400_000, 110, 112), [])
    assert buy_id in cancels and sell_id in cancels
    count = len(sent)
    s.on_order_book_change(book(600_000, 110, 112), [])
    assert len(sent) == count, 'Cancel must be confirmed before replacement'
    notify(buy_id, OrderStatus.REJECTED, remaining=3)
    assert buy_id in s.orders, 'Rejected cancel must keep the exposure reservation'
    notify(buy_id, OrderStatus.CANCELED, remaining=3)
    notify(sell_id, OrderStatus.FILLED, filled=5)
    assert s.inventory == -3 and not s.orders
    s.on_order_book_change(book(800_000), [])
    pending = list(s.orders)
    for key in pending:
        notify(key, OrderStatus.REJECTED)
    assert not s.orders, 'Rejected submissions must release the in-flight slot'

    # Near-limit sizing must include all reserved capacity, on either side.
    for inventory in (s.POSITION_LIMIT - 1, -s.POSITION_LIMIT + 1):
        s.inventory = inventory
        s.orders.clear()
        s.on_order_book_change(book(1_000_000 + len(sent) * 200_000), [])
        buy_reserved = sum(o['remaining'] for o in s.orders.values() if o['side'] == Side.BUY)
        sell_reserved = sum(o['remaining'] for o in s.orders.values() if o['side'] == Side.SELL)
        assert inventory + buy_reserved <= s.POSITION_LIMIT
        assert inventory - sell_reserved >= -s.POSITION_LIMIT
    # A one-sided market withdraws accepted quotes rather than inventing a price.
    for order in s.orders.values():
        order['accepted'] = True
    s.on_order_book_change(OrderBookSnapshot(99_000_000, [], []), [])
    assert all(o['canceling'] for o in s.orders.values())

    # Force every new quote on one side to fill: price alpha must never override risk.
    for filling_side in (Side.BUY, Side.SELL):
        s = load_strategy_from_file(path)
        sent.clear()
        s._attach_context(StrategyContext(send, cancels.append, lambda *a: None, lambda: 0))
        for step in range(250):
            s.on_order_book_change(book(step * 200_000), [])
            for key, order in list(s.orders.items()):
                if not order['accepted']:
                    notify(key, OrderStatus.ACCEPTED, remaining=order['remaining'])
                if order['side'] == filling_side:
                    notify(key, OrderStatus.FILLED, filled=order['remaining'])
            buy_reserved = sum(o['remaining'] for o in s.orders.values() if o['side'] == Side.BUY)
            sell_reserved = sum(o['remaining'] for o in s.orders.values() if o['side'] == Side.SELL)
            assert s.inventory + buy_reserved <= s.POSITION_LIMIT
            assert s.inventory - sell_reserved >= -s.POSITION_LIMIT
        assert abs(s.inventory) == s.POSITION_LIMIT


def make_stress_tape():
    """Alternating-flow fixture for risk tests, not performance evidence."""
    path = HERE / 'basic_strategy_checks' / 'stress.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ['Time,Operation,OrderId,Side,Volume,Price,Lifespan']
    rows += ['0,Insert,b,B,2000,100,G', '0,Insert,a,A,2000,102,G']
    # Sweeps repeatedly hit either side; replenishment prevents an empty market.
    for i in range(1, 2001):
        side = 'A' if (i // 100) % 2 == 0 else 'B'
        price = 100 if side == 'A' else 102
        rows.append(f'{i * .01:.3f},Insert,t{i},{side},7,{price},F')
        if i % 20 == 0:
            rows.append(f'{i * .01:.3f},Insert,b{i},B,200,100,G')
            rows.append(f'{i * .01:.3f},Insert,a{i},A,200,102,G')
    path.write_text('\n'.join(rows) + '\n', encoding='utf-8')
    return path


def main():
    strategies = sorted((REPO / 'strategies' / 'adam' / 'models').glob('basic_*.py'))
    assert len(strategies) == 4
    datasets = [Path(p).resolve() for p in sys.argv[1:]] or [REPO / 'data' / 'sample.csv']
    stress = make_stress_tape()
    records = []
    output = HERE / 'basic_strategy_checks'
    for path in strategies:
        check_source(path)
        check_lifecycle(path)
        for tape in [stress, *datasets]:
            strategy = load_strategy_from_file(path)
            simulator = AuditedSimulator(strategy, SimulationConfig(
                order_accept_delay_ms=5, max_position=100,
                max_potential_position=1000, position_breach_penalty_per_second=10_000,
            ))
            started = time.perf_counter()
            result = simulator.run(tape)
            assert simulator.max_abs_inventory_seen <= strategy.POSITION_LIMIT
            assert result.breach_penalty == 0
            record = {'strategy': path.name, 'dataset': tape.name,
                      'seconds': round(time.perf_counter() - started, 3),
                      'max_abs_inventory': simulator.max_abs_inventory_seen,
                      'rejections': simulator.rejections,
                      'cancel_race_rejections': simulator.cancel_race_rejections,
                      **result.model_dump(mode='json')}
            assert simulator.rejections == simulator.cancel_race_rejections
            records.append(record)
            print(json.dumps(record), flush=True)
            (output / f'{path.stem}_{tape.stem}_pnl.json').write_text(
                json.dumps(simulator.pnl_curve(result.pnl)), encoding='utf-8')
            (output / 'results.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print('PASS: source checks, lifecycle tests, sample/stress inventory bounds and penalties.')


if __name__ == '__main__':
    main()
