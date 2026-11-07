"""Multi-agent arena: several strategies trade on one backtest, against the tape and each other.

Usage:
    uv run python strategies/peter/arena.py --backtest data/splits/dev.csv --out logs/arena/dev.json \
        --agent base=strategies/peter/arb_mm.py \
        --agent passive=strategies/peter/arb_mm.py:'{"takes":0}'         --agent join=strategies/peter/bots.py#JoinMM

A research tool, not a submission. It mirrors `simulator.simulator.ExchangeSimulator` (5 ms accept
delay, price-time matching at the maker's price, the ±100 penalty and the 1000 worst-case reject,
the markout PnL) with "user" generalised to N agents:

- each agent sees its own orders as owner "user" and everything else, other agents included, as
  "market"; trades are relabelled the same way
- an agent's order can trade with anyone's resting order except its own (self-trade prevention
  skips them, as the simulator skips the user's own orders)
- agents are called in a shuffled order after every market event, so ties in the 5 ms queue are
  not always won by the same agent
- an agent's markout liquidates against every resting order that isn't its own

Each agent is loaded from its own copy of the strategy module, and `PARAMS.update(overrides)` is
applied to that copy before the strategy is built. With a single agent the result matches the
simulator's (see --check).
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from simulator.main import (
    MAX_POSITION,
    MAX_POTENTIAL_POSITION,
    ORDER_ACCEPT_DELAY_MS,
    POSITION_BREACH_PENALTY_PER_SECOND,
)
from simulator.models import (
    BookOrderView,
    LiveOrder,
    MarketOperation,
    OrderBookSnapshot,
    OrderRequest,
    OrderStatus,
    OrderType,
    PrivateOrderEvent,
    Side,
    Trade,
)
from simulator.simulator import PNL_SAMPLE_INTERVAL_US, ExchangeSimulator, OrderBook
from simulator.strategy import BaseStrategy, StrategyContext

BUY, SELL = Side.BUY, Side.SELL
DELAY_US = ORDER_ACCEPT_DELAY_MS * 1000


def load_agent(path: Path, overrides: dict, tag: str, cls_name: str | None = None) -> BaseStrategy:
    """Import a private copy of the strategy module, apply PARAMS overrides, build its strategy.

    `cls_name` picks a class; otherwise the alphabetically first subclass is used, as in the simulator.
    """
    spec = importlib.util.spec_from_file_location(f'arena_agent_{tag}', path.resolve())
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up here
    sys.path.insert(0, str(path.resolve().parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    if overrides:
        module.PARAMS.update(overrides)
    if cls_name is not None:
        return getattr(module, cls_name)()
    for name in sorted(dir(module)):
        cand = getattr(module, name)
        if isinstance(cand, type) and issubclass(cand, BaseStrategy) and cand is not BaseStrategy \
                and cand.__module__ == module.__name__:
            return cand()
    raise ValueError(f'no BaseStrategy subclass in {path}')


class Agent:
    def __init__(self, idx: int, name: str, strategy: BaseStrategy) -> None:
        self.idx, self.name, self.strategy = idx, name, strategy
        self.inventory = 0
        self.cash = 0.0
        self.penalty = 0.0
        self.volume = 0
        self.filled_ids: set[str] = set()
        self.resting = {BUY: 0, SELL: 0}
        self.inv_time: defaultdict[int, int] = defaultdict(int)
        self.samples: list[float] = []


class Arena:
    def __init__(self, agents: list[tuple[str, BaseStrategy]], seed: int = 0, public_depth: int = 20) -> None:
        self.book = OrderBook()
        self.agents = [Agent(i, n, s) for i, (n, s) in enumerate(agents)]
        self.owner_of: dict[str, int] = {}  # agent order id -> agent idx
        self.pending: list[tuple[int, int, str, int, object]] = []  # (ready_us, seq, kind, agent, payload)
        self.seq = 0
        self.now = 0
        self.trades: list[Trade] = []
        self.last_trade_price: float | None = None
        self.next_sample: int | None = None
        self.rng = random.Random(seed)
        self.depth = public_depth
        for a in self.agents:
            a.strategy._attach_context(StrategyContext(
                submit_order=lambda req, a=a: self.submit(a, req),
                cancel_order=lambda oid, a=a: self.queue(a, 'cancel', oid),
                modify_order=lambda oid, q, a=a: self.queue(a, 'modify', (oid, q)),
                now_us=lambda: self.now,
            ))

    # ------------------------------------------------------------ order entry
    def submit(self, a: Agent, req: OrderRequest) -> str:
        self.seq += 1
        oid = f'a{a.idx:02d}u{self.seq:09d}'
        self.owner_of[oid] = a.idx
        self.pending.append((self.now + DELAY_US, self.seq, 'order', a.idx, (oid, req)))
        return oid

    def queue(self, a: Agent, kind: str, payload) -> None:
        self.seq += 1
        self.pending.append((self.now + DELAY_US, self.seq, kind, a.idx, payload))

    def activate_up_to(self, ts: int | None) -> None:
        self.pending.sort(key=lambda x: (x[0], x[1]))
        due = [p for p in self.pending if ts is None or p[0] <= ts]
        self.pending = [p for p in self.pending if not (ts is None or p[0] <= ts)]
        for ready, _, kind, ai, payload in due:
            self.advance(max(self.now, ready))
            a = self.agents[ai]
            if kind == 'order':
                self.activate_order(a, *payload, ready)
            elif kind == 'cancel':
                self.activate_cancel(a, payload)
            else:
                self.activate_modify(a, *payload)

    def notify(self, a: Agent, ev: PrivateOrderEvent) -> None:
        if ev.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
            a.filled_ids.add(ev.order_id)
        a.strategy.on_private_order_update(ev)

    def activate_order(self, a: Agent, oid: str, req: OrderRequest, ready: int) -> None:
        sign = 1 if req.side == BUY else -1
        limit = MAX_POTENTIAL_POSITION if MAX_POTENTIAL_POSITION is not None else MAX_POSITION
        if limit is not None and abs(a.inventory + sign * (a.resting[req.side] + req.quantity)) > limit:
            self.notify(a, PrivateOrderEvent(timestamp_us=self.now, order_id=oid, status=OrderStatus.REJECTED))
            return
        order = LiveOrder(order_id=oid, owner='user', side=req.side, price=req.price,
                          initial_quantity=req.quantity, remaining_quantity=req.quantity,
                          order_type=req.order_type, created_at_us=ready)
        self.notify(a, PrivateOrderEvent(timestamp_us=self.now, order_id=oid, status=OrderStatus.ACCEPTED,
                                         remaining_quantity=order.remaining_quantity))
        self.match(order, a.idx)
        if order.order_type == OrderType.IOC or order.remaining_quantity == 0:
            return
        self.book.insert(order)
        a.resting[order.side] += order.remaining_quantity

    def activate_cancel(self, a: Agent, oid: str) -> None:
        order = self.book.get(oid)
        if order is None or self.owner_of.get(oid) != a.idx:
            self.notify(a, PrivateOrderEvent(timestamp_us=self.now, order_id=oid, status=OrderStatus.REJECTED,
                                             message='cancel rejected: no resting order with this id'))
            return
        self.book.remove(oid)
        a.resting[order.side] -= order.remaining_quantity
        self.notify(a, PrivateOrderEvent(timestamp_us=self.now, order_id=oid, status=OrderStatus.CANCELED,
                                         remaining_quantity=order.remaining_quantity))

    def activate_modify(self, a: Agent, oid: str, q: int) -> None:
        order = self.book.get(oid)
        if order is None or self.owner_of.get(oid) != a.idx:
            return
        if not (0 < q < order.remaining_quantity):
            self.notify(a, PrivateOrderEvent(timestamp_us=self.now, order_id=oid, status=OrderStatus.REJECTED,
                                             remaining_quantity=order.remaining_quantity))
            return
        diff = order.remaining_quantity - q
        self.book.reduce_quantity(order, diff)
        order.initial_quantity = q
        a.resting[order.side] -= diff
        self.notify(a, PrivateOrderEvent(timestamp_us=self.now, order_id=oid, status=OrderStatus.MODIFIED,
                                         remaining_quantity=order.remaining_quantity))

    # --------------------------------------------------------------- matching
    def best_maker(self, side: Side, taker_agent: int | None) -> LiveOrder | None:
        b = self.book
        if side == BUY:
            prices, levels = b._ask_prices, b._asks
        else:
            prices, levels = reversed(b._bid_prices), b._bids
        for price in prices:
            for o in levels[price].queue:
                if o.remaining_quantity > 0 and (taker_agent is None or self.owner_of.get(o.order_id) != taker_agent):
                    return o
        return None

    def match(self, taker: LiveOrder, taker_agent: int | None) -> None:
        while taker.remaining_quantity > 0:
            maker = self.best_maker(taker.side, taker_agent)
            if maker is None:
                break
            if not (taker.price >= maker.price if taker.side == BUY else taker.price <= maker.price):
                break
            q = min(taker.remaining_quantity, maker.remaining_quantity)
            taker.remaining_quantity -= q
            self.book.reduce_quantity(maker, q)
            maker_agent = self.owner_of.get(maker.order_id) if maker.owner == 'user' else None
            self.trades.append(Trade(
                timestamp_us=self.now, price=maker.price, quantity=q,
                taker_order_id=taker.order_id, maker_order_id=maker.order_id,
                taker_owner='market' if taker_agent is None else 'user',
                maker_owner='market' if maker_agent is None else 'user',
                book_side='bid' if maker.side == BUY else 'ask'))
            self.last_trade_price = float(maker.price)
            for ai, o in ((taker_agent, taker), (maker_agent, maker)):
                if ai is None:
                    continue
                a = self.agents[ai]
                s = 1 if o.side == BUY else -1
                a.inventory += s * q
                a.cash -= s * q * maker.price
                a.volume += q
            if maker_agent is not None:
                ma = self.agents[maker_agent]
                ma.resting[maker.side] -= q
                self.notify(ma, self.fill_event(maker, q))
            if taker_agent is not None:
                self.notify(self.agents[taker_agent], self.fill_event(taker, q))
            if maker.remaining_quantity == 0:
                self.book.remove(maker.order_id)
        if taker_agent is not None and taker.order_type == OrderType.IOC and taker.remaining_quantity > 0:
            self.notify(self.agents[taker_agent], PrivateOrderEvent(
                timestamp_us=self.now, order_id=taker.order_id, status=OrderStatus.CANCELED,
                remaining_quantity=taker.remaining_quantity))

    def fill_event(self, o: LiveOrder, q: int) -> PrivateOrderEvent:
        return PrivateOrderEvent(timestamp_us=self.now, order_id=o.order_id,
                                 status=OrderStatus.FILLED if o.remaining_quantity == 0 else OrderStatus.PARTIALLY_FILLED,
                                 filled_quantity=q, remaining_quantity=o.remaining_quantity, fill_price=int(o.price))

    def market_event(self, ev) -> None:
        if ev.operation == MarketOperation.INSERT:
            order = LiveOrder(order_id=ev.order_id, owner='market', side=ev.side, price=ev.price,
                              initial_quantity=ev.volume, remaining_quantity=ev.volume,
                              order_type=ev.order_type, created_at_us=ev.timestamp_us)
            self.match(order, None)
            if order.order_type == OrderType.LIMIT and order.remaining_quantity > 0:
                self.book.insert(order)
        else:
            self.book.remove(ev.order_id)  # unknown ids are ignored, as in the simulator

    # ----------------------------------------------------------- public feed
    def relabel_views(self, views: list[BookOrderView], ai: int) -> list[BookOrderView]:
        out = []
        for v in views:
            if v.owner == 'user' and self.owner_of.get(v.order_id) != ai:
                v = BookOrderView(order_id=v.order_id, side=v.side, price=v.price, quantity=v.quantity, owner='market')
            out.append(v)
        return out

    def relabel_trade(self, t: Trade, ai: int) -> Trade:
        if t.taker_owner == 'market' and t.maker_owner == 'market':
            return t
        tk = 'user' if self.owner_of.get(t.taker_order_id) == ai else 'market'
        mk = 'user' if self.owner_of.get(t.maker_order_id) == ai else 'market'
        if tk == t.taker_owner and mk == t.maker_owner:
            return t
        return t.model_copy(update={'taker_owner': tk, 'maker_owner': mk})

    def emit(self) -> None:
        bids = self.book.visible_views(BUY, self.depth)
        asks = self.book.visible_views(SELL, self.depth)
        trades, self.trades = self.trades, []
        order = list(range(len(self.agents)))
        self.rng.shuffle(order)
        multi = len(self.agents) > 1
        for ai in order:
            a = self.agents[ai]
            snap = OrderBookSnapshot(
                timestamp_us=self.now,
                bids=self.relabel_views(bids, ai) if multi else list(bids),
                asks=self.relabel_views(asks, ai) if multi else list(asks))
            a.strategy.on_order_book_change(snap, [self.relabel_trade(t, ai) for t in trades] if multi else trades)

    # ------------------------------------------------------- time and scoring
    def markout(self, a: Agent) -> float:
        if a.inventory == 0:
            return 0.0
        remaining = abs(a.inventory)
        last = self.last_trade_price
        value = 0.0
        for o in self.book.iter_levels(BUY if a.inventory > 0 else SELL):
            if o.owner == 'user' and self.owner_of.get(o.order_id) == a.idx:
                continue
            f = min(remaining, o.remaining_quantity)
            value += f * o.price
            last = float(o.price)
            remaining -= f
            if remaining == 0:
                break
        value += remaining * last
        return value if a.inventory > 0 else -value

    def pnl(self, a: Agent) -> float:
        return a.cash + self.markout(a) - a.penalty

    def accrue(self, ts: int) -> None:
        for a in self.agents:
            if MAX_POSITION is not None and abs(a.inventory) > MAX_POSITION:
                a.penalty += (ts - self.now) / 1e6 * POSITION_BREACH_PENALTY_PER_SECOND
        self.now = ts

    def advance(self, ts: int) -> None:
        if self.next_sample is None:
            self.now = ts
            for a in self.agents:
                a.samples.append(self.pnl(a))
            self.next_sample = ts + PNL_SAMPLE_INTERVAL_US
            return
        if ts > self.now:
            for a in self.agents:
                a.inv_time[abs(a.inventory)] += ts - self.now
            while self.next_sample <= ts:
                self.accrue(self.next_sample)
                for a in self.agents:
                    a.samples.append(self.pnl(a))
                self.next_sample += PNL_SAMPLE_INTERVAL_US
            self.accrue(ts)

    def run(self, path: Path) -> list[dict]:
        for ev in ExchangeSimulator.iter_backtest(path):
            if self.pending:
                self.activate_up_to(ev.timestamp_us)
            self.advance(ev.timestamp_us)
            self.market_event(ev)
            self.emit()
        self.activate_up_to(None)
        return [self.result(a) for a in self.agents]

    def result(self, a: Agent) -> dict:
        inc = [b - x for x, b in itertools.pairwise(a.samples)]
        se = statistics.stdev(inc) * math.sqrt(len(inc)) if len(inc) >= 2 else None
        total = sum(a.inv_time.values())

        def q(p: float) -> int:
            held = 0
            for lv in sorted(a.inv_time):
                held += a.inv_time[lv]
                if held >= p * total:
                    return lv
            return 0

        return {'name': a.name, 'pnl': self.pnl(a), 'pnl_se': se, 'cash': a.cash, 'inventory': a.inventory,
                'breach_penalty': a.penalty, 'fills': len(a.filled_ids), 'traded_volume': a.volume,
                'abs_inventory_q50': q(0.5), 'abs_inventory_q95': q(0.95)}


def parse_agent(spec: str) -> tuple[str, Path, dict, str | None]:
    """Parse name=path.py[#Class][:{json overrides}]."""
    name, rest = spec.split('=', 1)
    overrides: dict = {}
    if ':' in rest[2:]:  # allow a Windows drive letter
        i = rest.index(':', 2)
        rest, overrides = rest[:i], json.loads(rest[i + 1:])
    path, _, cls_name = rest.partition('#')
    return name, Path(path), overrides, cls_name or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--backtest', type=Path, required=True)
    parser.add_argument('--agent', action='append', required=True, help="name=path.py[#Class][:'{json overrides}']")
    parser.add_argument('--seed', type=int, default=0, help='seed of the per-event agent call order')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    specs = [parse_agent(s) for s in args.agent]
    agents = [(n, load_agent(p, ov, f'{i}_{n}', c)) for i, (n, p, ov, c) in enumerate(specs)]
    results = Arena(agents, seed=args.seed).run(args.backtest)
    for r in results:
        print(f"{r['name']:16} pnl={r['pnl']:>10.0f} se={r['pnl_se'] or 0:>7.0f} vol={r['traded_volume']:>6} "
              f"|inv| q50={r['abs_inventory_q50']:>3} q95={r['abs_inventory_q95']:>3} pen={r['breach_penalty']:.0f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({'backtest': str(args.backtest), 'seed': args.seed, 'results': results}, indent=2))


if __name__ == '__main__':
    main()
