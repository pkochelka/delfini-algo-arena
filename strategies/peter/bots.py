"""Simple, model-free bots of different families, for comparison runs in the arena.

A research file with several strategies, not a submission: pick one with
`arena.py --agent name=strategies/peter/bots.py#ClassName`. A bot worth keeping gets
copied into its own file (the simulator uses the alphabetically first class of a file).

Shared by all bots (`Bot`): own-order tracking, a hard guard keeping the worst case
|position + every open order on a side| within CAP, quote syncing without cancel stacking,
and IOC execution towards a target position. Book levels count market orders only, and an
event whose spread is wider than MAX_SPREAD is ignored (a side briefly backed only by stale
far-away orders).
"""

from __future__ import annotations

from simulator.models import (
    OrderBookSnapshot,
    OrderRequest,
    OrderStatus,
    OrderType,
    PrivateOrderEvent,
    Side,
    Trade,
)
from simulator.strategy import BaseStrategy

BUY, SELL = Side.BUY, Side.SELL


def market_levels(orders, n: int) -> list[list[int]]:
    """Aggregate market-owned orders (best first) into up to n [price, qty] levels."""
    levels: list[list[int]] = []
    for o in orders:
        if o.owner != 'market':
            continue
        if levels and levels[-1][0] == o.price:
            levels[-1][1] += o.quantity
        elif len(levels) == n:
            break
        else:
            levels.append([o.price, o.quantity])
    return levels


class Bot(BaseStrategy):
    """Base: order bookkeeping, guard, quoting and target-position helpers. Trades nothing itself."""

    CAP = 90
    MAX_SPREAD = 8

    def __init__(self) -> None:
        super().__init__()
        self.pos = 0
        self.orders: dict[str, list] = {}  # id -> [side, price, remaining, is_ioc]
        self.cancelling: set[str] = set()
        self.ioc_in_flight = False
        self.bb = self.ba = 0
        self.bq = self.aq = 0
        self.mid = 0.0

    # bookkeeping
    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        o = self.orders.get(event.order_id)
        if o is None:
            return
        st = event.status
        if st in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED):
            self.pos += event.filled_quantity if o[0] == BUY else -event.filled_quantity
            o[2] = event.remaining_quantity
        elif st == OrderStatus.MODIFIED:
            o[2] = event.remaining_quantity
        if st in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
            self.orders.pop(event.order_id, None)
            self.cancelling.discard(event.order_id)
            if o[3]:
                self.ioc_in_flight = False

    def room(self, side: Side) -> int:
        rest = sum(o[2] for o in self.orders.values() if o[0] == side)
        if side == BUY:
            return max(0, self.CAP - (self.pos + rest))
        return max(0, self.CAP + self.pos - rest)

    def send(self, side: Side, price: int, qty: int, ioc: bool) -> None:
        qty = min(qty, self.room(side))
        if qty <= 0 or price <= 0:
            return
        oid = self.send_order(OrderRequest(side=side, price=price, quantity=qty,
                                           order_type=OrderType.IOC if ioc else OrderType.LIMIT))
        self.orders[oid] = [side, price, qty, ioc]
        if ioc:
            self.ioc_in_flight = True

    def cancel(self, oid: str) -> None:
        if oid not in self.cancelling:
            self.cancelling.add(oid)
            self.cancel_order(oid)

    def quote(self, side: Side, price: int | None, qty: int) -> None:
        """Keep one resting order on `side` at `price` (None or qty 0: no order)."""
        mine = [(oid, o) for oid, o in self.orders.items()
                if o[0] == side and not o[3] and oid not in self.cancelling]
        if price is not None and qty > 0 and len(mine) == 1 and mine[0][1][1] == price:
            return
        for oid, _ in mine:
            self.cancel(oid)
        if price is None or qty <= 0:
            return
        if any(oid in self.orders and self.orders[oid][0] == side for oid in self.cancelling):
            return  # no stacking while a cancel on this side is in flight
        self.send(side, price, qty, False)

    def work_to(self, target: int, chunk: int, deadband: int = 0) -> None:
        """Move the position towards `target` with one IOC at the touch at a time."""
        target = max(-self.CAP, min(self.CAP, target))
        diff = target - self.pos
        if self.ioc_in_flight or abs(diff) <= deadband:
            return
        if diff > 0:
            self.send(BUY, self.ba, min(chunk, diff), True)
        else:
            self.send(SELL, self.bb, min(chunk, -diff), True)

    # feed
    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        bids = market_levels(snapshot.bids, 1)
        asks = market_levels(snapshot.asks, 1)
        if not bids or not asks or asks[0][0] - bids[0][0] > self.MAX_SPREAD:
            return
        (self.bb, self.bq), (self.ba, self.aq) = bids[0], asks[0]
        self.mid = (self.bb + self.ba) / 2
        self.step(trades)

    def step(self, trades: list[Trade]) -> None:
        """Called after every usable book update; subclasses trade here."""


class JoinMM(Bot):
    """Naive market maker: join the best bid and ask; stop adding to a side past MAX_INV."""

    SIZE = 5
    MAX_INV = 50

    def step(self, trades: list[Trade]) -> None:
        self.quote(BUY, self.bb, self.SIZE if self.pos < self.MAX_INV else 0)
        self.quote(SELL, self.ba, self.SIZE if self.pos > -self.MAX_INV else 0)


class PennyMM(Bot):
    """Step one tick inside a spread of >= 3 ticks, else join; shift quotes away from inventory."""

    SIZE = 5
    SKEW_PER_TICK = 20  # one tick of skew per this many lots

    def step(self, trades: list[Trade]) -> None:
        wide = self.ba - self.bb >= 3
        bid = self.bb + 1 if wide else self.bb
        ask = self.ba - 1 if wide else self.ba
        skew = int(self.pos / self.SKEW_PER_TICK)
        bid = min(bid - skew, self.ba - 1)
        ask = max(ask - skew, self.bb + 1)
        self.quote(BUY, bid, self.SIZE)
        self.quote(SELL, ask, self.SIZE)


class Momentum(Bot):
    """EMA crossover on the mid; hold +-TARGET in the direction of the trend, via IOCs."""

    FAST, SLOW = 10, 60
    THRESH = 0.5  # ticks
    TARGET = 60
    CHUNK = 10

    def __init__(self) -> None:
        super().__init__()
        self.fast: float | None = None
        self.slow = 0.0
        self.target = 0

    def step(self, trades: list[Trade]) -> None:
        if self.fast is None:
            self.fast = self.slow = self.mid
        self.fast += 2 / (self.FAST + 1) * (self.mid - self.fast)
        self.slow += 2 / (self.SLOW + 1) * (self.mid - self.slow)
        sig = self.fast - self.slow
        if sig > self.THRESH:
            self.target = self.TARGET
        elif sig < -self.THRESH:
            self.target = -self.TARGET
        self.work_to(self.target, self.CHUNK, deadband=self.CHUNK // 2)


class MeanRevert(Bot):
    """Fade the deviation of the mid from a slow EMA, in proportion to its size."""

    SLOW = 200
    FULL_DEV = 3.0  # ticks of deviation for the full position
    TARGET = 60
    CHUNK = 10

    def __init__(self) -> None:
        super().__init__()
        self.slow: float | None = None

    def step(self, trades: list[Trade]) -> None:
        if self.slow is None:
            self.slow = self.mid
        self.slow += 2 / (self.SLOW + 1) * (self.mid - self.slow)
        dev = self.mid - self.slow
        frac = max(-1.0, min(1.0, -dev / self.FULL_DEV))
        self.work_to(round(frac * self.TARGET), self.CHUNK, deadband=self.CHUNK // 2)


class Imbalance(Bot):
    """Trade in the direction of the smoothed top-of-book volume imbalance."""

    SPAN = 5
    TARGET = 40
    CHUNK = 10

    def __init__(self) -> None:
        super().__init__()
        self.imb = 0.0

    def step(self, trades: list[Trade]) -> None:
        raw = (self.bq - self.aq) / (self.bq + self.aq)
        self.imb += 2 / (self.SPAN + 1) * (raw - self.imb)
        self.work_to(round(self.imb * self.TARGET), self.CHUNK, deadband=self.CHUNK // 2)
