"""s_mm2: S-tracking market maker with a queue-aware expected-value rule. FAILED EXPERIMENT: best 35.3k/fold
(kappa 10) vs s_mm 45.7k; the per-taker EV ignores the option value of holding queue position.

Fair price S of the round-1 generator (strategies/peter/round1_generator.md) is tracked exactly:
S = off + U, U = c * sum(sign * sqrt(filled)) over market takers, `off` bounded by market inserts at the
touch (bid <= S - 0.05 <= S + 0.05 <= ask).

A resting ask at price a with A lots ahead of it (better prices + earlier in its level) and size z is
filled by the next buy taker (size Q ~ round(lognormal(mu, sd)), limit round(S + 1)) for min(z, (Q - A)+)
lots, each worth a - S - c*sqrt(Q) (S jumps by the taker's own impact). So per taker the order's value is

    EV = e * M1(A, z) - c * M2(A, z),   e = a - S,  reachable only if a <= round(S + 1)

M1 = E[min(z, (Q-A)+)], M2 = E[min(z, (Q-A)+) * sqrt(Q)], computed in O(1) from prefix sums of the
taker-size pmf. Keep an order while e >= M2/M1 * c + m_keep (+ inventory term); post a new order at the
most aggressive candidate price whose back-of-queue break-even passes with margin m_post.
Stale quotes (ask <= S - take) are taken with IOCs.
"""

from __future__ import annotations

import math

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

PARAMS = {
    'c': 0.026,
    'gap': 0.05,
    'relax': 0.000002,
    'max_width': 0.3,
    'cap': 100,
    'q_mu': 4.106,
    'q_sd': 1.396,
    'q_max': 200000,
    'kappa': 1.0,  # keep while EV per taker (lots*ticks) >= -kappa (option value of queue position)
    'theta': 0.3,  # min edge to post
    'm_keep': 0.0,  # keep while e >= breakeven(A, z) + m_keep
    'm_post': 0.05,  # post at a price while e >= breakeven(A_back, z) + m_post
    'inv_k': 0.0,  # margin += inv_k * pos / cap on the side that adds to |pos|
    'levels': 2,  # candidate prices: improve, touch, touch+1 ... (levels = how many at/behind touch)
    'deep_post': 0,  # 1: if no candidate passes m_post, post at the deepest candidate that passes m_keep
    'take': 0.2,
    'take_max': 200,
    'min_post': 1,
    'trade_start_us': 0,
}

BUY, SELL = Side.BUY, Side.SELL


class SMM2(BaseStrategy):
    def __init__(self) -> None:
        super().__init__()
        p = self.p = PARAMS
        self.c = p['c']
        self.pos = 0
        self.orders: dict[str, list] = {}  # id -> [side, price, remaining]
        self.cancelling: set[str] = set()
        self.U = 0.0
        self.lo = -1e18
        self.hi = 1e18
        self.max_id = ''
        self._build_pmf()

    def _build_pmf(self) -> None:
        mu, sd, n = self.p['q_mu'], self.p['q_sd'], self.p['q_max']

        def cdf(x):
            return 0.5 * math.erfc(-(math.log(x) - mu) / (sd * math.sqrt(2))) if x > 0 else 0.0

        # P(Q = k) for k = 1..n (k=1 absorbs everything below 1.5)
        P0 = [0.0] * (n + 2)  # prefix sums of p
        P1 = [0.0] * (n + 2)  # of k p
        Ph = [0.0] * (n + 2)  # of sqrt(k) p
        P3 = [0.0] * (n + 2)  # of k sqrt(k) p
        prev = 0.0
        s0 = s1 = sh = s3 = 0.0
        for k in range(1, n + 1):
            cur = cdf(k + 0.5) if k < n else 1.0
            pk = cur - prev
            prev = cur
            r = math.sqrt(k)
            s0 += pk
            s1 += k * pk
            sh += r * pk
            s3 += k * r * pk
            P0[k], P1[k], Ph[k], P3[k] = s0, s1, sh, s3
        P0[n + 1], P1[n + 1], Ph[n + 1], P3[n + 1] = s0, s1, sh, s3
        self.P0, self.P1, self.Ph, self.P3, self.n = P0, P1, Ph, P3, n
        self.be_cache: dict = {}

    def ev(self, A: int, z: int, e: float) -> float:
        """Expected value per taker arrival (ticks * lots) of an order with A ahead, size z, edge e."""
        m = self.moments(A, z)
        return e * m[0] - self.c * m[1]

    def moments(self, A: int, z: int):
        key = (A >> 2, z >> 2)
        v = self.be_cache.get(key)
        if v is not None:
            return v
        n = self.n
        A = min(A, n - 2)
        hi = min(A + z, n)
        P0, P1, Ph, P3 = self.P0, self.P1, self.Ph, self.P3
        m1 = (P1[hi] - P1[A]) - A * (P0[hi] - P0[A]) + z * (1.0 - P0[hi])
        m2 = (P3[hi] - P3[A]) - A * (Ph[hi] - Ph[A]) + z * (Ph[n] - Ph[hi])
        v = (m1, m2)
        self.be_cache[key] = v
        return v

    def breakeven(self, A: int, z: int) -> float:
        """Edge e at which EV = e*M1 - c*M2 = 0 for an order with A lots ahead and size z."""
        key = (A >> 3, z >> 3, 1)
        v = self.be_cache.get(key)
        if v is not None:
            return v
        n = self.n
        A = min(A, n - 2)
        hi = min(A + z, n)
        P0, P1, Ph, P3 = self.P0, self.P1, self.Ph, self.P3
        # Q in (A, hi]: weight Q - A ; Q > hi: weight z
        m1 = (P1[hi] - P1[A]) - A * (P0[hi] - P0[A]) + z * (1.0 - P0[hi])
        m2 = (P3[hi] - P3[A]) - A * (Ph[hi] - Ph[A]) + z * (Ph[n] - Ph[hi])
        v = self.c * m2 / m1 if m1 > 1e-12 else 99.0
        self.be_cache[key] = v
        return v

    # ---------------------------------------------------------------- private feed
    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        o = self.orders.get(event.order_id)
        if o is None:
            return
        st = event.status
        if st is OrderStatus.PARTIALLY_FILLED or st is OrderStatus.FILLED:
            self.pos += event.filled_quantity if o[0] is BUY else -event.filled_quantity
            o[2] = event.remaining_quantity
        elif st is OrderStatus.MODIFIED:
            o[2] = event.remaining_quantity
        if st is OrderStatus.FILLED or st is OrderStatus.CANCELED or st is OrderStatus.REJECTED:
            self.orders.pop(event.order_id, None)
            self.cancelling.discard(event.order_id)

    def _newer(self, oid: str) -> bool:
        m = self.max_id
        return len(oid) > len(m) or (len(oid) == len(m) and oid > m)

    # ---------------------------------------------------------------- public feed
    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        p = self.p
        if trades:
            q = 0
            sgn = 0
            tid = None
            for tr in trades:
                if tr.taker_owner == 'market':
                    q += tr.quantity
                    sgn = 1 if tr.book_side == 'ask' else -1
                    tid = tr.taker_order_id
            if q:
                self.U += sgn * self.c * math.sqrt(q)
                if self._newer(tid):
                    self.max_id = tid
        bids, asks = snapshot.bids, snapshot.asks
        if not bids or not asks:
            return
        U = self.U
        relax = p['relax']
        # new market insert at the touch -> bound on off
        bb = bids[0].price
        for o in bids:
            if o.price != bb:
                break
            if o.owner == 'market' and self._newer(o.order_id):
                self.max_id = o.order_id
                v = bb + p['gap'] - U
                self.lo -= relax
                self.hi += relax
                if v > self.lo:
                    self.lo = v
                    if self.lo > self.hi:
                        self.hi = self.lo + 0.2
        ba = asks[0].price
        for o in asks:
            if o.price != ba:
                break
            if o.owner == 'market' and self._newer(o.order_id):
                self.max_id = o.order_id
                v = ba - p['gap'] - U
                self.lo -= relax
                self.hi += relax
                if v < self.hi:
                    self.hi = v
                    if self.hi < self.lo:
                        self.lo = self.hi - 0.2
        if self.hi - self.lo > p['max_width'] or snapshot.timestamp_us < p['trade_start_us']:
            return
        S = U + 0.5 * (self.lo + self.hi)
        self._take(S, bids, asks)
        self._side(BUY, S, bids, bb, ba)
        self._side(SELL, S, asks, bb, ba)

    def _take(self, S: float, bids, asks) -> None:
        p = self.p
        tk = p['take']
        if not tk:
            return
        cap = self.p['cap']
        if asks[0].price <= S - tk:
            room = cap - self.pos - sum(o[2] for o in self.orders.values() if o[0] is BUY)
            lim = math.floor(S - tk)
            q = 0
            for o in asks:
                if o.price > lim:
                    break
                if o.owner == 'market':
                    q += o.quantity
            q = min(q, room, p['take_max'])
            if q > 0:
                oid = self.send_order(OrderRequest(side=BUY, price=lim, quantity=q, order_type=OrderType.IOC))
                self.orders[oid] = [BUY, lim, q]
                self.cancelling.add(oid)
        if bids[0].price >= S + tk:
            room = cap + self.pos - sum(o[2] for o in self.orders.values() if o[0] is SELL)
            lim = math.ceil(S + tk)
            q = 0
            for o in bids:
                if o.price < lim:
                    break
                if o.owner == 'market':
                    q += o.quantity
            q = min(q, room, p['take_max'])
            if q > 0:
                oid = self.send_order(OrderRequest(side=SELL, price=lim, quantity=q, order_type=OrderType.IOC))
                self.orders[oid] = [SELL, lim, q]
                self.cancelling.add(oid)

    def _side(self, side: Side, S: float, views, bb: int, ba: int) -> None:
        p = self.p
        cap = p['cap']
        buy = side is BUY
        sg = 1 if buy else -1
        touch = bb if buy else ba
        # reachability limit of the next taker on this side: a sell taker's limit is round(S - 1)
        reach = round(S - 1) if buy else round(S + 1)
        inv = sg * self.pos  # > 0: this side adds to |pos|
        m_inv = p['inv_k'] * inv / cap
        # scan the book: queue ahead of each of our orders, qty at each candidate level
        mine = {oid: o for oid, o in self.orders.items() if o[0] is side}
        depth_px = touch - sg * (p['levels'] - 1)
        ahead = {}
        cum = 0
        lvl_cum = {}  # price -> cumulative qty through the end of that level
        for v in views:
            px = v.price
            if (px < depth_px) if buy else (px > depth_px):
                break
            if v.owner == 'user' and v.order_id in mine:
                ahead[v.order_id] = cum
            cum += v.quantity
            lvl_cum[px] = cum
        rest = 0
        have = set()
        for oid, o in mine.items():
            rest += o[2]
            if oid in self.cancelling:
                continue
            px = o[1]
            e = (S - px) if buy else (px - S)
            A = ahead.get(oid)
            if (px < depth_px) if buy else (px > depth_px):  # too deep: frees room for better prices
                self.cancelling.add(oid)
                self.cancel_order(oid)
                continue
            if A is None:  # in flight or deeper than scanned
                if e >= p['m_keep'] + m_inv and ((px < ba) if buy else (px > bb)):
                    have.add(px)
                else:
                    self.cancelling.add(oid)
                    self.cancel_order(oid)
                continue
            reachable = (px >= reach) if buy else (px <= reach)
            ok = (self.ev(A, o[2], e - p['m_keep'] - m_inv) >= -p['kappa']) if reachable else e >= m_inv
            if ok:
                have.add(px)
            else:
                self.cancelling.add(oid)
                self.cancel_order(oid)
        room = (cap - self.pos - rest) if buy else (cap + self.pos - rest)
        if room < p['min_post']:
            return
        # candidates from most aggressive: improve (inside spread), touch, behind ...
        cands = []
        imp = touch + sg
        if (imp < ba) if buy else (imp > bb):
            cands.append(imp)
        for k in range(p['levels']):
            cands.append(touch - sg * k)
        z = room
        chosen = None
        last_ok = None
        prev_cum = 0
        for px in cands:
            e = (S - px) if buy else (px - S)
            Aback = lvl_cum.get(px)
            if Aback is None:
                # empty level (improve) or not seen: everything better than it is ahead
                Aback = 0 if px == imp else cum
            reachable = (px >= reach) if buy else (px <= reach)
            if e >= p['theta'] + m_inv and reachable and self.ev(Aback, z, e - p['m_post'] - m_inv) >= -p['kappa']:
                chosen = px
                break
            if e >= p['m_keep'] + m_inv:
                last_ok = px
        if chosen is None and p['deep_post'] and last_ok is not None:
            chosen = last_ok
        if chosen is None or chosen in have:
            return
        oid = self.send_order(OrderRequest(side=side, price=chosen, quantity=z, order_type=OrderType.LIMIT))
        self.orders[oid] = [side, chosen, z]
