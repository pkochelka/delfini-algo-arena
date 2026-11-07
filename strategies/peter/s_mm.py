"""s_mm: market maker that tracks the hidden fair price S of the round-1 generator.

S = S0 + c * sum(sign * sqrt(filled qty)) over market taker orders (strategies/peter/round1_generator.md).
S0 is learned online from maker inserts at the touch: every market bid satisfies bid <= S - 0.05 and every
market ask ask >= S + 0.05, so each new touch insert bounds off = S - U. The bounds relax very slowly
(`relax`, per observation) so a drift in U (our liquidity changes how much a taker fills) is corrected.
Converges within ~1 s; error ~0.01 tick afterwards. Must stay slow: relax 5e-4 gave 0.33-tick-wide bounds.

Quoting (defaults = best found, wf01-10 of data/splits_r1/wf, 745 s folds):
  * post at tb = floor(S - theta) / ta = ceil(S + theta), never crossing the opposite touch
  * keep a resting order while edge >= theta_keep AND the next taker can still reach it
    (bid >= round(S - 1), ask <= round(S + 1): takers' IOC limits), up to keep_depth ticks behind target.
    Orders left behind when S moves away are therefore kept while reachable (they hold queue priority);
    that rule alone was +4k/fold.
  * take stale quotes (ask <= S - take, bid >= S + take) with IOCs, up to take_max.
  * room is strict: pos + every order on the side (incl. cancelling / IOC in flight) <= cap.

Results (fold mean, wf01-10):  sig_ladder 30.2k | s_mm 45.7k (+51%)
  holdout wf11-12: 32.1k -> 44.6k;  synthetic generator tapes seeds 101-112: 34.7k -> 47.9k (12/12)
  full round1 tape: 727,048 (sig_ladder 489,589), 0 breaches, ~100 s runtime.
PnL mix: ~80% maker fills (65% of maker lots at the front of the queue, ~0.36 tick/lot vs S after the
taker's jump), ~20% IOC takes (~0.37 tick/lot).

Tested and flat or worse (all behind flags, defaults off): inventory skew (+-0.05..0.4), lower take
threshold for inventory-reducing takes (take_red), separate improve threshold (theta_imp1), pre_join /
size_top split, shrinking orders left behind (behind_keep: -1.6..-5.6k), queue-dependent keep
threshold (a_front/tk_front), reusing room of unreachable cancels (free_unreach: margin 0 breaches,
margin 1 no gain), v3 touch-relative logic (-8k). The position cap is the binding resource: PnL is almost
linear in cap (60: 30.7k, 100: 45.7k, 150: 59k gross of breach penalty), but resting beyond it breaches.
An EV-per-taker keep rule from the taker-size distribution (s_mm2.py) loses: it ignores the option value
of queue position.
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
    'gap': 0.05,  # min distance of a maker quote from S
    'relax': 0.000002,  # bound relaxation per S-observation (drift correction)
    'max_width': 0.3,  # trade only while hi - lo <= this
    'cap': 100,
    'theta': 0.30,  # post at the best price with edge >= theta
    'theta_keep': 0.10,  # keep a resting order while edge >= theta_keep
    'keep_depth': 1,  # keep orders up to this many ticks behind the target price
    'take': 0.15,  # IOC when the opposite touch is >= take on our side of S (0 = off)
    'take_max': 100,
    'min_post': 1,
    'inv_skew': 0.0,  # theta += inv_skew * pos/cap on the side that adds inventory
    'trade_start_us': 0,
    'v3': 0,
    'free_unreach': 0,
    'a_front': -1,  # >=0: orders at the touch with <= a_front market lots ahead use tk_front as keep threshold
    'tk_front': 0.1,
    'behind_keep': 10**9,  # shrink an order left behind the target price to this many lots
    'free_margin': 0,
    'theta_imp1': None,  # separate edge threshold for improving the market touch
    'take_red': 0.15,  # take threshold when the take reduces |pos| (at |pos| = cap)
    'take_pow': 1.0,
    'max_keep_e': 99.0,
    'reach': 1,
    'reach_off': 1.0,
    'size_top': 1000,
    'pre_join': 0,
    'pj_min': 10,
    'theta_imp': 0.3,
    'imp_cancel': 1,
    'deep_cancel': 0,
    'deep_post': 1,
    'size': 200,
}

BUY, SELL = Side.BUY, Side.SELL


class SMM(BaseStrategy):
    def __init__(self) -> None:
        super().__init__()
        self.p = PARAMS
        self.c = self.p['c']
        self.pos = 0
        self.orders: dict[str, list] = {}  # id -> [side, price, remaining]
        self.cancelling: set[str] = set()
        self.freed: set[str] = set()
        self.mod_req: dict[str, int] = {}
        self.ahead: dict[str, int] = {}
        self.U = 0.0
        self.lo = -1e18
        self.hi = 1e18
        self.max_id = ''
        self.last_taker = None
        self.taker_q = 0
        self.taker_sign = 0
        self.S = None

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
            self.mod_req.pop(event.order_id, None)
        if st is OrderStatus.REJECTED and event.order_id in self.mod_req:
            self.mod_req.pop(event.order_id, None)  # modify rejected: order still live
            return
        if st is OrderStatus.FILLED or st is OrderStatus.CANCELED or st is OrderStatus.REJECTED:
            self.orders.pop(event.order_id, None)
            self.cancelling.discard(event.order_id)
            self.freed.discard(event.order_id)
            self.mod_req.pop(event.order_id, None)

    # ---------------------------------------------------------------- S tracking
    def _newer(self, oid: str) -> bool:
        m = self.max_id
        return len(oid) > len(m) or (len(oid) == len(m) and oid > m)

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        p = self.p
        # impact of market takers: one market event per call, so at most one market taker
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
        # touch levels (market + ours), detect a new market insert at the touch
        bb = ba = None
        qb = qa = 0
        U = self.U
        relax = p['relax']
        ahead = self.ahead
        ahead.clear()
        for o in bids:
            if bb is None:
                bb = o.price
            elif o.price != bb:
                break
            if o.owner != 'market':
                ahead[o.order_id] = qb
            else:
                qb += o.quantity
                oid = o.order_id
                if self._newer(oid):
                    self.max_id = oid
                    v = bb + p['gap'] - U
                    self.lo -= relax
                    self.hi += relax
                    if v > self.lo:
                        self.lo = v
                        if self.lo > self.hi:
                            self.hi = self.lo + 0.2
        for o in asks:
            if ba is None:
                ba = o.price
            elif o.price != ba:
                break
            if o.owner != 'market':
                ahead[o.order_id] = qa
            else:
                qa += o.quantity
                oid = o.order_id
                if self._newer(oid):
                    self.max_id = oid
                    v = ba - p['gap'] - U
                    self.lo -= relax
                    self.hi += relax
                    if v < self.hi:
                        self.hi = v
                        if self.hi < self.lo:
                            self.lo = self.hi - 0.2
        if bb is None or ba is None:
            return
        if self.hi - self.lo > p['max_width'] or snapshot.timestamp_us < p['trade_start_us']:
            return
        S = U + 0.5 * (self.lo + self.hi)
        self.S = S
        self._quote(S, bb, ba, bids, asks)

    # ---------------------------------------------------------------- quoting
    def _quote(self, S: float, bb: int, ba: int, bids, asks) -> None:
        p = self.p
        cap = p['cap']
        pos = self.pos
        # taking stale quotes
        if p['take']:
            tkb = p['take'] if pos >= 0 else p['take_red'] + (p['take'] - p['take_red']) * (1 + pos / cap) ** p['take_pow']
            tka = p['take'] if pos <= 0 else p['take_red'] + (p['take'] - p['take_red']) * (1 - pos / cap) ** p['take_pow']
            if ba <= S - tkb:
                room = cap - pos - sum(o[2] for o in self.orders.values() if o[0] is BUY)
                lim = math.floor(S - tkb)
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
                    pass
            if bb >= S + tka:
                room = cap + pos - sum(o[2] for o in self.orders.values() if o[0] is SELL)
                lim = math.ceil(S + tka)
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
                    pass

        sk = p['inv_skew'] * pos / cap
        if p['v3']:
            self._side3(BUY, S, bb, ba, bids, sk)
            self._side3(SELL, S, bb, ba, asks, -sk)
            return
        th_b = p['theta'] + sk
        th_a = p['theta'] - sk
        tk_b = p['theta_keep'] + sk
        tk_a = p['theta_keep'] - sk
        # target prices
        tb = min(math.floor(S - th_b), ba - 1)
        ta = max(math.ceil(S + th_a), bb + 1)
        if p['theta_imp1'] is not None:
            mbb = mba = None
            for v in bids:
                if v.owner == 'market':
                    mbb = v.price
                    break
            for v in asks:
                if v.owner == 'market':
                    mba = v.price
                    break
            ti = p['theta_imp1']
            if mbb is not None:
                x = min(math.floor(S - ti - sk), ba - 1)
                if x > mbb:
                    tb = x
                elif tb > mbb:
                    tb = mbb
            if mba is not None:
                x = max(math.ceil(S + ti - sk), bb + 1)
                if x < mba:
                    ta = x
                elif ta < mba:
                    ta = mba
        kd = p['keep_depth']
        rb = round(S - p['reach_off'])  # lowest bid price the next sell taker reaches
        ra = round(S + p['reach_off'])
        rest_b = rest_a = 0
        have_b = have_a = have_b2 = have_a2 = False
        for oid, o in list(self.orders.items()):
            if oid in self.cancelling:
                if oid in self.freed:
                    continue
                if o[0] is BUY:
                    rest_b += o[2]
                else:
                    rest_a += o[2]
                continue
            side, px = o[0], o[1]
            fu = p['free_unreach']
            if p['a_front'] >= 0:
                fr = self.ahead.get(oid, 10**9) <= p['a_front']
                tk_b = (p['tk_front'] if fr else p['theta_keep']) + sk
                tk_a = (p['tk_front'] if fr else p['theta_keep']) - sk
            if side is BUY:
                ok = S - px >= tk_b and px >= tb - kd and px < ba and S - px <= p['max_keep_e'] and (not p['reach'] or px >= rb)
                if fu and not ok and px < rb - p['free_margin']:
                    self.freed.add(oid)
                else:
                    rest_b += o[2]
                if ok:
                    if px == tb:
                        have_b = True
                    elif px == tb - 1:
                        have_b2 = True
                    if px < tb and p['behind_keep'] < o[2] and oid not in self.mod_req:
                        nq = max(1, p['behind_keep'])
                        self.mod_req[oid] = nq
                        self.modify_order(oid, nq)
                else:
                    self.cancelling.add(oid)
                    self.cancel_order(oid)
            else:
                ok = px - S >= tk_a and px <= ta + kd and px > bb and px - S <= p['max_keep_e'] and (not p['reach'] or px <= ra)
                if fu and not ok and px > ra + p['free_margin']:
                    self.freed.add(oid)
                else:
                    rest_a += o[2]
                if ok:
                    if px == ta:
                        have_a = True
                    elif px == ta + 1:
                        have_a2 = True
                    if px > ta and p['behind_keep'] < o[2] and oid not in self.mod_req:
                        nq = max(1, p['behind_keep'])
                        self.mod_req[oid] = nq
                        self.modify_order(oid, nq)
                else:
                    self.cancelling.add(oid)
                    self.cancel_order(oid)
        mp = p['min_post']
        st = p['size_top']
        if not have_b:
            q = min(cap - pos - rest_b, st)
            if q >= mp:
                oid = self.send_order(OrderRequest(side=BUY, price=tb, quantity=q, order_type=OrderType.LIMIT))
                self.orders[oid] = [BUY, tb, q]
                rest_b += q
        if not have_a:
            q = min(cap + pos - rest_a, st)
            if q >= mp:
                oid = self.send_order(OrderRequest(side=SELL, price=ta, quantity=q, order_type=OrderType.LIMIT))
                self.orders[oid] = [SELL, ta, q]
                rest_a += q
        if p['pre_join'] and kd >= 1:
            if not have_b2 and tb - 1 > 0:
                q = cap - pos - rest_b
                if q >= p['pj_min']:
                    oid = self.send_order(OrderRequest(side=BUY, price=tb - 1, quantity=q, order_type=OrderType.LIMIT))
                    self.orders[oid] = [BUY, tb - 1, q]
            if not have_a2:
                q = cap + pos - rest_a
                if q >= p['pj_min']:
                    oid = self.send_order(OrderRequest(side=SELL, price=ta + 1, quantity=q, order_type=OrderType.LIMIT))
                    self.orders[oid] = [SELL, ta + 1, q]

    def _side3(self, side: Side, S: float, bb: int, ba: int, views, sk: float) -> None:
        """v3 per-side logic: improve / join / deep target with separate thresholds."""
        p = self.p
        buy = side is BUY
        sg = 1 if buy else -1
        # market-only touch on this side
        mt = None
        for v in views:
            if v.owner == 'market':
                mt = v.price
                break
        if mt is None:
            return
        opp = ba if buy else bb  # opposite touch incl. ours: never cross it
        # edge of a price: positive = good for us
        th_imp = p['theta_imp'] + sk
        th_join = p['theta'] + sk
        th_keep = p['theta_keep'] + sk
        if buy:
            best = min(math.floor(S - th_imp), opp - 1)
            if best > mt:
                tgt, mode = best, 0  # improve
            elif S - mt >= th_join:
                tgt, mode = mt, 1  # join
            else:
                tgt, mode = min(math.floor(S - th_join), opp - 1), 2  # behind
        else:
            best = max(math.ceil(S + th_imp), opp + 1)
            if best < mt:
                tgt, mode = best, 0
            elif mt - S >= th_join:
                tgt, mode = mt, 1
            else:
                tgt, mode = max(math.ceil(S + th_join), opp + 1), 2
        kd = p['keep_depth']
        rest = 0
        have = False
        cap = p['cap']
        for oid, o in list(self.orders.items()):
            if o[0] is not side:
                continue
            rest += o[2]
            if oid in self.cancelling:
                continue
            px = o[1]
            e = sg * (S - px)
            behind = sg * (mt - px)  # ticks behind the market touch (<0: inside it)
            ok = e >= th_keep and sg * (opp - px) > 0 and behind <= kd
            if ok and mode == 0 and p['imp_cancel'] and sg * (tgt - px) > 0:
                ok = False  # free the room for the improving price
            if ok and mode == 2 and p['deep_cancel'] and px != tgt:
                ok = False
            if ok:
                if px == tgt:
                    have = True
            else:
                self.cancelling.add(oid)
                self.cancel_order(oid)
        if have or (mode == 2 and not p['deep_post']):
            return
        room = (cap - self.pos - rest) if buy else (cap + self.pos - rest)
        q = min(room, p['size'])
        if q >= p['min_post']:
            oid = self.send_order(OrderRequest(side=side, price=tgt, quantity=q, order_type=OrderType.LIMIT))
            self.orders[oid] = [side, tgt, q]
