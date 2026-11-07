"""regime_mm: sig_ladder market making in calm stretches + a round-2 regime layer (strategies/peter/round2_findings.md).

Round 2's price process has calm stretches (round-1-like, where the ladder earns) and turbulent ones (exogenous
jumps of up to ~110 ticks, clustered) where every passive maker loses heavily. The big jumps are announced:
about every 200 s the market makers go quiet for ~3.0 s (inserts 180/s -> ~53/s, so all market events
~370/s -> ~120/s) and the jump comes as they return. Direction is a coin flip; |jump| ~ Exp(mean ~31).

States (each callback = one market event):
    CALM    sig_ladder quoting (the core below, unchanged)
    ALERT   market events in the trailing ev_w_ms <= ev_thr: cancel every quote, IOC-flatten the position
            (fl_slip ticks through the touch), remember ref = mid; arm the sniper for alert_max_s
    SNIPE   while armed: once the book or a market trade has moved >= sn_jth ticks from ref, IOC in that
            direction with limit ref +- sn_lim to take the stale quotes the jump left behind (sn_shots shots
            sn_gap_ms apart, up to sn_qty lots); then hold hold_s and unwind
    TURB    a mid move >= tb_move ticks within tb_win_ms (or a snipe) starts it; it lasts until tb_quiet_s pass
            without one. No quoting (tb_cap > 0: quote the core with that cap instead); a position is unwound
            passively at the touch, by IOC after uw_ioc_s
"""

from __future__ import annotations

from collections import deque

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
    # ladder core (ladder_mm defaults)
    'size_top': 200,
    'cap': 100,
    'max_spread': 8,
    'topup': 10,
    # guard
    'g_share': 0.10,  # 0 = off (all guard defaults 0 -> bit-identical to ladder_mm)
    'g_hyst': 0.0,
    'g_qty': 0,  # >0: unsafe while market touch qty on the side <= this
    'g_dep': 0.5,  # >0: unsafe also requires touch qty < g_dep * its peak since this price became the touch
    'flow_k': 0.10,
    'flow_ms': 1000,
    'g_mode': 'shrink',
    'g_keep': 0,
    'g_keep_flat': 1,
    'g_spread1': 1,  # only guard while the spread is 1 tick
    'pre_join': 1,
    'levels_keep': 1,  # ticks behind the touch where an existing (pre-joined) order is kept
    'pj_hold': 0,  # 1: keep a pre-joined order behind the touch after the guard releases (else cancel it)
    'trade_start_us': 0,
    # combination switches (all off = the defaults above, bit for bit)
    'size_deep': 0,  # >0: also rest this many lots one tick behind the touch while the side is safe (q_ladder levels 2)
    'pj_topup': 0,  # 1: while unsafe, top up an existing behind order with freed room (else only post if none)
    'improve_both': 1,  # 1: with a spread >= 3 ticks quote one tick inside on both sides
    'flow2_k': 0.5,  # extra score term: taker flow against the side over the last flow2_ms / 1000 (0 = off)
    'flow2_ms': 200,
    'sev_share': 0.0,  # >0: severe state (score < sev_share and touch qty < sev_dep * peak) also pulls the flattening part
    'sev_dep': 0.2,
    # ---- regime layer (regime_mm)
    'regime': 1,  # 0: plain sig_ladder
    'ev_w_ms': 300,  # outage detector window
    'ev_thr': 60,  # alert when market events in the window <= this (normal ~110, outage ~37)
    'ev_warm_s': 2,  # need this long since the first callback before alerting
    'alert_max_s': 6.0,  # sniper stays armed this long after the alert starts
    'fl_on': 1,  # flatten on alert
    'fl_slip': 2,  # flatten IOC limit: this many ticks through the touch
    'sn_on': 0,  # off: on dev it adds ~+1k (noise); the market's takers clear stale quotes before our 5 ms order lands
    'sn_jth': 4,  # trigger: ask/bid moved >= sn_jth from ref, or a market trade >= sn_jth from ref
    'sn_lim': 12,  # IOC limit = ref +- sn_lim
    'sn_qty': 100,
    'sn_trade': 0,  # also trigger on a market trade >= sn_jth from ref
    'sn_t0': 2.0,  # only trigger between sn_t0 and sn_t1 s after the alert (the jump lands ~2.7 s after it)
    'sn_t1': 4.5,
    'sn_shots': 3,
    'sn_gap_ms': 20,
    'hold_s': 1.0,  # hold the sniped position this long before unwinding
    'tb_on': 1,
    'tb_move': 5,  # mid range within tb_win_ms that starts / extends TURB
    'tb_win_ms': 500,
    'tb_quiet_s': 10.0,
    'tb_cap': 0,  # >0: keep quoting the core during TURB with this cap
    'uw_ioc_s': 5.0,  # passive unwind for this long, then IOC
    'uw_slip': 3,
}

BUY, SELL = Side.BUY, Side.SELL


class ZCoreLadder(BaseStrategy):
    def __init__(self) -> None:
        super().__init__()
        self.p = PARAMS
        self.pos = 0
        self.orders: dict[str, list] = {}  # id -> [side, price, remaining, seq]
        self.cancelling: set[str] = set()
        self.modifying: dict[str, int] = {}  # id -> latest requested quantity
        self.mod_n: dict[str, int] = {}  # id -> modifies in flight (a rejected one leaves the order live)
        self.seq = 0
        self.unsafe = {BUY: False, SELL: False}
        self.flow: deque = deque()  # (ts, signed lots, + = buy aggressor)
        self.flow_sum = 0
        self.peak = {BUY: (None, 0), SELL: (None, 0)}  # side -> (touch price, peak market qty there)
        self.severe = {BUY: False, SELL: False}

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        oid = event.order_id
        o = self.orders.get(oid)
        if o is None:
            return
        st = event.status
        if st in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED):
            self.pos += event.filled_quantity if o[0] == BUY else -event.filled_quantity
            o[2] = event.remaining_quantity
        elif st == OrderStatus.MODIFIED:
            o[2] = event.remaining_quantity
            self._mod_done(oid)
        if st == OrderStatus.REJECTED and self.mod_n.get(oid):
            self._mod_done(oid)  # modify rejected (order already that small): order still live
            if event.remaining_quantity:
                o[2] = event.remaining_quantity
            return
        if st in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
            self.orders.pop(oid, None)
            self.cancelling.discard(oid)
            self.modifying.pop(oid, None)
            self.mod_n.pop(oid, None)

    def _mod_done(self, oid: str) -> None:
        n = self.mod_n.get(oid, 0) - 1
        if n <= 0:
            self.mod_n.pop(oid, None)
            self.modifying.pop(oid, None)
        else:
            self.mod_n[oid] = n

    @staticmethod
    def _touch(orders) -> tuple[int | None, int]:
        price, qty = None, 0
        for o in orders:
            if o.owner != 'market':
                continue
            if price is None:
                price = o.price
            elif o.price != price:
                break
            qty += o.quantity
        return price, qty

    def _room(self, side: Side) -> int:
        rest = sum(o[2] for o in self.orders.values() if o[0] == side)
        if side == BUY:
            return self.p['cap'] - (self.pos + rest)
        return self.p['cap'] + self.pos - rest

    def _send(self, side: Side, price: int, q: int) -> None:
        oid = self.send_order(OrderRequest(side=side, price=price, quantity=q, order_type=OrderType.LIMIT))
        self.seq += 1
        self.orders[oid] = [side, price, q, self.seq]

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        p = self.p
        now = snapshot.timestamp_us
        if p['flow_k'] or p['flow2_k']:
            for tr in trades:
                if tr.taker_owner == 'market':
                    s = tr.quantity if tr.book_side == 'ask' else -tr.quantity
                    self.flow.append((now, s))
                    self.flow_sum += s
            lim = now - p['flow_ms'] * 1000
            while self.flow and self.flow[0][0] < lim:
                self.flow_sum -= self.flow.popleft()[1]

        (bb, qb), (ba, qa) = self._touch(snapshot.bids), self._touch(snapshot.asks)
        if bb is None or ba is None or ba - bb > p['max_spread'] or now < p['trade_start_us']:
            return

        f2 = 0
        if p['flow2_k']:
            lim2 = now - p['flow2_ms'] * 1000
            for ts, q in reversed(self.flow):
                if ts < lim2:
                    break
                f2 += q
        tot = qb + qa
        for side, q_side, against, against2, px in ((BUY, qb, -self.flow_sum, -f2, bb), (SELL, qa, self.flow_sum, f2, ba)):
            pk_px, pk = self.peak[side]
            pk = max(pk, q_side) if pk_px == px else q_side
            self.peak[side] = (px, pk)
            if not p['g_share'] and not p['g_qty']:
                self.unsafe[side] = False
                continue
            score = q_side / tot - p['flow_k'] * against / 1000.0 - p['flow2_k'] * against2 / 1000.0
            self.severe[side] = bool(p['sev_share'] and ba - bb == 1 and score < p['sev_share']
                                     and q_side < p['sev_dep'] * pk)
            if p['g_spread1'] and ba - bb > 1:
                bad, ok = False, True
            else:
                bad = (p['g_share'] and score < p['g_share']) or (p['g_qty'] and q_side <= p['g_qty'])
                if p['g_dep'] and q_side >= p['g_dep'] * pk:
                    bad = False
                ok = ((not p['g_share'] or score >= p['g_share'] + p['g_hyst']) and
                      (not p['g_qty'] or q_side > p['g_qty'])) or (p['g_dep'] and q_side >= p['g_dep'] * pk)
            if bad:
                self.unsafe[side] = True
            elif ok:
                self.unsafe[side] = False

        qb_px, qa_px = bb, ba
        if p['improve_both'] and ba - bb >= 3:
            qb_px, qa_px = bb + 1, ba - 1
        for side, touch, step in ((BUY, qb_px, -1), (SELL, qa_px, 1)):
            unsafe = self.unsafe[side]
            mine = sorted(((o[3], oid, o) for oid, o in self.orders.items()
                           if o[0] == side and oid not in self.cancelling), key=lambda x: x[0])
            at_touch, behind = [], []
            for _, oid, o in mine:
                d = (touch - o[1]) if side == BUY else (o[1] - touch)
                if d == 0:
                    at_touch.append((oid, o))
                elif 0 < d <= p['levels_keep'] and ((p['pre_join'] and (unsafe or p['pj_hold'])) or
                                                    (p['size_deep'] and d == 1)):
                    behind.append((oid, o))
                else:
                    self.cancelling.add(oid)
                    self.cancel_order(oid)
            if unsafe:
                mode = p['g_mode']
                if mode == 'cancel':
                    keep = 0
                elif mode == 'freeze':
                    keep = 10**9
                else:
                    keep = p['g_keep']
                    if p['g_keep_flat'] and not self.severe[side] and (
                            (side == SELL and self.pos > 0) or (side == BUY and self.pos < 0)):
                        keep = max(keep, abs(self.pos))
                for oid, o in at_touch:
                    cur = min(o[2], self.modifying.get(oid, o[2]))
                    if keep <= 0:
                        self.cancelling.add(oid)
                        self.cancel_order(oid)
                    elif cur > keep:
                        self.modifying[oid] = keep
                        self.mod_n[oid] = self.mod_n.get(oid, 0) + 1
                        self.modify_order(oid, keep)
                        keep = 0
                    else:
                        keep -= cur
                if p['pre_join'] and not behind:
                    q = min(p['size_top'], self._room(side))
                    if q > 0 and touch + step > 0:
                        self._send(side, touch + step, q)
                elif p['pre_join'] and p['pj_topup']:
                    q = self._room(side)
                    if q >= max(1, p['topup']) and touch + step > 0:
                        self._send(side, touch + step, q)
                continue
            have = sum(o[2] for _, o in at_touch)
            target = p['size_top']
            if have:
                short = target - have
                target = short if p['topup'] and short >= p['topup'] else 0
            q = min(target, self._room(side))
            if q > 0:
                self._send(side, touch, q)
            if p['size_deep'] and touch + step > 0:
                have_b = sum(o[2] for _, o in behind)
                target = p['size_deep']
                if have_b:
                    short = target - have_b
                    target = short if p['topup'] and short >= p['topup'] else 0
                q = min(target, self._room(side))
                if q > 0:
                    self._send(side, touch + step, q)


class RegimeMM(ZCoreLadder):
    """The class the simulator picks (alphabetically first): regime layer around the ZCoreLadder core."""

    def __init__(self) -> None:
        super().__init__()
        self.evq: deque = deque()
        self.first_us = None
        self.last_mid = None
        self.mids: deque = deque()
        self.cur_cap = self.p['cap']
        self.armed = False
        self.alert_until = -1
        self.ref = 0.0
        self.sn_dir = 0
        self.sn_left = 0
        self.sn_next = 0
        self.hold_until = -1
        self.turb_until = -1
        self.ioc_ids: set = set()
        self.uw_oid = None
        self.uw_start = None
        self.n_alert = 0
        self.n_snipe = 0

    # ------------------------------------------------------------ bookkeeping
    def _room(self, side: Side) -> int:
        rest = sum(o[2] for o in self.orders.values() if o[0] == side)
        if side == BUY:
            return self.cur_cap - (self.pos + rest)
        return self.cur_cap + self.pos - rest

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        super().on_private_order_update(event)
        oid = event.order_id
        if oid not in self.orders:
            self.ioc_ids.discard(oid)
            if oid == self.uw_oid:
                self.uw_oid = None

    def _ioc(self, side: Side, price: int, q: int) -> None:
        q = min(q, self._room(side))
        if q <= 0 or price <= 0:
            return
        oid = self.send_order(OrderRequest(side=side, price=price, quantity=q, order_type=OrderType.IOC))
        self.seq += 1
        self.orders[oid] = [side, price, q, self.seq]
        self.ioc_ids.add(oid)

    def _cancel_quotes(self) -> None:
        for oid in list(self.orders):
            if oid in self.cancelling or oid in self.ioc_ids or oid == self.uw_oid:
                continue
            self.cancelling.add(oid)
            self.cancel_order(oid)

    def _flatten(self, bb, ba, slip: int) -> None:
        if any(self.orders[o][0] == (SELL if self.pos > 0 else BUY) for o in self.ioc_ids if o in self.orders):
            return  # one in flight already
        if self.pos > 0 and bb is not None:
            self._ioc(SELL, bb - slip, self.pos)
        elif self.pos < 0 and ba is not None:
            self._ioc(BUY, ba + slip, -self.pos)

    def _unwind(self, now: int, bb, ba) -> None:
        p = self.p
        if self.pos == 0:
            if self.uw_oid is not None and self.uw_oid not in self.cancelling:
                self.cancelling.add(self.uw_oid)
                self.cancel_order(self.uw_oid)
            self.uw_start = None
            return
        if now < self.hold_until:
            return
        if self.uw_start is None:
            self.uw_start = now
        side = SELL if self.pos > 0 else BUY
        if now - self.uw_start > p['uw_ioc_s'] * 1e6:
            if self.uw_oid is not None:
                if self.uw_oid not in self.cancelling:
                    self.cancelling.add(self.uw_oid)
                    self.cancel_order(self.uw_oid)
                return
            self._flatten(bb, ba, p['uw_slip'])
            return
        px = ba if side == SELL else bb
        if px is None:
            return
        if self.uw_oid is not None:
            o = self.orders.get(self.uw_oid)
            if o is not None and o[1] != px and self.uw_oid not in self.cancelling:
                self.cancelling.add(self.uw_oid)
                self.cancel_order(self.uw_oid)
            return
        if self.ioc_ids:
            return
        q = min(abs(self.pos), self.cur_cap + (self.pos if side == SELL else -self.pos))
        if q > 0:
            oid = self.send_order(OrderRequest(side=side, price=px, quantity=q, order_type=OrderType.LIMIT))
            self.seq += 1
            self.orders[oid] = [side, px, q, self.seq]
            self.uw_oid = oid

    # ------------------------------------------------------------ main
    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        p = self.p
        if not p['regime']:
            super().on_order_book_change(snapshot, trades)
            return
        now = snapshot.timestamp_us
        if self.first_us is None:
            self.first_us = now
        evq = self.evq
        evq.append(now)
        lim = now - p['ev_w_ms'] * 1000
        while evq[0] < lim:
            evq.popleft()
        (bb, _), (ba, _) = self._touch(snapshot.bids), self._touch(snapshot.asks)
        mid = (bb + ba) / 2 if bb is not None and ba is not None else self.last_mid
        if mid is None:
            return
        self.last_mid = mid
        trading = now >= p['trade_start_us']

        # turbulence: mid range over the trailing tb_win_ms
        if p['tb_on']:
            ms = self.mids
            ms.append((now, mid))
            lim = now - p['tb_win_ms'] * 1000
            while ms[0][0] < lim:
                ms.popleft()
            lo = hi = mid
            for _, m in ms:
                if m < lo:
                    lo = m
                elif m > hi:
                    hi = m
            if hi - lo >= p['tb_move']:
                self.turb_until = now + p['tb_quiet_s'] * 1e6

        # outage alert
        if self.armed and now > self.alert_until:
            self.armed = False
        if (not self.armed and len(evq) <= p['ev_thr'] and now - self.first_us >= p['ev_warm_s'] * 1e6):
            self.armed = True
            self.alert_until = now + p['alert_max_s'] * 1e6
            self.ref = mid
            self.sn_dir = 0
            self.n_alert += 1
            if trading:
                self._cancel_quotes()
                if p['fl_on']:
                    self._flatten(bb, ba, p['fl_slip'])
        if not trading:
            return

        # sniper
        age = (now - (self.alert_until - p['alert_max_s'] * 1e6)) / 1e6
        if self.armed and p['sn_on'] and self.sn_dir == 0 and p['sn_t0'] <= age <= p['sn_t1']:
            j, ref = p['sn_jth'], self.ref
            up = (ba is not None and ba >= ref + 0.5 + j) or (ba is None and bb is not None and bb >= ref - 0.5 + j)
            dn = (bb is not None and bb <= ref - 0.5 - j) or (bb is None and ba is not None and ba <= ref + 0.5 - j)
            for tr in (trades if p['sn_trade'] else ()):
                if tr.taker_owner == 'market':
                    if tr.book_side == 'ask' and tr.price >= ref + j:
                        up = True
                    elif tr.book_side == 'bid' and tr.price <= ref - j:
                        dn = True
            if up != dn:
                self.sn_dir = 1 if up else -1
                self.sn_left = p['sn_shots']
                self.sn_next = now
                self.hold_until = now + p['hold_s'] * 1e6
                self.turb_until = max(self.turb_until, now + p['tb_quiet_s'] * 1e6)
                self.n_snipe += 1
                self._cancel_quotes()
        if self.sn_dir and self.sn_left > 0 and now >= self.sn_next:
            self.cur_cap = p['cap']
            if self.sn_dir > 0:
                self._ioc(BUY, int(self.ref + p['sn_lim']), p['sn_qty'] - self.pos)
            else:
                self._ioc(SELL, int(self.ref - p['sn_lim'] + 0.999), p['sn_qty'] + self.pos)
            self.sn_left -= 1
            self.sn_next = now + p['sn_gap_ms'] * 1000

        defensive = self.armed or now < self.turb_until
        if defensive:
            if p['tb_cap'] and not self.armed and now >= self.hold_until:
                self.cur_cap = p['tb_cap']
                if self.uw_oid is not None and self.uw_oid not in self.cancelling:
                    self.cancelling.add(self.uw_oid)
                    self.cancel_order(self.uw_oid)
                if not self.ioc_ids:
                    super().on_order_book_change(snapshot, trades)
                return
            self._cancel_quotes()
            self._unwind(now, bb, ba)
            return
        # calm
        self.sn_dir = 0
        self.uw_start = None
        if self.uw_oid is not None and self.uw_oid not in self.cancelling:
            self.cancelling.add(self.uw_oid)
            self.cancel_order(self.uw_oid)
        self.cur_cap = p['cap']
        if not self.ioc_ids:
            super().on_order_book_change(snapshot, trades)
