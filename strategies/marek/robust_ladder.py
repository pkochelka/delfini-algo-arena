"""robust_ladder: strategies/peter/ladder_mm.py plus cheap regime guards for unseen tapes.

The quoting logic is ladder_mm's (rest size_top lots at the touch on both sides, never re-send a
still-wanted order, worst-case |pos + open orders on a side| <= cap). On round1 its mark-to-market
PnL is extremely smooth (trailing 60 s PnL never below -420, trailing 300 s never below +5,400), so a
guard keyed on our own equity curve costs almost nothing there, while on tapes where the edge is
gone (momentum, informed flow) the same curve trends down within minutes.

Guards (all behind PARAMS; 0 disables):
    kill switch   equity E = cash + pos * mid, sampled each second. Enter DEFENSIVE when, since the
                  last mode switch, E drops ks_dd below its trailing ks_w1-second peak, or the PnL over
                  the trailing ks_w2 seconds is below -ks_l2. DEFENSIVE uses cap ks_cap for ks_cool s;
                  each re-trigger within ks_reset s multiplies the cooldown by ks_mult (<= ks_cool_max).
    vr guard      variance ratio VR(q) of 1 s mid returns over the trailing vr_win s; above vr_max the
                  tape is trending (round1 ~1.0, sample_v2 ~1.1, random_stress ~1.6, sample ~2.4) and
                  the cap is limited to vr_cap.
    shrink        when the effective cap falls, resting orders on the adding side are shrunk with
                  modify_order (keeps queue priority) so the worst case still fits.
    flatten       ks_flatten=1: on entering DEFENSIVE, IOC the position down to the defensive cap.
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
    # ladder_mm
    'size_top': 200,
    'cap': 100,
    'max_spread': 8,
    'topup': 10,
    'trade_start_us': 0,
    # kill switch on our own equity curve
    'ks_w1': 60,  # s, trailing-peak window
    'ks_dd': 1500,  # $ drop below the trailing ks_w1 peak that triggers
    'ks_w2': 300,  # s
    'ks_l2': 0,  # trailing ks_w2 PnL below -ks_l2 triggers (needs ks_w2 s of history since the last switch)
    'ks_cap': 20,  # cap while DEFENSIVE
    'ks_cool': 120,  # s in DEFENSIVE before probing NORMAL again
    'ks_mult': 2.0,  # cooldown multiplier on a re-trigger
    'ks_cool_max': 1800,
    'ks_cap_decay': 0.5,  # defensive cap = ks_cap * decay^(streak-1): 20, 10, 5, 2, 1, 0
    'ks_reset': 900,  # s in NORMAL after which the streak (cooldown and cap escalation) resets
    'ks_flatten': 0,
    # >0: ks_dd scales with max(1, sigma / ks_sigma_ref), sigma = EWMA of 1 s mid changes (ticks/sqrt s);
    # round1's sigma is ~0.92, so a tape with bigger tick moves doesn't trip the switch on noise
    'ks_sigma_ref': 0.92,
    'ks_sigma_hl': 300,
    # variance-ratio regime guard
    'vr_q': 10,
    'vr_win': 900,  # s of 1 s returns
    'vr_min_n': 300,  # need this many returns before acting
    'vr_max': 0.0,  # >0 enables: VR above this -> cap vr_cap
    'vr_cap': 30,
    'kill': 1,  # 0 disables the kill switch
    # depth-scaled cap: cap <= depth_k * EWMA(mean touch qty per side), floored at depth_min
    'depth_k': 0.1,  # round1 touch ~1300 -> cap 100 untouched; vscale0.3 (touch ~400) -> ~40
    'depth_min': 25,
    'depth_hl': 60,  # s, EWMA half-life (per-second samples)
}

BUY, SELL = Side.BUY, Side.SELL


class RobustLadder(BaseStrategy):
    def __init__(self) -> None:
        super().__init__()
        self.p = PARAMS
        self.pos = 0
        self.cash = 0.0
        self.orders: dict[str, list] = {}  # id -> [side, price, remaining]
        self.cancelling: set[str] = set()
        self.ioc: str | None = None
        self.modifying: dict[str, int] = {}  # id -> requested quantity
        # equity sampling
        self.next_sample_us = 0
        self.eq: deque = deque()  # (t_s, equity) since the last mode switch
        self.defensive_until = -1.0
        self.streak = 0  # triggers since the last long calm spell
        self.def_cap = PARAMS['ks_cap']
        self.last_switch = 0.0
        self.triggers = 0
        # vr
        self.mids: deque = deque()
        self.vr = 1.0
        self.depth = -1.0
        self.var1 = -1.0
        self.last_mid = None
        self.touch_qty = 0.0

    # ------------------------------------------------------------------ fills
    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        o = self.orders.get(event.order_id)
        if o is None:
            return
        st = event.status
        if st in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED):
            q = event.filled_quantity
            if o[0] == BUY:
                self.pos += q
                self.cash -= q * event.fill_price
            else:
                self.pos -= q
                self.cash += q * event.fill_price
            o[2] = event.remaining_quantity
        elif st == OrderStatus.MODIFIED:
            o[2] = event.remaining_quantity
            self.modifying.pop(event.order_id, None)
        if st == OrderStatus.REJECTED and event.order_id in self.modifying:
            # a modify that lost the race (order already smaller): the order itself is still live
            self.modifying.pop(event.order_id, None)
            o[2] = event.remaining_quantity
            return
        if st in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
            self.orders.pop(event.order_id, None)
            self.cancelling.discard(event.order_id)
            self.modifying.pop(event.order_id, None)
            if event.order_id == self.ioc:
                self.ioc = None

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

    def _rest(self, side: Side) -> int:
        # cancels and modifies can lose the race against a fill: count every live order in full, the
        # flatten IOC too (ladder_mm leaves it out, which is why its flat_at flag breaches the limit)
        return sum(o[2] for o in self.orders.values() if o[0] == side)

    def _room(self, side: Side, cap: int) -> int:
        rest = self._rest(side)
        if side == BUY:
            return cap - (self.pos + rest)
        return cap + self.pos - rest

    # ------------------------------------------------------------------ regime
    def _update_regime(self, t_s: float, mid: float) -> None:
        p = self.p
        eq = self.cash + self.pos * mid
        self.eq.append((t_s, eq))
        if self.last_mid is not None:
            d2 = (mid - self.last_mid) ** 2
            if self.var1 < 0:
                self.var1 = d2
            else:
                self.var1 += (1 - 0.5 ** (1 / p['ks_sigma_hl'])) * (d2 - self.var1)
        self.last_mid = mid
        if self.depth < 0:
            self.depth = self.touch_qty
        else:
            a = 1 - 0.5 ** (1 / p['depth_hl'])
            self.depth += a * (self.touch_qty - self.depth)
        keep = max(p['ks_w1'], p['ks_w2']) + 2
        while self.eq and self.eq[0][0] < t_s - keep:
            self.eq.popleft()
        if p['vr_max']:
            self.mids.append(mid)
            if len(self.mids) > p['vr_win'] + 1:
                self.mids.popleft()
            self._vr()
        if not p['kill'] or t_s < self.defensive_until:
            return
        if self.streak and t_s - self.defensive_until > p['ks_reset']:
            self.streak = 0  # calm for long enough: forget the escalation
        peak = max(e for t, e in self.eq if t >= t_s - p['ks_w1'])
        dd = p['ks_dd']
        if p['ks_sigma_ref'] and self.var1 > 0:
            dd *= max(1.0, self.var1 ** 0.5 / p['ks_sigma_ref'])
        hit = dd and eq < peak - dd
        if not hit and p['ks_w2'] and self.eq[0][0] <= t_s - p['ks_w2']:
            past = next(e for t, e in self.eq if t >= t_s - p['ks_w2'])
            hit = eq - past < -p['ks_l2']
        if hit:
            self.streak += 1
            self.triggers += 1
            cool = min(p['ks_cool'] * p['ks_mult'] ** (self.streak - 1), p['ks_cool_max'])
            self.def_cap = int(p['ks_cap'] * p['ks_cap_decay'] ** (self.streak - 1))
            self.defensive_until = t_s + cool
            self.last_switch = t_s
            self.eq.clear()
            self.eq.append((t_s, eq))

    def _vr(self) -> None:
        m = self.mids
        q = self.p['vr_q']
        n = len(m) - 1
        if n < max(self.p['vr_min_n'], 3 * q):
            self.vr = 1.0
            return
        ml = list(m)
        r1 = [ml[i + 1] - ml[i] for i in range(n)]
        mu = (ml[-1] - ml[0]) / n
        v1 = sum((x - mu) ** 2 for x in r1) / n
        if v1 <= 0:
            self.vr = 1.0
            return
        rq = [ml[i + q] - ml[i] - q * mu for i in range(n - q + 1)]
        vq = sum(x * x for x in rq) / len(rq)
        self.vr = vq / (q * v1)

    def _cap(self, t_s: float) -> int:
        cap = self.p['cap']
        if self.p['kill'] and t_s < self.defensive_until:
            cap = min(cap, self.def_cap)
        if self.p['depth_k'] and self.depth >= 0:
            cap = min(cap, max(self.p['depth_min'], int(self.p['depth_k'] * self.depth)))
        if self.p['vr_max'] and self.vr > self.p['vr_max']:
            cap = min(cap, self.p['vr_cap'])
        return cap

    # ------------------------------------------------------------------ main
    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        (bb, qb), (ba, qa) = self._touch(snapshot.bids), self._touch(snapshot.asks)
        p = self.p
        ts = snapshot.timestamp_us
        if bb is None or ba is None or ba - bb > p['max_spread'] or ts < p['trade_start_us']:
            return
        t_s = ts / 1e6
        if ts >= self.next_sample_us:
            self.touch_qty = (qb + qa) / 2
            was_def = t_s < self.defensive_until
            self._update_regime(t_s, (bb + ba) / 2)
            self.next_sample_us = ts + 1_000_000
            if p['ks_flatten'] and not was_def and t_s < self.defensive_until and self.ioc is None:
                excess = abs(self.pos) - self.def_cap
                if excess > 0:
                    side, px = (SELL, bb) if self.pos > 0 else (BUY, ba)
                    self.ioc = self.send_order(OrderRequest(side=side, price=px, quantity=excess,
                                                            order_type=OrderType.IOC))
                    self.orders[self.ioc] = [side, px, excess]
        cap = self._cap(t_s)

        for side, price in ((BUY, bb), (SELL, ba)):
            have = 0
            mine = []
            for oid, o in self.orders.items():
                if o[0] != side or oid in self.cancelling or oid == self.ioc:
                    continue
                if o[1] == price:
                    have += o[2]
                    mine.append(oid)
                else:
                    self.cancelling.add(oid)
                    self.cancel_order(oid)
            room = self._room(side, cap)
            if room < 0:
                # shrink the newest orders first (the oldest have the best queue position)
                for oid in reversed(mine):
                    if room >= 0:
                        break
                    o = self.orders[oid]
                    if oid in self.modifying:
                        room += o[2] - self.modifying[oid]  # already being shrunk
                        continue
                    cut = min(o[2], -room)
                    if cut >= o[2]:
                        self.cancelling.add(oid)
                        self.cancel_order(oid)
                    else:
                        self.modify_order(oid, o[2] - cut)
                        self.modifying[oid] = o[2] - cut  # o[2] stays in full until MODIFIED
                    room += cut
                continue
            target = p['size_top']
            if mine:
                short = target - have
                if not p['topup'] or short < p['topup']:
                    continue
                target = short
            q = min(target, room)
            if q <= 0:
                continue
            oid = self.send_order(OrderRequest(side=side, price=price, quantity=q, order_type=OrderType.LIMIT))
            self.orders[oid] = [side, price, q]
