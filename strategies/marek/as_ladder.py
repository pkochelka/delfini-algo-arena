"""as_ladder: sig_ladder plus Avellaneda-Stoikov inventory control (position sizing and active unwinding).

Base = strategies/marek/sig_ladder.py, defaults unchanged (ladder_mm core + depletion guard). With every
`as_*` switch at 0 this is sig_ladder bit for bit.

A-S quantity. With signed position q (lots), mid volatility sigma (ticks per sqrt-second, EWMA of squared
mid changes on a 100 ms grid, as in strategies/avellaneda_stoikov.py), risk aversion gamma and a constant
horizon tau (the infinite-horizon / stationary form; round1 has no meaningful end-of-session risk), the
reservation price sits

    skew = q * gamma * sigma^2 * tau        ticks below the mid (above it when short)

The book is 1 tick wide 93% of the time, so the continuous A-S quotes are turned into discrete actions,
all driven by `skew` (so every threshold scales with the measured volatility):

    as_size_k  >0  exponential sizing (Gueant / Fushimi): the side that ADDS to |q| targets
                   as_size_base * exp(-as_size_k * |skew|) lots instead of size_top (as_size_base 0 =
                   cap; size_top-based sizing never binds because the cap room is always smaller). as_shrink 1 also
                   modify_orders resting adding-side orders down to that target (oldest kept, queue
                   priority kept). The reducing side keeps its full size.
    as_back    >0  the adding side quotes floor(|skew| / as_back) ticks behind its touch (at most
                   as_back_max), with as_hyst ticks of hysteresis so the level does not flicker.
    as_improve 1   when the spread is >= 2 ticks, the reducing side steps inside by floor(|skew|) ticks
                   (at most spread - 1), i.e. it quotes toward the reservation price.
    as_take    >0  active unwind: once |skew| >= as_take ticks (the reservation price is far enough
                   past the far touch to pay the half spread), IOC the reducing side at the opposite
                   touch for min(as_take_lots, lots needed to bring |skew| back to as_take_to * as_take)
                   lots, at most once per as_take_ms. Guarded by the same worst-case room as quotes.

gamma*tau is what matters; both are exposed so the A-S form stays readable. Rule of thumb on round1
(sigma^2 ~ 0.85 ticks^2/s): as_gamma 1e-4, as_tau_s 100 gives skew ~0.85 ticks at q = 100.

Results (marek/research/round1/as_ladder/as_ladder.md; gamma 1e-4, tau 100 s): every switch LOSES on the
round1 folds, monotonically in strength (size k2 -7k, back 0.5 -15k, improve -97k, take 0.8 -96k), because
pinning at +-100 and being flipped by the next big taker is the round1 edge. On the trending / losing tapes
the passive switches help a little (back 0.5: random_stress -15.3k -> -9.0k; size k4 + shrink: -15.3k ->
-10.8k, sample -10.4k -> -7.6k), the aggressive ones (take, improve) lose everywhere. Defaults therefore keep
all switches off; as_back / as_size_k are candidates for a regime-gated overlay (cf. robust_ladder's kill switch).

sig_ladder docstring follows.

sig_ladder: ladder_mm plus quote-time adverse-selection signals (round1 research, research/round1/signals.md).

Core = strategies/peter/ladder_mm.py with its defaults (one big order per side at the market touch,
queue priority kept, worst case |position + open orders on a side| <= cap, top-up behind).

Overlay ("guard"): a side is *unsafe* when the predicted move runs through it. The score of a side is

    score = share - flow_k * (taker flow against this side over the last flow_ms) / 1000

where share = market qty at this side's touch / (both touches' market qty). score < `g_share` marks
the side unsafe (released at g_share + g_hyst). With `g_dep` > 0 the side is only unsafe while the
market qty at its touch is below g_dep * its peak since that price became the touch (an old level
being eaten, not a fresh level still filling up; a new touch price resets it). Optionally `g_qty`: also unsafe while the market
qty at this side's touch is <= g_qty (we are nearly alone at the level).

While a side is unsafe:
    g_mode 'shrink'  modify_order the oldest touch order down to `g_keep` lots (keeps its queue
                     priority), cancel younger ones, post / top up nothing new on that side
           'cancel'  cancel everything on that side (= ladder_mm imb_pull with hysteresis)
           'freeze'  keep the orders as they are, only stop top-ups / new orders
    g_keep_flat 1    keep instead min(resting, |pos|) on the side that reduces inventory (the
                     flattening part), g_keep on the adding side
    pre_join 1       also rest the freed room one tick behind the touch (the level that becomes the
                     touch if the move happens), so we are early in that queue; cancelled when the
                     guard releases unless `pj_hold`

Defaults = the best variant found (walk-forward wf01-10, research/round1/signals.md):
    g_share 0.10, g_dep 0.5, flow_k 0.10 (1 s window), g_hyst 0, g_keep_flat 1, pre_join 1
    473,066 vs ladder_mm 425,118 (+47.9k, +4.8k per fold, fold-delta sd 1.5k, 10/10 folds)
Set g_share = g_qty = 0 to get ladder_mm back bit for bit. What matters, in order: the depletion
condition (without it the guard fires on every fresh level and loses), keeping the flattening part
with its priority, pre-joining the next level, the flow term. Cancelling/freezing whole sides,
holding the pre-joined order after release, or guarding always all lose 5-270k.
"""

from __future__ import annotations

import math
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
    'improve_both': 0,  # 1: with a spread >= 3 ticks quote one tick inside on both sides
    # Avellaneda-Stoikov overlay (all 0 = sig_ladder)
    'as_gamma': 1e-4,  # risk aversion, per lot
    'as_tau_s': 100.0,  # constant horizon, seconds
    'as_sigma_init': 0.92,  # ticks per sqrt-second, before the estimate warms up
    'as_sigma_hl_s': 60.0,  # EWMA half-life of the variance estimate
    'as_size_k': 0.0,  # >0: adding-side size = as_size_base * exp(-as_size_k * |skew|)
    'as_size_base': 0,  # 0 = cap (size_top 200 never binds: the cap room is always smaller)
    'as_shrink': 0,  # 1: also modify resting adding-side orders down to that size
    'as_back': 0.0,  # >0: adding side floor(|skew| / as_back) ticks behind the touch
    'as_back_max': 2,
    'as_hyst': 0.2,  # ticks of hysteresis on the back-off level
    'as_improve': 0,  # 1: step the reducing side inside a >= 2 tick spread by floor(|skew|)
    'as_take': 0.0,  # >0: IOC-unwind once |skew| >= this many ticks
    'as_take_to': 0.5,  # unwind down to |skew| = as_take_to * as_take
    'as_take_lots': 50,
    'as_take_ms': 200,
}

BUY, SELL = Side.BUY, Side.SELL


class AsLadder(BaseStrategy):
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
        # A-S state
        self.var = PARAMS['as_sigma_init'] ** 2  # ticks^2 per second
        self.var_ts: int | None = None
        self.var_mid = 0.0
        self.back_lvl = {BUY: 0, SELL: 0}
        self.iocs: set[str] = set()
        self.last_take = -(10**18)

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
            self.iocs.discard(oid)
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

    def _send(self, side: Side, price: int, q: int, ioc: bool = False) -> None:
        typ = OrderType.IOC if ioc else OrderType.LIMIT
        oid = self.send_order(OrderRequest(side=side, price=price, quantity=q, order_type=typ))
        self.seq += 1
        self.orders[oid] = [side, price, q, self.seq]
        if ioc:
            self.iocs.add(oid)

    def _update_var(self, now: int, mid: float) -> None:
        if self.var_ts is None:
            self.var_ts, self.var_mid = now, mid
            return
        el = now - self.var_ts
        if el < 100_000:
            return
        dt = el / 1e6
        a = 1.0 - 0.5 ** (dt / self.p['as_sigma_hl_s'])
        self.var += a * ((mid - self.var_mid) ** 2 / dt - self.var)
        self.var_ts, self.var_mid = now, mid

    def _shrink_to(self, orders: list, keep: int) -> None:
        """Shrink the side's orders (oldest first kept) to `keep` lots in total via modify / cancel."""
        for oid, o in orders:
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

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        p = self.p
        now = snapshot.timestamp_us
        if p['flow_k']:
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

        self._update_var(now, 0.5 * (bb + ba))
        k_as = p['as_gamma'] * self.var * p['as_tau_s']  # ticks of skew per lot
        skew = self.pos * k_as  # > 0: reservation below mid (long, want to sell)
        askew = abs(skew)
        reduce_side = SELL if self.pos > 0 else BUY if self.pos < 0 else None
        add_side = BUY if self.pos > 0 else SELL if self.pos < 0 else None

        if (p['as_take'] and reduce_side is not None and askew >= p['as_take']
                and now - self.last_take >= p['as_take_ms'] * 1000 and not self.iocs):
            target_q = p['as_take_to'] * p['as_take'] / k_as
            q = min(p['as_take_lots'], abs(self.pos), max(1, math.ceil(abs(self.pos) - target_q)),
                    self._room(reduce_side))
            if q > 0:
                self._send(reduce_side, bb if reduce_side == SELL else ba, q, ioc=True)
                self.last_take = now

        for side in (BUY, SELL):
            lvl = 0
            if p['as_back'] and side == add_side:
                lvl = self.back_lvl[side]
                raw = askew / p['as_back']
                if raw >= lvl + 1 + p['as_hyst']:
                    lvl = int(raw - p['as_hyst'])
                elif raw < lvl - p['as_hyst']:
                    lvl = max(0, int(raw + p['as_hyst']))
                lvl = min(lvl, p['as_back_max'])
            self.back_lvl[side] = lvl

        tot = qb + qa
        for side, q_side, against, px in ((BUY, qb, -self.flow_sum, bb), (SELL, qa, self.flow_sum, ba)):
            pk_px, pk = self.peak[side]
            pk = max(pk, q_side) if pk_px == px else q_side
            self.peak[side] = (px, pk)
            if not p['g_share'] and not p['g_qty']:
                self.unsafe[side] = False
                continue
            score = q_side / tot - p['flow_k'] * against / 1000.0
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
        if p['as_improve'] and reduce_side is not None and ba - bb >= 2 and askew >= 1:
            imp = min(int(askew), ba - bb - 1)
            if reduce_side == BUY:
                qb_px = max(qb_px, bb + imp)
            else:
                qa_px = min(qa_px, ba - imp)
        qb_px -= self.back_lvl[BUY]
        qa_px += self.back_lvl[SELL]
        for side, touch, step in ((BUY, qb_px, -1), (SELL, qa_px, 1)):
            unsafe = self.unsafe[side]
            mine = sorted(((o[3], oid, o) for oid, o in self.orders.items()
                           if o[0] == side and oid not in self.cancelling and oid not in self.iocs),
                          key=lambda x: x[0])
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
                    if p['g_keep_flat'] and ((side == SELL and self.pos > 0) or (side == BUY and self.pos < 0)):
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
            if p['as_size_k'] and side == add_side:
                target = int((p['as_size_base'] or p['cap']) * math.exp(-p['as_size_k'] * askew))
                if p['as_shrink'] and have > target + max(1, p['topup']):
                    self._shrink_to(at_touch, target)
                    have = target
            if have:
                short = target - have
                target = short if p['topup'] and short >= p['topup'] else 0
            q = min(target, self._room(side))
            if q > 0 and touch > 0:
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
