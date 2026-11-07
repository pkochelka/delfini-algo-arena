"""sig_ladder: ladder_mm plus quote-time adverse-selection signals (round1 research, research/round1/signals.md).

Core = strategies/peter/ladder_mm.py with its defaults (one big order per side at the market touch,
queue priority kept, worst case |position + open orders on a side| <= cap, top-up behind).

Overlay ("guard"): a side is *unsafe* when the predicted move runs through it. The score of a side is

    score = share - flow_k * (taker flow against this side over the last flow_ms) / 1000
                  - flow2_k * (taker flow against this side over the last flow2_ms) / 1000

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
Short flow window (marek/research/round1/combo/): flow2_k 0.5 over 200 ms on top of the 1 s term:
    489,114 on wf01-10 (+16.0k vs the 1 s term alone, 10/10 folds, fold-delta sd 0.7k), full tape 489,589;
    plateau +12..+16k over flow2_k 0.25-0.75 x 200-500 ms. Neutral to +2.5k on the stress tapes, -0.9% on sample_v2.
    Set flow2_k = 0 for the previous 473,066 version.
Combination switches tested and left off: size_deep (100/100 touch/behind split; overlaps pre_join, -4..-81k),
improve_both (+3.3k on round1, but -9k on pscale2 and -5k on random_stress), sev_share (pull the flattening
part too when nearly gone: noise).
Set g_share = g_qty = flow2_k = 0 to get ladder_mm back bit for bit. What matters, in order: the depletion
condition (without it the guard fires on every fresh level and loses), keeping the flattening part
with its priority, pre-joining the next level, the flow term. Cancelling/freezing whole sides,
holding the pre-joined order after release, or guarding always all lose 5-270k.
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
}

BUY, SELL = Side.BUY, Side.SELL


class SigLadder(BaseStrategy):
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
