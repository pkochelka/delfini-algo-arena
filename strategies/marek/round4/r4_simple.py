"""r4_simple: an interpretable round-4 strategy. Every quantity has a name and a one-line meaning.

The generator (marek/research/round4/round4_generator.md) has a hidden price S. Base makers quote at S -+ D with
D = d0 + Exp(lam) (wider in the volatile regime), takers kick S by c * filled^e, S drifts with a velocity V in volatile
regimes, and every ~200 s a 3 s maker drought is followed by a jump of S. This file turns each of those facts into one
small rule:

  Tracker   a two-state Kalman filter (level S, velocity V). Every new market quote at price P is a noisy reading of
            S: a bid says S ~ P + 0.5 + (d0 + lam), an ask says S ~ P - 0.5 - (d0 + lam), noise lam^2 + 1/12; a bid
            also proves S >= P + d0 (an ask S <= P - d0). Every filled market taker moves S by c * q^e. V decays with
            time constant v_tau. A reading more than `gate` sd away is ignored (S is trusted less after each one);
            a reading >= `far` ticks away may be a jump: two of them that agree (or one inside a shock window)
            re-anchor S there.
  Regime    makers quote wide in the volatile regime: the EWMA (half-life w_hl s) of the observed offset D between
            a new quote and S is ~0.5 in calm and ~1.8 in volatile; w = how far it sits between w_lo and w_hi.
  Ladder    a bot posts chains of quotes exactly 2000 us apart, starting 2+ ticks behind the touch and one tick
            further out each. Rungs after the first are dropped; a quote 2+ ticks behind the touch is held 2 ms:
            if a rung follows it was the chain's start (dropped), otherwise it is a late reading.
  Quote     calm: rest the whole room at floor(F - theta) / ceil(F + theta), where F = S + k_now * V * w; keep a
            resting quote while its edge is >= 0, it is within 1 tick of the target and a taker can still reach it.
            volatile: no two-sided quotes (quotes get run over); only a reduce-only quote theta_t beyond F once the
            book has moved against the position by mom_gate (lets the trend flatten us at a good price).
  Take      IOC every market quote priced through F by more than tau + tau_sd * sd (tau larger in volatile);
            in calm, also reduce the position at any edge >= tau_sd * sd.
  Shock     fewer than dr_n market quotes in the last dr_win s = drought; estimate when the quote rate dropped;
            stop quoting. The jump lands ~3.0 s after the onset. From onset + ss_cw0 a bid clipped to the stale ask
            across a wide spread (or an ask to the stale bid) gives the direction; from onset + ss_w0 so does the
            first quote >= ss_far ticks beyond the old touch (earlier such quotes are the drought's own tails) or a
            one-sided taker sweep over sw_span+ ticks. Then IOC the stale side up to ss_lim ticks beyond the old
            touch, ss_shots times ss_gap_ms apart, for the whole room, plus up to xs_cap lots beyond the +-100 limit
            when the new level is known to be >= xs_edge ticks away (a clip or a big sweep counts as that). The
            excess is IOC'd back once the shots are done, at F -+ xu_lim, after xu_ts s at F -+ xu_lim2.

Knobs are in P; the research write-up is marek/research/round4/edge/README.md (section "interpretable model").
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

P = {
    # ---- tracker: maker law (round-4 fit) and taker kicks, calm / volatile
    'd_c': 0.049 + 0.4756,  # mean quote offset d0 + lam
    'r_c': 0.4756 ** 2 + 1 / 12,  # its variance (Exp + price rounding)
    'd_v': 0.699 + 1.06,
    'r_v': 1.06 ** 2 + 1 / 12,
    'kick_c': 0.0168, 'kick_e': 0.616, 'kick_var_c': 0.0169,  # S += c * q^e per filled taker, plus noise
    'kick_c_v': 0.05, 'kick_e_v': 0.64, 'kick_var_v': 0.076,
    'v_tau_c': 0.5, 'v_sd_c': 0.3,  # velocity: OU time constant (s) and stationary sd (ticks / s)
    'v_tau_v': 2.1, 'v_sd_v': 3.0,
    'q_s': 0.15 ** 2,  # unmodelled level noise, ticks^2 / s
    'y_half': 0.5,  # a bid at integer P was placed at S - D in [P, P + 1): reading S ~ P + y_half + d
    'gate': 3.0,  # ignore a reading more than this many sd from S
    'rej_n': 1,  # after this many ignored readings in a row, trust S less (so the gate reopens after a move)
    'far': 8.0,  # a reading this far from S may be a jump ...
    'jump_agree': 4.0,  # ... two of them within this of each other re-anchor S
    'tail_k': 1.0,  # > 1: a quote further out than S expects may be a wide (tail) quote: its reading noise is
                    # multiplied by tail_k; a quote closer in than expected cannot be, and keeps full weight
    'bound': 1,  # a bid at P proves S >= P + d0 (an ask: S <= P - d0): clamp S to it
    'bound_win': 0.0,  # > 0: keep the tightest such bound from each side over the last bound_win s, each bound
                       # relaxed by bound_relax ticks/s of age (S moves); contradicting sides drop the older bound
    'bound_relax_c': 1.0, 'bound_relax_v': 5.0,
    'd0_c': 0.049, 'd0_v': 0.699,
    # ---- regime from the makers' quoting width
    'w_hl': 0.15, 'w_lo': 0.8, 'w_hi': 1.4, 'turb_w': 0.5,
    'jump_hold': 3.0,  # s of volatile handling after a detected jump
    # ---- quoting
    'cap': 100,
    'theta': 0.25,  # calm quote distance from F
    'reach': 1.0,  # calm takers price at round(S -+ 1): a quote further than this from F is unreachable
    'k_now': 0.15,  # nowcast: F = S + k_now * V (volatile weight)
    'max_sd': 0.6,  # no new quotes while S is this uncertain
    'theta_t': 2.0, 'keep_t': 1.5, 'mom_gate': -0.5, 'flat_pos': 60,  # volatile reduce-only quote
    # ---- takes
    'tau_c': 0.10, 'tau_t': 0.30, 'tau_sd': 1.0,
    'red_on': 1,  # calm: IOC the position down at any edge >= tau_sd * sd (no tau)
    # ---- shocks
    'dr_win': 0.5, 'dr_n': 50, 'dr_cool': 20.0,
    'ss_w0': 2.99, 'ss_cw0': 2.93, 'ss_w1': 3.30, 'ss_far': 3, 'ss_spread': 3, 'ss_lim': 8, 'ss_shots': 3,
    'ss_gap_ms': 15,
    'xs_cap': 950, 'xs_edge': 4.0, 'xu_lim': 1.0, 'xu_lim2': 4.0, 'xu_ts': 0.08, 'xu_gap_ms': 6, 'xu_wait': 1,
    'sw_span': 3, 'sw_edge': 10.0,  # a one-sided sweep spanning sw_span+ ticks in the window: edge credit sw_edge
    'clip_edge': 4.0,  # edge credit for a clip trigger (>= xs_edge: the first shot carries the oversize)
    'lad_on': 1, 'lad_tol': 3,  # rung: exactly 2000 +- lad_tol us after a same-side quote one tick closer
    'lad_deep': 2, 'lad_wait': 2010,  # a quote lad_deep+ ticks behind the touch waits lad_wait us for a rung
}

BUY, SELL = Side.BUY, Side.SELL


class Tracker:
    """Kalman filter for (S, V). Attribute names mirror the big model's filter so the research tools can log it."""

    def __init__(self, s0: float) -> None:
        self.mean = s0
        self.kv = 0.0  # velocity
        self.pss, self.psv, self.pvv = 1.0, 0.0, 0.0
        self.sd = 1.0
        self.t = None
        self.w_off = 0.5  # EWMA of observed quote offsets
        self.pv = 0.0  # regime weight w in [0, 1]
        self.Y = 0.0  # unused (logging compatibility)
        self.rej = None  # (t, implied S) of the last ignored reading
        self.n_rej = 0  # ignored readings in a row
        self.hot = False  # inside a shock window: one far reading re-anchors
        self.drought = False  # makers quote wide during a drought
        self.lo: deque = deque()  # (t, bound) from bids, window bound_win
        self.hi: deque = deque()  # (t, bound) from asks
        self.jumps = 0
        self.last_jump = -1e9

    def predict(self, t: float) -> None:
        if self.t is None:
            self.t = t
            return
        dt = t - self.t
        if dt <= 0:
            return
        self.t = t
        w = self.pv
        tau = (1 - w) * P['v_tau_c'] + w * P['v_tau_v']
        s2 = (1 - w) * P['v_sd_c'] ** 2 + w * P['v_sd_v'] ** 2
        ph = math.exp(-dt / tau)
        self.mean += self.kv * dt
        self.kv *= ph
        self.pss += 2 * self.psv * dt + self.pvv * dt * dt + P['q_s'] * dt
        self.psv = ph * (self.psv + self.pvv * dt)
        self.pvv = ph * ph * self.pvv + s2 * (1 - ph * ph)
        self.sd = math.sqrt(self.pss)

    def _update(self, y: float, r: float) -> None:
        inn = y - self.mean
        s = self.pss + r
        ks, kvv = self.pss / s, self.psv / s
        pss, psv, pvv = self.pss, self.psv, self.pvv
        self.mean += ks * inn
        self.kv += kvv * inn
        self.pss = max(pss - ks * pss, 1e-8)
        self.psv = psv - ks * psv
        self.pvv = max(pvv - kvv * psv, 1e-6)
        self.sd = math.sqrt(self.pss)

    def quote(self, t: float, is_bid: bool, price: int) -> None:
        """A new market quote: a noisy reading of S."""
        self.predict(t)
        w = 1.0 if self.drought else self.pv
        d = (1 - w) * P['d_c'] + w * P['d_v']
        r = (1 - w) * P['r_c'] + w * P['r_v']
        y = price + P['y_half'] + d if is_bid else price - P['y_half'] - d
        inn = y - self.mean
        if P['tail_k'] > 1 and ((is_bid and inn < 0) or (not is_bid and inn > 0)):
            r *= P['tail_k']  # further out than S expects: may be a wide quote
        if abs(inn) > P['gate'] * math.sqrt(self.pss + r):
            if abs(inn) >= P['far'] or self.hot:
                rj = self.rej
                self.rej = (t, y)
                if self.hot or (rj is not None and t - rj[0] < 0.2 and abs(rj[1] - y) < P['jump_agree']):
                    self.mean = y if self.hot else 0.5 * (rj[1] + y)
                    self.kv, self.psv = 0.0, 0.0
                    self.pss, self.sd = 1.0, 1.0
                    self.jumps += 1
                    self.last_jump = t
                    self.rej = None
            else:
                self.n_rej += 1
                if self.n_rej >= P['rej_n']:
                    self.pss += 0.25 * r  # quotes keep disagreeing with S: trust S less, so the gate reopens
                    self.sd = math.sqrt(self.pss)
            return
        self.n_rej = 0
        # regime tell: the observed offset of this quote from S (clipped against tail quotes)
        off = min(abs((self.mean - 0.5 - price) if is_bid else (price - 0.5 - self.mean)), 5.0)
        a = 1 - 0.5 ** (1.0 / (P['w_hl'] * 180))  # ~180 quotes / s
        self.w_off += a * (off - self.w_off)
        self.pv = min(max((self.w_off - P['w_lo']) / (P['w_hi'] - P['w_lo']), 0.0), 1.0)
        self._update(y, r)
        if P['bound']:
            d0 = (1 - w) * P['d0_c'] + w * P['d0_v']
            b = price + d0 if is_bid else price - d0
            win = P['bound_win']
            if win:
                lo, hi = self.lo, self.hi
                (lo if is_bid else hi).append((t, b))
                while lo and lo[0][0] < t - win:
                    lo.popleft()
                while hi and hi[0][0] < t - win:
                    hi.popleft()
                rx = (1 - w) * P['bound_relax_c'] + w * P['bound_relax_v']
                L = max((x[1] - rx * (t - x[0]) for x in lo), default=-1e18)
                U = min((x[1] + rx * (t - x[0]) for x in hi), default=1e18)
                while L > U:  # the sides still contradict: S has moved, drop the older binding quote
                    xl = max(lo, key=lambda x: x[1] - rx * (t - x[0]))
                    xu = min(hi, key=lambda x: x[1] + rx * (t - x[0]))
                    if xl[0] < xu[0]:
                        lo.remove(xl)
                    else:
                        hi.remove(xu)
                    L = max((x[1] - rx * (t - x[0]) for x in lo), default=-1e18)
                    U = min((x[1] + rx * (t - x[0]) for x in hi), default=1e18)
                self.mean = min(max(self.mean, L), U)
            elif is_bid and self.mean < b:
                self.mean = b
            elif not is_bid and self.mean > b:
                self.mean = b

    def taker(self, t: float, is_buy: bool, q: int) -> None:
        """A filled market taker kicks S."""
        self.predict(t)
        w = self.pv
        k = (1 - w) * P['kick_c'] * q ** P['kick_e'] + w * P['kick_c_v'] * q ** P['kick_e_v']
        self.mean += k if is_buy else -k
        self.pss += (1 - w) * P['kick_var_c'] + w * P['kick_var_v']
        self.sd = math.sqrt(self.pss)


class Simple(BaseStrategy):
    def __init__(self) -> None:
        super().__init__()
        self.f: Tracker | None = None
        self.pos = 0
        self.orders: dict[str, list] = {}  # id -> [side, price, remaining]
        self.cancelling: set[str] = set()
        self.max_id = ''
        self.nb = self.na = 0
        self.mbb = self.mba = None  # market-only touch before this event
        self.now_t = 0.0
        self.turb = False
        self.ins_t: deque = deque()  # market quote times (drought detection, onset estimate)
        self.lad_h = {True: deque(), False: deque()}  # side -> (t_us, price) of market quotes in the last ~2 ms
        self.lad_pend: list = []  # deep quotes held back: (t, t_us, is_bid, price)
        # shock state
        self.sg_t = None  # drought detection time
        self.sg_next = 0.0
        self.onset = None
        self.ss_dir = 0
        self.ss_ref = 0
        self.ss_left = 0
        self.ss_next = 0.0
        self.ss_t = -1e9
        self.ss_hold = 0.0
        self.ss_edge = 0.0
        self.xu_next = 0.0

    # ------------------------------------------------------------------ private feed
    def on_private_order_update(self, ev: PrivateOrderEvent) -> None:
        o = self.orders.get(ev.order_id)
        if o is None:
            return
        st = ev.status
        if st is OrderStatus.PARTIALLY_FILLED or st is OrderStatus.FILLED:
            self.pos += ev.filled_quantity if o[0] is BUY else -ev.filled_quantity
            o[2] = ev.remaining_quantity
        if st is OrderStatus.FILLED or st is OrderStatus.CANCELED or st is OrderStatus.REJECTED:
            self.orders.pop(ev.order_id, None)
            self.cancelling.discard(ev.order_id)

    # ------------------------------------------------------------------ public feed
    def _scan(self, views):
        """Market touch of one side and the newest market quote in it (market ids are equal-length digit strings)."""
        top = None
        for v in views:
            if v.owner == 'market':
                top = v.price
                break
        new, mx = None, self.max_id
        for v in views:
            if v.owner == 'market' and len(v.order_id) >= len(mx) and v.order_id > mx:
                mx, new = v.order_id, v
        if new is not None:
            self.max_id = mx
        return top, new

    def on_order_book_change(self, snap: OrderBookSnapshot, trades: list[Trade]) -> None:
        bids, asks = snap.bids, snap.asks
        t = snap.timestamp_us * 1e-6
        self.now_t = t
        f = self.f
        if f is None:
            if not bids or not asks:
                return
            f = self.f = Tracker((bids[0].price + asks[0].price) / 2)
        # market taker fills in this event
        q, buy, tid = 0, False, None
        for tr in trades:
            if tr.taker_owner == 'market':
                q += tr.quantity
                buy = tr.book_side == 'ask'
                tid = tr.taker_order_id
        if self.lad_pend:  # deep quotes with no rung after them were real readings
            keep = []
            for e in self.lad_pend:
                if snap.timestamp_us - e[1] > P['lad_wait']:
                    f.quote(e[0], e[2], e[3])
                    self.ins_t.append(e[0])
                else:
                    keep.append(e)
            self.lad_pend = keep
        pbb, pba = self.mbb, self.mba
        nb, na = len(bids), len(asks)
        newb = newa = None
        if nb > self.nb or (nb and bids[0].price != pbb):
            self.mbb, newb = self._scan(bids)
        if na > self.na or (na and asks[0].price != pba):
            self.mba, newa = self._scan(asks)
        self.nb, self.na = nb, na
        if q:
            if tid is not None and len(tid) == len(self.max_id) and tid > self.max_id:
                self.max_id = tid
            rested = (newb is not None and newb.order_id == tid) or (newa is not None and newa.order_id == tid)
            if not rested:
                f.taker(t, buy, q)
        elif newb is not None or newa is not None:
            is_bid = newb is not None
            v = newb if is_bid else newa
            if not self._ladder_rung(snap.timestamp_us, is_bid, v.price):
                own = pbb if is_bid else pba
                deep = own is not None and ((own - v.price) if is_bid else (v.price - own)) >= P['lad_deep']
                if P['lad_deep'] and deep and self.sg_t is None:
                    # a quote 2+ ticks behind the touch may be a ladder start: hold it until a rung could have followed
                    self.lad_pend.append((t, snap.timestamp_us, is_bid, v.price))
                else:
                    f.quote(t, is_bid, v.price)
                    self.ins_t.append(t)
                    self._shock_trigger(t, is_bid, v.price, pbb, pba)
        if q and self.ss_dir == 0 and self._in_window(t):
            # a market sweep of 2+ levels inside the window also gives the direction: the informed taker hits the
            # stale side a few ms before the new quotes arrive. A one-sided sweep spanning sw_span+ ticks is proof
            # enough to take the stale side with the full oversize at once (sw_edge).
            lv = {tr.price for tr in trades if tr.taker_owner == 'market'}
            one_side = len({tr.book_side for tr in trades if tr.taker_owner == 'market'}) == 1
            if len(lv) >= 2:
                big = one_side and max(lv) - min(lv) >= P['sw_span']
                self._fire(t, 1 if buy else -1, pbb, pba, P['sw_edge'] if big else 0.0)
        if not bids or not asks:
            return
        self._trade(f, snap)

    def _ladder_rung(self, t_us: int, is_bid: bool, price: int) -> bool:
        """Ladder bot: a quote exactly 2000 us after a same-side quote one tick closer is a rung, not a reading
        (the chain's start, held back as a deep quote, is dropped with it)."""
        if not P['lad_on']:
            return False
        h = self.lad_h[is_bid]
        while h and t_us - h[0][0] > 2000 + P['lad_tol']:
            h.popleft()
        step = -1 if is_bid else 1
        hit = None
        for e in h:
            if abs(t_us - e[0] - 2000) <= P['lad_tol'] and price == e[1] + step:
                hit = e
                break
        h.append((t_us, price))
        if hit is None:
            return False
        self.lad_pend = [e for e in self.lad_pend if not (e[2] == is_bid and e[3] == hit[1])]
        return True

    # ------------------------------------------------------------------ trading
    def _send(self, side: Side, price: int, qty: int, kind: OrderType) -> None:
        if qty <= 0 or price <= 0:
            return
        oid = self.send_order(OrderRequest(side=side, price=price, quantity=qty, order_type=kind))
        self.orders[oid] = [side, price, qty]
        if kind is OrderType.IOC:
            self.cancelling.add(oid)

    def _rest(self) -> tuple[int, int]:
        b = a = 0
        for o in self.orders.values():
            if o[0] is BUY:
                b += o[2]
            else:
                a += o[2]
        return b, a

    def _cancel_all(self, side: Side | None = None) -> None:
        for oid, o in self.orders.items():
            if oid not in self.cancelling and (side is None or o[0] is side):
                self.cancelling.add(oid)
                self.cancel_order(oid)

    def _trade(self, f: Tracker, snap: OrderBookSnapshot) -> None:
        t = self.now_t
        bids, asks = snap.bids, snap.asks
        bb, ba = bids[0].price, asks[0].price
        if self._shock(f, snap):
            return
        w = f.pv
        turb = self.turb = w > P['turb_w'] or t - f.last_jump < P['jump_hold']
        F = f.mean + P['k_now'] * f.kv * w
        sd = f.sd
        pos = self.pos
        cap = P['cap']
        rest_b, rest_a = self._rest()
        # ---- takes: IOC everything priced through F by more than tau + tau_sd * sd
        thr = (P['tau_t'] if turb else P['tau_c']) + P['tau_sd'] * sd
        if ba <= F - thr:
            lim = math.floor(F - thr)
            q = min(sum(o.quantity for o in asks if o.price <= lim and o.owner == 'market'), cap - pos - rest_b)
            if q > 0:
                self._send(BUY, lim, q, OrderType.IOC)
                rest_b += q
        if bb >= F + thr:
            lim = math.ceil(F + thr)
            q = min(sum(o.quantity for o in bids if o.price >= lim and o.owner == 'market'), cap + pos - rest_a)
            if q > 0:
                self._send(SELL, lim, q, OrderType.IOC)
                rest_a += q
        if not turb and P['red_on']:  # calm: reduce the position at any edge >= tau_sd * sd
            e = P['tau_sd'] * sd
            if pos > 0 and bb >= F + e:
                lim = math.ceil(F + e)
                q = min(sum(o.quantity for o in bids if o.price >= lim and o.owner == 'market'), pos, cap + pos - rest_a)
                if q > 0:
                    self._send(SELL, lim, q, OrderType.IOC)
                    rest_a += q
            elif pos < 0 and ba <= F - e:
                lim = math.floor(F - e)
                q = min(sum(o.quantity for o in asks if o.price <= lim and o.owner == 'market'), -pos, cap - pos - rest_b)
                if q > 0:
                    self._send(BUY, lim, q, OrderType.IOC)
                    rest_b += q
        # ---- passive quotes
        mid = 0.5 * ((self.mbb if self.mbb is not None else bb) + (self.mba if self.mba is not None else ba))
        if turb:
            # reduce-only quote theta_t beyond F once the book has moved against the position
            against = (F - mid) * (1 if pos > 0 else -1) <= P['mom_gate']
            want = abs(pos) >= P['flat_pos'] and against and pos != 0
            side = SELL if pos > 0 else BUY
            tgt = max(math.ceil(F + P['theta_t']), bb + 1) if pos > 0 else min(math.floor(F - P['theta_t']), ba - 1)
            have = False
            for oid, o in list(self.orders.items()):
                if oid in self.cancelling:
                    continue
                edge = (o[1] - F) if o[0] is SELL else (F - o[1])
                if want and o[0] is side and edge >= P['keep_t'] and abs(o[1] - tgt) <= 1:
                    have = True
                    continue
                self.cancelling.add(oid)
                self.cancel_order(oid)
            if want and not have:
                room = (cap + pos - rest_a) if pos > 0 else (cap - pos - rest_b)
                self._send(side, tgt, min(abs(pos), room), OrderType.LIMIT)
            return
        tb = min(math.floor(F - P['theta']), ba - 1)
        ta = max(math.ceil(F + P['theta']), bb + 1)
        rb, ra = round(F - P['reach']), round(F + P['reach'])
        have_b = have_a = False
        for oid, o in list(self.orders.items()):
            if oid in self.cancelling:
                continue
            px = o[1]
            if o[0] is BUY and F - px >= 0 and tb - 1 <= px < ba and px >= rb:
                have_b = have_b or px == tb
                continue
            if o[0] is SELL and px - F >= 0 and ta + 1 >= px > bb and px <= ra:
                have_a = have_a or px == ta
                continue
            self.cancelling.add(oid)
            self.cancel_order(oid)
        if sd > P['max_sd']:
            return
        if not have_b:
            self._send(BUY, tb, cap - pos - rest_b, OrderType.LIMIT)
        if not have_a:
            self._send(SELL, ta, cap + pos - rest_a, OrderType.LIMIT)

    # ------------------------------------------------------------------ shocks
    def _onset_estimate(self, sg: float):
        """When did the market quote rate drop from ~180 /s to ~54 /s? Maximum-likelihood change point over the
        quotes seen in the last 1.5 s."""
        ts = [x for x in self.ins_t if sg - 1.5 <= x <= sg]
        if len(ts) < 20:
            return sg - 0.4
        r1, r2 = 180.0, 54.0
        lr1, lr2 = math.log(r1), math.log(r2)
        a, n = ts[0], len(ts)
        best, arg = -1e18, ts[-1]
        for k in range(n):
            c = ts[k]
            ll = k * lr1 - r1 * (c - a) + (n - k) * lr2 - r2 * (sg - c)
            if ll > best:
                best, arg = ll, c
        return arg

    def _in_window(self, t: float, w0: float | None = None) -> bool:
        w0 = P['ss_w0'] if w0 is None else w0
        return self.onset is not None and self.sg_t is not None and w0 <= t - self.onset <= P['ss_w1']

    def _shock_trigger(self, t: float, is_bid: bool, price: int, pbb, pba) -> None:
        """Inside the jump window, the first quote far beyond the old touch (or clipped to the other side's touch
        across a wide spread) tells the jump direction. Far quotes count only from ss_w0 on: earlier they are the
        drought's own tail quotes (a 3-tick tail is common), and a wrong-way shot cannot be undone in time."""
        if self.ss_dir or pbb is None or pba is None or not self._in_window(t, P['ss_cw0']):
            return
        far_ok = self._in_window(t)
        d, edge = 0, 0.0
        if is_bid:
            if price == pba - 1 and pba - pbb >= P['ss_spread']:
                d, edge = 1, P['clip_edge']  # a bid clipped to the stale ask across a wide spread: S is above it
            elif far_ok and price <= pbb - P['ss_far']:
                d, edge = -1, pbb - (price + 0.5 + P['d_c'])  # a new bid at P: S ~ P + 0.5 + d, far below the old bid
        else:
            if price == pbb + 1 and pba - pbb >= P['ss_spread']:
                d, edge = -1, P['clip_edge']
            elif far_ok and price >= pba + P['ss_far']:
                d, edge = 1, (price - 0.5 - P['d_c']) - pba
        if d:
            self._fire(t, d, pbb, pba, edge)

    def _fire(self, t: float, d: int, pbb, pba, edge: float = 0.0) -> None:
        if self.ss_dir or pbb is None or pba is None:
            return
        self.ss_dir, self.ss_ref = d, (pba if d > 0 else pbb)
        self.ss_left, self.ss_next, self.ss_t = P['ss_shots'], t, t
        self.ss_edge = edge  # how far the new level is known to be beyond the stale touch
        self._cancel_all(SELL if d > 0 else BUY)  # our quotes on the losing side

    def _shock(self, f: Tracker, snap: OrderBookSnapshot) -> bool:
        """Drought -> jump window -> snipe -> unwind. True while this takes over from normal trading."""
        t = self.now_t
        it = self.ins_t
        while it and it[0] < t - P['dr_win']:
            it.popleft()
        if self.sg_t is None:
            if t > 1.0 and len(it) < P['dr_n'] and t > self.sg_next:
                self.sg_t = t
                self.sg_next = t + P['dr_cool']
                self.onset = self._onset_estimate(t)
                self.ss_dir = 0
                f.drought = True
                self._cancel_all()
            elif self.ss_dir == 0 and abs(self.pos) <= P['cap']:
                return False
        if self.sg_t is not None:
            f.hot = self._in_window(t)
            if f.hot or self.ss_dir:
                f.drought = False  # the jump has landed (or is due): quotes follow the normal law again
            if t - self.onset > P['ss_w1'] + 0.3 and self.ss_left <= 0:
                self.sg_t, f.hot, f.drought = None, False, False  # window over: normal trading (unwind continues below)
        # snipe shots
        if self.ss_dir and self.ss_left > 0 and t >= self.ss_next:
            side = BUY if self.ss_dir > 0 else SELL
            px = self.ss_ref + P['ss_lim'] if side is BUY else self.ss_ref - P['ss_lim']
            rest = sum(o[2] for o in self.orders.values() if o[0] is side)
            self.ss_edge = max(self.ss_edge, (f.mean - self.ss_ref) * self.ss_dir)  # the tracker may know more by now
            cap = P['xs_cap'] if (P['xs_cap'] and self.ss_edge >= P['xs_edge']) else P['cap']
            room = cap - self.pos - rest if side is BUY else cap + self.pos - rest
            self._send(side, px, room, OrderType.IOC)
            self.ss_left -= 1
            self.ss_next = t + P['ss_gap_ms'] * 1e-3
            if self.ss_left == 0:
                self.ss_hold = t + 0.3
        if self.ss_dir and self.ss_left <= 0 and t >= self.ss_hold:
            self.ss_dir = 0
        # unwind any excess over the limit at once (10k / s penalty); with xu_wait, not before the shots are done
        # (the jump has landed by then; before it, F may still sit at the stale level and the exit would be a loss)
        if abs(self.pos) > P['cap'] and t >= self.xu_next and not (P['xu_wait'] and self.ss_dir and self.ss_left > 0):
            lim_off = P['xu_lim'] if t - self.ss_t < P['xu_ts'] else P['xu_lim2']
            if self.pos > 0:
                fl = sum(o[2] for oid, o in self.orders.items() if o[0] is SELL and oid in self.cancelling)
                q = self.pos - P['cap'] - fl
                lim = math.ceil(f.mean - lim_off)
                if q > 0 and any(o.price >= lim and o.owner == 'market' for o in snap.bids):
                    self._send(SELL, lim, q, OrderType.IOC)
            else:
                fl = sum(o[2] for oid, o in self.orders.items() if o[0] is BUY and oid in self.cancelling)
                q = -self.pos - P['cap'] - fl
                lim = math.floor(f.mean + lim_off)
                if q > 0 and any(o.price <= lim and o.owner == 'market' for o in snap.asks):
                    self._send(BUY, lim, q, OrderType.IOC)
            self.xu_next = t + P['xu_gap_ms'] * 1e-3
        return self.sg_t is not None or self.ss_dir != 0 or abs(self.pos) > P['cap']


PARAMS = P
