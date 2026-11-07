"""fv_mm: market maker + stale-quote taker around a Bayesian estimate of the generator's hidden price S.

Round 2 broke s_mm: S still drives the makers (bid = min(floor(S - D), ask - 1), ask mirrored, D = d0 + Exp(lam))
and the takers' impact (c * sign * sqrt(filled)), but S now also diffuses and jumps (194 jumps > 5 ticks,
clustered into turbulent regimes, round2_findings.md), so the exact bound tracker of s_mm keeps contradicting
itself. Here S is a posterior on a grid (class FV):
    predict   diffusion sigma + jumps (flat over the grid) at a self-exciting rate rho0 + exc
    makers    likelihood of each new market insert under the maker rule, censored when clamped at the
              opposite touch, plus an outlier share eta; two agreeing inserts far outside the grid restart it
    takers    impact shifts the grid (filled qty from the trades of one market event)
The posterior mean F is the fair value; its sd the uncertainty. After a jump the new level shows up in the
first one or two maker inserts on the far side, while the stale side of the book is only swept by takers
tens of ms later: the take rule (ask <= F - take) snipes it, and the keep rule cancels our own stale quotes.

Quoting (s_mm's structure, thresholds widened by sd and by turbulence):
    post   bid floor(F - th), ask ceil(F + th), th = theta + theta_sd * sd + theta_turb * [exc > 0], never crossing
    keep   a resting order while its edge >= theta_keep (+ the same widening), reachable by the next taker,
           at most keep_depth ticks behind target
    take   IOC every market level priced at least `take + take_sd * sd` through F, up to the room
    room   pos + every order on the side (incl. cancels and IOCs in flight) <= cap, so no breach is possible
Offline parameter fit of the filter: r2_fv_fit.py.
"""

from __future__ import annotations

import math

import numpy as np

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

FV_PARAMS = {
    'h': 0.05,  # grid cell (ticks), 1/integer
    'K': 240,  # grid half-width in cells
    'd0': 0.05,
    'lam': 0.6,
    'eta': 0.005,
    'sigma': 0.5,  # diffusion, ticks / sqrt(s)
    'rho0': 0.05,  # base jump rate (/s)
    'kick': 1.0,  # excitation added per detected jump (/s)
    'exc_hl': 1.5,  # s
    'exc_max': 3.0,
    'jump_sd': 0.5,  # a jump is detected only by a confident filter (sd below this) ...
    'jump_det': 2.0,  # ... whose mean one observation moves this far (ticks)
    'w_u': 0.6,  # taker-price model (offline only: the live feed does not show IOC limits)
    'tau': 1.5,
    'c': 0.037,
    'far_tol': 3.0,  # two far inserts must agree within this to restart the grid
    'taker_obs': 0,
}

PARAMS = FV_PARAMS | {
    'cap': 100,
    'theta': 0.30,
    'theta_sd': 0.0,
    'theta_turb': 1.5,
    'theta_keep': 0.10,
    'keep_depth': 1,
    'reach_off': 1.0,
    'take': 0.15,
    'take_sd': 2.0,
    'max_sd': 0.6,  # no new quotes while the posterior sd is above this
    'inv_skew': 0.0,
    'size_top': 1000,
    'min_post': 1,
    'trade_start_us': 0,
}

BUY, SELL = Side.BUY, Side.SELL


class FV:
    """Posterior of S on the grid (c + k) * h, k = -K..K, with c an integer. See the module docstring.

    Prices are integers and h = 1/n, so every likelihood is a slice of a table precomputed per (side, clamped).
    Taker impact moves the origin c by whole cells; the sub-cell rest `r` is carried in `mean` only (<= h/2).
    """

    def __init__(self, p: dict, s0: float) -> None:
        self.p = p
        h = self.h = p['h']
        self.inv_h = round(1 / h)
        K = self.K = p['K']
        n = 2 * K + 1
        self.k = (np.arange(n) - K) * h
        self.k2 = self.k * self.k
        self.c = round(s0 / h)
        self.r = 0.0
        self.w = np.full(n, 1.0 / n)
        self.t = None
        self.exc = 0.0
        self.mean = float(s0)
        self.sd = 1.0
        self.far = None  # (t, implied S) of a pending far insert
        self.jumps = 0
        self.ll = 0.0
        self.n_ll = 0
        # tables over x = j*h, j = -J..J, where x = S - bid (bids) or ask - S (asks)
        J = self.J = 2 * K + 4 * self.inv_h
        x = (np.arange(2 * J + 1) - J) * h
        il = 1.0 / p['lam']
        F = -np.expm1(-np.maximum(x - p['d0'], 0.0) * il)
        F1 = -np.expm1(-np.maximum(x - 1.0 - p['d0'], 0.0) * il)
        eta = p['eta']
        self.T_cl = (1 - eta) * F + eta / 30.0
        self.T_un = (1 - eta) * (F - F1) + eta / 30.0
        self.T_cl_r = self.T_cl[::-1].copy()
        self.T_un_r = self.T_un[::-1].copy()
        self.s2 = p['sigma'] ** 2 / (2 * h * h)

    def predict(self, t: float) -> None:
        if self.t is None:
            self.t = t
            return
        dt = t - self.t
        if dt <= 0:
            return
        self.t = t
        p = self.p
        w = self.w
        a = self.s2 * dt
        while a > 0:
            b = a if a < 0.45 else 0.45
            w[1:-1] = (1 - 2 * b) * w[1:-1] + b * (w[:-2] + w[2:])
            a -= b
        if self.exc:
            self.exc *= 0.5 ** (dt / p['exc_hl'])
            if self.exc < 1e-4:
                self.exc = 0.0
        pi = 1.0 - math.exp(-(p['rho0'] + self.exc) * dt)
        w *= 1.0 - pi
        w += pi / w.size

    def shift(self, d: float) -> None:
        r = self.r + d
        s = round(r * self.inv_h)
        if s:
            self.c += s
            r -= s * self.h
        self.r = r
        self.mean += d

    def _apply(self, L: np.ndarray, count_ll: bool) -> None:
        post = self.w * L
        z = float(post.sum())
        if z <= 1e-300:
            post = np.full(post.size, 1.0 / post.size)
            z = 1e-300
        else:
            post *= 1.0 / z
        self.w = post
        if count_ll:
            self.ll += math.log(z)
            self.n_ll += 1
        m = float(post @ self.k)
        v = float(post @ self.k2) - m * m
        old, old_sd = self.mean, self.sd
        self.mean = self.c * self.h + self.r + m
        self.sd = math.sqrt(v) if v > 1e-8 else 1e-4
        if abs(m) > 3.0:
            self._recentre(round(m * self.inv_h))
        if abs(self.mean - old) > self.p['jump_det'] and old_sd < self.p['jump_sd']:
            self._jumped()

    def _jumped(self) -> None:
        self.exc = min(self.exc + self.p['kick'], self.p['exc_max'])
        self.jumps += 1

    def _recentre(self, s: int) -> None:
        w = self.w
        n = w.size
        if abs(s) >= n:
            w[:] = 1.0 / n
        elif s > 0:
            w[:-s] = w[s:].copy()
            w[-s:] = 0.0
        else:
            w[-s:] = w[:s].copy()
            w[:-s] = 0.0
        self.c += s
        z = w.sum()
        if z > 0:
            w /= z

    def _L(self, is_bid: bool, price: int, clamped: bool):
        """Likelihood over the grid, or None if the insert is beyond the tables."""
        K, J = self.K, self.J
        # bids: x_k = (c + k - P) h -> table index c - P + k + J, increasing in k
        # asks: x_k = (P - c - k) h -> reversed-table index 2J - (P - c - k + J) = c - P + k + J
        i = self.c - price * self.inv_h - K + J
        if is_bid:
            T = self.T_cl if clamped else self.T_un
        else:
            T = self.T_cl_r if clamped else self.T_un_r
        if i < 0 or i + 2 * K + 1 > T.size:
            return None
        return T[i:i + 2 * K + 1]

    def maker(self, t: float, is_bid: bool, price: int, opp: int | None, count_ll: bool = True) -> None:
        """A market limit insert at `price`; `opp` is the best market price on the other side before it."""
        self.predict(t)
        clamped = opp is not None and (price == opp - 1 if is_bid else price == opp + 1)
        if not clamped:
            imp = price + 0.55 if is_bid else price - 0.55
            if abs(imp - self.mean) > self.K * self.h - 2:
                f = self.far
                if f is not None and t - f[0] < 0.2 and abs(f[1] - imp) < self.p['far_tol']:
                    self.c = round((f[1] + imp) / 2 * self.inv_h)
                    self.r = 0.0
                    self.w = np.full(self.w.size, 1.0 / self.w.size)
                    self.far = None
                    self.sd = 1.0  # not a "confident" detection: count the jump here
                    self._jumped()
                    L = self._L(is_bid, price, False)
                    if L is not None:
                        self._apply(L, False)
                else:
                    self.far = (t, imp)
                return
        L = self._L(is_bid, price, clamped)
        if L is not None:
            self._apply(L, count_ll)

    def taker(self, t: float, is_buy: bool, price, filled: int) -> None:
        self.predict(t)
        p = self.p
        if p['taker_obs'] and price is not None:
            S = self.c * self.h + self.r + self.k
            o = (price - S) if is_buy else (S - price)
            L = p['w_u'] * ((o >= 0.5) & (o < 1.5))
            L = L + (1 - p['w_u']) / p['tau'] * np.exp(-np.maximum(o - 0.5, 0.0) / p['tau']) * (o >= 0.5) + 0.01
            self._apply(L, False)
        if filled:
            self.shift((1 if is_buy else -1) * p['c'] * math.sqrt(filled))


def _newer(a: str, b: str) -> bool:
    return len(a) > len(b) or (len(a) == len(b) and a > b)


class FVMM(BaseStrategy):
    def __init__(self) -> None:
        super().__init__()
        self.p = PARAMS
        self.f: FV | None = None
        self.pos = 0
        self.orders: dict[str, list] = {}  # id -> [side, price, remaining]
        self.cancelling: set[str] = set()
        self.max_id = ''
        self.mbb: int | None = None  # market-only touch before the current event
        self.mba: int | None = None
        self.nb = self.na = 0
        self.now_t = 0.0
        self.last_new_t = 0.0

    # ---------------------------------------------------------------- private feed
    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        o = self.orders.get(event.order_id)
        if o is None:
            return
        st = event.status
        if st is OrderStatus.PARTIALLY_FILLED or st is OrderStatus.FILLED:
            self.pos += event.filled_quantity if o[0] is BUY else -event.filled_quantity
            o[2] = event.remaining_quantity
        if st is OrderStatus.FILLED or st is OrderStatus.CANCELED or st is OrderStatus.REJECTED:
            self.orders.pop(event.order_id, None)
            self.cancelling.discard(event.order_id)

    # ---------------------------------------------------------------- public feed
    def _scan(self, views) -> tuple[int | None, object]:
        """Market-only touch of one side and the newest market order in it (None if none is new).

        Market ids are equal-length digit strings (7 digits on every tape so far), so plain string order is
        id order; user ids ('u...') compare greater and are dropped by the owner check. If the id length ever
        grows, on_order_book_change notices that no insert was seen for a second and re-syncs max_id.
        """
        top = None
        for v in views:
            if v.owner == 'market':
                top = v.price
                break
        new = None
        mx = self.max_id
        for v in views:
            if v.order_id > mx and v.owner == 'market':
                mx = v.order_id
                new = v
        if new is not None:
            self.max_id = mx
            self.last_new_t = self.now_t
        return top, new

    def _resync(self, views) -> None:
        best = self.max_id
        for v in views:
            if v.owner == 'market' and _newer(v.order_id, best):
                best = v.order_id
        self.max_id = best

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        p = self.p
        bids, asks = snapshot.bids, snapshot.asks
        t = snapshot.timestamp_us * 1e-6
        self.now_t = t
        f = self.f
        if f is None:
            if not bids or not asks:
                return
            f = self.f = FV(p, (bids[0].price + asks[0].price) / 2)
        q = 0
        sgn = 0
        tid = None
        for tr in trades:
            if tr.taker_owner == 'market':
                q += tr.quantity
                sgn = 1 if tr.book_side == 'ask' else -1
                tid = tr.taker_order_id
        # new market inserts: only scan a side whose view list grew or whose top moved
        nb, na = len(bids), len(asks)
        scan_b = nb > self.nb or (nb and bids[0].price != self.mbb)
        scan_a = na > self.na or (na and asks[0].price != self.mba)
        self.nb, self.na = nb, na
        pbb, pba = self.mbb, self.mba
        newb = newa = None
        if scan_b:
            self.mbb, newb = self._scan(bids)
        if scan_a:
            self.mba, newa = self._scan(asks)
        if t - self.last_new_t > 1.0:  # ~180 inserts / s: a silent second means max_id is out of sync
            self._resync(bids)
            self._resync(asks)
            self.last_new_t = t
        if q:
            if self._newer_id(tid) and len(tid) == len(self.max_id):
                self.max_id = tid
            rested = (newb is not None and newb.order_id == tid) or (newa is not None and newa.order_id == tid)
            if not rested:  # a crossing limit order (only possible against us) is no IOC: no impact
                f.taker(t, sgn > 0, None, q)
        elif newb is not None:
            f.maker(t, True, newb.price, pba)
        elif newa is not None:
            f.maker(t, False, newa.price, pbb)
        if not bids or not asks or snapshot.timestamp_us < p['trade_start_us']:
            return
        self._quote(f, bids[0].price, asks[0].price, bids, asks)

    def _newer_id(self, oid: str | None) -> bool:
        return oid is not None and _newer(oid, self.max_id)

    # ---------------------------------------------------------------- trading
    def _send(self, side: Side, price: int, qty: int, kind: OrderType) -> None:
        oid = self.send_order(OrderRequest(side=side, price=price, quantity=qty, order_type=kind))
        self.orders[oid] = [side, price, qty]
        if kind is OrderType.IOC:
            self.cancelling.add(oid)  # counts against the room until it is done, never "kept"

    def _quote(self, f: FV, bb: int, ba: int, bids, asks) -> None:
        p = self.p
        F, sd = f.mean, f.sd
        cap = p['cap']
        pos = self.pos
        rest_b = rest_a = 0
        for o in self.orders.values():
            if o[0] is BUY:
                rest_b += o[2]
            else:
                rest_a += o[2]
        # ---- take stale quotes
        tk = p['take'] + p['take_sd'] * sd
        if p['take'] and ba <= F - tk:
            lim = math.floor(F - tk)
            q = 0
            for o in asks:
                if o.price > lim:
                    break
                if o.owner == 'market':
                    q += o.quantity
            q = min(q, cap - pos - rest_b)
            if q > 0:
                self._send(BUY, lim, q, OrderType.IOC)
                rest_b += q
        if p['take'] and bb >= F + tk:
            lim = math.ceil(F + tk)
            q = 0
            for o in bids:
                if o.price < lim:
                    break
                if o.owner == 'market':
                    q += o.quantity
            q = min(q, cap + pos - rest_a)
            if q > 0:
                self._send(SELL, lim, q, OrderType.IOC)
                rest_a += q
        # ---- passive quotes
        wide = p['theta_sd'] * sd + (p['theta_turb'] if f.exc > 0.05 else 0.0)
        sk = p['inv_skew'] * pos / cap
        th_b, th_a = p['theta'] + wide + sk, p['theta'] + wide - sk
        kp_b, kp_a = p['theta_keep'] + wide + sk, p['theta_keep'] + wide - sk
        tb = min(math.floor(F - th_b), ba - 1)
        ta = max(math.ceil(F + th_a), bb + 1)
        kd = p['keep_depth']
        rb = round(F - p['reach_off'])  # deepest bid the next sell taker reaches
        ra = round(F + p['reach_off'])
        have_b = have_a = False
        for oid, o in self.orders.items():
            if oid in self.cancelling:
                continue
            px = o[1]
            if o[0] is BUY:
                if F - px >= kp_b and tb - kd <= px < ba and px >= rb:
                    have_b = have_b or px == tb
                    continue
            else:
                if px - F >= kp_a and ta + kd >= px > bb and px <= ra:
                    have_a = have_a or px == ta
                    continue
            self.cancelling.add(oid)
            self.cancel_order(oid)
        if sd > p['max_sd']:
            return
        if not have_b:
            q = min(cap - pos - rest_b, p['size_top'])
            if q >= p['min_post']:
                self._send(BUY, tb, q, OrderType.LIMIT)
        if not have_a:
            q = min(cap + pos - rest_a, p['size_top'])
            if q >= p['min_post']:
                self._send(SELL, ta, q, OrderType.LIMIT)
