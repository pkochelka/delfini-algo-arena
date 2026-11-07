"""arb_mm: Avellaneda-Stoikov market maker around microprice + an online ridge prediction.
Every mechanic is behind a key in PARAMS so it can be switched off rather than deleted.
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
    # exchange
    'limit': 100,  # hard position limit (rules.md)
    'latency_us': 5_000,
    # signal
    'horizon': 20,
    'fast': 20,
    'slow': 200,
    'flow_span': 20,
    'memory': 2000,
    'ridge': 1.0,
    'warmup': 400,
    'refit_every': 50,
    'pred_clip': 3.0,
    'pred_weight': 1.0,
    'kalman': 1,
    'kalman_span': 500,
    # Avellaneda-Stoikov
    'tau': 50.0,
    'inv_risk': 8.0,
    'vol_span': 200,
    'k_span': 500,
    'adapt': 3.0,
    'size_frac': 0.1,
    'takes': 1,
    'take_frac': 0.2,
    'take_margin': 1.0,
    # latency buffer
    'post_k': 3.0,
    'post_adapt': 1,
    'post_target': 0.04,
    'post_k_min': 0.25,
    'post_k_max': 3.0,
    # telemetry
    'markout_us': 1_000_000,
    # risk
    'pos_limit_frac': 0.9,
    'dd_ticks': 150.0,  # x limit
    'kill_cooldown_s': 60.0,
    'max_loss_ticks': 100.0,  # x limit
}


class EWStat:
    """Online EW mean/variance, returning a clipped z-score."""

    def __init__(self, memory: float) -> None:
        self.a = 1.0 / memory
        self.mean = 0.0
        self.var = 0.0
        self.started = False

    def z(self, x: float) -> float:
        if not self.started:
            self.started = True
            self.mean = x
            return 0.0
        d = x - self.mean
        self.mean += self.a * d
        self.var = (1.0 - self.a) * (self.var + self.a * d * d)
        if self.var <= 1e-12:
            return 0.0
        return max(-5.0, min(5.0, d / math.sqrt(self.var)))


class OnlineRidge:
    """Exponentially forgetting ridge regression, solved by Gaussian elimination."""

    def __init__(self, k: int, memory: float, ridge: float) -> None:
        self.k = k
        self.lam = 1.0 - 1.0 / memory
        self.ridge = ridge
        self.xx = [[0.0] * k for _ in range(k)]
        self.xy = [0.0] * k
        self.coef = [0.0] * k

    def update(self, x: list[float], y: float) -> None:
        lam = self.lam
        for i in range(self.k):
            xi = x[i]
            row = self.xx[i]
            for j in range(self.k):
                row[j] = lam * row[j] + xi * x[j]
            self.xy[i] = lam * self.xy[i] + xi * y

    def fit(self) -> None:
        k = self.k
        a = [self.xx[i][:] + [self.xy[i]] for i in range(k)]
        for i in range(k):
            a[i][i] = a[i][i] * (1.0 + self.ridge) + 1e-9
        for c in range(k):
            p = max(range(c, k), key=lambda r: abs(a[r][c]))
            if abs(a[p][c]) < 1e-12:
                self.coef = [0.0] * k
                return
            a[c], a[p] = a[p], a[c]
            for r in range(c + 1, k):
                f = a[r][c] / a[c][c]
                if f:
                    for j in range(c, k + 1):
                        a[r][j] -= f * a[c][j]
        coef = [0.0] * k
        for i in range(k - 1, -1, -1):
            s = a[i][k] - sum(a[i][j] * coef[j] for j in range(i + 1, k))
            coef[i] = s / a[i][i]
        self.coef = coef

    def predict(self, x: list[float]) -> float:
        return sum(c * v for c, v in zip(self.coef, x))


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


class ArbMarketMaker(BaseStrategy):
    """Skewed AS quotes around mid + learned prediction, with a latency buffer and a hard guard."""

    N_FEATURES = 6

    def __init__(self) -> None:
        super().__init__()
        p = PARAMS
        self.p = p
        self.L = p['limit']
        # own state
        self.pos = 0
        self.cash = 0.0
        self.orders: dict[str, list] = {}  # id -> [side, price, remaining, is_ioc]
        self.cancelling: set[str] = set()
        self.ioc_in_flight = False
        # model state
        self.t = 0
        self.prev_mid: float | None = None
        self.var = 1.0
        self.dist = 2.0
        self.spread = 1.0
        self.ema_fast: float | None = None
        self.ema_slow = 0.0
        self.flow = 0.0
        self.level: float | None = None
        self.prev_y = 0.0
        self.prev_dy = 0.0
        self.k_var = 1.0
        self.k_cov = 0.0
        self.prev_ts: int | None = None
        self.dt_us = 1000.0  # EW mean µs between events
        self.stats = [EWStat(p['memory']) for _ in range(self.N_FEATURES)]
        self.ridge = OnlineRidge(self.N_FEATURES, p['memory'], p['ridge'])
        self.hist: deque = deque()
        self.pred = 0.0
        self.gamma: float | None = None
        # telemetry
        self.markouts: deque = deque()  # (due_us, sign, price, qty, maker)
        self.mk = 0.0
        self.mk_take = 0.0
        self.cross_share = p['post_target']
        self.post_k = p['post_k']
        self.mid = 0.0
        self.best_bid = 0
        self.best_ask = 0
        # risk
        self.peak = 0.0
        self.size_mult = 1.0
        self.kill_until = -1
        self.halted = False

    # ------------------------------------------------------------ private feed
    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        o = self.orders.get(event.order_id)
        if o is None:
            return
        st = event.status
        if st == OrderStatus.ACCEPTED:
            o[4] = event.timestamp_us
        elif st == OrderStatus.PARTIALLY_FILLED or st == OrderStatus.FILLED:
            q = event.filled_quantity
            px = event.fill_price
            sign = 1 if o[0] == Side.BUY else -1
            self.pos += sign * q
            self.cash -= sign * q * px
            o[2] = event.remaining_quantity
            # a passive order filled at the moment it was accepted crossed on arrival
            maker = not o[3] and event.timestamp_us > o[4]
            self.markouts.append((event.timestamp_us + self.p['markout_us'], sign, px, q, maker))
            if not o[3]:
                self.cross_share += 0.02 * ((0.0 if maker else 1.0) - self.cross_share)
        elif st == OrderStatus.MODIFIED:
            o[2] = event.remaining_quantity
        if st in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
            self.orders.pop(event.order_id, None)
            self.cancelling.discard(event.order_id)
            if o[3]:
                self.ioc_in_flight = False

    # ------------------------------------------------------------- public feed
    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        bids = market_levels(snapshot.bids, 3)
        asks = market_levels(snapshot.asks, 3)
        if not bids or not asks:
            return
        now = snapshot.timestamp_us
        p = self.p
        self.t += 1
        bb, bq = bids[0]
        ba, aq = asks[0]
        self.best_bid, self.best_ask = bb, ba
        mid = (bb + ba) / 2.0
        self.mid = mid
        micro = (bb * aq + ba * bq) / (aq + bq)

        if self.prev_ts is not None:
            self.dt_us += 0.01 * ((now - self.prev_ts) - self.dt_us)
        self.prev_ts = now

        prev_mid = self.prev_mid if self.prev_mid is not None else mid
        d = mid - prev_mid
        self.var += 2.0 / (p['vol_span'] + 1) * (d * d - self.var)
        ak = 2.0 / (p['k_span'] + 1)
        signed = 0
        for tr in trades:
            self.dist += ak * (abs(tr.price - prev_mid) - self.dist)
            signed += tr.quantity if tr.book_side == 'ask' else -tr.quantity
        k_as = 1.0 / max(self.dist, 0.5)
        self.prev_mid = mid

        # adaptive Kalman on the microprice
        y = micro
        if self.level is None:
            self.level = y
            self.prev_y = y
        if p['kalman']:
            dy = y - self.prev_y
            a = 1.0 / p['kalman_span']
            self.k_var += a * (dy * dy - self.k_var)
            self.k_cov += a * (dy * self.prev_dy - self.k_cov)
            self.prev_dy = dy
            r_n = max(0.0, -self.k_cov)
            q_n = max(1e-6, self.k_var - 2 * r_n)
            if r_n < 1e-9:
                gain = 1.0
            else:
                pk = (q_n + math.sqrt(q_n * q_n + 4 * q_n * r_n)) / 2
                gain = pk / (pk + r_n)
            self.level += gain * (y - self.level)
        else:
            self.level = y
        self.prev_y = y

        self.spread += 0.02 * ((ba - bb) - self.spread)
        if self.ema_fast is None:
            self.ema_fast = self.ema_slow = mid
        self.ema_fast += 2.0 / (p['fast'] + 1) * (mid - self.ema_fast)
        self.ema_slow += 2.0 / (p['slow'] + 1) * (mid - self.ema_slow)
        self.flow += 2.0 / (p['flow_span'] + 1) * (signed - self.flow)

        bq3 = sum(lv[1] for lv in bids)
        aq3 = sum(lv[1] for lv in asks)
        raw = (
            (bq - aq) / (bq + aq),
            (bq3 - aq3) / (bq3 + aq3),
            self.flow,
            self.ema_slow - mid,
            self.ema_fast - self.ema_slow,
            mid - self.level,
        )
        z = [s.z(v) for s, v in zip(self.stats, raw)]

        self.hist.append((z, mid))
        if len(self.hist) > p['horizon']:
            z0, m0 = self.hist.popleft()
            self.ridge.update(z0, mid - m0)
        if self.t >= p['warmup']:
            if self.t % p['refit_every'] == 0:
                self.ridge.fit()
            lim = p['pred_clip'] * max(self.spread, 1.0)
            self.pred = max(-lim, min(lim, self.ridge.predict(z)))

        self._telemetry(now)
        pnl = self.cash + self.pos * mid
        if self.halted:
            return
        if pnl <= -p['max_loss_ticks'] * self.L:
            self._halt()
            return
        self.peak = max(self.peak, pnl)
        if self.kill_until < 0 and self.peak - pnl > p['dd_ticks'] * self.L * self.size_mult:
            self.kill_until = now + int(p['kill_cooldown_s'] * 1e6)
        if 0 <= self.kill_until <= now:
            self.kill_until = -1
            self.size_mult *= 0.5
            self.peak = pnl
        if self.t < p['warmup']:
            return
        self._quote(mid, k_as)

    # ---------------------------------------------------------------- helpers
    def _telemetry(self, now: int) -> None:
        mq = self.markouts
        while mq and mq[0][0] <= now:
            _, sign, px, _, maker = mq.popleft()
            mo = sign * (self.mid - px)
            if maker:
                self.mk += 0.02 * (mo - self.mk)
            else:
                self.mk_take += 0.05 * (mo - self.mk_take)

    def _resting(self, side: Side) -> int:
        return sum(o[2] for o in self.orders.values() if o[0] == side)

    def _room(self, side: Side) -> int:
        """Guard: largest qty that keeps the worst-case position within the cap."""
        cap = int(self.p['pos_limit_frac'] * self.L)
        if side == Side.BUY:
            return max(0, cap - (self.pos + self._resting(Side.BUY)))
        return max(0, cap + self.pos - self._resting(Side.SELL))

    def _send(self, side: Side, price: int, qty: int, ioc: bool) -> None:
        qty = min(qty, self._room(side))
        if qty <= 0 or price <= 0:
            return
        oid = self.send_order(OrderRequest(side=side, price=price, quantity=qty,
                                           order_type=OrderType.IOC if ioc else OrderType.LIMIT))
        # [side, price, remaining, is_ioc, accepted_us]
        self.orders[oid] = [side, price, qty, ioc, 1 << 62]
        if ioc:
            self.ioc_in_flight = True

    def _cancel(self, oid: str) -> None:
        if oid not in self.cancelling:
            self.cancelling.add(oid)
            self.cancel_order(oid)

    def _sync(self, side: Side, price: int | None, qty: int) -> None:
        mine = [(oid, o) for oid, o in self.orders.items()
                if o[0] == side and not o[3] and oid not in self.cancelling]
        if price is not None and qty > 0 and len(mine) == 1:
            oid, o = mine[0]
            if o[1] == price and qty / 2 <= o[2] <= qty:
                return
        for oid, _ in mine:
            self._cancel(oid)
        if price is None or qty <= 0:
            return
        if any(self.orders[oid][0] == side for oid in self.cancelling if oid in self.orders):
            return  # no stacking while a cancel is in flight
        self._send(side, price, qty, False)

    def _quote(self, mid: float, k_as: float) -> None:
        p = self.p
        L = self.L
        var = max(self.var, 1e-3)
        if self.gamma is None:
            self.gamma = p['inv_risk'] / (L * var * p['tau'])
        g = self.gamma
        risk = var * p['tau']
        fair = mid + p['pred_weight'] * self.pred
        r = fair - self.pos * g * risk
        delta = max(1.0, g * risk / 2 + math.log(1 + g / k_as) / g + p['adapt'] * max(0.0, -self.mk))
        q = max(1, round(p['size_frac'] * L * self.size_mult))
        reduce_only = self.kill_until >= 0

        if p['post_adapt']:
            self.post_k *= math.exp(0.01 * (self.cross_share / p['post_target'] - 1))
            self.post_k = max(p['post_k_min'], min(p['post_k_max'], self.post_k))
        ev_per_lat = max(1.0, p['latency_us'] / max(self.dt_us, 1.0))
        buf = math.ceil(self.post_k * math.sqrt(var * ev_per_lat))

        bid_px = min(math.floor(r - delta), self.best_ask - 1 - buf)
        ask_px = max(math.ceil(r + delta), self.best_bid + 1 + buf)
        bid_q = q if not (reduce_only and self.pos >= 0) else 0
        ask_q = q if not (reduce_only and self.pos <= 0) else 0

        take_side = None
        if p['takes'] and not reduce_only and not self.ioc_in_flight:
            need = p['take_margin'] + p['adapt'] * max(0.0, -self.mk_take)
            qmax = max(1, round(p['take_frac'] * L * self.size_mult))
            if fair - self.best_ask > need:
                take_side, bid_q = Side.BUY, 0
            elif self.best_bid - fair > need:
                take_side, ask_q = Side.SELL, 0

        self._sync(Side.BUY, bid_px if bid_q else None, bid_q)
        self._sync(Side.SELL, ask_px if ask_q else None, ask_q)
        if take_side == Side.BUY:
            self._send(Side.BUY, self.best_ask, qmax, True)
        elif take_side == Side.SELL:
            self._send(Side.SELL, self.best_bid, qmax, True)

    def _halt(self) -> None:
        self.halted = True
        for oid, o in list(self.orders.items()):
            if not o[3]:
                self._cancel(oid)
        if self.pos > 0:
            self._send(Side.SELL, self.best_bid, self.pos, True)
        elif self.pos < 0:
            self._send(Side.BUY, self.best_ask, -self.pos, True)
