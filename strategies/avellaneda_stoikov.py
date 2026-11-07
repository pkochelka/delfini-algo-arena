"""Avellaneda-Stoikov market maker: inventory-skewed two-sided quotes around a reservation price.

Avellaneda & Stoikov (2008), "High-frequency trading in a limit order book".
With mid price s, signed inventory q, risk aversion gamma, mid volatility
sigma (ticks per sqrt-second), time to the horizon tau (seconds) and order
arrival decay k (per tick), where fills at distance d from the mid arrive at
rate A * exp(-k * d):

    reservation price   r     = s - q * gamma * sigma^2 * tau
    optimal spread      delta = gamma * sigma^2 * tau + (2 / gamma) * ln(1 + gamma / k)
    quotes              bid   = r - delta / 2,   ask = r + delta / 2

tau = T - t, with the horizon T = `HORIZON_S` (one hour, the length of a
round) counted from the first book update. tau never drops below `TAU_MIN`,
so the quotes keep some inventory skew and spread into the end of the round.
sigma is estimated online as an EWMA of squared mid changes sampled on a
fixed time grid. k is estimated online from how many ticks past the touch each
aggressive market order reaches. Arrivals of A * exp(-k * d) with d = half
spread + depth beyond the touch decay at the same k beyond the touch, so the
depth is geometric with P(depth >= n) = exp(-k * n), mean m = 1 / (e^k - 1),
and k = ln(1 + 1 / m).

Risk: the exchange charges $10,000 per second while |inventory| > 100. A
resting quote is cancelled once the current position plus that quote filling
would pass `MAX_INVENTORY`, and a new quote is sent only if the position plus
every open order on that side (cancels still in flight included) plus the new
one stays within it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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

# Model parameters
GAMMA = 0.1 * 30.0 / 3600.0  # risk aversion; gamma * tau at t = 0 equals 0.1 * 30 s
HORIZON_S = 3600.0  # terminal time T, seconds after the first book update
TAU_MIN = 900.0  # floor on tau = T - t, seconds
K_INIT = 1.5  # initial order-arrival decay per tick, before any estimate
K_MIN, K_MAX = 0.05, 20.0
SIGMA_INIT = 0.5  # initial mid volatility, ticks per sqrt-second
SIGMA_SAMPLE_US = 100_000  # mid sampling interval for the volatility estimate
SIGMA_HALFLIFE_S = 60.0  # EWMA half-life of the variance estimate
K_HALFLIFE_TRADES = 500.0  # EWMA half-life of the trade-distance estimate, in trades

# Execution and risk parameters
ORDER_SIZE = 5
MAX_INVENTORY = 60  # worst-case |inventory| allowed, well inside the 100 penalty threshold
MIN_HALF_SPREAD = 2.0  # never quote closer than this to the reservation price, ticks
REQUOTE_TICKS = 1  # move a resting quote once its target differs by at least this
MIN_REQUOTE_US = 20_000  # minimum time between replacements on one side


@dataclass(slots=True)
class _Order:
    """One of our open orders, from submission until it is filled, cancelled or rejected."""

    side: Side
    price: int
    remaining: int
    cancel_sent: bool = False


class AvellanedaStoikovStrategy(BaseStrategy):
    """Quote one order per side at the Avellaneda-Stoikov bid and ask, replacing them as the targets move."""

    def __init__(self) -> None:
        super().__init__()
        self._inventory: int = 0
        self._start_us: int | None = None  # time of the first book update, t = 0
        self._orders: dict[str, _Order] = {}
        self._last_requote_us: dict[Side, int] = {Side.BUY: -(10**18), Side.SELL: -(10**18)}

        # Volatility estimate (variance in ticks^2 per second)
        self._variance: float = SIGMA_INIT**2
        self._last_sample_us: int | None = None
        self._last_sample_mid: float | None = None

        # Arrival-decay estimate: EWMA of how many ticks past the touch aggressive orders reach
        self._mean_trade_depth: float = 1.0 / math.expm1(K_INIT)
        self._prev_touch: tuple[int, int] | None = None  # market best bid and ask before the event

    # ------------------------------------------------------------------ feeds

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        """Update the estimates from the market's own book and trades, then re-quote both sides."""
        best_bid = next((o.price for o in snapshot.bids if o.owner == 'market'), None)
        best_ask = next((o.price for o in snapshot.asks if o.owner == 'market'), None)
        if best_bid is None or best_ask is None or best_bid >= best_ask:
            return
        mid = 0.5 * (best_bid + best_ask)
        now = snapshot.timestamp_us
        if self._start_us is None:
            self._start_us = now

        self._update_k(trades)
        self._update_sigma(now, mid)
        self._prev_touch = (best_bid, best_ask)

        tau = max(TAU_MIN, HORIZON_S - (now - self._start_us) / 1e6)
        bid_px, ask_px = self._target_quotes(mid, best_bid, best_ask, tau)
        self._manage_side(Side.BUY, bid_px, now)
        self._manage_side(Side.SELL, ask_px, now)

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        """Track inventory from fills and forget orders once they are done."""
        order = self._orders.get(event.order_id)
        if order is None:
            return
        status = event.status
        if status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
            sign = 1 if order.side == Side.BUY else -1
            self._inventory += sign * event.filled_quantity
            order.remaining = event.remaining_quantity
            if status == OrderStatus.FILLED or order.remaining == 0:
                del self._orders[event.order_id]
        elif status in (OrderStatus.CANCELED, OrderStatus.REJECTED):
            # A rejected cancel means the order is no longer resting (already filled).
            del self._orders[event.order_id]

    # -------------------------------------------------------------- estimates

    def _update_sigma(self, now: int, mid: float) -> None:
        """Fold mid changes over each elapsed sampling interval into the variance EWMA."""
        if self._last_sample_us is None or self._last_sample_mid is None:
            self._last_sample_us, self._last_sample_mid = now, mid
            return
        elapsed_us = now - self._last_sample_us
        if elapsed_us < SIGMA_SAMPLE_US:
            return
        dt = elapsed_us / 1e6
        rate = (mid - self._last_sample_mid) ** 2 / dt
        alpha = 1.0 - 0.5 ** (dt / SIGMA_HALFLIFE_S)
        self._variance += alpha * (rate - self._variance)
        self._last_sample_us, self._last_sample_mid = now, mid

    def _update_k(self, trades: list[Trade]) -> None:
        """Fold each market taker's depth past the previous touch into the mean-depth EWMA.

        A taker sweeping several levels prints one trade per level but is one
        arrival, so only its deepest price counts.
        """
        if self._prev_touch is None:
            return
        prev_bid, prev_ask = self._prev_touch
        deepest: dict[str, int] = {}
        for t in trades:
            if t.taker_owner != 'market':
                continue
            depth = prev_bid - t.price if t.book_side == 'bid' else t.price - prev_ask
            # Fills against our quotes inside the spread count as at the touch.
            deepest[t.taker_order_id] = max(deepest.get(t.taker_order_id, 0), depth)
        alpha = 1.0 - 0.5 ** (1.0 / K_HALFLIFE_TRADES)
        for depth in deepest.values():
            self._mean_trade_depth += alpha * (depth - self._mean_trade_depth)

    @property
    def _k(self) -> float:
        return min(K_MAX, max(K_MIN, math.log1p(1.0 / max(self._mean_trade_depth, 1e-9))))

    # ---------------------------------------------------------------- quoting

    def _target_quotes(self, mid: float, best_bid: int, best_ask: int, tau: float) -> tuple[int | None, int]:
        """Return the integer bid and ask to quote with `tau` seconds left, or None for a bid that would not be positive."""
        var_tau = self._variance * tau
        reservation = mid - self._inventory * GAMMA * var_tau
        half_spread = 0.5 * GAMMA * var_tau + math.log1p(GAMMA / self._k) / GAMMA
        half_spread = max(half_spread, MIN_HALF_SPREAD)

        # Nearest tick; flooring/ceiling would push every quote a tick behind the touch.
        bid = math.floor(reservation - half_spread + 0.5)
        ask = math.ceil(reservation + half_spread - 0.5)
        # Stay passive: a quote crossing the touch would trade on arrival.
        bid = min(bid, best_ask - 1)
        ask = max(ask, best_bid + 1)

        return (bid if bid > 0 else None), ask

    def _manage_side(self, side: Side, target: int | None, now: int) -> None:
        """Keep one live order on `side` at `target`: cancel stale or too-risky quotes and place a new one.

        Inventory limits: a live quote stays only while the position plus that
        quote filling fits in `MAX_INVENTORY`; `_send` separately checks room
        for a new one. Both count only real exposure, so a quote that fits is
        never cancelled just to be re-sent (and cancelled again) next event.
        """
        live = [(oid, o) for oid, o in self._orders.items() if o.side == side and not o.cancel_sent]
        sign = 1 if side == Side.BUY else -1
        live_qty = sum(o.remaining for _, o in live)

        if target is None or sign * self._inventory + live_qty > MAX_INVENTORY:
            for oid, o in live:
                self._cancel(oid, o)
            return

        if live:
            if now - self._last_requote_us[side] < MIN_REQUOTE_US:
                return
            if all(abs(o.price - target) < REQUOTE_TICKS for _, o in live):
                return
            for oid, o in live:
                self._cancel(oid, o)

        # Cancel-and-replace without waiting for the cancel: `_send` counts the
        # order being cancelled in its worst-case exposure, in case it fills first.
        self._send(side, target, now)

    def _send(self, side: Side, price: int, now: int) -> None:
        # Worst-case exposure includes orders whose cancel may still lose to a fill.
        pending = sum(o.remaining for o in self._orders.values() if o.side == side)
        sign = 1 if side == Side.BUY else -1
        if sign * self._inventory + pending + ORDER_SIZE > MAX_INVENTORY:
            return
        oid = self.send_order(OrderRequest(side=side, price=price, quantity=ORDER_SIZE, order_type=OrderType.LIMIT))
        self._orders[oid] = _Order(side=side, price=price, remaining=ORDER_SIZE)
        self._last_requote_us[side] = now

    def _cancel(self, oid: str, order: _Order) -> None:
        if not order.cancel_sent:
            self.cancel_order(oid)
            order.cancel_sent = True
