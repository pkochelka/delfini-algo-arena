"""ACF momentum market maker: join the touch, step a side back when short-horizon momentum says it will be run over.

Research (sample_v2, 1 s mid returns, see the Volatility tab of data-explorer):

    ACF lag 1 = +0.062 (95% band +-0.026), positive in all four quarters (+0.04..+0.08).
    The visible waves (- at 2 s, + at 3-7 s, - at 8-12 s, + at 12-18 s) are inside the
    band and flip sign between halves of the tape.

Regressing the next 1 s mid change on the summed mid change over each band
(100 ms grid, fit on the first half, tested on the second):

    lag-1 band only     beta 0.054   out-of-sample R^2 +0.0010
    all five bands                   out-of-sample R^2 -0.0002

So only the 1 s band carries signal, and a small one: after a one-tick move in
the last second the mid continues by ~0.013-0.054 ticks, against a mean spread
of 1.6 ticks. Taking liquidity on it loses ~0.8 ticks per trade (`USE_TAKER`
stays off). The signal is used where it costs nothing: deciding which side of
a passive two-sided quote to keep at the touch.

Forecast:  f = sum_b BETA_b * (mid(t - a_b) - mid(t - b_b))   over the enabled bands
Signal:    s = FORECAST_GAIN * f - INV_SKEW * inventory        (ticks)
Quotes:    join the market best bid and ask; if s > SKEW_THRESHOLD the ask steps
           back one tick (don't sell into an up-move), if s < -SKEW_THRESHOLD the
           bid does.

Risk: same exposure accounting as avellaneda_stoikov.py; worst-case |inventory|
including open orders stays within `MAX_INVENTORY`, well inside the 100 limit.

Backtests on sample_v2 (--no-logs; avellaneda_stoikov.py makes 3,945 +- 162):

    variant                                   pnl     pnl_se  fills  |inv| q95
    no skew at all (pure touch joiner)        32,001  1,475   10,593   53
    inventory skew only (lag-1 band off)      24,209    341    7,931    7
    default: lag-1 momentum + inventory       23,682    357    7,603    9
    all five ACF bands + inventory            23,168    351    7,459    9
    lag-1 with the sign flipped (reversion)   21,290    340    7,141    8
    default, SKEW_THRESHOLD 0.05 / 0.07 / 0.10   26,683 / 28,638 / 30,215

Momentum beats reversion by ~2.4k, which agrees with the lag-1 ACF, but
stepping a side back loses more spread capture than it saves in adverse
selection: the momentum skew is ~0.5k worse than inventory skew alone. The
extra ACF bands make it worse again. The threshold sweep mostly loosens the
inventory control (more PnL, more inventory risk), not the momentum part.
"""

from __future__ import annotations

from collections import deque
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

# Forecast: (from_s, to_s, beta, enabled). Band = mid change from `to_s` ago to `from_s` ago.
# Betas fitted by OLS on the whole of sample_v2; only the lag-1 band survives out of sample.
BANDS: tuple[tuple[float, float, float, bool], ...] = (
    (0.0, 1.0, 0.0457, True),  # +ACF at 1 s
    (1.0, 2.0, -0.0052, False),  # -ACF at 2 s
    (2.0, 7.0, 0.0049, False),  # +ACF at 3-7 s
    (7.0, 12.0, -0.0033, False),  # -ACF at 8-12 s
    (12.0, 18.0, 0.0071, False),  # +ACF at 12-18 s
)
SAMPLE_US = 100_000  # mid sampling grid for the forecast
FORECAST_GAIN = 1.0
SKEW_THRESHOLD = 0.03  # ticks of signal before a side steps back; 0.03 < beta for a 1-tick move
INV_SKEW = 0.01  # ticks of signal per lot of inventory

# Optional taking (off: the forecast never covers half the spread on sample_v2)
USE_TAKER = False
TAKE_MARGIN = 0.25  # forecast must exceed half spread + this, ticks
TAKE_SIZE = 1
TAKE_COOLDOWN_US = 1_000_000

# Execution and risk
ORDER_SIZE = 20
MAX_INVENTORY = 100
MIN_REQUOTE_US = 20_000  # minimum time between replacements on one side


@dataclass(slots=True)
class _Order:
    side: Side
    price: int
    remaining: int
    order_type: OrderType = OrderType.LIMIT
    cancel_sent: bool = False


class AcfMomentumMarketMaker(BaseStrategy):
    """Two-sided touch quotes whose side selection follows the 1 s return autocorrelation."""

    def __init__(self) -> None:
        super().__init__()
        self._inventory: int = 0
        self._orders: dict[str, _Order] = {}
        self._last_requote_us: dict[Side, int] = {Side.BUY: -(10**18), Side.SELL: -(10**18)}
        self._last_take_us: int = -(10**18)

        self._bands = [(round(a * 1e6 / SAMPLE_US), round(b * 1e6 / SAMPLE_US), beta) for a, b, beta, on in BANDS if on]
        depth = max((b for _, b, _ in self._bands), default=0) + 1
        self._mids: deque[float] = deque(maxlen=depth)  # mids[-1] = latest grid sample
        self._next_sample_us: int | None = None
        self._last_mid: float | None = None

    # ------------------------------------------------------------------ feeds

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        best_bid = next((o.price for o in snapshot.bids if o.owner == 'market'), None)
        best_ask = next((o.price for o in snapshot.asks if o.owner == 'market'), None)
        if best_bid is None or best_ask is None or best_bid >= best_ask:
            return
        now = snapshot.timestamp_us
        self._sample(now, 0.5 * (best_bid + best_ask))

        forecast = self._forecast()
        if forecast is None:
            return

        if USE_TAKER and self._maybe_take(forecast, best_bid, best_ask, now):
            return

        signal = FORECAST_GAIN * forecast - INV_SKEW * self._inventory
        bid_px, ask_px = best_bid, best_ask
        if signal > SKEW_THRESHOLD:
            ask_px += 1
        elif signal < -SKEW_THRESHOLD:
            bid_px -= 1
        self._manage_side(Side.BUY, bid_px if bid_px > 0 else None, now)
        self._manage_side(Side.SELL, ask_px, now)

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        order = self._orders.get(event.order_id)
        if order is None:
            return
        status = event.status
        if status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
            self._inventory += (1 if order.side == Side.BUY else -1) * event.filled_quantity
            order.remaining = event.remaining_quantity
            if status == OrderStatus.FILLED or order.remaining == 0:
                del self._orders[event.order_id]
        elif status in (OrderStatus.CANCELED, OrderStatus.REJECTED):
            del self._orders[event.order_id]

    # --------------------------------------------------------------- forecast

    def _sample(self, now: int, mid: float) -> None:
        """Sample-and-hold the market mid onto a fixed grid: each grid point gets the mid in force at it."""
        if self._next_sample_us is None:
            self._next_sample_us = now
        while self._next_sample_us <= now:
            # Grid points passed since the last event saw the previous mid, not this one.
            self._mids.append(self._last_mid if self._last_mid is not None and self._next_sample_us < now else mid)
            self._next_sample_us += SAMPLE_US
        self._last_mid = mid

    def _forecast(self) -> float | None:
        """Expected mid change over the next second, ticks; None until the history is long enough."""
        if len(self._mids) < self._mids.maxlen:
            return None
        m = self._mids
        return sum(beta * (m[-1 - a] - m[-1 - b]) for a, b, beta in self._bands)

    # ---------------------------------------------------------------- trading

    def _maybe_take(self, forecast: float, best_bid: int, best_ask: int, now: int) -> bool:
        if now - self._last_take_us < TAKE_COOLDOWN_US:
            return False
        edge = abs(forecast) - 0.5 * (best_ask - best_bid) - TAKE_MARGIN
        if edge <= 0:
            return False
        side = Side.BUY if forecast > 0 else Side.SELL
        if not self._has_room(side, TAKE_SIZE):
            return False
        price = best_ask if side == Side.BUY else best_bid
        oid = self.send_order(OrderRequest(side=side, price=price, quantity=TAKE_SIZE, order_type=OrderType.IOC))
        self._orders[oid] = _Order(side=side, price=price, remaining=TAKE_SIZE, order_type=OrderType.IOC)
        self._last_take_us = now
        return True

    def _manage_side(self, side: Side, target: int | None, now: int) -> None:
        live = [(oid, o) for oid, o in self._orders.items() if o.side == side and not o.cancel_sent and o.order_type == OrderType.LIMIT]
        sign = 1 if side == Side.BUY else -1
        live_qty = sum(o.remaining for _, o in live)

        if target is None or sign * self._inventory + live_qty > MAX_INVENTORY:
            for oid, o in live:
                self._cancel(oid, o)
            return

        if live:
            if all(o.price == target for _, o in live):
                return
            if now - self._last_requote_us[side] < MIN_REQUOTE_US:
                return
            for oid, o in live:
                self._cancel(oid, o)

        if not self._has_room(side, ORDER_SIZE):
            return
        oid = self.send_order(OrderRequest(side=side, price=target, quantity=ORDER_SIZE, order_type=OrderType.LIMIT))
        self._orders[oid] = _Order(side=side, price=target, remaining=ORDER_SIZE)
        self._last_requote_us[side] = now

    def _has_room(self, side: Side, qty: int) -> bool:
        """Worst case: every open order on `side`, cancels in flight included, fills."""
        pending = sum(o.remaining for o in self._orders.values() if o.side == side)
        sign = 1 if side == Side.BUY else -1
        return sign * self._inventory + pending + qty <= MAX_INVENTORY

    def _cancel(self, oid: str, order: _Order) -> None:
        if not order.cancel_sent:
            self.cancel_order(oid)
            order.cancel_sent = True
