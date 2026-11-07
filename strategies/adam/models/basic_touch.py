"""Basic market-making strategy. Parameters below are deliberately easy to tune.

Standalone submission: imports only the standard library and the public simulator API.
Prices are ticks; quantities are lots; time constants are seconds unless named _US.
"""
from math import ceil, exp, floor, sqrt

from simulator.models import OrderRequest, OrderStatus, OrderType, Side
from simulator.strategy import BaseStrategy

class TouchMarketMaker(BaseStrategy):
    """Fixed-size at-touch reference; inventory management only through exposure caps."""

    BASE_SIZE = 5                 # Lots per new quote before size adjustment.
    POSITION_LIMIT = 80           # Conservative internal cap; competition cap is 100.
    REFRESH_US = 100_000           # Reconsider quotes every 100 ms of market events.
    REQUOTE_TICKS = 1              # Cancel only when target moves by this many ticks.
    MIN_HALF_SPREAD = 0.5          # Minimum distance from model center, in ticks.
    INVENTORY_SKEW_TICKS = 0.0    # Center shift at the internal inventory cap.
    SIZE_SKEW = 0.0               # 0 = fixed size; 1 = inventory-dependent size.
    IMBALANCE_TICKS = 0.0         # Center shift at full book imbalance.
    ADAPT_VOLATILITY = False
    VOL_TAU_S = 10.0               # Time decay of squared price moves.
    INIT_VARIANCE_RATE = 1.0       # Starting estimate in ticks squared per second.
    RISK_HORIZON_S = 1.0           # Exposure horizon used for volatility widening.
    VOL_MULT = 1.0                # Multiplier on horizon price standard deviation.
    MAX_HALF_SPREAD = 10.0         # Limit volatility-driven widening.


    def __init__(self):
        super().__init__()
        self.inventory = 0
        self.orders = {}
        self.last_refresh_us = -10**18
        self.last_mid = None
        self.last_mid_us = None
        self.variance_rate = self.INIT_VARIANCE_RATE

    def on_private_order_update(self, event):
        order = self.orders.get(event.order_id)
        if order is None:
            return
        if event.status == OrderStatus.ACCEPTED:
            order["accepted"] = True
        elif event.status in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED):
            sign = 1 if order["side"] == Side.BUY else -1
            self.inventory += sign * event.filled_quantity
            order["remaining"] = event.remaining_quantity
            if event.status == OrderStatus.FILLED:
                del self.orders[event.order_id]
        elif event.status == OrderStatus.CANCELED:
            del self.orders[event.order_id]
        elif event.status == OrderStatus.REJECTED:
            if order["accepted"]:
                # A rejected cancel must not erase the still-live order's reservation.
                order["canceling"] = False
            else:
                del self.orders[event.order_id]

    def on_order_book_change(self, snapshot, trades):
        # Exclude our quotes from the signal and reference prices.
        best_bid = None
        best_ask = None
        bid_volume = 0
        ask_volume = 0
        for order in snapshot.bids:
            if order.owner != "market":
                continue
            if best_bid is None:
                best_bid = order.price
            if order.price == best_bid:
                bid_volume += order.quantity
        for order in snapshot.asks:
            if order.owner != "market":
                continue
            if best_ask is None:
                best_ask = order.price
            if order.price == best_ask:
                ask_volume += order.quantity

        now = snapshot.timestamp_us
        if best_bid is None or best_ask is None or best_bid >= best_ask:
            for order_id, order in list(self.orders.items()):
                self.cancel_quote(order_id, order)
            return

        mid = (best_bid + best_ask) / 2
        if self.last_mid is not None and now > self.last_mid_us:
            dt = (now - self.last_mid_us) / 1_000_000
            weight = 1 - exp(-dt / self.VOL_TAU_S)
            squared_move_rate = (mid - self.last_mid) ** 2 / dt
            self.variance_rate = (
                (1 - weight) * self.variance_rate + weight * squared_move_rate
            )
        self.last_mid = mid
        self.last_mid_us = now

        if now - self.last_refresh_us < self.REFRESH_US:
            return
        self.last_refresh_us = now
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        half_width = self.MIN_HALF_SPREAD
        if self.ADAPT_VOLATILITY:
            half_width = min(
                self.MAX_HALF_SPREAD,
                max(half_width, self.VOL_MULT * sqrt(
                    self.variance_rate * self.RISK_HORIZON_S
                )),
            )
        center = (
            mid
            + self.IMBALANCE_TICKS * imbalance
            - self.INVENTORY_SKEW_TICKS * self.inventory / self.POSITION_LIMIT
        )
        # Passive prices: never intentionally cross the current market.
        bid_price = max(1, min(best_bid, floor(center - half_width)))
        ask_price = max(best_ask, ceil(center + half_width))
        ratio = self.inventory / self.POSITION_LIMIT
        buy_size = max(1, round(self.BASE_SIZE * max(
            0.1, min(2.0, 1 - self.SIZE_SKEW * ratio)
        )))
        sell_size = max(1, round(self.BASE_SIZE * max(
            0.1, min(2.0, 1 + self.SIZE_SKEW * ratio)
        )))
        self.maintain_quote(Side.BUY, bid_price, buy_size)
        self.maintain_quote(Side.SELL, ask_price, sell_size)

    def cancel_quote(self, order_id, order):
        if order["accepted"] and not order["canceling"]:
            order["canceling"] = True
            self.cancel_order(order_id)

    def maintain_quote(self, side, price, desired_size):
        reserved = sum(
            order["remaining"] for order in self.orders.values()
            if order["side"] == side
        )
        capacity = (
            self.POSITION_LIMIT - self.inventory - reserved
            if side == Side.BUY
            else self.POSITION_LIMIT + self.inventory - reserved
        )
        for order_id, order in list(self.orders.items()):
            if order["side"] != side:
                continue
            # Preserve queue priority while a quote is still appropriate.
            # A cancel stays reserved until its terminal confirmation.
            allowed_existing = capacity + order["remaining"]
            if (abs(order["price"] - price) >= self.REQUOTE_TICKS
                    or allowed_existing <= 0):
                self.cancel_quote(order_id, order)
            return

        quantity = min(desired_size, capacity)
        if quantity <= 0:
            return
        order_id = self.send_order(OrderRequest(
            side=side, price=price, quantity=quantity, order_type=OrderType.LIMIT
        ))
        self.orders[order_id] = {
            "side": side, "price": price, "remaining": quantity,
            "accepted": False, "canceling": False,
        }

