"""Inventory-safe market maker with an online market-condition-dependent spread.

One standalone submission. Parameters are updated only using past public/private feeds.
The spread combines market sweep depth, sampled volatility and optional delayed trade markouts.
The current defaults were selected on chronological development folds; toxicity is disabled.
It is an A-S-inspired heuristic, not an exact queue-aware optimal-control solution.
"""
from collections import deque
from math import ceil, exp, floor, log1p, sqrt

from simulator.models import OrderRequest, OrderStatus, OrderType, Side
from simulator.strategy import BaseStrategy


PARAMS = {
    'spread_mult': 2.0,
    'vol_halflife_s': 30.0,
    'tox_weight': 0.0,
    'inventory_skew': 2.0,
    'risk_aversion': 0.03,
    'risk_horizon_s': 5.0,
    'latency_weight': 1.0,
    'min_half_spread': 0.5,
    'max_half_spread': 6.0,
    'size': 10,
    'position_limit': 80,
    'refresh_us': 50_000,
    'requote_ticks': 1,
    'sample_us': 100_000,
    'flow_halflife_trades': 100.0,
    'markout_us': 1_000_000,
    'warmup_us': 60_000_000,
    'trade_start_us': 0,
    'adaptive': True,
    'fixed_half_spread': 0.5,
}


class AdaptiveSpreadStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.p = dict(PARAMS)
        self.inventory = 0
        self.orders = {}
        self.start = None
        self.last_refresh = -10**18
        self.last_sample = None
        self.sample_mid = None
        self.variance = 1.0
        self.mean_depth = 0.1
        self.toxicity = 0.2
        self.previous_bid = None
        self.previous_ask = None
        self.previous_mid = None
        self.pending_markouts = deque()
        self.last_half_spread = 0.5

    def on_private_order_update(self, event):
        order = self.orders.get(event.order_id)
        if order is None:
            return
        if event.status == OrderStatus.ACCEPTED:
            order['accepted'] = True
        elif event.status in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED):
            sign = 1 if order['side'] == Side.BUY else -1
            self.inventory += sign * event.filled_quantity
            order['remaining'] = event.remaining_quantity
            if event.status == OrderStatus.FILLED:
                del self.orders[event.order_id]
        elif event.status == OrderStatus.CANCELED:
            del self.orders[event.order_id]
        elif event.status == OrderStatus.REJECTED:
            if order['accepted']:
                order['canceling'] = False
            else:
                del self.orders[event.order_id]

    def cancel(self, oid, order):
        if order['accepted'] and not order['canceling']:
            order['canceling'] = True
            self.cancel_order(oid)

    def on_order_book_change(self, snapshot, trades):
        p = self.p
        now = snapshot.timestamp_us
        if self.start is None:
            self.start = now
        bid = next((o.price for o in snapshot.bids if o.owner == 'market'), None)
        ask = next((o.price for o in snapshot.asks if o.owner == 'market'), None)
        if bid is None or ask is None or bid >= ask:
            for oid, order in list(self.orders.items()):
                self.cancel(oid, order)
            self.previous_bid = None
            self.previous_ask = None
            self.previous_mid = None
            return
        mid = (bid + ask) / 2
        if self.last_sample is None:
            self.last_sample, self.sample_mid = now, mid
        elif now - self.last_sample >= p['sample_us']:
            dt = (now - self.last_sample) / 1_000_000
            weight = 1 - exp(-0.69314718056 * dt / p['vol_halflife_s'])
            self.variance += weight * ((mid-self.sample_mid)**2 / dt-self.variance)
            self.last_sample, self.sample_mid = now, mid

        flow_weight = 1 - exp(-0.69314718056 / p['flow_halflife_trades'])
        while self.pending_markouts and self.pending_markouts[0][0] <= now:
            due, sign, reference = self.pending_markouts.popleft()
            impact = max(-8.0, min(8.0, sign * (mid-reference)))
            self.toxicity += flow_weight * (impact-self.toxicity)
        if self.previous_bid is not None:
            groups = {}
            for trade in trades:
                if trade.taker_owner != 'market':
                    continue
                sign = 1 if trade.book_side == 'ask' else -1
                depth = (trade.price-self.previous_ask if sign == 1
                         else self.previous_bid-trade.price)
                old = groups.get(trade.taker_order_id)
                if old is None or depth > old[0]:
                    groups[trade.taker_order_id] = (max(0, depth), sign)
            for depth, sign in groups.values():
                self.mean_depth += flow_weight * (depth-self.mean_depth)
                self.pending_markouts.append((now+p['markout_us'], sign, self.previous_mid))
        self.previous_bid, self.previous_ask, self.previous_mid = bid, ask, mid
        if now < max(self.start+p['warmup_us'], p['trade_start_us']):
            return
        if now-self.last_refresh < p['refresh_us']:
            return
        self.last_refresh = now
        if p['adaptive']:
            k = min(10.0, max(0.15, log1p(1/max(self.mean_depth, 0.001))))
            gamma = p['risk_aversion']
            liquidity = log1p(gamma/k)/gamma
            inventory_risk = 0.5 * gamma * self.variance * p['risk_horizon_s']
            latency_risk = p['latency_weight'] * sqrt(max(0,self.variance)*.005)
            adverse_selection = p['tox_weight'] * max(0,self.toxicity)
            half = p['spread_mult'] * (liquidity+inventory_risk+latency_risk+adverse_selection)
            half = max(p['min_half_spread'], min(p['max_half_spread'], half))
        else:
            half = p['fixed_half_spread']
        self.last_half_spread = half
        skew = p['inventory_skew'] * self.inventory/p['position_limit']
        center = mid-skew
        # Nearest-tick rounding; allow inside-spread improvement but never intentional crossing.
        buy = max(1, min(ask-1, floor(center-half+.5)))
        sell = max(bid+1, ceil(center+half-.5))
        self.quote(Side.BUY, buy)
        self.quote(Side.SELL, sell)

    def quote(self, side, price):
        p = self.p
        reserved = sum(o['remaining'] for o in self.orders.values() if o['side'] == side)
        capacity = (p['position_limit']-self.inventory-reserved if side == Side.BUY
                    else p['position_limit']+self.inventory-reserved)
        for oid, order in list(self.orders.items()):
            if order['side'] == side:
                if abs(order['price']-price) >= p['requote_ticks'] or capacity+order['remaining'] <= 0:
                    self.cancel(oid, order)
                return  # Includes unaccepted orders and cancellations still in flight.
        quantity = min(p['size'], capacity)
        if quantity <= 0:
            return
        oid = self.send_order(OrderRequest(side=side, price=price, quantity=quantity, order_type=OrderType.LIMIT))
        self.orders[oid] = dict(side=side,price=price,remaining=quantity,accepted=False,canceling=False)
