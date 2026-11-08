"""Research only: r3_best (or CALM_SRC) + logging of own orders, fills and 50 ms state samples.
Output: $CALM_LOG (default scratch) as .npz-like pickles at exit."""
import atexit, os, pickle, pathlib
_SRC = os.environ.get('CALM_SRC', str(pathlib.Path(__file__).resolve().parents[1] / 'r3_best.py'))
exec(compile(open(_SRC).read(), _SRC, 'exec'))
_OUT = os.environ.get('CALM_LOG', '/tmp/calm_log.pkl')


class AALog(FVAShock):
    def __init__(self):
        super().__init__()
        self.L_sent = []   # t_us, oid, side(1 buy), price, qty, ioc, F, sd, turb, bb, ba, pos
        self.L_fill = []   # t_us, oid, side, price, qty, F, sd, pos_after
        self.L_seen = {}   # oid -> (t_us, ahead market qty, level qty incl. us, behind)
        self.L_canc = []   # t_us, oid
        self.L_stale = []  # t_us, side(+1 = stale ask, we would buy), touch px, qty through F-+0.25, pos, rest_same_side, turb
        self._lst = {1: None, -1: None}
        self.L_samp = []   # t_us, pos, turb, F, sd, bb, ba, bq_mkt, aq_mkt, our_b_px, our_b_q, our_b_ahead, our_a_px, our_a_q, our_a_ahead
        self._unseen = set()
        self._next_s = 0
        self._bb = self._ba = None
        atexit.register(self._dump)

    def _dump(self):
        with open(_OUT, 'wb') as fh:
            pickle.dump({'sent': self.L_sent, 'fill': self.L_fill, 'seen': self.L_seen, 'canc': self.L_canc,
                         'samp': self.L_samp, 'stale': self.L_stale}, fh)

    def send_order(self, order):
        oid = super().send_order(order)
        f = self.f
        ioc = order.order_type is OrderType.IOC
        self.L_sent.append((self.now_us(), oid, 1 if order.side is BUY else -1, order.price, order.quantity, ioc,
                            f.mean, f.sd, self.turb, self._bb, self._ba, self.pos))
        if not ioc:
            self._unseen.add(oid)
        return oid

    def cancel_order(self, oid):
        self.L_canc.append((self.now_us(), oid))
        super().cancel_order(oid)

    def on_private_order_update(self, event):
        o = self.orders.get(event.order_id)
        if o is not None and event.filled_quantity and event.fill_price is not None:
            f = self.f
            sg = 1 if o[0] is BUY else -1
            self.L_fill.append((event.timestamp_us, event.order_id, sg, event.fill_price, event.filled_quantity,
                                f.mean, f.sd, self.pos + sg * event.filled_quantity))
        super().on_private_order_update(event)

    def on_order_book_change(self, snapshot, trades):
        if snapshot.bids:
            self._bb = snapshot.bids[0].price
        if snapshot.asks:
            self._ba = snapshot.asks[0].price
        if self._unseen:
            for views in (snapshot.bids, snapshot.asks):
                ahead = {}
                for v in views:
                    if v.owner == 'market':
                        ahead[v.price] = ahead.get(v.price, 0) + v.quantity
                    elif v.order_id in self._unseen:
                        tot = sum(w.quantity for w in views if w.price == v.price and w.owner == 'market')
                        self.L_seen[v.order_id] = (snapshot.timestamp_us, ahead.get(v.price, 0), tot)
                        self._unseen.discard(v.order_id)
            self._unseen = {o for o in self._unseen if o in self.orders}
        super().on_order_book_change(snapshot, trades)
        tu = snapshot.timestamp_us
        if self.f is not None and snapshot.bids and snapshot.asks:
            F = self.f.mean
            for sgn, views in ((1, snapshot.asks), (-1, snapshot.bids)):
                q = 0
                for v in views:
                    if (v.price > F - 0.25) if sgn > 0 else (v.price < F + 0.25):
                        break
                    if v.owner == 'market':
                        q += v.quantity
                key = (views[0].price, q) if q else None
                if key != self._lst[sgn]:
                    self._lst[sgn] = key
                    if q:
                        rest = sum(o[2] for o in self.orders.values() if (o[0] is BUY) == (sgn > 0))
                        self.L_stale.append((tu, sgn, views[0].price, q, self.pos, rest, self.turb))
        if tu >= self._next_s and self.f is not None and snapshot.bids and snapshot.asks:
            self._next_s = tu + 50000
            bb, ba = snapshot.bids[0].price, snapshot.asks[0].price
            bq = sum(v.quantity for v in snapshot.bids if v.price == bb and v.owner == 'market')
            aq = sum(v.quantity for v in snapshot.asks if v.price == ba and v.owner == 'market')
            ob = oa = None
            for views, isb in ((snapshot.bids, True), (snapshot.asks, False)):
                ah = 0
                for v in views:
                    if v.owner == 'market':
                        ah += v.quantity
                    else:
                        r = (v.price, v.quantity, ah)
                        if isb and ob is None:
                            ob = r
                        if not isb and oa is None:
                            oa = r
                        break
            self.L_samp.append((tu, self.pos, self.turb, self.f.mean, self.f.sd, bb, ba, bq, aq,
                                *(ob or (None, 0, None)), *(oa or (None, 0, None))))
