import sys, pathlib
src = pathlib.Path(sys.argv[1]).read_text()
out = sys.argv[2]; logp = sys.argv[3]
def rep(a, b, n=1):
    global src
    assert src.count(a) == n, (a, src.count(a))
    src = src.replace(a, b)
# log buffers
rep("BUY, SELL = Side.BUY, Side.SELL\n", "BUY, SELL = Side.BUY, Side.SELL\nimport atexit, pickle\nLOG = {'take': [], 'samp': [], 'send': [], 'fill': []}\natexit.register(lambda: pickle.dump(LOG, open(%r, 'wb')))\n" % logp)
# take buy
rep("""            q = min(q, (p['xcap'] if p['xcap'] and ba <= F - p['xedge'] else cap) - pos - rest_b)
            if q > 0:""", """            LOG['take'].append((self.now_t, 1, ba, lim, q, cap - pos - rest_b, F, sd, turb, pos, f.pv))
            q = min(q, (p['xcap'] if p['xcap'] and ba <= F - p['xedge'] else cap) - pos - rest_b)
            if q > 0:""")
rep("""            q = min(q, (p['xcap'] if p['xcap'] and bb >= F + p['xedge'] else cap) + pos - rest_a)
            if q > 0:""", """            LOG['take'].append((self.now_t, -1, bb, lim, q, cap + pos - rest_a, F, sd, turb, pos, f.pv))
            q = min(q, (p['xcap'] if p['xcap'] and bb >= F + p['xedge'] else cap) + pos - rest_a)
            if q > 0:""")
# sample every 20ms
rep("""        self.turb = turb
        if turb:""", """        self.turb = turb
        if self.now_t >= getattr(self, '_ns', 0.0):
            self._ns = self.now_t + 0.02
            LOG['samp'].append((self.now_t, self.pos, turb, f.pv, F, sd, bb, ba, f.exc, self.sg_t is not None, f.Y))
        if turb:""")
rep("""    def _send(self, side: Side, price: int, qty: int, kind: OrderType) -> None:
        oid = self.send_order(OrderRequest(side=side, price=price, quantity=qty, order_type=kind))""",
"""    def _send(self, side: Side, price: int, qty: int, kind: OrderType) -> None:
        oid = self.send_order(OrderRequest(side=side, price=price, quantity=qty, order_type=kind))
        LOG['send'].append((self.now_t, oid, 1 if side is BUY else -1, price, qty, kind is OrderType.IOC, self.f.mean if self.f else 0, self.pos, self.turb))""")
rep("""        st = event.status
        if st is OrderStatus.PARTIALLY_FILLED or st is OrderStatus.FILLED:""", """        st = event.status
        if st is OrderStatus.PARTIALLY_FILLED or st is OrderStatus.FILLED:
            LOG['fill'].append((self.now_us() * 1e-6, event.order_id, 1 if o[0] is BUY else -1, event.fill_price, event.filled_quantity, self.pos))""")
pathlib.Path(out).write_text(src)
