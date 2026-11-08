"""Research only (ceiling angle): r3_best with (a) instrumentation and (b) an exact-S oracle.

PARAMS (all off = identical to r3_best):
    instr_out  path prefix: at exit write <prefix>_ev.npz (per public event: t_us, filter F, sd, pv, turb, pos)
               and <prefix>_ord.csv (per order sent: id, t_us, label = calling function:line, side, price, qty, type)
    or_on      1: every trading decision (quotes, takes, sniper, unwind) uses the exact S of
               s_dynamics/s_path.parquet (data/.cache/round3_S_oracle.npz, built by marek/research/round3/ideas/ceiling/ceiling_mk_oracle.py) instead of
               the filter mean F. The filter itself still runs (its regime / jump state is kept).
    or_sd      None: keep the filter sd in the thresholds; a number: use this sd instead
    or_pv      1: also replace P(volatile) by the exact regime label (0/1)
Only valid on data/round3.csv (the oracle is that tape's S).
"""
from __future__ import annotations

import atexit
import importlib.util
import sys
from array import array
from pathlib import Path

import numpy as np

ROOT = Path('/Users/marek/Documents/delfini-algo-arena')
_spec = importlib.util.spec_from_file_location('r3_best_base', ROOT / 'strategies/marek/round3/r3_best.py')
r3 = importlib.util.module_from_spec(_spec)
sys.modules['r3_best_base'] = r3
_spec.loader.exec_module(r3)

PARAMS = r3.PARAMS
PARAMS.update({'instr_out': None, 'or_on': 0, 'or_sd': None, 'or_pv': 0})


class AOracle(r3.FVAShock):
    def __init__(self) -> None:
        super().__init__()
        p = self.p
        self._fm = None
        if p['or_on'] or p['or_pv']:
            z = np.load(ROOT / 'data/.cache/round3_S_oracle.npz')
            self._ot = z['t_us'].tolist()
            self._os = z['S'].tolist()
            self._ov = z['vol'].tolist()
            self._oi = 0
        self._instr = p['instr_out'] is not None
        if self._instr:
            self.ev_t = array('q')
            self.ev = {k: array('d') for k in ('F', 'sd', 'pv', 'turb', 'pos', 'exc')}
            self.ords = []
            atexit.register(self._dump)

    def _dump(self) -> None:
        pre = self.p['instr_out']
        np.savez(pre + '_ev.npz', t_us=np.frombuffer(self.ev_t, dtype=np.int64),
                 **{k: np.frombuffer(v, dtype=np.float64) for k, v in self.ev.items()})
        with open(pre + '_ord.csv', 'w') as fh:
            fh.write('oid,t_us,label,side,price,qty,kind,F,pos\n')
            for r in self.ords:
                fh.write(','.join(str(x) for x in r) + '\n')

    def send_order(self, order):
        oid = super().send_order(order)
        if self._instr:
            fr = sys._getframe(1)
            if fr.f_code.co_name == '_send':
                fr = fr.f_back
            f = self.f
            self.ords.append((oid, self.now_us(), f'{fr.f_code.co_name}:{fr.f_lineno}', order.side.value,
                              order.price, order.quantity, order.order_type.value,
                              round(f.mean, 3) if f is not None else '', self.pos))
        return oid

    def _oracle(self, t_us: int):
        t, i = self._ot, self._oi
        n = len(t)
        while i + 1 < n and t[i + 1] <= t_us:
            i += 1
        self._oi = i
        return self._os[i], self._ov[i]

    def on_order_book_change(self, snapshot, trades) -> None:
        f = self.f
        if f is not None and self._fm is not None:
            f.mean, f.sd, f.pv = self._fm
            self._fm = None
        super().on_order_book_change(snapshot, trades)
        if self._instr and self.f is not None:
            f = self.f
            self.ev_t.append(snapshot.timestamp_us)
            e = self.ev
            e['F'].append(f.mean)
            e['sd'].append(f.sd)
            e['pv'].append(f.pv)
            e['turb'].append(float(self.turb))
            e['pos'].append(float(self.pos))
            e['exc'].append(f.exc)

    def _quote(self, f, bb, ba, bids, asks) -> None:
        p = self.p
        if (p['or_on'] or p['or_pv']) and self._fm is None:
            self._fm = (f.mean, f.sd, f.pv)
            S, v = self._oracle(self.now_us())
            if p['or_on']:
                f.mean = S
                if p['or_sd'] is not None:
                    f.sd = p['or_sd']
            if p['or_pv']:
                f.pv = 0.9999 if v else 1e-4
        super()._quote(f, bb, ba, bids, asks)
