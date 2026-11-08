"""Research only: r3_best (or CALM_SRC) quoting/taking on the exact S (s_dynamics/s_path.parquet s_hat, smoothed:
uses the future) while the filter is not turbulent. Upper bound of what a better S estimate is worth in calm."""
import os, pathlib
_SRC = os.environ.get('CALM_SRC', str(pathlib.Path(__file__).resolve().parents[1] / 'r3_best.py'))
exec(compile(open(_SRC).read(), _SRC, 'exec'))
_a = np.load(os.environ['CALM_SHAT'])  # np.c_[t_us, s_hat] exported from s_path.parquet
_ST, _SS = _a[:, 0], _a[:, 1]
_SD = float(os.environ.get('CALM_ORACLE_SD', '0.03'))


class AAOracle(FVAShock):
    def _quote(self, f, bb, ba, bids, asks):
        if f.pv > 0.5 or f.exc > 0.2 or self.sg_t is not None:
            return super()._quote(f, bb, ba, bids, asks)
        i = int(np.searchsorted(_ST, self.now_us(), side='right')) - 1
        m, s = f.mean, f.sd
        f.mean, f.sd = float(_SS[max(i, 0)]) + self.p['y_k'] * f.Y, _SD
        try:
            super()._quote(f, bb, ba, bids, asks)
        finally:
            f.mean, f.sd = m, s
