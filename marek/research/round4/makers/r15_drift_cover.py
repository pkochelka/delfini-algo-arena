"""Do calm-regime S trends happen without the 15-lot agent? Big 5 s S moves in calm, flagged by agent / TWAP / shock."""
import numpy as np, pandas as pd
S = pd.read_parquet('../s_dynamics/s_path.parquet')
st = S.t_us.to_numpy(); sh = S.s_hat.to_numpy(); vol = S.vol.to_numpy(); shock = S.shock.to_numpy()
grid = np.arange(0, 6000e6, 0.5e6).astype(np.int64)
ix = np.clip(np.searchsorted(st, grid, 'right') - 1, 0, len(st) - 1)
sg_ = sh[ix]; vg = vol[ix]; kg = shock[ix]
a = pd.read_pickle('cache/r15_v2.pkl')
g = a.groupby('chain').agg(t0=('t_us', 'min'), t1=('end_t_us', 'max'))
tw = pd.read_pickle('cache/tw.pkl'); tw = tw[tw.tw_L >= 2]
twg = tw.groupby('tw_chain').agg(t0=('t_us', 'min'), t1=('t_us', 'max'))
def cover(iv, pad=0):
    m = np.zeros(len(grid), bool)
    for r in iv.itertuples():
        m[(grid >= r.t0 - pad) & (grid <= r.t1 + pad)] = True
    return m
ag = cover(g); twm = cover(twg)
W = 10  # 5 s
d = np.full(len(grid), np.nan); d[:-W] = sg_[W:] - sg_[:-W]
calm = np.zeros(len(grid), bool); calm[:-W] = ~vg[:-W] & ~vg[W:] & ~kg[:-W] & ~kg[W:]
win_ag = np.zeros(len(grid), bool); win_tw = np.zeros(len(grid), bool)
for i in range(len(grid) - W):
    win_ag[i] = ag[i:i + W + 1].mean() > 0.5; win_tw[i] = twm[i:i + W + 1].any()
m = calm & ~np.isnan(d)
print('calm 5 s windows', m.sum(), 'sd ΔS all', np.nanstd(d[m]).round(2),
      'agent-window sd', np.nanstd(d[m & win_ag]).round(2), 'TWAP-window sd', np.nanstd(d[m & win_tw & ~win_ag]).round(2),
      'neither sd', np.nanstd(d[m & ~win_ag & ~win_tw]).round(2))
for thr in [3, 4, 5, 7, 10]:
    big = m & (np.abs(d) > thr)
    print(f'|ΔS_5s|>{thr}: {big.sum()} windows; agent {(big & win_ag).mean() / big.mean():.2f}, TWAP {(big & win_tw).mean() / big.mean():.2f}, '
          f'neither {(big & ~win_ag & ~win_tw).mean() / big.mean():.2f}; share of calm time agent {win_ag[m].mean():.3f} TWAP {win_tw[m].mean():.3f}')
# lag-1 autocorr of 1 s calm S changes, outside agent+TWAP
d1 = np.diff(sg_[::2]); c1 = (~vg[::2] & ~kg[::2])[1:]
agm = ag[::2][1:]; twm2 = twm[::2][1:]
for lab, mm in [('all calm', c1), ('calm no agent/TWAP', c1 & ~agm & ~twm2), ('calm agent', c1 & agm)]:
    x = d1.copy(); x[~mm] = np.nan
    ok = ~np.isnan(x[:-1]) & ~np.isnan(x[1:])
    print(lab, 'lag-1 acf of 1 s ΔS', np.corrcoef(x[:-1][ok], x[1:][ok])[0, 1].round(3), 'n', ok.sum())
