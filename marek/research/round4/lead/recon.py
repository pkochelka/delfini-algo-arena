"""Lead recon: round 4 vs round 3 headline stats."""
import sys
import numpy as np, pandas as pd

def stats(path):
    d = pd.read_csv(path)
    out = {}
    ins = d[d.Operation == 'Insert']; can = d[d.Operation == 'Cancel']
    out['rows'] = len(d); out['ops'] = d.Operation.value_counts().to_dict()
    out['lifespan'] = ins.Lifespan.value_counts().to_dict()
    T = d.Time.max(); out['T'] = T
    tk = ins[ins.Lifespan != 'G']; mk = ins[ins.Lifespan == 'G']
    out['taker_rate'] = len(tk) / T; out['maker_rate'] = len(mk) / T; out['cancel_rate'] = len(can) / T
    out['taker_side'] = tk.Side.value_counts(normalize=True).round(3).to_dict()
    out['taker_logsz'] = (np.log(tk.Volume).mean().round(3), np.log(tk.Volume).std().round(3))
    out['maker_logsz'] = (np.log(mk.Volume).mean().round(3), np.log(mk.Volume).std().round(3))
    out['maker_sz_q'] = mk.Volume.quantile([.1, .5, .9, .99]).tolist()
    out['price_range'] = (mk.Price.min(), mk.Price.max())
    # exact dt between consecutive maker inserts (us)
    t_us = np.round(mk.Time.values * 1e6).astype(np.int64)
    dt = np.diff(t_us)
    vc = pd.Series(dt[(dt > 0) & (dt <= 20000)]).value_counts()
    bg = vc.median()
    out['exact_dt_top'] = vc.head(12).to_dict(); out['dt_bg_median'] = bg
    # all-op exact dt
    ta = np.round(d.Time.values * 1e6).astype(np.int64); dta = np.diff(ta)
    out['all_dt_top'] = pd.Series(dta[(dta > 0) & (dta <= 50000)]).value_counts().head(8).to_dict()
    # droughts
    b = np.floor(mk.Time.values / 0.5).astype(int); cnt = np.bincount(b)
    low = np.where(cnt < 50)[0]
    on = [low[0]] + [low[i] for i in range(1, len(low)) if low[i] != low[i - 1] + 1] if len(low) else []
    out['droughts'] = [x * 0.5 for x in on]
    # per-minute
    mm = np.bincount(np.floor(mk.Time.values / 60).astype(int)); tm = np.bincount(np.floor(tk.Time.values / 60).astype(int))
    out['maker_per_min_sd'] = mm.std().round(1); out['taker_per_min_sd'] = tm.std().round(1)
    out['taker_vol_q'] = tk.Volume.quantile([.5, .9, .99]).tolist()
    # cancel delay distribution (life of cancelled makers)
    ti = mk.set_index('OrderId').Time
    life = (can.set_index('OrderId').Time - ti.reindex(can.OrderId).values).dropna()
    out['cancel_life_q'] = life.quantile([.1, .5, .9, .99]).round(3).tolist()
    out['frac_makers_cancelled'] = round(len(can) / len(mk), 4)
    out['orderid_gaps'] = int((np.diff(ins.OrderId.values) != 1).sum())
    return out

for p in sys.argv[1:]:
    print('=====', p)
    for k, v in stats(p).items():
        print(f'{k}: {v}')
