"""Validation of the final model on the full-tape smoothed path (tag): rule violations, offsets, D fit, impact residuals."""
import numpy as np, pandas as pd, sys, json
import prep
tag = sys.argv[1] if len(sys.argv) > 1 else 'v3'
P = json.load(open(prep.common.HERE / sys.argv[2]))
d = prep.build(); r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']; k = d['kind']; p = d['price']; sg = d['sign']
vol = r['vol'].astype(bool); sh = r['shock'].astype(bool)
lad = np.load(prep.common.HERE / '../flow/cache/ladder_member.npy')[:, 0] >= 0
out = {}
for nm, msk in [('calm', ~vol & ~sh), ('vol', vol & ~sh), ('shock_window', sh)]:
    q = P['vol' if nm == 'vol' else 'calm']
    mk = msk & (k <= 3) & ~lad; tk = msk & (k >= 4) & (k <= 5)
    gap = sg[mk] * (m[mk] - p[mk])                      # S - bid, ask - S
    gl = np.where(sg[mk] > 0, r['hi'][mk] - p[mk], p[mk] - r['lo'][mk])  # most favourable S inside 99% CI
    x = sg[tk] * (p[tk] - m[tk])
    xin_lo = sg[tk] * (p[tk] - np.where(sg[tk] > 0, r['hi'][tk], r['lo'][tk]))
    xin_hi = sg[tk] * (p[tk] - np.where(sg[tk] > 0, r['lo'][tk], r['hi'][tk]))
    off = q['off']
    band = (x >= off - 0.5 - 1e-9) & (x <= off + 0.5 + 1e-9)
    band_ci = (xin_hi >= off - 0.5) & (xin_lo <= off + 0.5)
    o = dict(n_makers=int(mk.sum()), maker_gap_below_d0_pointS=int((gap < q['d0'] - 1e-9).sum()),
             maker_gap_below_0_pointS=int((gap < 0).sum()),
             maker_inconsistent_with_99CI=int((gl < q['d0']).sum()),
             maker_gap_min=float(gap.min()) if len(gap) else None,
             n_takers=int(tk.sum()), taker_in_band_pointS=float(band.mean()), taker_in_band_99CI=float(band_ci.mean()),
             taker_passive_side=int((xin_hi < off - 0.5).sum()), taker_aggr_tail=float((xin_lo > off + 0.5).mean()),
             post_sd_median=float(np.median(r['sd'][msk])), post_sd_p90=float(np.percentile(r['sd'][msk], 90)),
             ci99_width_median=float(np.median(r['hi'][msk] - r['lo'][msk])),
             maker_gap_q=[float(x) for x in np.percentile(gap, [0.01, 0.1, 1, 50])] if len(gap) else None,
             n_ladder_excluded=int((msk & (k <= 3) & lad).sum()))
    out[nm] = o
print(json.dumps(out, indent=1))
json.dump(out, open(prep.common.HERE / f'validation_{tag}.json', 'w'), indent=1)
