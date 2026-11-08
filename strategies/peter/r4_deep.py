"""Round-4 deep dive: momentum by split/regime, 2000 s cycle, taker informativeness. Compare tapes given as args."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis import load_tape

def run(csv, folds):
    t = load_tape(csv, verbose=False); ev, o = t.events, t.orders
    T = ev.t_us.to_numpy() / 1e6
    bid, ask = ev.bid.to_numpy().astype(float), ev.ask.to_numpy().astype(float)
    two = (bid > 0) & (ask > 0)
    MID = pd.Series(np.where(two, (bid + ask) / 2, np.nan)).ffill().bfill().to_numpy()
    dur = T[-1]; g = np.arange(0, dur, 0.1)
    m = MID[np.maximum(np.searchsorted(T, g, 'right') - 1, 0)]
    r1 = m[10::10] - m[:-10:10]                        # 1 s returns
    sec = np.arange(len(r1))
    rv10 = pd.Series(np.abs(np.diff(m))).rolling(100).sum().to_numpy()[9::10][:len(r1)]
    turb = np.nan_to_num(rv10) > 15
    f = o[o.ioc]; ft = f.t_us.to_numpy() / 1e6; sg = np.where(f.side == 'B', 1, -1); fq = f.filled.to_numpy()
    def mid_at(x): return MID[np.maximum(np.searchsorted(T, x, 'right') - 1, 0)]
    m0 = mid_at(ft + 0.005)
    out = {'tape': csv}
    for H in (1, 5, 30):
        out[f'taker_mk{H}'] = float(np.mean(sg * (mid_at(ft + H) - m0)))
    big = fq >= np.quantile(fq[fq > 0], 0.9)
    out['taker_mk5_big'] = float(np.mean((sg * (mid_at(ft + 5) - m0))[big]))
    print(f"\n######## {csv}")
    print({k: round(v, 3) for k, v in out.items() if k != 'tape'})
    # per split
    fj = json.load(open(folds)); wins = [(x['name'], x['eval_start_s'], x['end_s']) for x in fj['folds']]
    h0 = fj['held_out_from_s']; hb = np.linspace(h0, dur, 5)
    wins += [(f'hold{i+1}', hb[i], hb[i + 1]) for i in range(4)]
    rows = []
    for nm, a, b in wins:
        s = (sec >= a) & (sec < b - 1)
        x = r1[s]
        def acf(v, l=1): return float(np.corrcoef(v[:-l], v[l:])[0, 1])
        r10 = m[int(a*10)+100:int(b*10):100] - m[int(a*10):int(b*10)-100:100]
        vr10 = np.var(r10) / (10 * np.var(x))
        c = s & ~turb; tb = s & turb
        xc = r1[c]; xt = r1[tb]
        k = (ft >= a) & (ft < b)
        mk = sg[k] * (mid_at(ft[k] + 5) - m0[k])
        ts = sg[k]; tacf = float(np.corrcoef(ts[:-1], ts[1:])[0, 1])
        rows.append(dict(split=nm, acf1=acf(x), acf1_calm=acf(xc) if len(xc) > 20 else np.nan,
                         acf1_turb=acf(xt) if len(xt) > 20 else np.nan, vr10=vr10, turb=tb.sum() / s.sum(),
                         taker_mk5=mk.mean(), taker_sign_acf=tacf, drift=m[min(int(b*10), len(m)-1)] - m[int(a*10)]))
    df = pd.DataFrame(rows).set_index('split'); print(df.round(3).to_string())
    print('acf1>0 splits:', (df.acf1 > 0.05).sum(), '/', len(df), '  vr10>1.2:', (df.vr10 > 1.2).sum())
    print('corr(turb, acf1)=%.2f corr(turb, taker_mk5)=%.2f' % (df.turb.corr(df.acf1), df.turb.corr(df.taker_mk5)))
    # calm vs turbulent overall
    for lab, msk in (('calm', ~turb), ('turb', turb)):
        x = r1[msk]; print(f'{lab}: share {msk.mean():.2f}  acf1 {np.corrcoef(x[:-1], x[1:])[0,1]:.3f}  sd {x.std():.2f}')
    # momentum after big 1 s moves (|r|>10): continuation next 1/5/10 s
    idx = np.flatnonzero(np.abs(r1) > 10); idx = idx[idx < len(r1) - 11]
    for H in (1, 5, 10):
        cont = np.sign(r1[idx]) * (m[(idx + 1 + H) * 10] - m[(idx + 1) * 10])
        print(f'after |1s move|>10 (n={len(idx)}): signed next {H}s {cont.mean():+.2f}')
    # 2000 s / 1500 s cycle: maker inserts per 100 s and turbulence per 100 s
    gm = o[~o.ioc]; mt = gm.t_us.to_numpy() / 1e6
    cnt = np.bincount((mt // 100).astype(int), minlength=60)[:60]
    tb100 = np.array([turb[i*100:(i+1)*100].mean() for i in range(60)])
    jb100 = np.array([np.abs(r1[i*100:(i+1)*100]).sum() for i in range(60)])
    print('maker ins /100s:', ' '.join(f'{c//1000}' for c in cnt))
    print('turb share/100s:', ' '.join(f'{int(v*9)}' for v in tb100))
    for P in (1500, 2000, 3000):
        ph = (np.arange(60) * 100 + 50) % P / P
        cs, sn = np.cos(2*np.pi*ph), np.sin(2*np.pi*ph)
        X = np.c_[np.ones(60), cs, sn]
        for nm, y in (('maker', cnt), ('turb', tb100)):
            b, res, *_ = np.linalg.lstsq(X, y, rcond=None); r2 = 1 - res[0] / ((y - y.mean())**2).sum()
            print(f'  period {P}: {nm} sinusoid R2 {r2:.2f} amp {np.hypot(b[1], b[2])/b[0]*100:.1f}%')
    print('corr(maker rate, turb) per 100 s: %.2f' % np.corrcoef(cnt, tb100)[0, 1])

for csv, fo in zip(sys.argv[1::2], sys.argv[2::2]):
    run(csv, fo)
