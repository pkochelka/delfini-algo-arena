import numpy as np, pandas as pd
for n in ['round3', 'round4']:
    d = pd.read_csv(f'data/{n}.csv')
    mk = d[(d.Operation == 'Insert') & (d.Lifespan == 'G')]
    tk = d[(d.Operation == 'Insert') & (d.Lifespan == 'F')]
    sec = np.floor(mk.Time.values).astype(int)
    df = pd.DataFrame({'s': sec, 'side': mk.Side.values, 'p': mk.Price.values})
    bb = df[df.side == 'B'].groupby('s').p.quantile(0.9); aa = df[df.side == 'A'].groupby('s').p.quantile(0.1)
    mid = ((bb + aa) / 2).reindex(range(6000)).ffill().values
    sg = np.where(tk.Side == 'B', 1, -1)
    ts = np.floor(tk.Time.values).astype(int)
    imb = np.bincount(ts, weights=sg, minlength=6000)[:6000]
    cnt = np.bincount(ts, minlength=6000)[:6000]
    print('=====', n)
    for L in [5, 20, 60]:
        I = pd.Series(imb).rolling(L).sum().values
        past = mid - np.roll(mid, L); fut = np.roll(mid, -L) - mid
        ok = np.isfinite(I) & (np.arange(6000) > L) & (np.arange(6000) < 6000 - L)
        print(f'L={L}: corr(imb, past move)={np.corrcoef(I[ok], past[ok])[0,1]:.3f}  corr(imb, fut move)={np.corrcoef(I[ok], fut[ok])[0,1]:.3f}  corr(past move, fut move)={np.corrcoef(past[ok], fut[ok])[0,1]:.3f}  sign-count corr past|move| {np.corrcoef(pd.Series(cnt).rolling(L).sum().values[ok], np.abs(past[ok]))[0,1]:.3f}')
    # taker sign acf
    print('taker sign acf lags1-5', [round(np.corrcoef(sg[:-k], sg[k:])[0, 1], 3) for k in range(1, 6)])
