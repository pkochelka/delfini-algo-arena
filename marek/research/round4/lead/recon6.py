import numpy as np, pandas as pd
for n in ['round3', 'round4']:
    d = pd.read_csv(f'data/{n}.csv')
    mk = d[(d.Operation == 'Insert') & (d.Lifespan == 'G')]
    tk = d[(d.Operation == 'Insert') & (d.Lifespan == 'F')]
    # proxy mid: per 1s, (max bid insert + min ask insert)/2 using median of top-5
    sec = np.floor(mk.Time.values).astype(int)
    df = pd.DataFrame({'s': sec, 'side': mk.Side.values, 'p': mk.Price.values})
    bb = df[df.side == 'B'].groupby('s').p.quantile(0.9); aa = df[df.side == 'A'].groupby('s').p.quantile(0.1)
    mid = ((bb + aa) / 2).reindex(range(6000)).ffill()
    b = np.floor(mk.Time.values / 0.5).astype(int); cnt = np.bincount(b)
    low = np.where(cnt < 50)[0]
    on = [low[0]] + [low[i] for i in range(1, len(low)) if low[i] != low[i - 1] + 1]
    on = np.array(on) * 0.5
    J = [mid[int(o) + 6] - mid[int(o) - 2] for o in on]
    print('=====', n, 'shocks', len(on))
    print('jumps', np.round(J, 1).tolist())
    print('sum jumps', round(sum(J), 1), 'sd', round(np.std(J), 1), 'total move', mid.iloc[-1] - mid.iloc[0])
    dm = mid.diff().fillna(0).values
    mask = np.ones(6000, bool)
    for o in on: mask[int(o) - 2:int(o) + 7] = False
    nm = dm[mask]
    print('non-shock 1s move sd', nm.std().round(3), 'sum', nm.sum().round(1), 'kurt', pd.Series(nm).kurt().round(1))
    for L in [10, 60, 300]:
        x = pd.Series(np.where(mask, dm, 0)).rolling(L).sum().values[L::L]
        print(f'  non-shock {L}s move sd', np.nanstd(x).round(2), ' sqrt-scaling from 1s:', (nm.std() * np.sqrt(L)).round(2))
    # biggest 1s non-shock moves
    big = np.argsort(-np.abs(np.where(mask, dm, 0)))[:12]
    print('  biggest non-shock 1s moves (t,move):', [(int(i), round(dm[i], 1)) for i in sorted(big)])
