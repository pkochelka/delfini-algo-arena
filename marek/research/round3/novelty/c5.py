import numpy as np, pandas as pd
m=pd.read_parquet('cache/round3_makers_chains.parquet'); m['life']=m.end_t-m.t
mid=(m.pbid+m.pask)/2; m['off']=np.where(m.side==0, mid-m.price, m.price-mid)
m['touchoff']=np.where(m.side==0, m.pbid-m.price, m.price-m.pask)
B=5; nb=1200
deep=np.bincount((m.t[m.off>=3.5]//B).astype(int),minlength=nb)[:nb]; on=deep>=150
np.save('cache/round3_on5s.npy',on)
m['on']=on[(m.t//B).astype(int).clip(0,nb-1)]
for flag in [False,True]:
    C=m[m.chain&(m.on==flag)]; T=(on==flag).sum()*B
    g=C.groupby('cid').agg(n=('vol','size'),start=('touchoff','first'))
    print(f'on={flag} chain orders/s={len(C)/T:.2f} chains/s={len(g)/T:.3f} vol q={np.percentile(C.vol,[5,25,50,75,95]).tolist()} life q={np.percentile(C.life,[25,50,75,95]).round(2).tolist()} fill={(C.filled>0).mean():.3f}')
    print('   len dist',(g.n.value_counts(normalize=True).sort_index().round(3).to_dict()),' start',g.start.value_counts(normalize=True).sort_index().round(3).to_dict())
    print('   touchoff dist', C.touchoff.value_counts(normalize=True).sort_index().round(3).to_dict())
    N=m[~m.chain&(m.on==flag)]
    print(f'   nonchain: rate={len(N)/T:.1f} lv={np.log(N.vol).mean():.3f} touchoff dist', N.touchoff.clip(-1,9).value_counts(normalize=True).sort_index().round(3).to_dict())
