import numpy as np, pandas as pd
for n in ['round2','round3']:
    o=pd.read_parquet(f'cache/{n}_orders.parquet'); g=pd.read_parquet(f'cache/{n}_grid.parquet')
    m=o[~o.ioc].copy(); k=o[o.ioc]
    mid=(m.pbid+m.pask)/2
    m['off']=np.where(m.side==0, mid-m.price, m.price-mid); m['life']=m.end_t-m.t
    B=5; nb=6000//B
    deep=np.bincount((m.t[m.off>=3.5]//B).astype(int),minlength=nb)[:nb]
    on=deep>=150
    # episodes
    d=np.diff(np.r_[0,on.astype(int),0]); st=np.where(d==1)[0]; en=np.where(d==-1)[0]
    print(n,'on frac',on.mean(),'episodes',len(st)); 
    if n=='round3': print([(s*B,e*B) for s,e in zip(st,en)])
    mb=(m.t//B).astype(int).clip(0,nb-1); kb=(k.t//B).astype(int).clip(0,nb-1)
    m['on']=on[mb]; k=k.assign(on=on[kb])
    gb=(g.t//B).astype(int).clip(0,nb-1); g['on']=on[gb]
    mid=g.mid.ffill().to_numpy(); r=np.r_[np.nan,np.diff(mid[::10])]  # 1s returns
    gon=on[(np.arange(0,6000,1)//B)]
    for flag in [False,True]:
        mm=m[m.on==flag]; kk=k[k.on==flag]; T=(on==flag).sum()*B
        sel=gon==flag
        print(f' on={flag} T={T} mk_rate={len(mm)/T:.1f} tk_rate={len(kk)/T:.2f} mk_lv={np.log(mm.vol).mean():.3f} tk_lv={np.log(kk.vol).mean():.3f} off_mean={mm.off.mean():.3f} off<=0.5:{(mm.off<=0.5).mean():.3f} life_mean={mm.life.mean():.3f} ret1s_sd={np.nanstd(r[sel]):.3f} spread={(g.ask-g.bid)[g.on==flag].mean():.3f} touchq={np.nanmedian(g.bq[g.on==flag])}')
