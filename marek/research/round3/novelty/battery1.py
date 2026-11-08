import numpy as np, pandas as pd
from scipy import stats
def L(n): 
    o=pd.read_parquet(f'cache/{n}_orders.parquet'); c=pd.read_parquet(f'cache/{n}_cancels.parquet'); g=pd.read_parquet(f'cache/{n}_grid.parquet'); tr=pd.read_parquet(f'cache/{n}_trades.parquet'); return o,c,g,tr
res={}
for n in ['round2','round3']:
    o,c,g,tr=L(n); R={}
    m=o[~o.ioc]; k=o[o.ioc]
    R['n_maker']=len(m); R['n_taker']=len(k); R['n_cancel']=len(c)
    R['maker_rate']=len(m)/6000; R['taker_rate']=len(k)/6000
    R['maker_ask_frac']=m.side.mean(); R['taker_ask_frac']=k.side.mean()
    lv=np.log(m.vol); R['mk_logv_mean']=lv.mean(); R['mk_logv_sd']=lv.std()
    lt=np.log(k.vol); R['tk_logv_mean']=lt.mean(); R['tk_logv_sd']=lt.std()
    R['mk_vol_max']=m.vol.max(); R['tk_vol_max']=k.vol.max()
    R['mk_vol1_frac']=(m.vol==1).mean(); R['tk_vol1_frac']=(k.vol==1).mean()
    R['mk_even_frac']=(m.vol%2==0).mean(); R['mk_mult5']=(m.vol%5==0).mean(); R['mk_mult10']=(m.vol%10==0).mean()
    R['tk_mult5']=(k.vol%5==0).mean(); R['tk_mult10']=(k.vol%10==0).mean()
    R['px_mod10_0_mk']=(m.price%10==0).mean(); R['px_mod5_0_mk']=(m.price%5==0).mean()
    R['px_mod10_0_tk']=(k.price%10==0).mean()
    # oid gaps
    d=np.diff(o.oid.to_numpy()); R['oid_gap_not1']=(d!=1).sum(); R['oid_first']=o.oid.iloc[0]
    # end codes
    for code,nm in enumerate(['alive','cancelled','filled','ioc_expired']):
        R['mk_end_'+nm]=(m.end==code).mean()
    R['mk_partial']=((m.filled>0)&(m.filled<m.vol)).mean()
    R['tk_fill_any']=(k.filled>0).mean(); R['tk_full']=(k.filled==k.vol).mean()
    R['mk_crossed']=m.crossed.mean()
    life=(m.end_t-m.t)[m.end==1]; R['life_med']=life.median(); R['life_mean']=life.mean(); R['life_p90']=life.quantile(.9); R['life_p99']=life.quantile(.99)
    # interarrival
    ia=np.diff(m.t.to_numpy()); R['mk_ia_cv']=ia.std()/ia.mean(); R['mk_ia_zero']=(ia==0).mean()
    ia=np.diff(k.t.to_numpy()); R['tk_ia_cv']=ia.std()/ia.mean(); R['tk_ia_zero']=(ia==0).mean()
    # spread, depth
    sp=(g.ask-g.bid).dropna(); R['spread_mean']=sp.mean(); R['spread1']=(sp==1).mean(); R['spread_ge3']=(sp>=3).mean()
    R['touch_qty_med']=np.nanmedian(g.bq); 
    # mid returns
    mid=g.mid.ffill().to_numpy()
    for h in [1,10,100,600]:
        r=mid[h::h]-mid[:-h:h]; r=r[~np.isnan(r)]
        R[f'ret_sd_{h/10}s']=r.std(); R[f'ret_acf1_{h/10}s']=np.corrcoef(r[1:],r[:-1])[0,1]
    R['mid_range']=np.nanmax(mid)-np.nanmin(mid); R['mid_start']=mid[0]; R['mid_end']=mid[-1]
    # taker offset from touch
    s=np.where(k.side==0,1,-1)
    tch=np.where(k.side==0,k.pask,k.pbid); R['tk_lim_minus_touch_mean']=np.nanmean(s*(k.price-tch))
    R['tk_marketable']=np.nanmean(s*(k.price-tch)>=0)
    # maker dist from same-side touch (dist>0 inside)
    R['mk_dist_inside']=(m.dist>0).mean(); R['mk_dist0']=(m.dist==0).mean(); R['mk_dist_mean']=m.dist.mean()
    res[n]=R
df=pd.DataFrame(res); df['ratio']=df.round3/df.round2
pd.set_option('display.width',200); print(df.to_string(float_format=lambda x:f'{x:.4g}'))
