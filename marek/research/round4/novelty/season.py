import numpy as np, pandas as pd
from scipy import stats
from common import *
for n in ['round3','round4']:
    o=L(n,'orders'); v=regime_1s(n)
    if True:
        ch=pd.read_parquet(f'cache/{n}_makers_chains.parquet')[['oid','chain']]; o=o.merge(ch,on='oid',how='left'); o['chain']=o.chain.fillna(False)
    else: o['chain']=False
    sec=o.t.astype(int).clip(0,5999); calm=~v[sec]
    out=[]
    for nm,sel in [('taker',o.ioc),('maker_reg',(~o.ioc)&(~o.chain)),('taker_lv',o.ioc)]:
        s=o[sel&calm]
        b=(s.t//300).astype(int)
        if nm=='taker_lv': y=s.groupby(b).vol.apply(lambda x:np.log(x).mean())
        else:
            T=pd.Series(~v[:6000]).groupby(np.arange(6000)//300).sum(); y=s.groupby(b).size()/T
        r=stats.linregress(y.index,y.values)
        out.append(f'{nm}: slope/5min={r.slope:+.4f} p={r.pvalue:.3f} first={y.iloc[:4].mean():.2f} last={y.iloc[-4:].mean():.2f}')
    print(n,'(calm only, 5-min bins) ',' | '.join(out))
    # hour-of-tape: volatile occupancy by 1000s
    print('   vol occupancy per 1000s', [round(v[i:i+1000].mean(),2) for i in range(0,6000,1000)])
