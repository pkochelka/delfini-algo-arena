import numpy as np, pandas as pd
m=pd.read_parquet('cache/round3_makers_chains.parquet'); m['life']=m.end_t-m.t
C=m[m.chain].sort_values(['cid','cidx'])
# cancel times within chain
C['eus']=np.rint(C.end_t*1e6)
d=C.groupby('cid').eus.diff().dropna()
print('within-chain end-time diff (us) top:',d.value_counts().head(10).to_dict())
print('end diff quantiles ms',np.percentile(d/1e3,[5,25,50,75,95]).round(2))
# chain mean log size vs time (60s bins)
g=pd.read_parquet('cache/round3_chains.parquet')
g['lv']=np.log(g.vmean); g['min']=(g.t//60).astype(int)
s=g.groupby('min').agg(n=('n','size'),lv=('lv','mean'),lvsd=('lv','std'))
print(s.T.round(2).to_string())
