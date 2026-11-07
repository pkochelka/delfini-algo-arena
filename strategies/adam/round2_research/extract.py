"""Causal public-book/print features and purged forward targets for Round 2."""
import gc
import hashlib
import json
import sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
REPO=next(p for p in HERE.parents if (p/'pyproject.toml').exists())
sys.path.insert(0,str(REPO))
from analysis import load_tape
from analysis import features as F
from analysis import execution as E

GEN=HERE/'generated'
GEN.mkdir(exist_ok=True)
OUT=HERE/'results'
OUT.mkdir(exist_ok=True)
SOURCE=REPO/'data'/'round2_public.csv'
DT=.1
HORIZONS=[.1,1.,5.]

def dump(name,data):
    def clean(v):
        if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [clean(x) for x in v]
        if isinstance(v,np.generic):v=v.item()
        if isinstance(v,float) and not np.isfinite(v):return None
        return v
    (OUT/name).write_text(json.dumps(clean(data),indent=2),encoding='utf-8')

def quote(tape,times,col,strict=False):
    q=tape.quotes
    ix=np.searchsorted(q.t_us.to_numpy(),times,side='left' if strict else 'right')-1
    out=q[col].to_numpy(dtype=float)[np.maximum(0,ix)].copy()
    valid=(ix>=0)&(times<=tape.end_us)
    if col in ('mid','bid','ask','spread','imbalance','microprice'):
        valid&=q.mid.notna().to_numpy()[np.maximum(ix,0)]
    out[~valid]=np.nan
    return out

def trailing(a,n):
    c=np.r_[0,np.cumsum(a)]
    return c[1:]-c[np.maximum(0,np.arange(len(a))+1-n)]

def main():
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),data=str(SOURCE),
        data_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),grid_s=.1,latency_s=.005,
        horizons_s=HORIZONS,train_s=[60,3590],validation_s=[3600,4790],test_s=[4800,5990],
        purge_s=10,models=['training mean','microprice residual','ridge(alpha=10)','HGB leaves15','HGB leaves31'],
        hgb=dict(max_iter=150,learning_rate=.05,min_samples_leaf=100,l2_regularization=10,early_stopping=False,seed=1729),
        selection='For each horizon select fitted model by validation MSE; freeze before test. No trading strategy is selected by this analysis.',
        fill_models='Historical passive-order any-fill within 1s, conditional first-fill 1s markout; ridge/logistic vs HGB; train-only sampling and chronological evaluation.',
        observability='Predictors use book within 20 ticks and executed public prints. Raw submitted IOC sizes and off-screen insert/cancel messages are diagnostics only, never predictors.',
        limitations='Public dataset already opened descriptively. Chronological fitting holdout, not organizer-private blind data. Historical maker fill cohorts are policy-dependent, not counterfactual fills.')
    dump('protocol.json',protocol)
    tape=load_tape(SOURCE)
    print('LOADED',len(tape.events),len(tape.orders),len(tape.trades),flush=True)
    start=tape.start_us
    times=np.arange(start+50_000,tape.end_us-5_000_000,100_000,dtype=np.int64)
    seconds=(times-start)/1e6
    mid=quote(tape,times,'mid')
    bid=quote(tape,times,'bid'); ask=quote(tape,times,'ask')
    # Only past values are carried through transient one-sided gaps; current validity is retained.
    filled=pd.Series(mid).ffill().to_numpy()
    X={}
    bq=quote(tape,times,'bid_qty'); aq=quote(tape,times,'ask_qty')
    X['spread']=ask-bid
    X['imb_touch']=(bq-aq)/np.maximum(1,bq+aq)
    X['micro_offset']=quote(tape,times,'microprice')-mid
    X['log_bid_touch']=np.log1p(bq);X['log_ask_touch']=np.log1p(aq)
    X['touch_order_imb']=(quote(tape,times,'bid_n')-quote(tape,times,'ask_n'))/np.maximum(1,quote(tape,times,'bid_n')+quote(tape,times,'ask_n'))
    di=np.maximum(0,np.searchsorted(tape.depth_t_us,times,side='right')-1)
    for k in (3,5,10,20):
        vals=[]
        for px,qty in ((tape.depth_bid_px,tape.depth_bid_qty),(tape.depth_ask_px,tape.depth_ask_qty)):
            p=px[di];v=qty[di]
            val=np.where((p>0)&(abs(p-p[:,0,None])<k),v,0).sum(axis=1)
            vals.append(val)
        b,a=vals
        X[f'imb_depth{k}']=(b-a)/np.maximum(1,b+a)
        X[f'log_depth{k}']=np.log1p(b+a)
    for h in (.1,.5,1,5,30,60):
        n=round(h/DT)
        lag=np.r_[np.full(n,np.nan),filled[:-n]]
        X[f'return_{h:g}s']=filled-lag
    r=np.r_[0,np.diff(filled)]
    for h in (1,5,30,60):
        n=round(h/DT)
        X[f'rms_{h}s']=np.sqrt(np.maximum(0,trailing(np.nan_to_num(r)**2,n)/h))
        X[f'moves_{h}s']=trailing((np.nan_to_num(r)!=0).astype(float),n)/h
    # Flow windows from event-level best-quote OFI; all components observable from public snapshots.
    q=tape.quotes
    qt=q.t_us.to_numpy()
    b=q.bid.fillna(0).to_numpy();a=q.ask.fillna(0).to_numpy()
    bqty=q.bid_qty.to_numpy(dtype=float);aqty=q.ask_qty.to_numpy(dtype=float)
    good=(b>0)&(a>b)
    ofi=np.zeros(len(q));changes=np.zeros(len(q))
    ofi[1:]=(np.where(b[1:]>=b[:-1],bqty[1:],0)-np.where(b[1:]<=b[:-1],bqty[:-1],0)
             -np.where(a[1:]<=a[:-1],aqty[1:],0)+np.where(a[1:]>=a[:-1],aqty[:-1],0))
    ofi[1:]*=good[1:]&good[:-1]
    changes[1:]=((b[1:]!=b[:-1])|(a[1:]!=a[:-1])|(bqty[1:]!=bqty[:-1])|(aqty[1:]!=aqty[:-1]))&(good[1:]&good[:-1])
    def aggregate(ts,weights,h):
        c=np.r_[0,np.cumsum(weights,dtype=float)]
        return c[np.searchsorted(ts,times,side='right')]-c[np.searchsorted(ts,times-round(h*1e6),side='right')]
    tr=tape.trades
    tt=tr.t_us.to_numpy();vol=tr.qty.to_numpy(dtype=float)
    sg=np.where(tr.aggressor.to_numpy()=='B',1.,-1.)
    for h in (.1,.5,1,5,30):
        flow=aggregate(tt,sg*vol,h);total=aggregate(tt,vol,h)
        X[f'trade_imb_{h:g}s']=flow/np.maximum(1,total)
        X[f'log_tradevol_{h:g}s']=np.log1p(total)
        X[f'ofi_{h:g}s']=aggregate(qt,ofi,h)/np.maximum(1,bq+aq)
        X[f'book_updates_{h:g}s']=np.log1p(aggregate(qt,changes,h))
    # No time/absolute price/order IDs as predictors: avoid learning this particular trajectory.
    names=list(X)
    matrix=np.column_stack([X[k] for k in names]).astype(np.float32)
    valid=np.isfinite(matrix).all(axis=1)&np.isfinite(mid)&(seconds>=60)
    ys=[]
    for h in HORIZONS:
        ys.append(quote(tape,times+5000+round(h*1e6),'mid')-quote(tape,times+5000,'mid'))
    Y=np.column_stack(ys)
    valid&=np.isfinite(Y).all(axis=1)
    np.savez_compressed(GEN/'features.npz',X=matrix,t_us=times,seconds=seconds,Y=Y,
        valid=valid,mid=mid,bid=bid,ask=ask,entry_bid=quote(tape,times+5000,'bid'),entry_ask=quote(tape,times+5000,'ask'),
        exit_bid=np.column_stack([quote(tape,times+5000+round(h*1e6),'bid') for h in HORIZONS]),
        exit_ask=np.column_stack([quote(tape,times+5000+round(h*1e6),'ask') for h in HORIZONS]))
    dump('feature_schema.json',dict(names=names,features=len(names),grid_rows=len(times),valid_rows=int(valid.sum())))
    print('FEATURES',len(times),len(names),int(valid.sum()),flush=True)
    # Historical passive-order outcomes, predictors strictly before the order event.
    o=tape.orders
    passive=o.loc[(~o.ioc)&(~o.crossed)&(o.queue_ahead>=0)&(o.dist>=-3)&(o.dist<=1)].copy()
    # Deterministic subsample controls cost without selecting on outcomes.
    passive=passive.iloc[::5].copy()
    ot=passive.t_us.to_numpy()
    ix=np.searchsorted(times,ot,side='left')-1
    keep=(ix>=0)&(ot+2_000_000<=tape.end_us)
    passive=passive.loc[keep].copy();ix=ix[keep];ot=ot[keep]
    side=np.where(passive.side.to_numpy()=='B',1.,-1.)
    orderX=matrix[ix].copy()
    ordernames=names+['maker_side','log_order_size','log_queue_ahead','placement_ticks']
    orderX=np.column_stack([orderX,side,np.log1p(passive.volume),np.log1p(passive.queue_ahead),passive.dist]).astype(np.float32)
    first=tr.groupby('maker_row',sort=False).t_us.min()
    ft=first.reindex(passive.row).to_numpy()
    yf=(ft<=ot+1_000_000).astype(float)
    futuremid=quote(tape,np.nan_to_num(ft,nan=tape.end_us+1).astype(np.int64)+1_000_000,'mid')
    ym=side*(futuremid-passive.price.to_numpy())
    ym[(yf==0)|~np.isfinite(ft)]=np.nan
    ov=np.isfinite(orderX).all(axis=1)&valid[ix]
    np.savez_compressed(GEN/'order_features.npz',X=orderX,seconds=(ot-start)/1e6,
        any_fill=yf,markout=ym,valid=ov,side=side,volume=passive.volume.to_numpy(),queue=passive.queue_ahead.to_numpy())
    dump('order_schema.json',dict(names=ordernames,rows=len(passive),valid_rows=int(ov.sum()),
        label='Any historical first fill within 1s, observed competing cancellation policy. Conditional markout 1s after that first fill. Not counterfactual strategy fill probability.'))
    print('ORDER FEATURES',len(passive),flush=True)
    # Exploratory diagnostics use only the fitting period; validation/test get descriptive summaries later.
    window=(60.,3590.)
    diagnostics=dict(summary=F.summary(tape,window),vol_signature=F.vol_signature(tape,window).to_dict('records'),
        return_acf=F.acf(F.returns(tape,.1,window),50).tolist(),
        sign_acf=F.trade_sign_acf(tape,30,window).tolist(),
        variance_ratio=F.variance_ratio(tape,1.,window=window).to_dict('records'),
        one_sided_share=F.one_sided_share(tape,window))
    ins=o[(o.t_us>=start+60e6)&(o.t_us<start+3590e6)]
    for tag,g in [('ioc',ins[ins.ioc]),('limit',ins[~ins.ioc])]:
        diagnostics[tag]=dict(count=len(g),rate=len(g)/(3590-60),mean_size=g.volume.mean(),
            median_size=g.volume.median(),q95=g.volume.quantile(.95),q99=g.volume.quantile(.99),max_size=g.volume.max(),
            buy_share=(g.side=='B').mean(),lifetime_quantiles_ms=((g.end_t_us-g.t_us)/1000).quantile([.1,.5,.9,.99]).to_dict())
    for bin_s in (.01,.1,1,10,60):
        diagnostics[f'fano_{bin_s:g}s']={}
        for tag,g in [('ioc',ins[ins.ioc]),('limit',ins[~ins.ioc])]:
            bins=np.floor((g.t_us.to_numpy()-start-60e6)/(bin_s*1e6)).astype(int)
            c=np.bincount(bins,minlength=int((3590-60)/bin_s))
            c=c[:int((3590-60)//bin_s)]  # equal exposure: exclude the partial final bin
            diagnostics[f'fano_{bin_s:g}s'][tag]=float(np.var(c)/np.mean(c))
    for h in (.005,.05,.1,1.,5.):
        tables,s=E.maker_markouts(tape,window,h)
        diagnostics[f'maker_markout_{h:g}s']=s
        for tag,frame in tables.items():frame.to_csv(OUT/f'maker_{h:g}s_{tag}.csv',index=False)
        frame,s=E.queue_fills(tape,window,h)
        frame.to_csv(OUT/f'queue_{h:g}s.csv',index=False)
        diagnostics[f'queue_{h:g}s']=s
    for age in (0,50,100,500):
        quotes,cancels,s=E.latency_risk(tape,window,5,age)
        quotes.to_csv(OUT/f'latency_quotes_age{age}.csv',index=False)
        cancels.to_csv(OUT/f'latency_cancels_age{age}.csv',index=False)
    F.activity(tape,60,window).to_csv(OUT/'activity_train_60s.csv',index=False)
    dump('train_diagnostics.json',diagnostics)
    print('DIAGNOSTICS COMPLETE',flush=True)

if __name__=='__main__':main()
