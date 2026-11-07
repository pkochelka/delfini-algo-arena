"""Dependence-aware uncertainty, observable cohort diagnostics and feature audits."""
import ast
import json
import pickle
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from extract import HERE,OUT,GEN,SOURCE,load_tape,quote,dump,HORIZONS
from models import masks

def main():
    d=np.load(GEN/'features.npz');X=d['X'];Y=d['Y'];times=d['t_us'];sec=d['seconds']
    names=json.loads((OUT/'feature_schema.json').read_text())['names'];m=masks(sec,d['valid'])
    models=pickle.loads((GEN/'fitted_models.pkl').read_bytes())
    ordernames=json.loads((OUT/'order_schema.json').read_text())['names']
    for kind,model,nms in [('markout_ridge',models['markout']['ridge'],ordernames),('fill_logistic',models['fill']['logistic'],ordernames)]:
        coefs=model.steps[-1][1].coef_.ravel()
        pd.DataFrame(dict(feature=nms,standardized_coefficient=coefs)).sort_values('standardized_coefficient',key=abs,ascending=False).to_csv(OUT/(kind+'_coefficients.csv'),index=False)
    # Resample 10-second wall-clock blocks to account for correlated adjacent forecast labels.
    p=np.load(GEN/'price_predictions_0.1s.npz')['prediction']
    y=Y[:,0];mean=y[m['train']].mean();mask=m['test']
    b=np.floor((sec[mask]-4800)/10).astype(int)
    n=int(b.max())+1
    sums=np.zeros((n,5))
    for k in range(n):
        ix=np.flatnonzero(mask)[b==k];moving=y[ix]!=0
        sums[k]=[len(ix),np.sum((y[ix]-mean)**2),np.sum((y[ix]-p[ix])**2),moving.sum(),np.sum((p[ix][moving]>0)==(y[ix][moving]>0))]
    rng=np.random.default_rng(1729);boot=sums[rng.integers(0,n,size=(3000,n))].sum(axis=1)
    improvement=1-boot[:,2]/boot[:,1];accuracy=boot[:,4]/boot[:,3]
    uncertainty=dict(block_s=10,blocks=n,replicates=3000,
        mse_improvement_95=np.quantile(improvement,[.025,.975]).tolist(),
        moving_direction_accuracy_95=np.quantile(accuracy,[.025,.975]).tolist(),
        caveat='Conditional on this tape, fixed model and these partitions. Does not account for research/model-family selection or unseen market regimes.')
    dump('uncertainty.json',uncertainty)
    risk=pickle.loads((GEN/'risk_models.pkl').read_bytes())
    riskp=risk['jump'][1].predict_proba(np.nan_to_num(X))[:,1]
    cutoff=np.quantile(riskp[m['train']],[.5,.8,.9,.95,.99])
    tape=load_tape(SOURCE);tr=tape.trades
    ft=tr.t_us.to_numpy()
    # Risk state must be available at least five milliseconds BEFORE the historical fill.
    ix=np.searchsorted(times,ft-5000,side='right')-1
    ix=np.maximum(0,ix)
    future=quote(tape,ft+1_000_000,'mid')
    side=np.where(tr.aggressor.to_numpy()=='A',1.,-1.)
    markout=side*(future-tr.price.to_numpy())
    rowmask=np.isfinite(markout)&d['valid'][ix]&(times[ix]<=ft-5000)
    cohort=[]
    for split,(lo,hi) in dict(train=(60,3590),validation=(3600,4790),test=(4800,5990)).items():
        keep=rowmask&(ft>=tape.start_us+lo*1e6)&(ft+1e6<tape.start_us+hi*1e6)
        buckets=np.searchsorted(cutoff,riskp[ix],side='right')
        for k in range(6):
            z=keep&(buckets==k);qty=tr.qty.to_numpy()[z]
            if not len(qty):continue
            cohort.append(dict(split=split,training_risk_bucket=k,prints=int(z.sum()),lots=int(qty.sum()),
                volume_weighted_markout=float(np.average(markout[z],weights=qty)),
                adverse_volume_share=float(np.average(markout[z]<0,weights=qty)),
                average_prior_risk=float(np.mean(riskp[ix[z]]))))
    pd.DataFrame(cohort).to_csv(OUT/'prior_risk_maker_markouts.csv',index=False)
    dump('maker_risk_cohort_protocol.json',dict(train_risk_quantiles=[.5,.8,.9,.95,.99],thresholds=cutoff.tolist(),
        latency_ms=5,horizon_s=1,interpretation='Historical fill selection conditioned on a prior risk state. Not a counterfactual cancel simulation or strategy PnL; no cohort threshold selected on validation/test.'))
    # Observable feature and target indexing checks, independent direct source lookups.
    q=tape.quotes;qt=q.t_us.to_numpy();checked=0
    rng=np.random.default_rng(302)
    indices=rng.choice(np.flatnonzero(d['valid']),size=100,replace=False)
    for i in indices:
        qi=np.searchsorted(qt,times[i],side='right')-1
        assert qt[qi]<=times[i]
        assert qi+1==len(qt) or qt[qi+1]>times[i]
        bq=float(q.bid_qty.iloc[qi]);aq=float(q.ask_qty.iloc[qi])
        assert abs(X[i,names.index('imb_touch')]-(bq-aq)/(bq+aq))<1e-6
        assert abs(X[i,names.index('spread')]-q.spread.iloc[qi])<1e-6
        for j,h in enumerate(HORIZONS):
            ti=times[i]+5000+round(h*1e6);ei=times[i]+5000
            v=q.mid.iloc[np.searchsorted(qt,ti,side='right')-1]-q.mid.iloc[np.searchsorted(qt,ei,side='right')-1]
            assert Y[i,j]==v
        di=np.searchsorted(tape.depth_t_us,times[i],side='right')-1
        assert tape.depth_t_us[di]<=times[i]
        checked+=1
    validation=dict(sampled_decisions_checked=checked,latency_target_checks=checked*len(HORIZONS),
        book_and_depth_asof_checks=True,features_exclude_absolute_price_clock_ids_and_submitted_IOC_sizes=True,
        label_purge_seconds=10,max_price_label_seconds=5.005,max_historical_order_label_seconds=2,
        note='Direct indexing audits of representative features/targets, not a full independent exchange/replay parity test.')
    dump('validation.json',validation)
    ev=tape.events;fit=(ev.t_us>=tape.start_us+60e6)&(ev.t_us<tape.start_us+3590e6)
    e=ev.loc[fit]
    correlations=[]
    for seconds in (1,10,60):
        a=[]
        for select in ((e.op=='I')&(e.lifespan=='G'),(e.op=='I')&(e.lifespan=='F'),e.op=='C'):
            b=np.floor((e.loc[select,'t_us'].to_numpy()-tape.start_us-60e6)/(seconds*1e6)).astype(int)
            a.append(np.bincount(b,minlength=int(np.ceil(3530/seconds)))[:int(3530//seconds)])
        c=np.corrcoef(a)
        correlations.append(dict(bin_s=seconds,limit_ioc=float(c[0,1]),limit_cancel=float(c[0,2]),ioc_cancel=float(c[1,2])))
    o=tape.orders
    births=o.set_index('order_id').t_us
    canc=e.loc[e.op=='C'];born=canc.order_id.map(births)
    lag=(canc.t_us.to_numpy()-born.to_numpy())/1e6
    life=lag[np.isfinite(lag)]
    dump('flow_structure.json',dict(count_correlations=correlations,cancel_messages=len(canc),linked_cancel_fraction=float(np.isfinite(lag).mean()),
        cancel_message_lifetime_s=dict(mean=float(life.mean()),median=float(np.median(life)),q90=float(np.quantile(life,.9))),
        note='Cancellation-message lag from original insertion, including messages for orders already filled. Not necessarily actual resting lifetime. Count correlations are contemporaneous, not advance forecasts.'))
    print('INTERPRETATION/AUDITS COMPLETE',json.dumps(uncertainty),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
