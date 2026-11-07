"""Fixed supplemental risk models and structural diagnostics; no trading-model retuning."""
import json
import pickle
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor,HistGradientBoostingClassifier
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_pinball_loss,brier_score_loss,roc_auc_score
from threadpoolctl import threadpool_limits
from extract import GEN,OUT,SOURCE,HORIZONS,load_tape,quote,dump
from models import masks

def main():
    dump('risk_protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Supplementary risk and regime analysis after initial directional study. Same chronological partitions. No retuning of the earlier price models.',
        quantile=.9,models='constant training quantile, rolling Gaussian q90, HGB quantile 15 leaves/150 iterations',
        jump_thresholds={'.1s':3,'1s':5,'5s':10},jump_models='training rate vs HGB classifier',
        regimes='3-component Gaussian mixture on current log RMS, spread, depth and print activity; trained on fitting block only; descriptive clusters, not identified latent generator states.'))
    d=np.load(GEN/'features.npz');X=d['X'];Y=d['Y'];sec=d['seconds']
    names=json.loads((OUT/'feature_schema.json').read_text())['names']
    m=masks(sec,d['valid'])
    results=[];jumpresults=[];risk_models={};jump_models={};locks={}
    for j,h in enumerate(HORIZONS):
        y=abs(Y[:,j]);threshold={.1:3,1.:5,5.:10}[h]
        model=HistGradientBoostingRegressor(loss='quantile',quantile=.9,max_leaf_nodes=15,
            max_iter=150,learning_rate=.05,min_samples_leaf=100,l2_regularization=10,
            early_stopping=False,random_state=1729)
        model.fit(X[m['train']],y[m['train']]);risk_models[j]=model
        vals={'constant':np.full(m['validation'].sum(),np.quantile(y[m['train']],.9)),
            'rolling_gaussian':X[m['validation'],names.index('rms_30s')]*np.sqrt(h)*1.644854,
            'hgb':np.maximum(0,model.predict(X[m['validation']]))}
        for name,p in vals.items():
            yy=y[m['validation']]
            results.append(dict(horizon_s=h,model=name,split='validation',pinball=mean_pinball_loss(yy,p,alpha=.9),coverage=float(np.mean(yy<=p)),mean_prediction=float(np.mean(p))))
        locks[f'q90_{h:g}s']=min(vals,key=lambda k:mean_pinball_loss(y[m['validation']],vals[k],alpha=.9))
        yy=(y>=threshold).astype(int)
        c=HistGradientBoostingClassifier(max_leaf_nodes=15,max_iter=150,learning_rate=.05,
            min_samples_leaf=100,l2_regularization=10,early_stopping=False,random_state=1729)
        c.fit(X[m['train']],yy[m['train']]);jump_models[j]=c
        vp={'constant':np.full(m['validation'].sum(),yy[m['train']].mean()),'hgb':c.predict_proba(X[m['validation']])[:,1]}
        for name,p in vp.items():jumpresults.append(dict(horizon_s=h,threshold=threshold,model=name,split='validation',
            brier=brier_score_loss(yy[m['validation']],p),auc=roc_auc_score(yy[m['validation']],p),base_rate=float(yy[m['validation']].mean())))
        locks[f'jump_{h:g}s']=min(vp,key=lambda k:brier_score_loss(yy[m['validation']],vp[k]))
    dump('locked_risk_models.json',dict(locked_utc=datetime.now(timezone.utc).isoformat(),selections=locks,test_not_yet_scored=True))
    for j,h in enumerate(HORIZONS):
        y=abs(Y[:,j]);yy=y[m['test']]
        ps={'constant':np.full(len(yy),np.quantile(y[m['train']],.9)),
            'rolling_gaussian':X[m['test'],names.index('rms_30s')]*np.sqrt(h)*1.644854,
            'hgb':np.maximum(0,risk_models[j].predict(X[m['test']]))}
        for name,p in ps.items():results.append(dict(horizon_s=h,model=name,split='test',selected=name==locks[f'q90_{h:g}s'],
            pinball=mean_pinball_loss(yy,p,alpha=.9),coverage=float(np.mean(yy<=p)),mean_prediction=float(np.mean(p))))
        threshold={.1:3,1.:5,5.:10}[h];yj=(y>=threshold).astype(int)
        ps={'constant':np.full(len(yy),yj[m['train']].mean()),'hgb':jump_models[j].predict_proba(X[m['test']])[:,1]}
        for name,p in ps.items():jumpresults.append(dict(horizon_s=h,threshold=threshold,model=name,split='test',selected=name==locks[f'jump_{h:g}s'],
            brier=brier_score_loss(yj[m['test']],p),auc=roc_auc_score(yj[m['test']],p),base_rate=float(yj[m['test']].mean())))
    pd.DataFrame(results).to_csv(OUT/'tail_risk_models.csv',index=False)
    pd.DataFrame(jumpresults).to_csv(OUT/'jump_models.csv',index=False)
    # Regime clustering sees only present features, with normalization/cluster fitting on train.
    cols=[names.index(n) for n in ('rms_1s','rms_30s','spread','log_depth5','log_tradevol_1s')]
    z=X[:,cols].copy();z[:,:2]=np.log1p(z[:,:2])
    scaler=StandardScaler().fit(z[m['train']]);z=np.nan_to_num(scaler.transform(z))
    gm=GaussianMixture(n_components=3,covariance_type='full',reg_covar=.01,n_init=3,random_state=1729).fit(z[m['train']])
    (GEN/'risk_models.pkl').write_bytes(pickle.dumps(dict(tail=risk_models,jump=jump_models,regime=gm,scaler=scaler)))
    states=gm.predict(z)
    cluster=[]
    for split,mask in m.items():
        for state in range(3):
            ix=mask&(states==state)
            cluster.append(dict(split=split,state=state,n=int(ix.sum()),share=float(ix.sum()/mask.sum()),
                mean_spread=float(X[ix,names.index('spread')].mean()),
                current_rms30=float(X[ix,names.index('rms_30s')].mean()),
                future_abs_move_1s=float(abs(Y[ix,1]).mean()),future_move_1s_std=float(Y[ix,1].std()),
                future_5tick_jump_rate=float((abs(Y[ix,1])>=5).mean())))
    pd.DataFrame(cluster).to_csv(OUT/'regime_clusters.csv',index=False)
    # Source-event diagnostics, kept separate from model predictors.
    tape=load_tape(SOURCE);ev=tape.events;q=tape.quotes
    mid=q.mid.to_numpy();good=np.isfinite(mid)
    delta=np.r_[0,np.diff(mid)];ok=good&np.r_[False,good[:-1]]
    train=(q.t_us.to_numpy()>=tape.start_us+60e6)&(q.t_us.to_numpy()<tape.start_us+3590e6)&ok
    types=np.where(ev.op.to_numpy()=='C','cancel',np.where(ev.lifespan.to_numpy()=='F','IOC','limit insert'))
    jump_sources=[]
    for threshold in (.5,1,3,5,10):
        for typ in ('cancel','IOC','limit insert'):
            ix=train&(abs(np.nan_to_num(delta))>=threshold)&(types==typ)
            jump_sources.append(dict(threshold_ticks=threshold,event_type=typ,count=int(ix.sum()),
                sum_squared_changes=float(np.nansum(delta[ix]**2)),mean_absolute_change=float(np.nanmean(abs(delta[ix]))) if ix.any() else None))
    pd.DataFrame(jump_sources).to_csv(OUT/'jump_event_sources_train.csv',index=False)
    tails=[]
    for split,mask in m.items():
        for j,h in enumerate(HORIZONS):
            a=abs(Y[mask,j]);var=Y[mask,j]**2
            qs=np.quantile(a,[.5,.9,.95,.99,.999])
            tails.append(dict(split=split,horizon_s=h,median=qs[0],q90=qs[1],q95=qs[2],q99=qs[3],q999=qs[4],max=float(a.max()),
                top1pct_squared_move_share=float(np.sort(var)[-max(1,round(len(a)*.01)):].sum()/var.sum()),
                kurtosis=float(pd.Series(Y[mask,j]).kurt())))
    pd.DataFrame(tails).to_csv(OUT/'return_tails.csv',index=False)
    # Ex-post 5-minute regime summaries aid transfer checks; not model selection inputs.
    blocks=[]
    for lo in range(0,6000,300):
        mask=(sec>=lo)&(sec<lo+300)&d['valid']
        oi=tape.orders[(tape.orders.t_us>=tape.start_us+lo*1e6)&(tape.orders.t_us<tape.start_us+(lo+300)*1e6)]
        ioc=oi[oi.ioc]
        blocks.append(dict(start_s=lo,n=int(mask.sum()),mean_mid=float(d['mid'][mask].mean()),
            mean_spread=float(X[mask,names.index('spread')].mean()),rms30=float(X[mask,names.index('rms_30s')].mean()),
            ioc_rate=len(ioc)/300,mean_ioc_size=float(ioc.volume.mean()),buy_ioc_fraction=float((ioc.side=='B').mean()),
            return1s_std=float(Y[mask,1].std()),return1s_mean=float(Y[mask,1].mean())))
    pd.DataFrame(blocks).to_csv(OUT/'five_minute_regimes.csv',index=False)
    dump('risk_results.json',dict(locks=locks,quantiles=results,jump=jumpresults,clusters=cluster,return_tails=tails))
    print('RISK/STRUCTURE COMPLETE',locks,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
