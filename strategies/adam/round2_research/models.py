"""Chronological model comparisons. Freeze validation selections before test scoring."""
import hashlib
import json
import pickle
import time
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor,HistGradientBoostingClassifier
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,roc_auc_score,brier_score_loss,log_loss
from threadpoolctl import threadpool_limits
from extract import HERE,OUT,GEN,HORIZONS,dump

def masks(seconds,valid):
    return {k:valid&(seconds>=lo)&(seconds<hi) for k,(lo,hi) in
        dict(train=(60,3590),validation=(3600,4790),test=(4800,5990)).items()}

def reg_metrics(y,p):
    nz=y!=0
    return dict(n=len(y),mse=mean_squared_error(y,p),mae=mean_absolute_error(y,p),r2=r2_score(y,p),
        correlation=np.corrcoef(y,p)[0,1] if np.std(p)>1e-10 else None,
        moving_direction_accuracy=float(np.mean((p[nz]>0)==(y[nz]>0))) if nz.any() else None,
        moving_fraction=float(nz.mean()),target_std=float(np.std(y)),pred_std=float(np.std(p)))

def cls_metrics(y,p):
    return dict(n=len(y),base_rate=float(y.mean()),brier=brier_score_loss(y,p),log_loss=log_loss(y,p,labels=[0,1]),
        auc=roc_auc_score(y,p) if len(np.unique(y))==2 else None)

def regressors():
    return dict(ridge=make_pipeline(StandardScaler(),Ridge(alpha=10)),
        hgb15=HistGradientBoostingRegressor(max_leaf_nodes=15,max_iter=150,learning_rate=.05,
            min_samples_leaf=100,l2_regularization=10,early_stopping=False,random_state=1729),
        hgb31=HistGradientBoostingRegressor(max_leaf_nodes=31,max_iter=150,learning_rate=.05,
            min_samples_leaf=100,l2_regularization=10,early_stopping=False,random_state=1729))

def deciles(y,p,edges,extra=None):
    b=np.searchsorted(edges,p,side='right')
    rows=[]
    for i in np.unique(b):
        m=b==i
        r=dict(bucket=int(i),n=int(m.sum()),pred=float(p[m].mean()),realized=float(y[m].mean()),positive=float((y[m]>0).mean()))
        if extra:
            for k,v in extra.items():r[k]=float(np.mean(v[m]))
        rows.append(r)
    return rows

def main():
    np.seterr(invalid='ignore',divide='ignore')
    data=np.load(GEN/'features.npz')
    X,Y=data['X'],data['Y']
    names=json.loads((OUT/'feature_schema.json').read_text())['names']
    m=masks(data['seconds'],data['valid'])
    selections={}; price=[]; fitted={}; predictions={}
    for j,h in enumerate(HORIZONS):
        y=Y[:,j]
        candidates=regressors()
        p={name:np.zeros(len(y)) for name in ['mean','micro']+list(candidates)}
        p['mean'][:]=np.mean(y[m['train']])
        p['micro'][:]=X[:,names.index('micro_offset')]
        for name,model in candidates.items():
            t=time.perf_counter()
            model.fit(X[m['train']],y[m['train']])
            for split in ('train','validation'):
                p[name][m[split]]=model.predict(X[m[split]])
            print('FIT PRICE',h,name,round(time.perf_counter()-t,2),flush=True)
        val={name:reg_metrics(y[m['validation']],v[m['validation']]) for name,v in p.items()}
        winner=min(val,key=lambda name:val[name]['mse'])
        selections[f'price_{h:g}s']=winner
        fitted[j]=candidates
        predictions[j]=p
        for name in p:
            price.append(dict(horizon_s=h,model=name,split='validation',**val[name]))
        coefs=candidates['ridge'].named_steps['ridge'].coef_
        pd.DataFrame(dict(feature=names,standardized_coefficient=coefs)).sort_values('standardized_coefficient',key=abs,ascending=False).to_csv(OUT/f'ridge_{h:g}s.csv',index=False)
    orders=np.load(GEN/'order_features.npz')
    O=orders['X']; yf=orders['any_fill'];ym=orders['markout']
    om=masks(orders['seconds'],orders['valid'])
    fill_models=dict(logistic=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=500)),
        hgb=HistGradientBoostingClassifier(max_leaf_nodes=15,max_iter=150,learning_rate=.05,
            min_samples_leaf=100,l2_regularization=10,early_stopping=False,random_state=1729))
    fill_p={'mean':np.full(len(yf),yf[om['train']].mean())}
    fill_val={}
    for name,model in fill_models.items():
        t=time.perf_counter();model.fit(O[om['train']],yf[om['train']])
        fill_p[name]=np.full(len(yf),np.nan)
        fill_p[name][om['validation']]=model.predict_proba(O[om['validation']])[:,1]
        print('FIT FILL',name,round(time.perf_counter()-t,2),flush=True)
    for name,p in fill_p.items():fill_val[name]=cls_metrics(yf[om['validation']],p[om['validation']])
    selections['fill']=min(fill_val,key=lambda k:fill_val[k]['brier'])
    mark_models=regressors();mark_models.pop('hgb31')
    mm={k:v&np.isfinite(ym) for k,v in om.items()}
    mark_p={'mean':np.full(len(ym),np.mean(ym[mm['train']]))}
    mark_val={}
    for name,model in mark_models.items():
        t=time.perf_counter();model.fit(O[mm['train']],ym[mm['train']])
        mark_p[name]=np.full(len(ym),np.nan)
        mark_p[name][om['validation']]=model.predict(O[om['validation']])
        print('FIT MARKOUT',name,round(time.perf_counter()-t,2),flush=True)
    for name,p in mark_p.items():mark_val[name]=reg_metrics(ym[mm['validation']],p[mm['validation']])
    selections['markout']=min(mark_val,key=lambda k:mark_val[k]['mse'])
    # Serialize the actual fitted predictors and lock their fingerprints BEFORE test predictions.
    artifact=GEN/'fitted_models.pkl'
    artifact.write_bytes(pickle.dumps(dict(price=fitted,fill=fill_models,markout=mark_models)))
    lock=dict(locked_utc=datetime.now(timezone.utc).isoformat(),selections=selections,
        fitted_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),test_not_yet_scored=True,
        validation_price=price,validation_fill=fill_val,validation_markout=mark_val)
    dump('locked_models.json',lock)
    print('LOCKED',selections,flush=True)
    # Only after the lock do we score the withheld final 20%.
    curves=[];crossing=[];importance=[]
    for j,h in enumerate(HORIZONS):
        y=Y[:,j];p=predictions[j]
        for name,model in fitted[j].items():p[name][m['test']]=model.predict(X[m['test']])
        for name,v in p.items():
            price.append(dict(horizon_s=h,model=name,split='test',selected=name==selections[f'price_{h:g}s'],**reg_metrics(y[m['test']],v[m['test']])))
        chosen=selections[f'price_{h:g}s']
        edges=np.quantile(p[chosen][m['validation']],np.arange(.1,1,.1))
        for split in ('validation','test'):
            ix=m[split];pred=p[chosen][ix]
            curves.extend([dict(horizon_s=h,split=split,**r) for r in deciles(y[ix],pred,edges)])
            # Independent crossing opportunities, NOT an inventory-constrained strategy backtest.
            # Thinning by horizon reduces overlapping holding intervals.
            inds=np.flatnonzero(ix)[::max(1,math_ceil(h/.1))]
            ps=p[chosen][inds]
            pnl=np.where(ps>0,data['exit_bid'][inds,j]-data['entry_ask'][inds],data['entry_bid'][inds]-data['exit_ask'][inds,j])
            for extra in (0.,.25,.5,1.):
                allowed=abs(ps)>(X[inds,names.index('spread')]/2+extra)
                good=allowed&np.isfinite(pnl)
                crossing.append(dict(horizon_s=h,split=split,extra_edge_threshold=extra,opportunities=len(inds),
                    selected=int(good.sum()),mean_crossing_ticks=float(np.mean(pnl[good])) if good.any() else None,
                    win_rate=float(np.mean(pnl[good]>0)) if good.any() else None))
        # Joint group permutation on validation only; descriptive, no repeated selection.
        if chosen in fitted[j]:
            model=fitted[j][chosen];base=reg_metrics(y[m['validation']],p[chosen][m['validation']])['mse']
            groups={'book':[i for i,n in enumerate(names) if n.startswith(('spread','imb_','micro','log_bid','log_ask','touch_','log_depth'))],
                'returns':[i for i,n in enumerate(names) if n.startswith('return_')],
                'volatility':[i for i,n in enumerate(names) if n.startswith(('rms_','moves_'))],
                'prints':[i for i,n in enumerate(names) if n.startswith(('trade_imb','log_trade'))],
                'quote_flow':[i for i,n in enumerate(names) if n.startswith(('ofi_','book_updates'))]}
            xx=X[m['validation']].copy();rng=np.random.default_rng(1729)
            for group,cols in groups.items():
                z=xx.copy();block=100;order=rng.permutation(math_ceil(len(z)/block))
                perm=np.concatenate([np.arange(k*block,min((k+1)*block,len(z))) for k in order])
                z[:,cols]=xx[perm][:,cols]
                importance.append(dict(horizon_s=h,group=group,validation_mse_increase=mean_squared_error(y[m['validation']],model.predict(z))-base))
        np.savez_compressed(GEN/f'price_predictions_{h:g}s.npz',seconds=data['seconds'],prediction=p[chosen],target=y,valid=data['valid'])
    fill_test={};mark_test={}
    for name,model in fill_models.items():fill_p[name][om['test']]=model.predict_proba(O[om['test']])[:,1]
    for name,p in fill_p.items():fill_test[name]=cls_metrics(yf[om['test']],p[om['test']])
    for name,model in mark_models.items():mark_p[name][om['test']]=model.predict(O[om['test']])
    for name,p in mark_p.items():mark_test[name]=reg_metrics(ym[mm['test']],p[mm['test']])
    for split in ('validation','test'):
        ix=om[split];p=fill_p[selections['fill']]
        edges=np.linspace(.1,.9,9)
        pd.DataFrame(deciles(yf[ix],p[ix],edges)).to_csv(OUT/f'fill_calibration_{split}.csv',index=False)
        ix=mm[split];p=mark_p[selections['markout']]
        ve=mark_p[selections['markout']][mm['validation']]
        pd.DataFrame(deciles(ym[ix],p[ix],np.quantile(ve,np.arange(.1,1,.1)))).to_csv(OUT/f'markout_deciles_{split}.csv',index=False)
    pd.DataFrame(price).to_csv(OUT/'price_models.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'price_deciles.csv',index=False)
    pd.DataFrame(crossing).to_csv(OUT/'crossing_opportunities.csv',index=False)
    pd.DataFrame(importance).to_csv(OUT/'feature_group_importance.csv',index=False)
    dump('model_results.json',dict(lock=lock,price=price,fill_validation=fill_val,fill_test=fill_test,
        markout_validation=mark_val,markout_test=mark_test,crossing_opportunities=crossing,
        row_counts={k:int(v.sum()) for k,v in m.items()},order_row_counts={k:int(v.sum()) for k,v in om.items()},
        conditional_markout_counts={k:int(v.sum()) for k,v in mm.items()}))
    print('COMPLETE MODEL TEST',flush=True)

def math_ceil(x):return int(np.ceil(x))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
