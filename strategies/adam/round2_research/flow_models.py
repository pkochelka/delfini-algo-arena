"""Activity spectrum and causal one-second executed-aggressor intensity forecasts."""
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_poisson_deviance
from threadpoolctl import threadpool_limits
from extract import load_tape,SOURCE,OUT,dump

def main():
    tape=load_tape(SOURCE)
    taker=tape.trades.groupby('taker_row',sort=False).t_us.first().to_numpy()
    c=np.bincount(((taker-tape.start_us)//1_000_000).astype(int),minlength=6000)[:6000]
    times=np.arange(60,5990)
    fit=(times<3590);val=(times>=3600)&(times<4790);test=times>=4800
    train_counts=c[60:3590]
    freq,power=welch(train_counts,fs=1,nperseg=512,detrend='linear')
    select=(freq>1/300)&(freq<1/2)
    ix=np.flatnonzero(select)[np.argsort(power[select])[-10:][::-1]]
    spectrum=pd.DataFrame(dict(period_s=1/freq[ix],power=power[ix]))
    spectrum.to_csv(OUT/'activity_spectral_peaks_train.csv',index=False)
    period=float(spectrum.period_s.iloc[0])
    lag=np.column_stack([np.log1p(c[times-k]) for k in (1,2,5,10)])
    cs=np.r_[0,np.cumsum(c)]
    lag=np.column_stack([lag]+[np.log1p((cs[times]-cs[times-k])/k) for k in (5,30,60)])
    phase=np.column_stack([np.sin(2*np.pi*times/period),np.cos(2*np.pi*times/period)])
    xs=dict(lagged=lag,phase_only=phase,lagged_plus_phase=np.column_stack([lag,phase]))
    models={k:make_pipeline(StandardScaler(),PoissonRegressor(alpha=.01,max_iter=500)) for k in xs}
    models['hgb_lagged']=HistGradientBoostingRegressor(loss='poisson',max_leaf_nodes=7,max_iter=100,
        min_samples_leaf=50,l2_regularization=10,early_stopping=False,random_state=1729)
    xs['hgb_lagged']=lag
    for k,model in models.items():model.fit(xs[k][fit],c[times][fit])
    validation={k:mean_poisson_deviance(c[times][val],model.predict(xs[k][val])) for k,model in models.items()}
    validation['constant']=mean_poisson_deviance(c[times][val],np.full(val.sum(),train_counts.mean()))
    winner=min(validation,key=validation.get)
    dump('locked_flow_model.json',dict(train_peak_period_s=period,selection=winner,validation_deviance=validation,
        test_not_yet_scored=True,note='Exploratory extension. One period selected using training Welch spectrum only; absolute-clock phase is tested explicitly and must not be assumed transferable to a new session.'))
    result=[]
    for split,mask in [('validation',val),('test',test)]:
        y=c[times][mask]
        for k in ['constant']+list(models):
            p=np.full(mask.sum(),train_counts.mean()) if k=='constant' else models[k].predict(xs[k][mask])
            result.append(dict(split=split,model=k,selected=k==winner,n=int(mask.sum()),
                poisson_deviance=mean_poisson_deviance(y,p),correlation=float(np.corrcoef(y,p)[0,1]) if np.std(p)>1e-10 else None,
                mean_observed=float(y.mean()),mean_predicted=float(p.mean())))
    pd.DataFrame(result).to_csv(OUT/'arrival_intensity_models.csv',index=False)
    dump('flow_model_results.json',dict(selected=winner,period_s=period,results=result,
        train_count_acf={str(k):float(np.corrcoef(train_counts[:-k],train_counts[k:])[0,1]) for k in (1,2,5,10,30,60,120)}))
    print('FLOW MODELS COMPLETE',winner,period,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
