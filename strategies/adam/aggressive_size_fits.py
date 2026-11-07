"""Integer-aware size-distribution fits and refitted-cutoff power-law bootstrap.

All research outputs stay in strategies/adam/aggressive_size_results. No package installation
or repository source changes. Sizes are counted once per incoming order, never per fill.
"""
import hashlib
import json
import csv
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special, stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
REPO = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').exists() and (p / 'simulator').is_dir())
sys.path.insert(0, str(REPO))

OUT = HERE / 'aggressive_size_results'
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20261107)


def hist(x):
    return np.unique(np.asarray(x, dtype=np.int64), return_counts=True)


def zipf_fit_counts(k, counts, xmin):
    n = int(counts.sum())
    logsum = float(np.dot(counts, np.log(k)))
    fit = optimize.minimize_scalar(lambda alpha: alpha*logsum+n*np.log(special.zeta(alpha,xmin)),
                                   bounds=(1.00001,15), method='bounded', options={'xatol':1e-7})
    alpha = float(fit.x)
    z = special.zeta(alpha, xmin)
    after = np.cumsum(counts)/n
    before = (np.cumsum(counts)-counts)/n
    D = max(np.max(np.abs(after-(1-special.zeta(alpha,k+1)/z))),
            np.max(np.abs(before-(1-special.zeta(alpha,k)/z))))
    return dict(xmin=int(xmin), n=n, alpha=alpha, survival_exponent=alpha-1,
                D=float(D), loglik=float(-fit.fun), success=bool(fit.success))


def zipf_fit(x, xmin):
    k, counts = hist(x[x >= xmin])
    return zipf_fit_counts(k,counts,xmin)


def select_tail(x, min_tail=1000):
    # Fixed rule rerun on every bootstrap replicate, including cutoff selection.
    quantiles = np.unique(np.r_[np.linspace(0,.9,30),1-np.geomspace(.1,min_tail/len(x),35)])
    candidates = np.unique(np.quantile(x, quantiles, method='nearest')).astype(int)
    values, counts = hist(x)
    fits = []
    for v in candidates:
        start = np.searchsorted(values,v)
        if counts[start:].sum() >= min_tail:
            fits.append(zipf_fit_counts(values[start:],counts[start:],int(v)))
    best = min(fits, key=lambda f:f['D'])
    return best, fits


def read_ioc_sizes(path):
    """Stream price-time matching with no full-event or snapshot arrays in memory."""
    levels = {'B':{},'A':{}}
    live = {}
    records = []
    crossing_G = 0
    with path.open(newline='') as file:
        reader = csv.reader(file)
        assert next(reader) == ['Time','Operation','OrderId','Side','Volume','Price','Lifespan']
        for event_row,row in enumerate(reader):
            _,op,oid,side,vol,px,life = row
            if op == 'Cancel':
                old = live.pop(oid,None)
                if old is not None:
                    s,p,entry = old
                    levels[s][p].remove(entry)
                    if not levels[s][p]:
                        del levels[s][p]
                continue
            volume,price = int(vol),int(px)
            remaining = volume
            opposite = levels['A' if side=='B' else 'B']
            while remaining and opposite:
                best = min(opposite) if side=='B' else max(opposite)
                if (side=='B' and price<best) or (side=='A' and price>best):
                    break
                queue = opposite[best]
                while remaining and queue:
                    maker = queue[0]
                    qty = min(remaining,maker[1])
                    remaining -= qty
                    maker[1] -= qty
                    if maker[1] == 0:
                        queue.popleft()
                        del live[maker[0]]
                if not queue:
                    del opposite[best]
            if life == 'F':
                records.append((event_row,volume,volume-remaining,side))
            elif volume>remaining:
                crossing_G += 1
            if remaining and life != 'F':
                entry = [oid,remaining]
                levels[side].setdefault(price,deque()).append(entry)
                live[oid] = (side,price,entry)
    return pd.DataFrame(records,columns=['event_row','submitted_size','executed_size','side']),crossing_G


def zipf_sample(alpha, xmin, n):
    """Exact inverse-CDF sampling from a Hurwitz-zeta-normalized integer tail."""
    u = RNG.random(n)
    target = u * special.zeta(alpha,xmin)
    lo = np.full(n,xmin,dtype=np.int64)
    hi = np.maximum(xmin+1,np.ceil(xmin*u**(-1/(alpha-1))*2).astype(np.int64))
    while np.any(special.zeta(alpha,hi)>target):
        mask = special.zeta(alpha,hi)>target
        hi[mask] *= 2
    while np.any(hi-lo>1):
        mid = (hi+lo)//2
        move_lo = special.zeta(alpha,mid)>=target
        lo = np.where(move_lo,mid,lo)
        hi = np.where(move_lo,hi,mid)
    return lo


def tail_bootstrap(x, best, simulations=199, min_tail=1000):
    body = x[x<best['xmin']]
    prob = best['n']/len(x)
    Ds = []
    for i in range(simulations):
        n_tail = RNG.binomial(len(x),prob)
        tail = zipf_sample(best['alpha'],best['xmin'],n_tail)
        below = RNG.choice(body,len(x)-n_tail,replace=True) if len(body) else np.empty(0,dtype=int)
        synthetic = np.concatenate([below,tail])
        fitted,_ = select_tail(synthetic,min_tail=min_tail)
        Ds.append(fitted['D'])
        if (i+1)%50 == 0:
            print(f'  bootstrap {i+1}/{simulations}',flush=True)
    return dict(simulations=simulations, p=(1+sum(d>=best['D'] for d in Ds))/(simulations+1),
                p_resolution=1/(simulations+1), xmin_refitted=True, min_tail=min_tail)


def lognormal_logpmf(k,mu,sigma,offset=0.0,xmin=1):
    """Q=max(1,floor(X+offset)), X lognormal. offset .5 means nearest rounding."""
    k = np.asarray(k)
    lower = np.maximum(k-offset,1e-300)
    upper = k+1-offset
    zl = (np.log(lower)-mu)/sigma
    zu = (np.log(upper)-mu)/sigma
    mass = np.where(zl>0, special.ndtr(-zl)-special.ndtr(-zu), special.ndtr(zu)-special.ndtr(zl))
    mass = np.where(k==1,special.ndtr(zu),mass)
    norm = stats.norm.logsf((np.log(xmin-offset)-mu)/sigma) if xmin>1 else 0.0
    return np.log(np.maximum(mass,1e-300))-norm


def lognormal_fit(x,xmin=1,offset=0.0):
    values,counts = hist(x[x>=xmin])
    start = [float(np.log(values).dot(counts)/counts.sum()),float(np.std(np.log(x[x>=xmin])))]
    if xmin>1:
        start = [4.1,1.4]
    def objective(p):
        if not (-20<p[0]<20 and .05<np.exp(p[1])<8):
            return 1e100
        return -float(np.dot(counts,lognormal_logpmf(values,p[0],np.exp(p[1]),offset,xmin)))
    fitted = optimize.minimize(objective,[start[0],np.log(max(start[1],.2))],
                               method='Nelder-Mead', options={'maxiter':3000,'xatol':1e-8,'fatol':1e-6})
    mu, sigma = float(fitted.x[0]),float(np.exp(fitted.x[1]))
    return dict(mu=mu,sigma=sigma,offset=offset,loglik=-float(fitted.fun),
                success=bool(fitted.success),parameters=2)


def full_models(x):
    values,counts = hist(x)
    n = len(x)
    models = {}
    for offset in (0.0,.5,1.0):
        models[f'Lognormal offset {offset:g}'] = lognormal_fit(x,offset=offset)
    power = zipf_fit(x,1)
    models['Discrete power law'] = dict(**power,parameters=1)
    # Geometric is the exact positive-integer version of an exponential size law.
    p = 1/float(np.mean(x))
    models['Geometric (exponential)'] = dict(p=p,loglik=float(np.dot(counts,np.log(p)+(values-1)*np.log1p(-p))),parameters=1)
    # A rounded Lomax (Pareto II) is a two-parameter heavy-tailed alternative.
    def lomax_logpmf(k,beta,scale):
        logleft = -beta*np.log1p(k/scale)
        logright = -beta*np.log1p((k+1)/scale)
        mass = logleft + np.log(-np.expm1(logright-logleft))
        return np.where(k==1,np.log(-np.expm1(-beta*np.log1p(2/scale))),mass)
    def objective(params):
        beta,scale = np.exp(params)
        return -float(np.dot(counts,lomax_logpmf(values,beta,scale)))
    fit = optimize.minimize(objective,np.log([2.,100.]),method='Nelder-Mead',options={'maxiter':2000})
    beta,scale = np.exp(fit.x)
    models['Rounded Lomax (Pareto II)'] = dict(beta=float(beta),scale=float(scale),loglik=-float(fit.fun),
                                             parameters=2,success=bool(fit.success))
    for model in models.values():
        model['AIC'] = 2*model['parameters']-2*model['loglik']
        model['n'] = n
    winner = min(v['AIC'] for v in models.values())
    for model in models.values():
        model['delta_AIC'] = model['AIC']-winner
    return models


def summary(x):
    return dict(n=len(x),mean=float(x.mean()),std=float(x.std(ddof=1)),
                quantiles={str(q):float(np.quantile(x,q)) for q in (0,.1,.25,.5,.75,.9,.95,.99,.999,1)},
                count_one=int((x==1).sum()))


def main():
    source = REPO/'data'/'round1.csv'
    def digest(path):
        with path.open('rb') as file:
            return hashlib.file_digest(file,'sha256').hexdigest()
    original_hash = digest(source)
    print('Reconstructing IOC executions in streaming mode...',flush=True)
    order_table,crossing_G = read_ioc_sizes(source)
    assert crossing_G == 0
    submitted = order_table.submitted_size.to_numpy(dtype=np.int64)
    matched = order_table.executed_size > 0
    matched_submitted = order_table.loc[matched,'submitted_size'].to_numpy(dtype=np.int64)
    executed_sizes = order_table.loc[matched,'executed_size'].to_numpy(dtype=np.int64)
    samples = {'IOC submitted':submitted,'Executed per aggressive order':executed_sizes}
    results = dict(source=str(source),sha256=original_hash,seed=20261107,
                   ioc_orders=len(submitted),aggressors_with_fills=int(matched.sum()),
                   no_fill_ioc_orders=int((~matched).sum()),crossing_G_orders=crossing_G,
                   matched_submitted=summary(matched_submitted),datasets={})
    chart = make_subplots(rows=2,cols=2,subplot_titles=(
        'Submitted IOC size: empirical survival and full lognormal fit',
        'Submitted sizes: Pareto tail fit',
        'Tail exponent versus lower cutoff',
        'Submitted versus executed size (one point per incoming order)'))
    for name,x in samples.items():
        print(name,summary(x),flush=True)
        models = full_models(x)
        best,curve = select_tail(x)
        print('Models',json.dumps(models),flush=True)
        print('Selected tail',json.dumps(best),flush=True)
        boot = tail_bootstrap(x,best)
        tail = x[x>=best['xmin']]
        best_ln_name = min((key for key in models if key.startswith('Lognormal')),key=lambda key:models[key]['AIC'])
        ln = models[best_ln_name]
        truncated_ln = lognormal_fit(tail,xmin=best['xmin'],offset=ln['offset'])
        difference = truncated_ln['loglik']-best['loglik']
        # Chronological 75/25 holdout; select cutoff and fit both models only on training orders.
        train,test = np.split(x,[int(.75*len(x))])
        train_tail,_ = select_tail(train,min_tail=750)
        cutoff = train_tail['xmin']
        train_offset = min((lognormal_fit(train,offset=o) for o in (0.0,.5,1.0)),
                           key=lambda fit:-fit['loglik'])['offset']
        train_ln = lognormal_fit(train,xmin=cutoff,offset=train_offset)
        held = test[test>=cutoff]
        lp_power = -train_tail['alpha']*np.log(held)-np.log(special.zeta(train_tail['alpha'],cutoff))
        lp_ln = lognormal_logpmf(held,train_ln['mu'],train_ln['sigma'],train_offset,cutoff)
        data = dict(summary=summary(x),models=models,power_tail=best,cutoff_scan=curve,bootstrap=boot,
                    conditional_lognormal=truncated_ln,loglik_lognormal_minus_power_tail=difference,
                    holdout=dict(train_fraction=.75,xmin=cutoff,n_test_tail=len(held),rounding_offset=train_offset,
                                 mean_logscore_lognormal_minus_power=float(np.mean(lp_ln-lp_power)),
                                 total_logscore_lognormal_minus_power=float(np.sum(lp_ln-lp_power))))
        results['datasets'][name] = data
        print('Bootstrap',boot,'tail loglik difference',difference,'holdout',data['holdout'],flush=True)
        pd.DataFrame(curve).to_csv(OUT/(name.lower().replace(' ','_')+'_thresholds.csv'),index=False)
        if name == 'IOC submitted':
            values,counts = hist(x)
            surv = np.cumsum(counts[::-1])[::-1]/len(x)
            chart.add_trace(go.Scatter(x=values,y=surv,name='Submitted empirical',line=dict(color='#2563eb')),row=1,col=1)
            lower = np.maximum(values-ln['offset'],1e-300)
            predicted = stats.norm.sf((np.log(lower)-ln['mu'])/ln['sigma'])
            predicted[values==1] = 1
            chart.add_trace(go.Scatter(x=values,y=predicted,name='Fitted lognormal',line=dict(color='#08916a',dash='dash')),row=1,col=1)
            keep = values>=best['xmin']
            chart.add_trace(go.Scatter(x=values[keep],y=surv[keep]/(best['n']/len(x)),name='Empirical conditional tail',line=dict(color='#2563eb')),row=1,col=2)
            chart.add_trace(go.Scatter(x=values[keep],y=special.zeta(best['alpha'],values[keep])/special.zeta(best['alpha'],best['xmin']),
                                       name='Discrete power-law tail',line=dict(color='#d97706',dash='dash')),row=1,col=2)
        chart.add_trace(go.Scatter(x=[v['xmin'] for v in curve],y=[v['survival_exponent'] for v in curve],
                                   name=name+' tail exponent',mode='lines+markers'),row=2,col=1)
        values,counts = hist(x)
        chart.add_trace(go.Scatter(x=values,y=np.cumsum(counts[::-1])[::-1]/len(x),name=name),row=2,col=2)
        (OUT/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    # Check buy/sell and chronological consistency of the lognormal reconstruction.
    first = results['datasets']['IOC submitted']['models']
    offset = min((v for k,v in first.items() if k.startswith('Lognormal')),key=lambda v:v['AIC'])['offset']
    sides = order_table.side.to_numpy()
    results['lognormal_subgroups'] = {side:lognormal_fit(submitted[sides==side],offset=offset) for side in ('B','A')}
    results['lognormal_subgroups'].update({f'quarter{k+1}':lognormal_fit(x,offset=offset)
                                         for k,x in enumerate(np.array_split(submitted,4))})
    # Absolute goodness-of-fit of the complete rounded lognormal, refitting its parameters.
    def ln_distance(x,fit):
        k,c = hist(x)
        before = (np.cumsum(c)-c)/len(x)
        after = np.cumsum(c)/len(x)
        lo = np.maximum(k-fit['offset'],1e-300)
        hi = k+1-fit['offset']
        F0 = special.ndtr((np.log(lo)-fit['mu'])/fit['sigma'])
        F0[k==1] = 0
        F1 = special.ndtr((np.log(hi)-fit['mu'])/fit['sigma'])
        return float(max(np.abs(before-F0).max(),np.abs(after-F1).max()))
    ln = min((v for k,v in first.items() if k.startswith('Lognormal')),key=lambda v:v['AIC'])
    D = ln_distance(submitted,ln)
    distances=[]
    for i in range(199):
        synthetic=np.maximum(1,np.floor(RNG.lognormal(ln['mu'],ln['sigma'],len(submitted))+offset)).astype(np.int64)
        fitted=lognormal_fit(synthetic,offset=offset)
        distances.append(ln_distance(synthetic,fitted))
    results['full_lognormal_bootstrap'] = dict(D=D,simulations=199,p=(1+sum(d>=D for d in distances))/200,
                                               offset_fixed_after_model_selection=True)
    (OUT/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    order_table.to_csv(OUT/'ioc_order_sizes.csv',index=False)
    for row,col in ((1,1),(1,2),(2,2)):
        chart.update_xaxes(type='log',title_text='Order size (lots)',row=row,col=col)
        chart.update_yaxes(type='log',title_text='P(size >= x)',row=row,col=col)
    chart.update_xaxes(type='log',title_text='Lower cutoff (lots)',row=2,col=1)
    chart.update_yaxes(title_text='Survival exponent beta (PMF alpha = beta+1)',row=2,col=1)
    chart.update_layout(template='plotly_white',height=1000,title='Round 1 aggressive-order sizes: lognormal versus Pareto',
                        legend=dict(orientation='h',y=-.13),margin=dict(b=150))
    (OUT/'diagnostics.html').write_text(chart.to_html(full_html=True,include_plotlyjs=True),encoding='utf-8')
    print('Full lognormal GOF',results['full_lognormal_bootstrap'],flush=True)
    print('Saved',OUT,flush=True)


if __name__ == '__main__':
    main()
