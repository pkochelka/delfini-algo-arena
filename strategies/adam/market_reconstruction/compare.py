"""Development selection, five fresh validation seeds, one full-length synthetic tape."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp,wasserstein_distance
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from engine import summarize
from simulate import Model,generate

HERE = Path(__file__).resolve().parent
OUT = HERE/'results'
SEEDS = [101,202,303,404,505]


def error(model,true):
    fields = ['spread_mean','mean_touch_quantity','mean_depth_5ticks','sigma_1s',
              'executed_size_mean','traded_lots_per_s','fano_100ms','maker_markout_1s']
    return float(np.mean([abs(model[k]-true[k])/max(abs(true[k]),.1) for k in fields]))


def distribution(rec,start,end,key):
    mask = (rec['grid_t']>=start)&(rec['grid_t']<end)
    x = rec[key][mask]
    return x[np.isfinite(x)]


def main():
    original = np.load(OUT/'original_measurements.npz')
    true_dev = summarize(original,7200,9600)
    true_test = summarize(original,9600,12000)
    candidates = []
    for conditioned in (False,True):
        label = 'conditional' if conditioned else 'unconditional'
        result,rec = generate(Model(conditioned),20261107,2430,record_path=OUT/('dev_'+label+'.npz'))
        result['dev_error'] = error(result,true_dev)
        candidates.append(result)
        print('DEV',label,json.dumps(result),flush=True)
    chosen = min(candidates,key=lambda r:r['dev_error'])
    protocol = dict(seeds=SEEDS,selected_conditioned=chosen['conditioned'],
                    selection_uses='original 7200-9600s only',calibration_uses='30-7200s; complete order lifetimes only',
                    validation_uses='original 9600-12000s',full_seed=20261107,
                    caveat='Earlier descriptive analyses inspected the whole tape; this is held out from current fitting, not a pristine blind dataset.')
    (OUT/'locked_model.json').write_text(json.dumps(protocol,indent=2))
    model = Model(chosen['conditioned'])
    repetitions = []
    for seed in SEEDS:
        r,rec = generate(model,seed,2430,record_path=OUT/f'validation_{seed}.npz')
        for key in ('spread','touch_qty','mid'):
            target = distribution(original,9600,12000,key)
            sample = distribution(rec.data(),30,2430,key)
            if key=='mid':
                target,sample = np.diff(target[::10]),np.diff(sample[::10])
            r['KS_'+key] = float(ks_2samp(target,sample).statistic)
            r['Wasserstein_'+key] = float(wasserstein_distance(target,sample))
        repetitions.append(r)
        print('VALIDATION',json.dumps(r),flush=True)
    full,full_rec = generate(model,20261107,12000,path=OUT/'synthetic_round1.csv',record_path=OUT/'synthetic_round1_measurements.npz')
    pd.DataFrame(repetitions).to_csv(OUT/'replicate_statistics.csv',index=False)
    comparison=[]
    for name,value in true_test.items():
        simulated = np.array([r[name] for r in repetitions])
        comparison.append(dict(metric=name,true=value,simulated_mean=float(simulated.mean()),
                               simulated_sd=float(simulated.std(ddof=1)),
                               relative_error=float((simulated.mean()-value)/max(abs(value),.01)),
                               simulated_min=float(simulated.min()),simulated_max=float(simulated.max())))
    pd.DataFrame(comparison).to_csv(OUT/'comparison.csv',index=False)
    outcome = dict(protocol=protocol,development=candidates,true_test=true_test,replicates=repetitions,
                   comparison=comparison,full_synthetic=full)
    (OUT/'outcome.json').write_text(json.dumps(outcome,indent=2))
    fig = make_subplots(rows=2,cols=2,subplot_titles=('Spread distribution','Touch-depth distribution',
                                                    'One-second midpoint returns','Selected summary ratios (simulation / original)'))
    first = np.load(OUT/f'validation_{SEEDS[0]}.npz')
    for rec,start,end,label,color in ((original,9600,12000,'Original validation','#2563eb'),
                                     (first,30,2430,'Synthetic seed 101','#08916a')):
        spread = distribution(rec,start,end,'spread')
        vals,counts = np.unique(spread,return_counts=True)
        fig.add_trace(go.Bar(x=vals,y=counts/counts.sum(),name=label,legendgroup=label,marker_color=color),row=1,col=1)
        depth = np.sort(distribution(rec,start,end,'touch_qty'))
        idx = np.linspace(0,len(depth)-1,min(1500,len(depth))).astype(int)
        fig.add_trace(go.Scatter(x=depth[idx],y=idx/(len(depth)-1),name=label,legendgroup=label,showlegend=False,line=dict(color=color)),row=1,col=2)
        returns = np.diff(distribution(rec,start,end,'mid')[::10])
        vals,counts = np.unique(returns,return_counts=True)
        fig.add_trace(go.Bar(x=vals,y=counts/counts.sum(),name=label,legendgroup=label,showlegend=False,marker_color=color),row=2,col=1)
    chosen_metrics = ['events_per_s','mean_ioc_size','spread_mean','mean_touch_quantity','sigma_1s','mean_sweep_ticks','maker_markout_1s']
    comp = pd.DataFrame(comparison).set_index('metric').loc[chosen_metrics]
    fig.add_trace(go.Bar(x=chosen_metrics,y=comp.simulated_mean/comp.true,
                         error_y=dict(array=comp.simulated_sd/comp.true),name='Five-seed mean +/- SD',marker_color='#8b5cf6'),row=2,col=2)
    fig.add_hline(y=1,line_dash='dash',row=2,col=2)
    fig.update_xaxes(type='log',title_text='Lots per side at touch',row=1,col=2)
    fig.update_layout(template='plotly_white',height=950,barmode='group',title='Synthetic market validation: matches and failures',
                      legend=dict(orientation='h',y=-.22),margin=dict(b=160))
    (OUT/'comparison.html').write_text(fig.to_html(full_html=True,include_plotlyjs=True),encoding='utf-8')
    lines = ['# Synthetic Round 1 reconstruction','',
             'A semi-parametric marked-Poisson order-book simulator, not recovered original source code.',
             'Fresh exponential G/IOC clocks and random sides; G volume/placement/cancellation-lifetime triplets '
             'resampled from training cohorts, optionally conditioned on spread and own-side touch imbalance; '
             'IOC sizes generated from a fitted rounded lognormal; IOC price reach conditioned on submitted-size bucket. '
             'Each G order receives a cancellation message, including after it has filled. FIFO matching and distinct '
             'microsecond timestamps are enforced. No original event-time or price path is replayed.','',
             'Calibration uses 30-7200 seconds, model selection uses 7200-9600 seconds, and the following '
             'comparison uses 9600-12000 seconds. Five fresh 2400-second simulations follow a 30-second burn-in. '
             'Earlier descriptive work used the whole tape, so this is a chronological fitting holdout rather than a blind validation.','',
             '| Metric | Original validation | Simulated mean | Between-seed SD | Relative error |',
             '| --- | ---: | ---: | ---: | ---: |']
    for row in comparison:
        lines.append(f'| {row["metric"]} | {row["true"]:.5g} | {row["simulated_mean"]:.5g} | {row["simulated_sd"]:.4g} | {100*row["relative_error"]:+.1f}% |')
    lines += ['', '## Interpretation','',
              'Arrival and size marginals alone are insufficient. Pay particular attention to volatility, '
              'sweep depth and maker markouts. If these disagree materially, synthetic strategy profitability '
              'is not trustworthy even when message rates and volumes match. Between-seed SD is not a '
              'confidence interval for model correctness. KS and Wasserstein distances are descriptive; '
              'dependent market observations do not support naive iid KS p-values.','',
              'The latent fair-price process, state-dependent cancellation hazard, and cross-order cancellation '
              'dependence are not identified by the present model. The markout statistic uses 100 ms sampled '
              'midpoints and averages per aggressive order, consistently for both tapes. One-sided gaps are '
              'not forward-filled. The full synthetic tape is seed 20261107, selected in advance, not the best seed.','',
              '## Files','',
              '- `synthetic_round1.csv`: full 12000-second tape in the repository simulator format.',
              '- `comparison.html`: interactive figures; use a local HTTP server to open as a page.',
              '- `comparison.csv`, `replicate_statistics.csv`, `outcome.json`: numeric results.',
              '- `training_marks.npz`, `parameters.json`, `locked_model.json`: reproducible fitted model.',
              '- Run `calibrate.py` then `compare.py` using the repo .venv Python.', '']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('COMPLETE',json.dumps(dict(selected_conditioned=model.conditioned,comparison=comparison)),flush=True)


if __name__=='__main__':
    main()
