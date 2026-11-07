"""Reproducible arrival-process diagnostics for Round 1; no repository edits.

Run from the repo with .venv/Scripts/python.exe -B strategies/adam/poisson_round1.py.
All plots/results are saved beside this script under poisson_round1_results/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
SOURCE = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').exists() and (p / 'simulator').is_dir()) / 'data' / 'round1.csv'
OUT = HERE / 'poisson_round1_results'
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20261107)


def ks_exp(x):
    x = np.sort(x)
    n = len(x)
    cdf = -np.expm1(-x / x.mean())
    return float(max(np.max(np.arange(1, n+1)/n - cdf),
                     np.max(cdf - np.arange(n)/n)))


def calibrated_gap_test(gaps_us, size=20_000, simulations=499):
    """Fit-aware parametric bootstrap, including microsecond timestamp rounding.

    Random subsampling limits trivial rejection from millions of observations.
    Full-sample effect sizes and chronological dependence are reported separately.
    The bootstrap tests the iid exponential clock null, not an arbitrary NHPP.
    """
    n = min(size, len(gaps_us))
    x = (gaps_us if n == len(gaps_us) else RNG.choice(gaps_us, n, replace=False)).astype(float)
    observed = ks_exp(x)
    extreme = 0
    for _ in range(simulations):
        continuous = RNG.exponential(x.mean(), n+1)
        rounded = np.rint(np.cumsum(continuous) + RNG.random())
        simulated = np.diff(rounded)
        extreme += ks_exp(simulated) >= observed
    return dict(n=n, simulations=simulations, D=observed,
                p=(extreme+1)/(simulations+1), mean_gap_us=float(x.mean()))


def corr_at(x, lag):
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def rate_permutation(counts, bin_s, block_s=5, simulations=999):
    """Block-permutation check for slow rate variation, preserving local clustering.

    Conditional/approximate: exchangeable stationary blocks under the null;
    valid interpretation requires dependence short relative to the block length.
    This is intentionally separate from the joint homogeneous-Poisson test.
    """
    n = len(counts) // bin_s * bin_s
    counts = counts[:n]
    blocks = counts.reshape(-1, block_s)
    grouped = counts.reshape(-1, bin_s).sum(axis=1)
    mean = grouped.mean()
    statistic = float(np.mean((grouped-mean)**2))
    extreme = 0
    for _ in range(simulations):
        permuted = blocks[RNG.permutation(len(blocks))].reshape(-1)
        new = permuted.reshape(-1, bin_s).sum(axis=1)
        extreme += np.mean((new-mean)**2) >= statistic
    return dict(bin_s=bin_s, block_s=block_s, simulations=simulations,
                p=(extreme+1)/(simulations+1), min_rate=float(grouped.min()/bin_s),
                max_rate=float(grouped.max()/bin_s), mean_rate=float(grouped.mean()/bin_s),
                relative_rate_sd=float(grouped.std(ddof=1)/grouped.mean()))


def analyze(name, times, duration_us):
    gaps = np.diff(times)
    rate = len(times) / (duration_us/1e6)
    n_seconds = duration_us // 1_000_000
    counts = np.bincount((times//1_000_000).astype(int), minlength=n_seconds)[:n_seconds]
    scales = []
    for width_us in (10_000, 100_000, 1_000_000, 10_000_000, 60_000_000):
        n = duration_us // width_us
        x = times[times < n * width_us]
        bins = np.bincount((x // width_us).astype(int), minlength=n)
        fano = float(bins.var(ddof=1) / bins.mean())
        test_stat = (len(bins)-1) * fano
        p = min(1.0, 2*min(stats.chi2.cdf(test_stat, len(bins)-1), stats.chi2.sf(test_stat, len(bins)-1)))
        scales.append(dict(bin_s=width_us/1e6, bins=len(bins), mean=float(bins.mean()), fano=fano,
                           dispersion_p_approx=float(p), count_acf1=corr_at(bins, 1)))
    bootstrap = calibrated_gap_test(gaps, size=len(gaps) if name == 'IOC inserts' else 20_000)
    count_acfs = np.array([corr_at(counts, k) for k in range(1,21)])
    lb = len(counts) * (len(counts)+2) * np.sum(count_acfs**2/(len(counts)-np.arange(1,21)))
    normalized = gaps.astype(float) / gaps.mean()
    quantiles = [.01, .1, .25, .5, .75, .9, .99, .999]
    summary = dict(name=name, events=len(times), rate_per_s=rate, gap_mean_us=float(gaps.mean()),
                   gap_cv=float(gaps.std(ddof=1)/gaps.mean()), zero_gap_count=int((gaps==0).sum()),
                   full_gap_KS_D=ks_exp(gaps.astype(float)),
                   gap_acf={str(lag): corr_at(gaps, lag) for lag in (1,2,5,10,50,100)},
                   counts_ljung_box_lags20=dict(Q=float(lb),p=float(stats.chi2.sf(lb,20))),
                   bootstrap=bootstrap, scales=scales,
                   homogeneity=[rate_permutation(counts, k) for k in (60,300,600)],
                   gap_qq=dict(theoretical=(-np.log1p(-np.array(quantiles))).tolist(),
                               observed=np.quantile(normalized, quantiles).tolist()))
    series = pd.DataFrame({'second': np.arange(len(counts)), 'events': counts})
    series.to_csv(OUT / (name.lower().replace(' ', '_') + '_counts_1s.csv'), index=False)
    print(json.dumps(summary), flush=True)
    return summary, counts


def main():
    raw = pd.read_csv(SOURCE, usecols=['Time','Operation','Lifespan','OrderId','Side'],
                      dtype={'Operation':'category','Lifespan':'category','Side':'category','OrderId':'int64'})
    times = np.rint(raw.Time.to_numpy() * 1e6).astype('int64')
    assert np.all(np.diff(times) >= 0)
    # Natural session boundary inferred from timestamps; report exact observed range too.
    duration_us = int(np.ceil(times[-1]/1e6) * 1e6)
    insert = (raw.Operation == 'Insert').to_numpy()
    cancel = (raw.Operation == 'Cancel').to_numpy()
    limit = insert & (raw.Lifespan == 'G').to_numpy()
    ioc = insert & (raw.Lifespan == 'F').to_numpy()
    streams = {'All events': np.ones(len(times), dtype=bool), 'Limit inserts':limit,
               'IOC inserts':ioc, 'Cancellations':cancel}
    summaries, counts = {}, {}
    for name, mask in streams.items():
        summaries[name], counts[name] = analyze(name, times[mask], duration_us)
    ids = raw.OrderId.to_numpy()
    insert_ids = ids[insert]
    assert np.all(np.diff(insert_ids) > 0), 'Need a general ID join for this tape'
    indices = np.searchsorted(insert_ids, ids[cancel])
    valid = indices < len(insert_ids)
    valid[valid] &= insert_ids[indices[valid]] == ids[cancel][valid]
    birth = times[insert][indices[valid]]
    ages = (times[cancel][valid]-birth)/1e6
    order_was_limit = limit[insert][indices[valid]]
    lifetimes = dict(matched_cancels=int(valid.sum()), unmatched=int((~valid).sum()),
                     all_matched_to_limit=bool(order_was_limit.all()),
                     negative_lifetimes=int((ages<0).sum()), mean_s=float(ages.mean()),
                     cv=float(ages.std(ddof=1)/ages.mean()),
                     quantile_s={str(q):float(np.quantile(ages,q)) for q in (0,.01,.1,.25,.5,.75,.9,.99,1)},
                     corr_limit_cancel_1s=float(np.corrcoef(counts['Limit inserts'], counts['Cancellations'])[0,1]),
                     corr_limit_ioc_1s=float(np.corrcoef(counts['Limit inserts'], counts['IOC inserts'])[0,1]),
                     corr_limit_cancel_60s=float(np.corrcoef(counts['Limit inserts'].reshape(-1,60).sum(axis=1),
                                                              counts['Cancellations'].reshape(-1,60).sum(axis=1))[0,1]))
    # Mark thinning and local stationarity: buy/sell splits and six subperiod gap CVs.
    marks = {}
    for label, mask in [('Limit inserts',limit),('IOC inserts',ioc)]:
        t = times[mask]
        buy = (raw.Side.to_numpy()[mask] == 'B').astype(float)
        segment_cv = []
        for start in np.linspace(0,duration_us,7)[:-1]:
            g = np.diff(t[(t>=start)&(t<start+duration_us/6)])
            segment_cv.append(float(g.std(ddof=1)/g.mean()))
        marks[label] = dict(buy_fraction=float(buy.mean()), side_acf1=corr_at(buy,1),
                            gap_CV_six_segments=segment_cv)
    results = dict(source=str(SOURCE), seed=20261107, first_s=float(times[0]/1e6),
                   last_s=float(times[-1]/1e6), assumed_observation_window_s=[0,duration_us/1e6],
                   streams=summaries, cancellations=lifetimes, marks=marks)
    # Familywise adjustment of the four planned marginal gap tests.
    ranked = sorted((s['bootstrap']['p'],name) for name,s in summaries.items())
    last = 0.0
    for rank,(p,name) in enumerate(ranked):
        last = max(last,min(1.0,p*(len(ranked)-rank)))
        summaries[name]['bootstrap']['holm_p_four_streams'] = last
    (OUT/'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print('CANCELLATION_LINK',json.dumps(lifetimes),flush=True)
    print('MARKS',json.dumps(marks),flush=True)

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        'Arrival rates in complete 60-second windows', 'Count dispersion: variance / mean',
        'Interarrival Q-Q against fitted exponential', 'Limit-insert vs cancellation counts (60 seconds)'))
    colors = ['#2563eb','#08916a','#d97706','#8b5cf6']
    for (name,s), color in zip(summaries.items(),colors):
        c = counts[name].reshape(-1,60).sum(axis=1)
        fig.add_trace(go.Scatter(x=np.arange(len(c))*60+30,y=c/60,name=name,
                                 legendgroup=name,line=dict(color=color)),row=1,col=1)
        fig.add_trace(go.Scatter(x=[x['bin_s'] for x in s['scales']],y=[x['fano'] for x in s['scales']],
                                 name=name,legendgroup=name,showlegend=False,mode='lines+markers',
                                 line=dict(color=color)),row=1,col=2)
        fig.add_trace(go.Scatter(x=s['gap_qq']['theoretical'],y=s['gap_qq']['observed'],name=name,
                                 legendgroup=name,showlegend=False,mode='lines+markers',line=dict(color=color)),row=2,col=1)
    fig.add_hline(y=1,line_dash='dash',line_color='#64748b',row=1,col=2)
    fig.add_trace(go.Scatter(x=[0,7],y=[0,7],mode='lines',name='Exponential reference',
                             line=dict(dash='dash',color='#64748b')),row=2,col=1)
    x = counts['Limit inserts'].reshape(-1,60).sum(axis=1)
    y = counts['Cancellations'].reshape(-1,60).sum(axis=1)
    fig.add_trace(go.Scatter(x=x,y=y,mode='markers',name='Matched clock counts',marker=dict(color='#08916a',size=6)),row=2,col=2)
    fig.update_xaxes(title_text='Elapsed seconds',row=1,col=1)
    fig.update_yaxes(title_text='Events per second',row=1,col=1)
    fig.update_xaxes(title_text='Bin width (seconds)',type='log',row=1,col=2)
    fig.update_yaxes(title_text='Fano factor; Poisson = 1',row=1,col=2)
    fig.update_xaxes(title_text='Theoretical normalized gap',row=2,col=1)
    fig.update_yaxes(title_text='Observed gap / mean gap',row=2,col=1)
    fig.update_xaxes(title_text='Limit inserts',row=2,col=2)
    fig.update_yaxes(title_text='Cancellations',row=2,col=2)
    fig.update_layout(template='plotly_white',height=950,title='Round 1: Poisson arrival diagnostics',
                      legend=dict(orientation='h',y=-.12),margin=dict(b=110))
    header = '<h1>Round 1 arrival-process diagnostics</h1><p>Hover to inspect. The aggregate can have a stable mean rate without independent Poisson increments. Full statistics and test assumptions are in results.json.</p>'
    (OUT/'diagnostics.html').write_text(header + fig.to_html(full_html=False,include_plotlyjs=True),encoding='utf-8')


if __name__ == '__main__':
    main()
