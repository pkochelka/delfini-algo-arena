"""Write a readable report and sensitivity checks from the arrival diagnostics."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import poisson_round1 as P

root = Path(__file__).parent / 'poisson_round1_results'
r = json.loads((root / 'results.json').read_text())
checks = {}
for name in r['streams']:
    frame = pd.read_csv(root / (name.lower().replace(' ', '_') + '_counts_1s.csv'))
    c = frame.events.to_numpy()
    trimmed = c[60:-60]
    checks[name] = dict(trimmed_rate=float(trimmed.mean()),
                        trimmed_fano_1s=float(trimmed.var(ddof=1)/trimmed.mean()),
                        rate_checks=[P.rate_permutation(c, 300, block_s=k, simulations=499) for k in (10,30)])
raw = pd.read_csv(r['source'], usecols=['Operation', 'OrderId'], dtype={'Operation':'category','OrderId':'int64'})
cancel_ids = raw.loc[raw.Operation == 'Cancel', 'OrderId'].to_numpy()
cancel_unique = len(np.unique(cancel_ids))
assert cancel_unique == len(cancel_ids)
supplement = dict(edge_and_block_sensitivity=checks, unique_cancel_ids=cancel_unique)
(root/'sensitivity.json').write_text(json.dumps(supplement, indent=2))

def fmt_p(p):
    return f'{p:.3g}' if p else '< floating-point resolution'

lines = ['# Round 1: are the events Poisson?', '',
         '**Conclusion:** a single homogeneous Poisson clock is a poor model for all CSV rows. '
         'The limit-insert and IOC-insert streams individually fit stationary Poisson clocks well '
         'at ordinary trading time scales. Cancellations are not an independent Poisson clock. '
         'There is no detected slow variation in the marginal mean arrival rates.', '',
         '## Data and observation interval', '',
         f'- Source: `{r["source"]}`',
         f'- Events: {r["streams"]["All events"]["events"]:,}; first timestamp {r["first_s"]:.6f}s, last {r["last_s"]:.6f}s.',
         '- Window used: [0, 12000) seconds (3h 20m). The natural end is inferred by rounding the last timestamp up to the next second.',
         '- Timestamps are sorted, integer microseconds, and strictly distinct. No batching at identical timestamps was observed.',
         '- Edge sensitivity trims the first and last 60 seconds. Findings remain materially unchanged.', '',
         '## Results', '',
         '| Stream | Events | Rate / s | Gap CV | Fano 1 s | Fano 10 s | Exponential-gap bootstrap p | Holm p (4 streams) |',
         '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for name,s in r['streams'].items():
    f = {v['bin_s']:v['fano'] for v in s['scales']}
    lines.append(f'| {name} | {s["events"]:,} | {s["rate_per_s"]:.5f} | {s["gap_cv"]:.5f} | {f[1]:.5f} | {f[10]:.5f} | {fmt_p(s["bootstrap"]["p"])} | {fmt_p(s["bootstrap"]["holm_p_four_streams"])} |')
lines += ['', 'Poisson reference: independent exponential gaps have CV 1, and independent equal-width '
          'Poisson counts have variance/mean (Fano factor) 1. Aggregate 10-second count dispersion '
          'rejects the joint homogeneous-Poisson null (approximate two-sided p = '
          f'{r["streams"]["All events"]["scales"][3]["dispersion_p_approx"]:.3g}). '
          'No rejection establishes a model; it means these diagnostics did not detect a departure.', '',
          '## Constant mean rate versus independent arrivals', '',
          'These are separate questions. A stationary clustered process can have a constant mean rate '
          'while violating Poisson independence. Under a genuinely inhomogeneous Poisson process, '
          'the rate may vary deterministically with time; independence is still required.', '',
          '| Stream | 60 s rate-check p | 300 s p | 600 s p | 1 s count Ljung-Box p (20 lags) |',
          '| --- | ---: | ---: | ---: | ---: |']
for name,s in r['streams'].items():
    ps = [fmt_p(x['p']) for x in s['homogeneity']]
    lines.append(f'| {name} | {" | ".join(ps)} | {fmt_p(s["counts_ljung_box_lags20"]["p"])} |')
lines += ['', 'The rate checks randomly permute 5-second blocks of observed 1-second counts and '
          'compare variation across 60-, 300-, and 600-second windows. They preserve local clustering '
          'instead of incorrectly treating overdispersion itself as proof of time-varying intensity. '
          'Their null assumes approximately exchangeable stationary blocks; long dependence can '
          'invalidate this approximation. Additional 10- and 30-second block sensitivity checks '
          'are in sensitivity.json. These do not exclude state-dependent intensity or subtle/high-frequency rate modulation.', '',
          '## Why cancellations matter', '',
          f'- All {r["cancellations"]["matched_cancels"]:,} cancellation rows match an earlier G insert by order ID; there are no duplicate cancellation IDs.',
          f'- There are {r["streams"]["Limit inserts"]["events"] - cancel_unique} limit inserts without a cancellation in the observed window.',
          f'- Limit-insert/cancellation count correlation is {r["cancellations"]["corr_limit_cancel_1s"]:.4f} in 1-second bins and {r["cancellations"]["corr_limit_cancel_60s"]:.4f} in 60-second bins.',
          f'- Time from insertion to the cancellation row: mean {r["cancellations"]["mean_s"]:.6f}s, median {r["cancellations"]["quantile_s"]["0.5"]:.6f}s, 99th percentile {r["cancellations"]["quantile_s"]["0.99"]:.6f}s.',
          '- These are cancellation-message lifetimes, not necessarily the time an order remains live in the book: it may have filled before its cancellation row.',
          '- A birth plus a later linked cancellation contributes roughly two messages over long windows, naturally explaining aggregate Fano factors near 2. Do not generate cancels as an unrelated 180/s clock.', '',
          '## Starting model for reconstruction (inference, not recovered source code)', '',
          '1. Generate G inserts from a constant-rate clock near 180/s.',
          '2. Generate IOC inserts from a constant-rate clock near 8/s.',
          '3. Couple cancellations to the earlier G orders, using a separately fitted lifetime/state-dependent mechanism.',
          '4. Merge the streams and match the timestamp-uniquification/rounding convention.', '',
          'If a homogeneous-Poisson model is assumed, exact count-based 95% rate intervals are:', '']
for name in ('Limit inserts','IOC inserts'):
    s = r['streams'][name]
    n = s['events']
    lo = stats.chi2.ppf(.025,2*n)/24000
    hi = stats.chi2.ppf(.975,2*(n+1))/24000
    lines.append(f'- {name}: {lo:.5f} to {hi:.5f} events/s.')
lines += ['', 'The two exogenous clocks have not been proven jointly independent: their 1-second '
          f'count correlation is {r["cancellations"]["corr_limit_ioc_1s"]:.5f}. '
          'The round-number rates 180 and 8 fall within the model-based intervals, making them '
          'reasonable reconstruction candidates rather than known generator constants.', '',
          '## Test calibration and limitations', '',
          '- Gap tests refit the exponential mean in each of 499 parametric bootstrap samples. '
          'Simulations include cumulative-timestamp rounding to microseconds. The IOC test uses '
          'all 96,087 gaps; other streams use reproducible random samples of 20,000 gaps. '
          'The smallest attainable p is 0.002; that value is a Monte Carlo floor, not an exact p.',
          '- All-event and cancellation samples are dependent under the alternative; rejection '
          'is evidence against the joint iid exponential null, not a diagnosis of its cause. '
          'Full-sample KS distances, count dispersion and chronological correlations are also saved.',
          '- Four planned gap tests are adjusted with Holm correction. Rate checks and autocorrelations '
          'are exploratory diagnostics; interpret isolated marginal p-values cautiously.',
          '- There are zero identical-timestamp gaps. A rounded continuous Poisson process can produce '
          'ties, so microsecond-level reconstruction may need a minimum-gap or tie-resolution rule. '
          'The clock fit is an engineering approximation above that resolution, not proof of exact continuous-time Poisson generation.',
          '- One realization cannot prove homogeneity or identify a unique generator. Marginal '
          'stationarity does not establish a constant history-conditional intensity. Cancellation '
          'lifetime dependence on price, size, book state, and market moves is a next investigation.', '',
          '## Files and reproduction', '',
          '- `diagnostics.html`: interactive rate, dispersion, gap Q-Q and insertion/cancellation plots.',
          '- `results.json`: all primary statistics and seeds.',
          '- `sensitivity.json`: edge trimming, block-length sensitivity, cancellation ID uniqueness.',
          '- `*_counts_1s.csv`: per-stream event counts.',
          '- From the repo run `.venv/Scripts/python.exe -B strategies/adam/poisson_round1.py`, '
          'then `.venv/Scripts/python.exe -B strategies/adam/summarize_poisson_round1.py`.', '',
          '## Method references', '',
          '- NIST, Homogeneous Poisson Process: https://itl.nist.gov/div898/handbook/apr/section1/apr171.htm',
          '- NIST, Poisson Dispersion Test: https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/poisdisp.htm',
          '- Brown et al., Time-Rescaling Theorem: https://www.stat.cmu.edu/~kass/papers/rescaling.pdf', '']
(root/'report.md').write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps(supplement, indent=2))
print('Wrote',root/'report.md')
