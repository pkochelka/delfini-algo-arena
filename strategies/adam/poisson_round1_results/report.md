# Round 1: are the events Poisson?

**Conclusion:** a single homogeneous Poisson clock is a poor model for all CSV rows. The limit-insert and IOC-insert streams individually fit stationary Poisson clocks well at ordinary trading time scales. Cancellations are not an independent Poisson clock. There is no detected slow variation in the marginal mean arrival rates.

## Data and observation interval

- Source: `C:\Users\adam\Documents\Codex\2026-11-07\new-chat-2\delfini-algo-arena\data\round1.csv`
- Events: 4,411,336; first timestamp 0.008661s, last 11999.999025s.
- Window used: [0, 12000) seconds (3h 20m). The natural end is inferred by rounding the last timestamp up to the next second.
- Timestamps are sorted, integer microseconds, and strictly distinct. No batching at identical timestamps was observed.
- Edge sensitivity trims the first and last 60 seconds. Findings remain materially unchanged.

## Results

| Stream | Events | Rate / s | Gap CV | Fano 1 s | Fano 10 s | Exponential-gap bootstrap p | Holm p (4 streams) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All events | 4,411,336 | 367.61133 | 1.07741 | 1.78960 | 1.97583 | 0.002 | 0.008 |
| Limit inserts | 2,157,657 | 179.80475 | 1.00035 | 0.99301 | 1.01581 | 0.712 | 0.712 |
| IOC inserts | 96,088 | 8.00733 | 1.00261 | 0.99516 | 1.04137 | 0.194 | 0.388 |
| Cancellations | 2,157,591 | 179.79925 | 1.07755 | 1.34924 | 1.05362 | 0.002 | 0.008 |

Poisson reference: independent exponential gaps have CV 1, and independent equal-width Poisson counts have variance/mean (Fano factor) 1. Aggregate 10-second count dispersion rejects the joint homogeneous-Poisson null (approximate two-sided p = 5.72e-79). No rejection establishes a model; it means these diagnostics did not detect a departure.

## Constant mean rate versus independent arrivals

These are separate questions. A stationary clustered process can have a constant mean rate while violating Poisson independence. Under a genuinely inhomogeneous Poisson process, the rate may vary deterministically with time; independence is still required.

| Stream | 60 s rate-check p | 300 s p | 600 s p | 1 s count Ljung-Box p (20 lags) |
| --- | ---: | ---: | ---: | ---: |
| All events | 0.915 | 0.078 | 0.828 | 0.0493 |
| Limit inserts | 0.945 | 0.069 | 0.798 | 0.71 |
| IOC inserts | 0.706 | 0.81 | 0.984 | 0.431 |
| Cancellations | 0.965 | 0.089 | 0.864 | 1.94e-39 |

The rate checks randomly permute 5-second blocks of observed 1-second counts and compare variation across 60-, 300-, and 600-second windows. They preserve local clustering instead of incorrectly treating overdispersion itself as proof of time-varying intensity. Their null assumes approximately exchangeable stationary blocks; long dependence can invalidate this approximation. Additional 10- and 30-second block sensitivity checks are in sensitivity.json. These do not exclude state-dependent intensity or subtle/high-frequency rate modulation.

## Why cancellations matter

- All 2,157,591 cancellation rows match an earlier G insert by order ID; there are no duplicate cancellation IDs.
- There are 66 limit inserts without a cancellation in the observed window.
- Limit-insert/cancellation count correlation is 0.5075 in 1-second bins and 0.9856 in 60-second bins.
- Time from insertion to the cancellation row: mean 0.467237s, median 0.336615s, 99th percentile 2.128236s.
- These are cancellation-message lifetimes, not necessarily the time an order remains live in the book: it may have filled before its cancellation row.
- A birth plus a later linked cancellation contributes roughly two messages over long windows, naturally explaining aggregate Fano factors near 2. Do not generate cancels as an unrelated 180/s clock.

## Starting model for reconstruction (inference, not recovered source code)

1. Generate G inserts from a constant-rate clock near 180/s.
2. Generate IOC inserts from a constant-rate clock near 8/s.
3. Couple cancellations to the earlier G orders, using a separately fitted lifetime/state-dependent mechanism.
4. Merge the streams and match the timestamp-uniquification/rounding convention.

If a homogeneous-Poisson model is assumed, exact count-based 95% rate intervals are:

- Limit inserts: 179.56491 to 180.04483 events/s.
- IOC inserts: 7.95678 to 8.05813 events/s.

The two exogenous clocks have not been proven jointly independent: their 1-second count correlation is 0.01845. The round-number rates 180 and 8 fall within the model-based intervals, making them reasonable reconstruction candidates rather than known generator constants.

## Test calibration and limitations

- Gap tests refit the exponential mean in each of 499 parametric bootstrap samples. Simulations include cumulative-timestamp rounding to microseconds. The IOC test uses all 96,087 gaps; other streams use reproducible random samples of 20,000 gaps. The smallest attainable p is 0.002; that value is a Monte Carlo floor, not an exact p.
- All-event and cancellation samples are dependent under the alternative; rejection is evidence against the joint iid exponential null, not a diagnosis of its cause. Full-sample KS distances, count dispersion and chronological correlations are also saved.
- Four planned gap tests are adjusted with Holm correction. Rate checks and autocorrelations are exploratory diagnostics; interpret isolated marginal p-values cautiously.
- There are zero identical-timestamp gaps. A rounded continuous Poisson process can produce ties, so microsecond-level reconstruction may need a minimum-gap or tie-resolution rule. The clock fit is an engineering approximation above that resolution, not proof of exact continuous-time Poisson generation.
- One realization cannot prove homogeneity or identify a unique generator. Marginal stationarity does not establish a constant history-conditional intensity. Cancellation lifetime dependence on price, size, book state, and market moves is a next investigation.

## Files and reproduction

- `diagnostics.html`: interactive rate, dispersion, gap Q-Q and insertion/cancellation plots.
- `results.json`: all primary statistics and seeds.
- `sensitivity.json`: edge trimming, block-length sensitivity, cancellation ID uniqueness.
- `*_counts_1s.csv`: per-stream event counts.
- From the repo run `.venv/Scripts/python.exe -B strategies/adam/poisson_round1.py`, then `.venv/Scripts/python.exe -B strategies/adam/summarize_poisson_round1.py`.

## Method references

- NIST, Homogeneous Poisson Process: https://itl.nist.gov/div898/handbook/apr/section1/apr171.htm
- NIST, Poisson Dispersion Test: https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/poisdisp.htm
- Brown et al., Time-Rescaling Theorem: https://www.stat.cmu.edu/~kass/papers/rescaling.pdf
