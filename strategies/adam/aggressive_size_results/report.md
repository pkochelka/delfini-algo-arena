# Round 1 aggressive order sizes: lognormal versus Pareto

**Finding:** the full submitted IOC size distribution is well described by a rounded lognormal, not a pure power law. A Pareto-like approximation is statistically plausible only for the top roughly 1.1% of submissions. Executed sizes have a thinner tail and reject the fitted pure power-law tail at the tested cutoffs.

## What is being counted?

- 96,088 IOC submissions; 94,697 execute at least once and 1,391 execute nothing.
- No G inserts crossed the market in the replay. Thus executed aggressors are IOC orders in this tape.
- Submitted size is incoming Volume. Executed size is summed across all matches of the incoming order. A sweep across many resting orders counts once, not once per fill.
- The full IOC sample, including unfilled submissions, is the appropriate starting point for reconstructing the size generator.
- Source SHA256: `362e8208be1eeec32ad190d097774e26448f0d5149575a8e1b1efd1db1735940`.

## Distribution summary (lots)

| Statistic | IOC submitted | Executed, conditional on a fill |
| --- | ---: | ---: |
| Minimum | 1.00 | 1.00 |
| Median | 61.00 | 60.00 |
| 90th percentile | 360.00 | 345.00 |
| 95th percentile | 603.00 | 556.00 |
| 99th percentile | 1,543.26 | 1,215.00 |
| 99.9th percentile | 4,630.17 | 2,504.13 |
| Maximum | 23,224.00 | 6,716.00 |
| Mean | 160.674 | 142.000 |

## Complete submitted-size fit

The best full-distribution model among the compared families is:

```text
X = exp(Normal(mu=4.105759, sigma=1.395941))
Q = max(1, floor(X + 0.5))
```

The parameters describe the natural logarithm of the latent continuous size, not the mean and standard deviation of Q. Nearest rounding wins over floor/ceil rounding on the same data. Integer-bin probabilities and clipping at one are included in the likelihood.

Full rounded-lognormal KS distance: 0.001862; 199-replicate refitted bootstrap p = 0.500. This is conditional on the selected rounding scheme and is not proof of a unique generator.

| Full-distribution model | Delta AIC versus best |
| --- | ---: |
| Lognormal offset 0.5 | 0.00 |
| Lognormal offset 0 | 101.52 |
| Lognormal offset 1 | 202.30 |
| Rounded Lomax (Pareto II) | 3,742.41 |
| Geometric (exponential) | 41,979.14 |
| Discrete power law | 151,424.15 |

All full-distribution models use the same 96,088 integer observations and normalized integer probabilities. Lower AIC is better. The Pareto II (Lomax) alternative is included because it can describe a body plus power-law tail; it also loses decisively here.

Buy/sell lognormal fits are similar, as are four chronological quarters; their fitted parameters are in results.json. A sensible reconstruction candidate is therefore `mu=4.1, sigma=1.4` with nearest rounding and a minimum of one lot. This is inference, not recovered generator code.

## Can the upper tail be Pareto?

- Selected lower cutoff: **1,465 lots**.
- Tail observations: **1,070**, or **1.114%** of submissions.
- Exact discrete power-law PMF exponent: **alpha=3.05771**.
- Corresponding asymptotic survival/Pareto exponent: **beta=alpha-1=2.05771**.
- KS distance: 0.01967; bootstrap goodness-of-fit p = **0.405**.

The discrete model is `P(Q=k | Q>=xmin) = k^(-alpha) / zeta(alpha,xmin)`. For an intuitive continuous approximation, `P(Q>=q | Q>=xmin) ~ (q/xmin)^(-beta)`. It is therefore important not to confuse the PMF exponent alpha with the survival exponent beta.

The tail fit is not rejected. However, it covers only about 1.2 decades of observed size and discards almost 99% of the data. A truncated lognormal is also plausible on this same support: its log likelihood exceeds the power law by only 2.139 with one extra parameter. This is weak evidence for choosing between those two tail families, not evidence that the whole distribution is Pareto.

A separate chronological 75/25 check selects its cutoff, rounding scheme and all parameters using training orders only. The lognormal has a small held-out tail log-score advantage of 4.105 across 746 held-out tail orders (training cutoff 776). This advantage is descriptive; no significance claim is made from it.

## Submitted versus executed

The best tested pure power-law fit to executed quantities starts at 849 lots, with alpha=3.6116, but is rejected (bootstrap p=0.005, the Monte Carlo floor). The maximum executed quantity is 6,716 versus 23,224 submitted. Available liquidity limits execution, and IOC residuals expire; do not estimate the submitted-size generator by treating executed sizes as uncensored draws from it.

## Method and limitations

- Follow the likelihood/KS/bootstrap approach of Clauset, Shalizi and Newman rather than a straight-line regression on a log-log histogram: https://arxiv.org/abs/0706.1062.
- Power-law likelihoods use exact Hurwitz-zeta normalization for integer sizes. The lower cutoff minimizes KS on a reproducible quantile grid, with at least 1,000 tail observations. It is not an exhaustive search over every integer or over very tiny tails.
- Each of 199 semiparametric bootstrap replicates resamples the empirical body, simulates an exact discrete power-law tail, and reselects the cutoff and exponent using the same rule. Resolution is 0.005. Tail p about 0.405 has Monte Carlo standard error about 0.035; a p of 0.005 is a simulation floor, not an exact probability.
- Bootstrap tests assume independent order sizes. Log-size autocorrelations and buy/sell/quarter fits are recorded as checks, but these cannot exclude every dependence.
- A successful fit does not establish the source distribution uniquely or guarantee that extreme-size extrapolation is safe. A lognormal can resemble a Pareto over a limited upper range.
- The research streaming matcher was checked against the repository replay on partial fills, cancellations and unfilled IOCs. Counts and quantiles also match the full Round 1 replay.

## Reproduction and files

- `../aggressive_size_fits.py`: matching, fits, bootstrap and plots (seed 20261107).
- `../summarize_aggressive_sizes.py`: report and training-only holdout verification.
- `../test_size_fit_methods.py`: independent matcher/sampler/MLE checks.
- `diagnostics.html`: interactive empirical/model survival plots and threshold sensitivity.
- `results.json`: parameters and all test results.
- `ioc_order_sizes.csv`: one row per IOC, including submitted and executed quantities.
- Run from the repo with `.venv/Scripts/python.exe -B strategies/adam/aggressive_size_fits.py` and then `.venv/Scripts/python.exe -B strategies/adam/summarize_aggressive_sizes.py`.
