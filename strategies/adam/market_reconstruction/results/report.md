# Synthetic Round 1 reconstruction

A semi-parametric marked-Poisson order-book simulator, not recovered original source code.
Fresh exponential G/IOC clocks and random sides; G volume/placement/cancellation-lifetime triplets resampled from training cohorts, optionally conditioned on spread and own-side touch imbalance; IOC sizes generated from a fitted rounded lognormal; IOC price reach conditioned on submitted-size bucket. Each G order receives a cancellation message, including after it has filled. FIFO matching and distinct microsecond timestamps are enforced. No original event-time or price path is replayed.

Calibration uses 30-7200 seconds, model selection uses 7200-9600 seconds, and the following comparison uses 9600-12000 seconds. Five fresh 2400-second simulations follow a 30-second burn-in. Earlier descriptive work used the whole tape, so this is a chronological fitting holdout rather than a blind validation.

| Metric | Original validation | Simulated mean | Between-seed SD | Relative error |
| --- | ---: | ---: | ---: | ---: |
| events_per_s | 367.68 | 367.78 | 0.8822 | +0.0% |
| limit_per_s | 179.85 | 179.87 | 0.4329 | +0.0% |
| ioc_per_s | 7.9792 | 8.0372 | 0.05217 | +0.7% |
| cancels_per_s | 179.85 | 179.87 | 0.4311 | +0.0% |
| mean_limit_size | 71.918 | 72.104 | 0.6121 | +0.3% |
| mean_ioc_size | 157.89 | 161.03 | 3.651 | +2.0% |
| ioc_size_p99 | 1529 | 1537.3 | 57.34 | +0.5% |
| ioc_fill_probability | 0.98611 | 0.98488 | 0.001008 | -0.1% |
| executed_size_mean | 140.86 | 138.75 | 2.496 | -1.5% |
| executed_size_p99 | 1217.2 | 1188.6 | 34.8 | -2.3% |
| traded_lots_per_s | 1108.3 | 1098.2 | 13.38 | -0.9% |
| mean_sweep_ticks | 0.093307 | 0.035272 | 0.001739 | -62.2% |
| spread_mean | 1.07 | 1.1962 | 0.03017 | +11.8% |
| spread_median | 1 | 1 | 0 | +0.0% |
| spread_one_tick_share | 0.93104 | 0.90875 | 0.008599 | -2.4% |
| one_sided_share | 8.3333e-05 | 0.000125 | 9.317e-05 | +0.4% |
| mean_touch_quantity | 1325.4 | 1393.6 | 17.07 | +5.1% |
| mean_depth_5ticks | 2872.5 | 2760.6 | 22 | -3.9% |
| returns_acf_100ms | -0.0015637 | -0.052864 | 0.01064 | -513.0% |
| returns_acf_1s | 0.011274 | -0.047407 | 0.0117 | -520.5% |
| ioc_side_acf | 0.0097913 | -0.0017901 | 0.004253 | -115.8% |
| sigma_100ms | 0.89934 | 0.55646 | 0.009222 | -38.1% |
| sigma_1s | 0.9063 | 0.50075 | 0.01861 | -44.7% |
| sigma_10s | 0.90973 | 0.47063 | 0.021 | -48.3% |
| fano_100ms | 1.2654 | 1.0637 | 0.004826 | -15.9% |
| fano_1s | 1.7381 | 1.6205 | 0.04471 | -6.8% |
| fano_10s | 1.8479 | 2.0293 | 0.1768 | +9.8% |
| maker_markout_1s | 0.31647 | 0.5718 | 0.01148 | +80.7% |

## Interpretation

Arrival and size marginals alone are insufficient. Pay particular attention to volatility, sweep depth and maker markouts. If these disagree materially, synthetic strategy profitability is not trustworthy even when message rates and volumes match. Between-seed SD is not a confidence interval for model correctness. KS and Wasserstein distances are descriptive; dependent market observations do not support naive iid KS p-values.

The latent fair-price process, state-dependent cancellation hazard, and cross-order cancellation dependence are not identified by the present model. The markout statistic uses 100 ms sampled midpoints and averages per aggressive order, consistently for both tapes. One-sided gaps are not forward-filled. The full synthetic tape is seed 20261107, selected in advance, not the best seed.

## Files

- `synthetic_round1.csv`: full 12000-second tape in the repository simulator format.
- `comparison.html`: interactive figures; use a local HTTP server to open as a page.
- `comparison.csv`, `replicate_statistics.csv`, `outcome.json`: numeric results.
- `training_marks.npz`, `parameters.json`, `locked_model.json`: reproducible fitted model.
- Run `calibrate.py` then `compare.py` using the repo .venv Python.
