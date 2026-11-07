# Round 2: market structure, predictive information and trading implications

## Main conclusions

The useful complexity here is in conditional execution and risk control. A short-horizon direction model generalizes within this public session, while longer-horizon directional models fail the validation comparison. A model that predicts moves can still lose money crossing the spread. No new profitable trading strategy is established by these prediction results.

- The unchanged Round1 native ladder loses 136,037 in validation and 62,763 in the final test (19m50s each), with zero penalties. The failure is not an inventory-limit penalty. The smaller tuned ladder also loses in both periods. Adaptive spread loses 6,656 in validation and earns 6,662 in test; this mixed result is not robust proof of profitability.
- Round2 order-size marginals resemble Round1: fitting-period IOC mean 160.42, median 61, q99 about 1,501. But future price risk and fill-conditioned outcomes are much worse. Marginal size matching is insufficient for selecting strategies or validating a data generator.
- Mean historical maker markout declines from +0.471 ticks at 5ms to +0.216 at 100ms and −0.259 at 1s. These are signed future-mid minus actual fill-price, volume weighted. They include initial spread capture. Fast positive markouts do not imply profitable aggressive liquidation: paying the exit half-spread can consume that edge.
- The largest 1% of one-second endpoint moves contribute 77.9% of squared movement in the final period. Mean/variance or Gaussian assumptions alone are fragile; rare jumps matter heavily.
- The locked 100ms HGB model reaches test correlation 0.328, R² 0.107 and 76.3% correct direction CONDITIONAL ON a nonzero move. Only 21.5% of these targets move. This is not 76.3% accuracy over every decision or an achievable trading win rate.
- Five-millisecond entry latency and bid/ask crossing remove the apparent easy trading edge. The fixed crossing-opportunity screens have negative mean test results, including high-score subsets. No screen was optimized on test.
- Historical passive-order fill prediction generalizes: selected HGB AUC 0.751. Conditional first-fill markout is harder: selected ridge R² only 0.011, correlation 0.127. The fill model estimates behavior under historical cancellation policy, not the fill probability of arbitrary new quotes.
- Magnitude is more predictable than longer-horizon direction: the one-second ≥5-tick endpoint-move classifier reaches test AUC 0.943 and lowers Brier loss from 0.0702 to 0.0467. AUC measures ranking; it does not guarantee calibrated probabilities or profitable quote suppression.
- Historical fills preceded by intermediate/high risk states often have much worse 1s markouts. The extreme top bucket is small and reverses sign in test. Do not fit a monotonic cancellation rule or choose its cutoff from these test cohorts.
- Executed-aggressor arrival intensity is also forecastable. A lag-only nonlinear Poisson model reduces test deviance from 1.720 to 1.404 (18.4%) with correlation 0.437. A fixed clock-phase model provides little benefit. Prefer adaptive flow estimates over hardcoded session timing.

## Data and evaluation design

Source: data/round2_public.csv, 2,193,064 events, 1,123,282 inserts and 197,588 matches over about 6,000 seconds. Data SHA256 is in protocol.json. Timestamps are sorted; duplicate timestamps: 0; duplicate insert IDs: 0; nonpositive prices/sizes: 0/0.

Split on time, never shuffled: fit 60–3590s; validation 3600–4790s; final scoring 4800–5990s. Ten-second gaps purge overlapping labels. The first 60s supplies history. Price labels span at most 5.005s; historical-order labels at most 2s. All targets require complete future observations. State/history can carry across fitting boundaries, but targets do not.

Price decisions use a 100ms grid offset by 50ms. Features are observed at decision t; the price label is mid(t+5ms+h)−mid(t+5ms), with h=100ms, 1s or 5s. Entry/exit crossing diagnostics use actual bid/ask at those times. The 5ms simulator latency is never assumed to be zero.

Price row counts: {'train': 35199, 'validation': 11867, 'test': 11861}. Historical-order counts: {'train': 124485, 'validation': 41130, 'test': 41726}. Conditional markout counts: {'train': 18561, 'validation': 6924, 'test': 6314}.

The public tape had already been opened descriptively. This is a chronological fitting holdout, not a new blind sample. Supplemental risk, activity and cohort analyses reuse the same final block after the initial model comparison; no new strategy was selected on those test numbers. Independent private data is still required.

## What a live strategy can actually observe

The 48 market predictors exclude absolute price, absolute clock phase, order IDs, future cancellation/lifetime information and submitted IOC size. The simulator exposes market orders within 20 ticks, not merely 20 individual orders; depth features use that tick restriction.

| Feature family | Construction and plausible use |
|---|---|
| Spread, touch size/order imbalance, microprice offset | Fair-value adjustment, touch competition, liquidity fragility |
| Depth imbalance and total depth within 3/5/10/20 ticks | Sweep risk, asymmetric replenishment, size capacity |
| Past returns over 0.1/0.5/1/5/30/60s | Short continuation, reversion and shock persistence |
| Trailing RMS and move frequency over 1/5/30/60s | Conditional spread, exposure limits, jump-risk state |
| Executed volume/sign imbalance over 0.1/0.5/1/5/30s | Aggressive flow direction and intensity |
| Best-quote OFI and update counts over the same windows | Visible replenishment/depletion and cancellation pressure |
| Historical order side, size, price placement and actual queue ahead | Fill-probability and first-fill toxicity models; policy dependence must be respected |

Depth snapshots are as-of and may be up to 100ms old; tick/event quote and trade features use information through t. Transient one-sided states invalidate current prediction rows. Past-price history carries only already observed prices through those gaps. Raw all-depth insert/cancel information and submitted IOC volumes appear in diagnostics only.

## Market dynamics, activity and distribution tails

Fitting mean spread 1.168 ticks; one-tick spread share 87.8%; mean touch quantity 1190.9; RMS volatility signature at 1s 2.918 ticks/sqrt-second. Volatility changes substantially through the session; see five_minute_regimes.csv.

Submitted IOC arrivals: 8.570/s. Limit inserts: 178.135/s. Cancellation-message mean lag 0.449s, median 0.320s; all fitting cancellations link to an earlier insertion. Message lag is not actual resting lifetime if the order filled earlier.

Count Fano factors near one at 10ms grow with aggregation: IOC/limit factors 1.91/1.96 at 1s and 6.00/3.41 at 60s, using complete equal-exposure bins only. This is inconsistent with a simple constant-rate Poisson description at every scale. It does not identify whether the cause is changing intensity, self-excitation, common shocks or another dependent mechanism. A Hawkes/Cox explanation needs a separately checked likelihood/residual model.

Limit-insert/cancel contemporaneous correlation is 0.9956 at 60s; this does not mean future cancels are perfectly predictable. Cancel lag, current visible order age and quote-state transitions should be modeled explicitly before using such correlations operationally.

Short-return and trade-sign autocorrelations, variance ratios, volatility signature, intensity spectra and 5-minute summaries are exported. The strongest fitting activity spectral peak is around 256s, but a model using that fixed phase does not improve validation. Do not hardcode a periodic generator from a visual peak.

Endpoint-return tail summaries (overlapping observations; maxima and kurtosis are descriptive):

| split | horizon_s | median | q95 | q99 | q999 | max | top1pct_squared_move_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0.1 | 0 | 1 | 2 | 6 | 54.5 | 0.7929 |
| train | 1 | 1 | 3.5 | 10.5 | 31.5 | 60.5 | 0.6265 |
| train | 5 | 2 | 10 | 29 | 56 | 72 | 0.4951 |
| validation | 0.1 | 0 | 2 | 4 | 9.567 | 86.5 | 0.6938 |
| validation | 1 | 1 | 9 | 14.17 | 39.84 | 95.5 | 0.5004 |
| validation | 5 | 2.5 | 22.5 | 35.84 | 106.6 | 116.5 | 0.4125 |
| test | 0.1 | 0 | 1 | 3 | 6 | 47.5 | 0.6636 |
| test | 1 | 1 | 6.5 | 13 | 100.5 | 117.5 | 0.7792 |
| test | 5 | 2 | 16.5 | 45.5 | 102 | 106 | 0.6605 |

Source-event attribution is in jump_event_sources_train.csv. Large quote changes occur after cancellations, IOC trades and limit insertions. The fitting tape has no crossing G orders, so a jump associated with a G insertion need not be an aggressive trade: restoring a new best quote can move the midpoint. Association with an event is not evidence of the unobserved fundamental cause.

## Directional models and conditional uncertainty

Fixed candidates: training mean; unscaled microprice residual; standardized ridge alpha=10; histogram gradient boosting with 15 or 31 leaves, 150 iterations, learning rate .05, minimum 100 samples per leaf and L2=10. No time/price trajectory input. Validation MSE chooses each horizon; fitted model bytes and choices are fingerprinted in locked_models.json before test scoring.

| horizon_s | model | mse | r2 | correlation | moving_direction_accuracy | moving_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1 | hgb15 | 0.5459 | 0.1067 | 0.3279 | 0.7629 | 0.2155 |
| 1 | mean | 24.64 | -0.0009457 | nan | 0.4888 | 0.6464 |
| 5 | mean | 105.9 | -0.006296 | nan | 0.4928 | 0.8514 |

A 3,000-replicate bootstrap using 10-second wall-clock blocks gives a conditional 95% interval of 7.1%–16.6% for the 100ms model's MSE reduction versus the training mean. Moving-case direction accuracy interval: 74.2%–78.6%. These intervals do not include model-family/research selection or new-regime uncertainty.

Joint feature-group permutation on validation identifies useful book, print-flow, quote-flow, volatility and return information for the short-horizon model. Permutation can create unrealistic feature combinations and is descriptive importance, not a causal effect. Correlated ridge coefficients are also not individual economic effects.

The 1s/5s mean forecast was selected because fitted directional models overfit the training relationships and increased validation MSE. Do not substitute a larger model simply because it has more capacity. Possible future work is shrinkage or mixture-of-experts conditional on verified regimes; it needs a new evaluation design.

### Crossing-cost diagnostic

Select sign(prediction), then cross at the actual ask/bid after 5ms and mark an opposite-side exit at horizon. Apply fixed score thresholds above the current half-spread plus {0,.25,.5,1} ticks. Thin decisions by the holding horizon. These remain independent hypothetical opportunities, without inventory, fill interaction or actual IOC replay; they are not dollar PnL.

| horizon_s | split | extra_edge_threshold | opportunities | selected | mean_crossing_ticks | win_rate |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1 | test | 0 | 11861 | 367 | -0.4332 | 0.2262 |
| 0.1 | test | 0.25 | 11861 | 221 | -0.3122 | 0.2398 |
| 0.1 | test | 0.5 | 11861 | 142 | -0.2183 | 0.2606 |
| 0.1 | test | 1 | 11861 | 30 | -0.03333 | 0.3667 |

## Queue fills, adverse selection and cancellation

Queue tables follow real non-crossing G orders placed at the touch, with actual competing cancellations and complete follow-up. At 1s, any-fill rate drops from about 61.6% with 1–5 units ahead to 17.3% with more than 100 ahead. These are selected historical order cohorts; quote size, price policy and cancellation policy differ between them.

| queue | orders | any_fill | full_fill | mean_fraction | median_first_ms |
| --- | --- | --- | --- | --- | --- |
| 1-5 | 2503 | 0.6157 | 0.5166 | 0.5576 | 93.17 |
| 6-20 | 4375 | 0.595 | 0.5017 | 0.5394 | 104.9 |
| 21-100 | 15595 | 0.5087 | 0.4357 | 0.4656 | 133.1 |
| >100 | 312345 | 0.1731 | 0.1505 | 0.1599 | 344 |

Historical fill model: logistic vs HGB classifier; validation Brier score selects HGB. Test scores:

| model | n | base_rate | brier | log_loss | auc |
| --- | --- | --- | --- | --- | --- |
| mean | 41726 | 0.1514 | 0.1285 | 0.4251 | 0.5 |
| logistic | 41726 | 0.1514 | 0.1157 | 0.3826 | 0.728 |
| hgb | 41726 | 0.1514 | 0.1117 | 0.3688 | 0.7505 |

Conditional first-fill markout model: features known before insertion; first fill must occur within 1s; signed markout is measured 1s after that fill. This uses first fills, unweighted per order, unlike the volume-weighted all-match maker table. Consequently its average is not directly comparable to −0.259 ticks above. Validation MSE selects ridge. Test scores:

| model | n | mse | mae | r2 | correlation | moving_direction_accuracy | moving_fraction | target_std | pred_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean | 6314 | 10.2 | 1.82 | -0.002496 | nan | 0.5415 | 0.9804 | 3.19 | 3.469e-18 |
| ridge | 6314 | 10.07 | 1.782 | 0.01078 | 0.1268 | 0.5648 | 0.9804 | 3.19 | 0.6256 |
| hgb15 | 6314 | 10.21 | 1.791 | -0.003089 | 0.1413 | 0.5775 | 0.9804 | 3.19 | 0.9332 |

At a hypothetical cancel request age of 100ms, about 0.24% of eligible resting historical orders receive a fill during the next 5ms. Average extra fill quantity is about .091 lots for buys and .069 for sells. These unconditional averages can hide much worse conditional loss during jumps. A cancel is effective after its delay, not when requested.

Maker markout by horizon, fitting data, signed ticks per unit including initial spread capture:

| horizon_s | lots | mean | p10 | median | p90 | adverse |
| --- | --- | --- | --- | --- | --- | --- |
| 0.005 | 4020079 | 0.4706 | -0 | 0.5 | 1 | 0.05602 |
| 0.05 | 4043623 | 0.2929 | -0.5 | 0.5 | 0.5 | 0.1574 |
| 0.1 | 4045267 | 0.2156 | -0.5 | 0.5 | 0.5 | 0.1991 |
| 1 | 4046044 | -0.2595 | -2 | -0.5 | 1.5 | 0.5023 |
| 5 | 4039433 | -0.2455 | -4.5 | -0.5 | 3.5 | 0.5098 |

Conditional tables by spread, own-side touch share, volatility and aggressor submitted size are included. In fitting data, own-side touch-share >.8 has about −.555 tick 1s maker markout, versus −.079 below .2. This relationship differs from the simple Round1 imbalance narrative. It is descriptive and confounded by other book states; use the multivariate, out-of-time model before writing a one-variable rule.

## Risk forecasts and changing regimes

Predict absolute endpoint moves with a 90% quantile HGB; compare fixed training q90 and rolling Gaussian approximations. Predict endpoint-move events |Δmid|≥3 ticks at 100ms, ≥5 at 1s and ≥10 at 5s with a fixed HGB classifier. These are endpoint events, not the probability that the price touches a barrier anywhere inside the interval. Supplemental model selections are locked using validation before test scoring.

| horizon_s | threshold | model | brier | auc | base_rate | selected |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 3 | constant | 0.01061 | 0.5 | 0.01071 | False |
| 0.1 | 3 | hgb | 0.009712 | 0.9319 | 0.01071 | True |
| 1 | 5 | constant | 0.07024 | 0.5 | 0.07436 | False |
| 1 | 5 | hgb | 0.04675 | 0.9427 | 0.07436 | True |
| 5 | 10 | constant | 0.08317 | 0.5 | 0.08971 | False |
| 5 | 10 | hgb | 0.06315 | 0.8751 | 0.08971 | True |

Tail model test coverage and pinball loss (target coverage 90%):

| horizon_s | model | pinball | coverage | mean_prediction | selected |
| --- | --- | --- | --- | --- | --- |
| 0.1 | constant | 0.146 | 0.8889 | 0.5 | False |
| 0.1 | rolling_gaussian | 0.173 | 0.9118 | 1.402 | False |
| 0.1 | hgb | 0.08916 | 0.9271 | 0.575 | True |
| 1 | constant | 0.7683 | 0.8561 | 2 | False |
| 1 | rolling_gaussian | 0.7212 | 0.8602 | 4.435 | False |
| 1 | hgb | 0.4471 | 0.8767 | 3.129 | True |
| 5 | constant | 1.952 | 0.8509 | 5.5 | False |
| 5 | rolling_gaussian | 1.959 | 0.8298 | 9.917 | False |
| 5 | hgb | 1.32 | 0.8849 | 7.269 | True |

The 1s/5s conditional q90 models cover only 87.7%/88.5% of test absolute moves; they are not reliable 90% bounds without further calibration. Extreme paths can exceed endpoint-based forecasts by more. Keep a deterministic inventory reserve and hard cap regardless of model confidence.

A three-component Gaussian mixture is fit on current volatility, spread, depth and print activity from the fitting period only. It is a descriptive clustering, not a proven hidden-state model. In test, cluster 0 occupies about 63.5% of observations and has 0.36% ≥5-tick 1s moves; the wider-spread cluster 1 occupies 16.3% and has 31.2%. A single spread/risk setting is unlikely to be appropriate everywhere.

| split | state | n | share | mean_spread | current_rms30 | future_abs_move_1s | future_move_1s_std | future_5tick_jump_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 0 | 7533 | 0.6351 | 1 | 0.9559 | 0.7581 | 1.235 | 0.003584 |
| test | 1 | 1939 | 0.1635 | 2.388 | 4.961 | 4.602 | 11.47 | 0.3115 |
| test | 2 | 2389 | 0.2014 | 1 | 6.346 | 1.717 | 3.177 | 0.1051 |

Historical maker fills grouped by a risk state available at least 5ms before execution; boundaries are the training distribution's 50/80/90/95/99 percentiles. No threshold is chosen on test.

| split | training_risk_bucket | prints | lots | volume_weighted_markout | adverse_volume_share | average_prior_risk |
| --- | --- | --- | --- | --- | --- | --- |
| test | 0 | 15692 | 564299 | -0.1508 | 0.5088 | 0.001084 |
| test | 1 | 10503 | 381011 | -0.008706 | 0.4544 | 0.002421 |
| test | 2 | 3999 | 140557 | -0.04243 | 0.4878 | 0.009393 |
| test | 3 | 3816 | 127354 | -2.106 | 0.6223 | 0.1906 |
| test | 4 | 4796 | 155138 | -1.544 | 0.6068 | 0.4212 |
| test | 5 | 123 | 4149 | 1.161 | 0.4437 | 0.7419 |

Risk is not monotonically related to realized maker reward in every bucket: the highest bucket has only 123 test matches and positive mean markout. These cohorts do not simulate cancellations or displaced queue fills. A strategy-level experiment must establish whether risk suppression improves PnL and whether re-entry loses too much queue priority.

## Transfer of the existing trading strategies

The following are actual official-simulator replays, with 60s seeded warmup, fixed Round1 parameters, 5ms latency and liquidation-book marking. They do not use the newly trained models. Each validation/test replay starts with zero cash/inventory and trades for 1190 seconds. All have zero penalties and respect their strategy cap.

| model | split | pnl | traded_volume | max_abs_inventory | abs_inventory_q95 | inventory | breach_penalty |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ladder_native | validation | -1.36e+05 | 329578 | 100 | 100 | -20 | 0 |
| ladder_native | test | -6.276e+04 | 289570 | 100 | 100 | 100 | 0 |
| ladder_tuned_round1 | validation | -3.158e+04 | 62368 | 80 | 55 | -26 | 0 |
| ladder_tuned_round1 | test | -7,101 | 51049 | 80 | 50 | -11 | 0 |
| adaptive_spread | validation | -6,656 | 4604 | 68 | 43 | -2 | 0 |
| adaptive_spread | test | 6,662 | 6094 | 52 | 22 | -4 | 0 |

## Model architecture to test next

1. Use the 100ms predictor as a bounded reservation-price/side-selection signal. Shrink it toward zero and require simulated net utility, rather than using its raw sign to cross.
2. Estimate the candidate quote's fill probability from queue, order size, placement, public book and recent flow. Model cancellation as a competing outcome. Validate counterfactual quote policies in the official exchange; historical cohorts alone cannot identify their fill probabilities.
3. Estimate signed future markout conditional on a fill, preferably separately for buys/sells and multiple horizons. The present ridge signal is weak. Use a conservative floor or shrinkage rather than treating its output as certain profit.
4. Use conditional tail forecasts and jump probability to set quote size, width and inventory tolerance. Compare an interpretable regime gate against a full nonlinear model. Both must reserve capacity for pending/canceling orders and respect |inventory|≤100 under one-sided fills.
5. Score quote candidates using expected fill reward minus inventory/hedging cost. Preserve queue position when the incremental expected benefit of moving is small; explicitly price the 5ms cancel/re-entry exposure. Do not assume small unconditional cancel-fill probability means small tail loss.
6. Test the integrated model against no-trade and simple conservative baselines in expanding chronological folds. Freeze every choice before a newly reserved test segment; public test reuse in this notebook is not a substitute for private-data validation.

This is the architecture to investigate, not an optimized or profitable submission. An online feature engine must match the research definitions, filter own quotes correctly, and handle market impact: training used a market-only replay, while the strategy's fills can change future book states. The sandbox requires a single file ≤1MiB, no file/network access, and bounded memory/runtime; fitted pickle files are local research artifacts and cannot simply be loaded in a submission. A compact tree/linear representation must be embedded and parity-tested.

## Reproducibility and limits

Run from the repository root:

```powershell
& strategiesdamound2_researchun_research.ps1
```

The runner executes extraction, price/fill/markout models, tail/jump/regime models, fixed strategy baselines, interpretation/audits, arrival-intensity modeling, integrity diagnostics and this report. It stops on an error. The original data, public replay helpers and trading strategies are unchanged. Generated feature arrays, fitted estimators, split CSVs and full simulator logs live under generated/ and are ignored by Git. Compact CSV/JSON evidence and source code are shareable.

Validation checks independently re-index 100 sampled decisions, 300 delayed targets and the corresponding depth timestamps. Source integrity checks pass. These are targeted causality/index audits, not a complete independent simulator-parity test. Historical probability labels are policy-dependent; overlapping targets reduce effective sample size; single-session uncertainty does not cover future regime changes; an AUC or R² gain is not a dollar-profit claim.

Files: protocol.json / locked_models.json / locked_risk_models.json / locked_flow_model.json document decisions; feature_schema.json and order_schema.json describe arrays; model_results.json and all CSV tables supply exact values; baseline_results.json supplies actual PnL; uncertainty.json and validation.json document checks. The report uses unrounded source values in these files.
