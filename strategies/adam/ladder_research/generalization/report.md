# Does Peter's ladder generalize?

## Assessment

The evidence supports a profitable rule across the observed round1 regime, rather than a profit confined to one time window. It does not establish generalization to unseen organizer data or different market regimes. The older sample is a separate available stress tape, not a genuinely blind validation sample. Fresh synthetic seeds check dependence on one random realization only under an approximate fitted generator.

## Frozen implementation and protocol

The current source was copied to ladder_frozen.py before tests. Current defaults use size_top=400, cap=100, levels=1, topup=10, imb_pull=0; the new imbalance filter is disabled. The previous defaults used size_top=200. With cap 100, each order is clipped to at most 200, but the larger target can trigger small topups differently. Current native earns 107,559 vs the previous 107,532 (+27, or 0.025%); a size_top=200 control exactly reproduces 107,532. The previously selected tuned settings exactly reproduce 22,254.

The native configuration and the already-selected tuned configuration were held fixed. No parameter was chosen or changed based on this audit. All 51 runs use the official simulator matching engine at the competition's 5 ms latency, penalty settings and inventory limits; all had zero penalties and stayed within their strategy cap. A streaming audit subclass records inventory and order statistics without modifying matching. The source file is unchanged.

The seven chronological slices use Peter's existing seeded splits, 60 seconds without trading and fresh zero cash/inventory per slice. First four slices were used for earlier fitting, next two for earlier validation, and the final slice for earlier testing. They are non-overlapping trading periods of one session, not seven independent trading days. Results across them are diagnostic because the public tape has been repeatedly inspected.

## Temporal stability

| Window | Native PnL | Native PnL/hour | Tuned PnL | Tuned PnL/hour |
|---|---:|---:|---:|---:|
| wf01 | 50,502 | 122,018 | 10,175 | 24,584 |
| wf02 | 46,146 | 111,494 | 10,605 | 25,623 |
| wf03 | 53,854 | 130,117 | 10,836 | 26,181 |
| wf04 | 51,566 | 124,589 | 11,654 | 28,157 |
| wf05 | 57,060 | 137,863 | 13,758 | 33,241 |
| wf06 | 55,017 | 132,927 | 13,727 | 33,166 |
| test | 107,559 | 131,705 | 22,254 | 27,250 |

Native is profitable in 7/7 periods, with PnL/hour ranging 111,494–137,863.

## Nearby native parameters

One-at-a-time changes, evaluated on wf05 and wf06. Only the named parameter changes; no winning value is selected.

| Change | Combined PnL / default PnL | Positive windows |
|---|---:|---:|
| size100 | 76.8% | 2/2 |
| size150 | 93.6% | 2/2 |
| size200 | 100.0% | 2/2 |
| topup0 | 59.1% | 2/2 |
| topup5 | 100.0% | 2/2 |
| topup20 | 100.0% | 2/2 |
| topup40 | 100.0% | 2/2 |

The active changes (smaller targets and disabling topups) remain profitable, which argues against a narrow parameter peak. Positive topup thresholds 5–40 produce identical results because size_top 400 is well above the per-side room allowed by cap 100, so those thresholds do not bind; that flatness is mechanical, not four independent robustness successes. Size 200 and 400 have the same inventory ceiling but differ slightly in topup behavior. These checks cannot remove bias from the original choice of strategy family.

## Other tape and fresh synthetic realizations

| Tape | Native PnL/hour | Tuned PnL/hour |
|---|---:|---:|
| legacy_sample | -8,525 | -18,950 |
| synthetic_7101 | 304,794 | 45,822 |
| synthetic_7102 | 318,054 | 47,964 |
| synthetic_7103 | 339,891 | 51,048 |
| synthetic_7104 | 348,987 | 55,026 |
| synthetic_7105 | 334,383 | 52,908 |

Synthetic tapes use independent, pre-recorded seeds 7101–7105, with 1260 seconds generated and the first 60 seconds excluded from trading. Fresh exponential event clocks, generated lognormal IOC sizes and resampled limit-order marks produce new paths. They do not replay round1 event times or prices.

The generator was fitted to round1, and earlier diagnostics found about 45% too little one-second volatility and 81% too-favorable maker markouts. Positive synthetic PnL therefore supports survival of fresh draws from this approximate model, not survival of the true unknown distribution. Do not average synthetic and real PnLs into one generalization estimate.

### Observed real-market differences

| Tape | IOC/s | Mean IOC size | Mean spread | Mean touch depth | 1s mid-change SD |
|---|---:|---:|---:|---:|---:|
| round1_test | 7.99 | 157.7 | 1.07 | 1335.7 | 0.903 |
| legacy_sample | 0.17 | 12.8 | 2.00 | 24.7 | 2.554 |

Quote statistics use a 100 ms grid offset by 50 ms, after the same 60-second warmup. The 1s statistic is the SD of differences of one-second-spaced mid samples (not annualized). This avoids sampling the legacy tape exactly on its scheduled cancellation timestamps.

## Structural stress tests

Same 49-minute final interval; change only IOC rows, preserve all original limit-order messages, cancellation times and prices. These are counterfactual stress replays, not independently regenerated markets: changing executions without changing later order decisions is only an approximation. Directional thinning drops 50% of one IOC side using a fixed seed, rather than changing its price to manufacture crossings.

| Stress | Native PnL | Native / original | Tuned PnL |
|---|---:|---:|---:|
| ioc_half | 38,802 | 36.1% | 13,080 |
| ioc_double | 208,540 | 193.9% | 38,408 |
| ioc_cap500 | 106,381 | 98.9% | 21,797 |
| drop_half_buy_ioc | 56,297 | 52.3% | 13,156 |
| drop_half_sell_ioc | 52,844 | 49.1% | 14,011 |

## Conditional uncertainty and limits

For each run, record three thousand circular moving-block bootstrap replicates of one-minute PnL increments, using five-minute blocks. This preserves some short-run dependence. Reported intervals concern the mean PnL rate conditional on that tape/regime, omit incomplete minutes and do not include model-selection, source-version, generator or regime uncertainty. Seven slices and five synthetic seeds are too few to estimate rare blow-ups reliably.

| Configuration | Final real-test PnL/hour | Conditional block-bootstrap 95% interval |
|---|---:|---|
| native | 131,705 | 123,004 to 143,695 |
| tuned | 27,250 | 23,637 to 30,258 |

The mechanism is plausible for a queue-priority maker: large uninformed aggressive orders reach its resting quotes, spread capture pays, and retaining live orders avoids repeatedly losing queue priority. This depends on the joint behavior of fills, future prices and cancellations, not merely matching the marginal order-size distribution. High inventory-limit occupancy means changes in adverse selection or persistent directional order flow matter. The strategy has no hardcoded future time/price path and reads only the current/public book and its private fills. That rules out an obvious lookahead mechanism, not broader research overfitting.

A stronger generalization test needs additional organizer tapes or genuinely independent days generated by the same unknown process. Freeze the code and parameters before seeing them; evaluate without retuning; check maker markouts, drawdowns, profitability and time at the inventory limit, including stressed/volatile periods. Any subsequent retuning needs another untouched evaluation set.

## Reproduce

From the repository:
```powershell
.\.venv\Scripts\python.exe -X utf8 -B strategies\adam\ladder_research\generalization.py
.\.venv\Scripts\python.exe -X utf8 -B strategies\adam\ladder_research\generalization_context.py
.\.venv\Scripts\python.exe -X utf8 -B strategies\adam\ladder_research\generalization_report.py
```

protocol.json records the frozen source hash and pre-specified test grid; outcome.json and individual JSONs include all audited results and conditional intervals; comparison.csv is a compact numeric export. No new candidate or final submission is selected by this work.
