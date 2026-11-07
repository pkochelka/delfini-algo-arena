# Round 1: the generator, reverse-engineered

Round 1 (`data/round1.csv`, 12000 s, 4.41M rows) is synthetic. This note recovers the process that generated it
and records the evidence.

The whole tape follows from one hidden fair price **S**, and S is computable exactly from public trades.
- **Makers** quote around S.
- **Takers** price their IOCs off S.
- **Only filled taker volume moves S.** Nothing else moves it.

A generator built from the fitted model reproduces the real tape's book statistics. It also reproduces
`queue_mm`'s PnL to within 2%.

**Tools:**
- [r1_model.py](r1_model.py): fits every parameter below from any tape. It writes [tuned/round1_model.json](tuned/round1_model.json).
- [r1_synth.py](r1_synth.py): generates new tapes from the model, with any parameter overridden.

Run both on the next round's data first: the fit reports how many orders break the price rules.

## The model

| Part | Rule | Round-1 values |
| --- | --- | --- |
| Hidden price | S(t) = S0 + c · Σ sign · √filled, over the IOCs before t. No other diffusion. | S0 = 10000, **c = 0.026** |
| Taker arrivals | Two independent Poisson streams, one per side | 4.00 /s per side |
| Taker size | max(1, round(lognormal(μ, σ))) | μ = 4.106, σ = 1.396 |
| Taker price | Buy IOC at round(S + 1), sell at round(S − 1), using S before its own jump | |
| Maker arrivals | Poisson, side 50/50 | 180 /s |
| Maker size | max(1, round(lognormal(μ, σ))) | μ = 2.658, σ = 1.799 |
| Maker price | Bid at min(floor(S − D), best ask − 1), ask at max(ceil(S + D), best bid + 1), so it never crosses | D = 0.05 + Exp(mean 0.45) |
| Maker cancel | A cancel decision arrives at hazard α + β · max(0, mv). The cancel row follows Exp(rate b) later, and is written even if the order has already filled. | α ≈ 1.72 /s, β ≈ 5.8 /s per tick, b ≈ 40 /s |

`mv` is the adverse move of S since the order was inserted: S down for a bid, S up for an ask.

**Things the model does not contain:**
- No momentum or mean reversion in S, since taker signs are i.i.d. coin flips (ACF ≈ 0).
- No informed takers: a taker's side is independent of the book, of recent flow and of S − mid.
- No intraday seasonality: rates are flat across 20 bins of 600 s.
- No dependence of maker size or lifetime on placement.

## Evidence

### Arrivals and sizes
- **Poisson arrivals.** Inter-arrival gaps have CV 1.00 for every stream. The dispersion index is 0.97–1.04 at 1, 10 and 60 s. The KS test against exponential gives p = 0.88 (sells) and 0.98 (buys).
- **No self-excitation.** The number of IOCs within 1, 10 and 100 ms after an IOC equals the Poisson expectation.
- **Order IDs** run consecutively from 1000001, inserts only. All timestamps are distinct.
- **Sizes are rounded lognormals.** The rounded-lognormal MLE matches P(v = 1…6) to 4 decimals, for both makers and takers. IOC log-size has skew 0.00 and kurtosis −0.02.

### S is exact
- **The rules leave no slack.** Constraints: every maker bid ≤ S ≤ every maker ask, and taker price − S ∈ [0.5, 1.5]. Under them, c is identified as a single point, 0.02600, and S0 lies in [9999.99999, 10000.00000]. There are **0 violations among 2,253,745 orders**. Moving c by ±0.0001 creates about 34k violations.
- **Taker price − S** is uniform on [0.5, 1.5]: ten equal bins hold 9,425 to 9,779 orders each.
- **Impact uses filled quantity.**
  - With 10 s windows, √filled explains R² = 0.954 of mid changes. Ordered size gives 0.893.
  - IOCs that fill nothing have zero impact.
  - Exponent 0.5 beats 0.4, 0.45, 0.55 and 0.6.
  - Impact is per taker. Using √ of each separate fill gives R² 0.81.
- **No other source of S movement.** With S0 = 10000 and c = 0.026, mid − S is stationary with sd 0.408. Its 100 s block means have sd 0.034, so the implied diffusion is about 0.

### Maker placement
- **Offset D.** The exact likelihood, including the no-cross censoring, gives D = 0.0500 + Exp(0.4503). A shifted gamma converges to shape 0.999, so the exponential is right.
- **Price never reaches S.** The smallest observed gap S − price is 0.0509.
- D is independent of side (E[D] 0.529 for asks vs 0.524 for bids), of size (corr 0.0001) and of lifetime.
- **Side choice.** P(buy) = 0.500 in every bin of book imbalance and of S − mid. Crossing orders are clipped, never dropped.

### State-dependent cancels
This piece is invisible if you look only at lifetimes at insert. Lifetime does not depend on where the order was
placed: mean 0.463–0.471 s across all distance bins.

The hazard of a live order depends only on mv:

| mv (ticks) | ≤ 0 | 0–0.1 | 0.2–0.3 | 0.4–0.5 | 0.6–0.8 | 0.8–1.0 | 1.0–1.5 | > 1.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hazard (/s) | 1.73 | 2.01 | 2.98 | 3.89 | 5.10 | 6.25 | 7.44 | 10.1 |

- **Distance doesn't matter.** Within a given mv bin, the hazard is the same for every insert distance and every current distance (2-D tables in the analysis).
- **The response is delayed.** The extra hazard builds up over 20–40 ms after the jump. Orders under 50 ms old have hazard 1.02 instead of 1.73. That is the Exp(b) step between decision and cancel row.
- **How the parameters were found.** α, β and b were calibrated by matching these hazards, the lifetime quantiles and the book statistics in synthetic tapes. They are not a direct MLE: the measured slope of 5.0 is lower than β because the delay smooths it.
- **What happens without it.** With the hazard set flat, the synthetic book lags S much more: sd(mid − S) is 0.62 instead of 0.41, and 5.9% of takers fill nothing instead of 1.45%.

## Validation of the generator

Book statistics: real tape vs one synthetic tape, 12000 s, seed 1, default parameters.

| Statistic | round1 | synthetic |
| --- | --- | --- |
| sd(mid − S) | 0.408 | 0.411 |
| Time at 1-tick / 2-tick spread | 93.0% / 6.9% | 93.1% / 6.8% |
| Mean touch qty | 1316 | 1316 |
| Mid volatility (1 s) | 0.921 | 0.917 |
| IOCs with no fill | 1.45% | 1.57% |
| Share of IOC volume filled | 87.1% | 86.1% |
| Cancel rows for already-filled orders | 13.2% | 13.3% |
| Order lifetime p10 / p50 / p90 (s) | 0.079 / 0.337 / 1.016 | 0.081 / 0.339 / 1.018 |
| Hazard at mv ≤ 0 / 0.3–0.6 / 0.6–1 | 1.72 / 3.87 / 5.44 | 1.72 / 3.81 / 5.36 |

`queue_mm` defaults on 12 folds of 805 s, each traded from t = 0:

| Data | mean PnL | sd | min |
| --- | --- | --- | --- |
| Real walk-forward folds wf01–wf12 | +30,152 | 1,291 | +27,460 |
| Synthetic, final model (seeds 101–112) | +30,685 | 1,557 | +28,903 |
| Synthetic, state-free cancels (α = 2.335, β = 0, b = 25.66) | +20,267 | 2,353 | +17,670 |

The generator is good enough to develop and test strategies on. The cancel hazard is the part that matters most for PnL.

**Known gap:** synthetic IOCs fill 86.1% of their volume against 87.1% in the real tape. The real book holds slightly
more depth within an IOC's limit. The cause hasn't been found.

## What this means for the strategy

1. **We can know S exactly.**
   - Every trade in `on_order_book_change` carries `taker_order_id`. Sum the quantity per taker, then add 0.026 · √q, with sign by aggressor side.
   - Start S at 10000. To check it, every visible maker insert must satisfy bid ≤ S ≤ ask, with a gap of at least 0.05.
   - Caveat: in the simulator our own resting orders change how much a taker fills, so the S in the tape (computed from the original fills) can drift slightly from ours. The maker-insert bounds correct that drift.
2. **The book lags S.**
   - sd(mid − S) is 0.41 tick.
   - After a jump, half the gap closes within about 0.2 s and all of it within about 0.5 s. The closing happens through new maker inserts at 90 /s per side, and through makers pulling quotes that S moved against.
   - Past mid changes predict nothing (corr ≈ 0.005). Signed maker-insert aggressiveness over the last 0.1 s correlates 0.49 with the mid change over the next 0.5 s: that is S showing through.
3. **There are mispriced quotes to take.**
   - Episodes with ask < S − 0.25 (or bid > S + 0.25) start 0.18 /s per side. 100% of them begin with a taker's jump.
   - They last a median of 150 ms, and 97.5% last longer than our 5 ms latency.
   - At a 0.5 threshold: 0.057 /s per side.
   - S is a martingale, so buying below S has positive expected value. The ±100 position limit caps the size per episode.
4. **Which maker fills make money.** Edge per lot is measured against S after the jump. The table splits fills by how far the fill price was from S before the trade:

   | Price − S before the fill | ≤ 0 | 0–0.25 | 0.25–0.5 | 0.5–0.75 | 0.75–1.0 | 1.0–1.25 | 1.25–1.5 |
   | --- | --- | --- | --- | --- | --- | --- | --- |
   | Share of market fills | 5.7% | 11.0% | 18.8% | 22.3% | 20.5% | 14.7% | 7.0% |
   | Maker edge after the jump | −0.67 | −0.34 | −0.13 | +0.10 | +0.33 | +0.50 | +0.66 |

   - An IOC only reaches prices up to round(S ± 1). An ask at price a is reachable only while S ≥ a − 1.5.
   - Quoting with S in mind means two things: pull or requote within 5 ms when S jumps toward our order (the market's makers take about 40 ms or more), and avoid quotes less than about 0.5 tick from S.
5. **Why queue seniority works.**
   - Every market order is cancelled after a mean of 0.47 s, and sooner once S moves against it. An order of ours that stays put therefore reaches the front of the queue quickly. That explains the age and queue markouts in [queue_hypotheses.md](queue_hypotheses.md).
   - It also means the orders ahead of ours are cleared out exactly when S has moved against the level. That is when our own order is stale.

## Using the generator

```
uv run python strategies/peter/r1_synth.py --seeds 1 2 3                       # round-1 model, 12000 s each
uv run python strategies/peter/r1_synth.py --duration 805 --tag fast --seeds 1 2 --set taker_rate=6 c=0.035
uv run python strategies/peter/r1_model.py data/synth/r1_fast_seed1.csv --out data/synth/r1_fast_seed1_fit.json
uv run python strategies/peter/fold_eval.py strategies/peter/queue_mm.py --files data/synth/r1_fast_seed1.csv,data/synth/r1_fast_seed2.csv
```

The knobs are `s0 c impact_exp s_sigma maker_rate maker_mu maker_sd d_shift d_scale cancel_alpha cancel_beta
life_b taker_rate taker_mu taker_sd taker_off duration`.
- `s_sigma` adds Brownian motion to S (0 in round 1), for testing robustness to a fair price that drifts without trades.
- Synthetic tapes go in `data/synth/`, which is git-ignored. Each comes with a `.params.json` beside it.

Compared with [augment.py](augment.py), this generator changes the *causes* (rates, impact, quoting rule) rather than
transforming the recorded tape. The book's reaction is then consistent with the model: for example, a higher taker rate
also produces more impact and more stale-quote cancels.
