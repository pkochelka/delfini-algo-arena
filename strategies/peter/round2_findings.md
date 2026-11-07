# Round 2 (public tape): what changed versus round 1

`data/round2_public.csv`: 6000 s, 2.19M rows. It has the same participants as round 1: about 180 maker inserts/s, about 4.5 taker IOCs/s per side, and the same size distributions. Its price process is different, though, and **the round-1 strategies lose money on it.**

## How to reproduce

```
uv run python -m analysis.fit data/round2_public.csv
uv run python strategies/peter/split_data.py --backtest data/round2_public.csv --folds 12 --warmup-s 60 --end-frac 0.75 --out-dir data/splits_r2/wf
uv run python strategies/peter/split_data.py --backtest data/round2_public.csv --ratios 3 1 --names pre75 hold25 --out-dir data/splits_r2
uv run python strategies/peter/r1_model.py data/round2_public.csv            # the round-1 generator: fails, see below
PYTHONIOENCODING=utf-8 uv run data-explorer --data data/round2_public.csv --export                    # -> data_explorer/round2_public/
PYTHONIOENCODING=utf-8 uv run data-explorer --data data/round1.csv data/round2_public.csv --export    # -> data_explorer/compare-round1-round2_public/
uv run python strategies/peter/r2_structure.py data/round2_public.csv --folds data/splits_r2/wf/folds.json --hold-parts 4
uv run walk-forward --strategy strategies/marek/sig_ladder.py --backtest data/round2_public.csv --folds 12 --warmup-s 60 --end-frac 0.75 --out-dir logs/wf/sig_ladder-round2_public
```

The Windows console needs `PYTHONIOENCODING=utf-8`; without it the export crashes on "σ".

**Splits:**
- `data/splits_r2/wf/wf01-12`: 370 s eval windows over 60–4500 s, each with a 60 s warmup.
- Holdout: 4500–6000 s, in `hold25.csv`. The audit cuts it into four 375 s parts.

## The round-1 generator does not fit

`r1_model.py` cannot identify a hidden price S:
- 535k of 1.12M orders violate the round-1 price rules. Round 1 had 0.
- sd(mid − S) is 139 ticks.

A greedy segmentation with S = c · Σ sign · √filled + J(t), with J piecewise constant, needs about 15.9k pieces. That is about 2.6 extra price moves per second that no taker caused.

The arrivals are no longer Poisson. The dispersion index of counts per 10 s is 4.1 for makers, 5.5 for takers and 3.9 for cancels. In round 1 it was about 1.0.

## The price process: impact plus exogenous jumps, in two regimes

The structural price used here is P, the maker-implied price: the median over 100 ms of bid insert + 0.55 and ask insert − 0.55.

**Takers still move the price.**
- In calm periods a taker moves P by +0.28 ticks in its direction, and the move is permanent.
- The slope is about 0.037 per √filled (round 1: c = 0.026). Fitted per split, it is 0.034–0.046.
- In turbulent periods impact is about 4× larger, because takers become informed (see the next list).

**On top of that, P jumps by itself.**
- 194 jumps of more than 5 ticks within 100 ms, or 1.9 per minute.
- Size: median 7 ticks, p90 11. The largest are +110 (t ≈ 5399 s), +100 (5570 s), −87 (3873 s), −55 (2763 s and 3423 s).
- Direction is 50/50 up and down. Consecutive jump signs are uncorrelated.

**What a big jump looks like:**
- No taker flow comes before it, and the book shows no tell.
- Within about 50 ms, maker inserts move to the new level.
- Takers then sweep the stale side. The taker rate goes from 9/s to about 38/s in the first second, and 57% of those takers trade in the jump's direction.
- The spread can briefly reach 100+ ticks, and one side of the book can empty.

**Jumps cluster into turbulent regimes.**
- "Turbulent" means the trailing 10 s realised volatility of P is above 2 ticks/√s. That is 24% of the time, in 91 episodes: median length 3 s, longest 226 s.
- The gap between jumps has CV 2.4. P(another jump within 10 s | a jump now) is 0.70, against 0.28 unconditionally.
- The worst stretch is 3500–4500 s: 83 jumps, and P falls by 475 ticks.

| | calm | turbulent |
| --- | --- | --- |
| 1 s mid volatility | 1.08 | 7.5 |
| Takers per second | 8.0 | 11.8 |
| Share of IOC volume filled | 0.86 | 0.71 |
| Touch quantity | 1272 | 637 |
| Mean spread | 1.07 | 1.75 |

Whole tape: σ at 1 s is 3.80 ticks/√s (round 1: 0.92), and the kurtosis of 1 s returns is 232.

## Market-maker economics

Maker markout at 1 s, against P, weighted by volume:

| Fills | Share of volume | Markout per lot | Total |
| --- | ---: | ---: | ---: |
| All | 100% | −0.41 | −2.86M |
| Calm | 72% | −0.07 | |
| Turbulent | 28% | −1.27 | |
| 100 ms bins containing a jump | 9% | −3.4 | −2.2M |

In the explorer's split by taker size, fills against takers of 1–100 lots mark out positive (+0.2 to +0.5). Fills against takers above 100 lots mark out −0.54.

## Round-1 strategies on the round-2 folds (12 × 370 s)

| Strategy | Total | Folds won | PnL, folds with turbulent share < 0.2 | PnL, folds with turbulent share ≥ 0.2 |
| --- | ---: | ---: | ---: | ---: |
| `marek/sig_ladder.py` | −174k | 40% | +20k | −194k |
| `marek/q_ladder.py` | −178k | 20% | +9k | −187k |
| `peter/queue_mm.py` | −134k | 30% | +14k | −148k |
| `marek/robust_ladder.py` | **+12.7k** | 70% | +29k | −16k |

- Fold PnL against the fold's turbulent share: Spearman −0.80 to −0.87.
- The kill switch is what keeps `robust_ladder` positive.
- Even in calm folds, the edge is roughly 10× smaller per hour than on round 1.

## Periodicity, trends, anomalies

**Periodicity: none usable.**
- Timestamp phase is uniform at every modulus from 1 ms to 60 s (KS p ≥ 0.25). Jump times are uniform mod 60 s and mod 10 s.
- The spectrum of maker counts is red: peaks at 50 s, 21 s and 14 s, with ANOVA p = 4e-6 for phase mod 50.4 s. Round 1 is flat. This looks like rate changes driven by the regimes, not a clock.
- The "2000 s period" in taker counts and jumps is only 3 cycles in 6000 s. Treat it as regime clustering.

**Trends:**
- The second half is busier: taker count +10% over the tape (p < 0.001), jumps +186% (p = 0.001). With one tape, this cannot be told apart from regime clustering.
- Net P drift is −311 ticks, z = −1.0 against a martingale. Taker buy share is 0.499. No directional trend.

**Volatility and returns:**
- Strong volatility clustering: the ACF of |1 s P returns| is 0.21 at lag 1 and still 0.08 at 30 s.
- Mid variance ratio is 1.25 at 30–60 s and 1.6 at 300 s. That comes from turbulent stretches that keep moving one way. It is noisy: only 20 windows of 300 s.
- After a jump the mid lags P by about 1.1 ticks and catches up by about 0.45 ticks over the next second.

**Anomalies:**
- One side of the book empties 274 times: 3.9× round 1's rate per hour. Median 8.6 ms, max 136 ms. 229 of them start with an IOC.
- The busiest second has 71 takers (Poisson p ≈ 3e-35): post-jump sweeps.
- The longest gap with no events is 168 ms, at 5844 s.
- The largest single orders are a 72,482-lot maker and a 26,005-lot taker. Both are consistent with lognormal tails.

## Per-split stability: the results do change

**Stable across the 16 splits (cross-split CV):**
- Maker rate: 0.6%.
- Taker and maker log-size: about 0.5%.
- Buy share (chi² p = 0.90).
- Calm volatility: 5.5%.
- Impact c: 9%.
- Order lifetime: 6%.

**Not stable:**

| Statistic | Range across splits | Homogeneity test |
| --- | --- | --- |
| Jumps per split | 1 to 48 | p = 7e-32 |
| Turbulent share | 3% to 65% | |
| Taker rate (/s) | 8.1 to 11.0 | p = 2e-72 |
| Touch quantity | 720 to 1270 | |
| Share of time at a 1-tick spread | 59% to 93% | |
| 1 s volatility | 1.4 to 7.3 | |

The unstable statistics move together through one latent variable, the turbulent share. Across splits it correlates with taker rate (+0.89), the 1-tick spread share (−0.95) and touch quantity (−0.85).

The splits fall into two groups:
- **Calm:** wf03, wf04, wf08, hold1, hold2. These look like round 1, with volatility around 1.
- **Turbulent:** wf06, wf10, wf11, hold3. These look like a different market.

## Predictability

Out-of-sample R² of the mid change, leave-one-split-out:

| Horizon | Best public model | P − mid | Same models, calm periods only |
| --- | ---: | ---: | --- |
| 0.5 s | 0.05 | 0.01 | public 0.13 |
| 1 s | 0.02 | | public 0.08 |
| 5 s | ≤ 0 | ≤ 0 | |

- The best public model combines imbalance, microprice, past returns, insert aggressiveness and 100 ms taker flow.
- Insert aggressiveness carries most of the signal. Imbalance alone gives 0.05 in calm periods at 0.5 s.
- Jumps cannot be predicted from public data.
- The P − mid proxy is weak: P is a noisy median and the book reprices within 50 ms. On round 1 the exact S − mid reached 0.32.

## Implications for strategy work

1. **Survival matters more than spread capture.** One 100-tick jump against a 100-lot position costs 10k, about a third of a good round-1 fold. Kill switches and regime detection (trailing realised volatility, spread above 1, touch depth falling, taker rate rising) come first.
2. **Pull or shrink quotes when turbulence starts.** P(another jump within 10 s | a jump) is 0.70, so the first jump is the warning. In turbulent periods makers lose 1.3 ticks/lot.
3. **Make thresholds relative.** The volatility-scaled kill threshold in `robust_ladder` already helps. Re-tune it on `data/splits_r2/wf`, and keep the holdout untouched.
4. **Treat round-1 structural tricks as calm-only.** That covers exact S, queue seniority and pre-join: the round-1 generator still describes the calm parts reasonably well.
