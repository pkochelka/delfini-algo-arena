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

## Arrivals: which point process

The scripts are in the scratchpad and were not kept. The numbers below come from `regime_split.py` and the analysis described here.

- **Calm periods are plain Poisson.**
  - Maker inserts arrive at 179.8/s: gap CV 1.004, KS p 0.91, dispersion index 1.0–1.3.
  - Takers arrive at **4.0/s per side**: CV 1.00, KS p 0.6–0.98. These are round 1's numbers.
  - No self-excitation: after a calm taker, the taker rate is 0.91–1.0× the baseline.
- **Turbulent periods bring extra takers.**
  - Takers run at 12.4/s, and their gaps have CV 1.10.
  - The rate is driven by recent price moves. After a 1 s mid move of 0–2 ticks it is 8.2/s, 4–8 ticks gives 12.4/s, 8–16 ticks 13.9/s, and above 16 ticks 30.8/s.
  - A Hawkes fit to all takers gives a branching ratio of 0.52 with a 0.73 s half-life, but its residuals are only borderline (KS p 0.065).
  - A 3-state Poisson hidden-Markov model fits better. Its states are 8.0/s for 84% of the time (mean stay 112 s), 12.1/s for 15% (21 s) and 36/s for 1% (2.2 s), and each state is Poisson inside (dispersion 0.97–1.04).
  - Best description: **a regime-modulated Poisson process whose intensity reacts to price moves.**
- **Maker outages.** 27 times on the tape, the maker insert rate drops from 180/s to **53/s, on both sides equally**, for **3.0 s**.
  - Inside an outage, arrivals are Poisson again: KS p 0.19.
  - The cancel rate falls to about 60/s, so all market events go from about 370/s to about 120/s.
  - Outages come about every 200 s. The gaps between them have CV 0.40 (KS against exponential p < 0.001), so they are quasi-regular rather than Poisson.
  - **Each outage announces a jump.** Makers return 2.95 ± 0.28 s after the onset, and the price jumps exactly then: 23 of 27 outages are followed by a jump of 5 ticks or more, and every big jump had one.
  - The jump at the end of an outage is roughly Exp(mean 31): KS p 0.96, median 18.5. Up share 0.57. Its direction could not be read from anything during the outage.

## Regime split and the regime strategy

### Regime split

`regime_split.py` writes `data/splits_r2/regime/`: one file per stretch, with a seeded book and a 30 s no-trade warmup. Dev is before 4500 s and test is from 4500 s on.

| Regime | Dev | Test |
| --- | --- | --- |
| Calm | 23 stretches, 2,766 s | 6 stretches, 775 s |
| Turbulent | 23 stretches, 1,539 s | 8 stretches, 526 s |

- A turbulent stretch runs from 5 s before an outage or jump to 30 s after it, with neighbours merged if they are closer than 30 s.
- Run any strategy on it with `regime_eval.py`.

### The round-1 strategies, split by regime (dev)

| Strategy | Calm | Turbulent |
| --- | ---: | ---: |
| `sig_ladder` | +78k | −262k |
| `q_ladder` | +69k | −256k |
| `queue_mm` | +62k | −204k |
| `robust_ladder` | +70k | −87k |

### `regime_mm.py`: `sig_ladder` plus a regime layer

**How it detects the regime:**
- **Outage alert.** It counts market events (callbacks) over the trailing 300 ms; 60 or fewer triggers the alert. The alert fires about 0.2–0.3 s after the outage starts, about 2.7 s before the jump.
- **Turbulence.** A mid range of 5 ticks or more within 500 ms starts it, and it ends after 10 s without another.

**What it does:** in an alert or in turbulence it cancels all quotes and unwinds the position passively, switching to an IOC after 5 s. Otherwise it quotes `sig_ladder` unchanged; `regime: 0` gives `sig_ladder` back bit for bit.

| | Calm | Turbulent |
| --- | ---: | ---: |
| Dev | +88.9k (23/23 stretches won) | +11.8k |
| Test, untouched | +29.0k (6/6, same as `sig_ladder`: no false alarms) | +5.0k (`sig_ladder` −89.7k, `robust_ladder` −33.2k) |

- **Round-2 walk-forward, 12 folds:** **+108.9k, 12/12 folds won**, t 12.9, stitched drawdown 2.3k. Compare `sig_ladder` −174k and `robust_ladder` +12.7k.
- **Full round-2 tape:** +149.4k, no breach.
- **Round-1 regression:** costs 0.4% (301.2k vs 302.5k over wf01–10).

### What the sweeps showed (dev, one parameter at a time)

**What matters:**
- **The turbulence filter.** Without it, turbulent stretches lose −240k.
- **The outage alert.** It is worth about +15k in turbulent stretches: −12k without it.
- **Staying out.** Quoting through turbulence loses, even at a small cap: −33k with cap 20, −106k with cap 50.

**What doesn't:**
- Flattening on the alert: ±0.
- An event threshold of 45 instead of 60: ±0.
- An event threshold of 75: costs 19k in calm stretches through false alarms.
- A move threshold of 7: about −8k.
- A move threshold of 10: −90k in turbulent stretches.

### Market taking: tried, left off (`sn_on`)

The sniper takes stale quotes by IOC once the book has moved at least `sn_jth` ticks from the price at the alert.

- On dev it adds about +1k at best, which is noise; its worst stretch is −6.5k.
- With the trade-based trigger on, it fires falsely inside outages, where the book is thin and normal takers sweep levels.
- When it does fire correctly, the market's takers (IOCs at round(S ± 1), arriving at about 38/s after a jump) have already removed every stale quote before our order lands 5 ms later.
- Winning that race would mean reading the jump from the first maker inserts at the new level within about 20 ms. That was not achieved: the inserts are clipped at the old touch, and for jumps above 20 ticks the new level is outside the visible depth.

## Turbulence: what the other strategies do, and the shock sniper (`fvs_mm.py`)

### All round-2 strategies on the regime split

| Strategy | Calm, dev | Calm, test | Turbulent, dev | Turbulent, test |
| --- | ---: | ---: | ---: | ---: |
| `regime_mm` (sit out) | +89k | +29k | +12k | +5k |
| `marek/round2/r2_ladder` | +92k | +22k | +176k | +57k |
| `peter/r2_mm` | +85k | +24k | −169k | −56k |
| `peter/fv_mm` (working tree, 20:40) | +162k | +42k | +273k | +57k |

`r2_ladder` and `fv_mm` were developed on the whole tape, so their test columns are not fully clean. Their per-stretch turbulent PnL correlates 0.97: they earn from the same places.

### Where turbulent PnL comes from

- **Taking quotes that are stale against a fair-price estimate.** In the volatile regime makers never quote within 0.70 of S (Marek's generator fit, `marek/research/round2/round2_generator.md`), so S stays trackable while stale touches are frequent.
- **`fv_mm` without taking:** turbulent dev drops from +176k to +9.5k.
- **`r2_ladder` by mode:**

  | Volatile mode | Turbulent dev |
  | --- | ---: |
  | Flat only | +14k |
  | Wide quotes only | +41k |
  | Snipe only | +44k |
  | Both (the default) | +176k |

- **Sitting out (`regime_mm`) leaves almost all of it on the table.**

### What `fv_mm` misses: the shocks

- Over 27 shocks `fv_mm` captured about 18k, against 79k for 100 lots × |jump|.
- Its grid re-centres only on far maker inserts. For big jumps those are outside the 20-tick public depth.

### The visible tell

- The makers' no-cross clip gives the jump away: after an up-jump every new bid is clipped to stale ask − 1, several ticks above the old bid of the wide drought spread. A down-jump mirrors it.
- The signal is common inside the drought itself, so it is usable only when time-locked: the jump lands at onset + 3.00 s.
- Detection lags the onset by a jittery 0.27–0.41 s. A change-point MLE of the insert rate (180/s → 54/s) over the last 1.5 s of inserts gives the onset to within about 10 ms (2760.067 against 2760.07).
- With a window of [onset + 2.99, onset + 3.10] s: 10 triggers, **10 of 10 in the right direction, all 3–65 ms after the jump**. The other shocks show no clipped insert in the window.

### What it is worth

- The sniper IOC-sweeps the stale side up to stale touch ± 8 ticks with the full room, then lets `fv_mm` manage the position after 0.3 s.
- Turbulent test: +56.5k → **+75.0k**. Turbulent dev: +273.4k → +274.0k, since `fv_mm` already took those shocks. Calm: unchanged.
- Walk-forward over 12 folds: 437.1k → 439.3k (12/12 folds won).
- Full tape: **+591k**, no breach.

### Notes

- The gain is concentrated in a few big shocks (the +96 at 5396 s gives +18k), so expect noise.
- An untimed version, triggering anywhere from 0.8 to 4 s after detection, fired 1–2 s before every jump and cost −21k on dev. The time-lock is the essential part.
- `fvs_mm.py` is a snapshot of `fv_mm.py` (working tree at about 20:40) with the subclass `FVAShock` appended. Re-apply the layer if `fv_mm` changes: it is the block after "fvs_mm:" at the end of the file.
