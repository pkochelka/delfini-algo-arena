# Round 4 public tape: structure audit (2026-11-08)

Tape `data/round4_public.csv`: 6000 s, 2.40M events, 1.23M orders; same format and duration as round 3.

## What was run

| Output | Command |
| --- | --- |
| `data/splits_r4/wf/` (12 folds, warmup 60 s, first 75%), `hold16/` (16 folds over the whole tape), `pre75.csv`, `hold25.csv` | `split_data.py` (same layout as `splits_r3`) |
| `data/splits_r4/regime/` (25 calm + 30 turbulent stretches, test from 4500 s) | `regime_split.py` |
| `data_explorer/round4_public/`, `data_explorer/compare-round3_public-round4_public/` | `python -m data_explorer.export` (needs `PYTHONUTF8=1` on Windows) |
| `logs/r4_structure.log` (and `logs/r3_structure.log` as the baseline) | `r2_structure.py --folds data/splits_r4/wf/folds.json --hold-parts 4` |
| `tuned/round4_public_model.json` | `r1_model.py` (the round-1 S model does not fit, as on r2/r3: mid − S sd 200) |
| round-3 filter fit | `r3x_fit.py --windows 600:1200,2400:3000,4800:5400` |
| momentum / cycle / taker markouts per split | scratch script, numbers below |

## Headline: round 4 = round 3 + momentum

The microstructure is round 3's. Rates, sizes, spread (81% at 1 tick), touch depth (965), cancel/insert 0.95, gap AR(1) 0.27 and the maker-insert likelihood under the r3x filter (ll −1.01 vs −1.08 on r3) are all unchanged. The price dynamics are not:

| | round 3 | round 4 |
| --- | ---: | ---: |
| σ 1 s / 10 s (ticks/√s) | 4.49 / 4.63 | 6.17 / 8.74 |
| variance ratio of mid at 5 / 10 / 30 s | 1.17 / 1.11 / 0.95 | **1.76 / 1.99 / 1.78** |
| ACF of 1 s mid returns, lag 1 / 2 / 3 | 0.11 / 0.02 / −0.02 | **0.29 / 0.15 / 0.10** |
| jump sign ACF lag 1 / 2 | −0.06 / 0.05 | **+0.26 / +0.17** |
| signed P move after a jump, next 1 s / 5 s | +0.3 / +0.2 | **+5.4 / +13.9** |
| signed mid move after a 1 s move > 10 ticks, next 1 / 5 / 10 s | +1.8 / +0.6 / −2.8 | **+7.0 / +17.2 / +14.1** |
| taker sign ACF (per order) | 0.002 | **0.079** |
| taker markout 1 s / 5 s / 30 s (mid) | 0.47 / 0.41 / 0.23 | **1.17 / 2.00 / 0.81** |
| big-taker (top 10%) markout 5 s | 1.57 | **3.76** |
| jumps (>5 ticks / 100 ms) | 349 | 499 |
| P(another jump within 10 s, given a jump) | 0.77 | 0.82 |
| r3x filter slope of mid change on F − mid, 0.5 / 2 / 10 s | 1.0 / 1.5 / 0.7 | 1.3 / 2.8 / **4.1** |
| public-feature R² for the 1 s mid change (OOS) | 0.06 | 0.14 (AR on past returns alone 0.08) |

- **Moves continue.** Jumps come in same-direction runs, and the price keeps drifting the way it just moved for about 5–10 s, then gives back part of it (the variance ratio peaks at 10 s). The hidden price itself trends: F − mid predicts *more* than the gap at longer horizons.
- **Takers are informed / herding.** Their sign is autocorrelated, and they earn about 2 ticks at 5 s. A passive maker that gets filled in a move is adversely selected much harder than on r3, while a trend follower or a sniper that joins a move earns more. Expect r3-tuned makers to lose edge, and expect `trend_follower`-style or shock-continuation logic to pay.
- The momentum is mostly in turbulence (1 s ACF 0.29 turbulent vs 0.11 calm, where calm is also up from r3's 0.08).

## Per split (12 wf folds + holdout in 4)

- **The momentum is in every split.** 1 s ACF > 0.05 in 16/16 splits (range 0.13–0.47), VR(10 s) > 1.2 in 13/16. On r3 it was 4/16.
- **Taker markout at 5 s is positive in all 16 splits** (0.06–3.5), and higher in turbulent splits (corr 0.64).
- **What varies across splits is the amount of turbulence, not the mechanics.** Jumps per split are 12–72 (CV 0.55, chi² p 3e-22) and the turbulent share is 0.21–0.80. Taker/maker counts are also non-homogeneous (p 4e-59 / 1e-181) because they rise with turbulence (Spearman: taker rate 0.89, touch −0.96, spread −0.91). Calm volatility (CV 0.07), sizes (0.7%), lifetimes and buy share are stable.
- Calm-only 1 s ACF swings between −0.05 and 0.19 by split, so do not tune a calm momentum term on one fold.
- The taker buy share across splits has p = 0.006 (wf03 0.47, wf05 0.48, hold3 0.51). It is borderline and probably tied to the trending folds.

## Anomalies

- **One-sided book:** 188 episodes (r3: 118), median 10 ms, max 222 ms; 136 caused by an IOC sweep, 51 by cancels. Keep the `None`-touch guard.
- **Largest 1 s move: 139.5 ticks at t = 171 s** (jump max 136 ticks/100 ms). Mid − P ranges from −50.5 to +21.5.
- Max inter-event gap 141 ms at t = 5632.6 s (the maker outage before a shock, as in r2/r3; regime_split found 28 outages and 319 jumps; turbulent share 0.545).
- Burstiness: 125 events in one 100 ms bin and 65 takers in one second, far beyond Poisson (dispersion index 3.9/16/40 at 1/10/60 s). This is regime-driven clustering, as on r3.
- Max sizes are normal for the lognormal fits (taker max 14,395, p = 0.92; maker 74,020, p = 0.38). There is no 50k taker outlier like r3's.
- Spread max 138 ticks (one-sided moments). Timestamps: no duplicates, no order-id gaps, sub-second phase uniform (KS p ≥ 0.09).

## Periodicity and trends

- **No usable periodicity.** The strongest periodogram line for maker/taker/cancel counts is 2000 s (3 cycles per tape; r3 gave 1500 s = 4 cycles). A 2000 s sinusoid explains only 19–22% of the 100 s maker-rate and turbulence variance (amplitude 3% of rate). It is the slow wandering of the turbulence level (maker rate vs turbulence corr 0.91 per 100 s), not a clock. Jump times mod 60 s: KS p = 0.019, weak and probably multiple testing. Squared returns: no period.
- **Volatility clusters strongly** (|r| ACF 0.27 at lag 1, still 0.14 at 10 s).
- **No significant trend in rates** (all |trend| ≤ 2%, p ≥ 0.3), unlike r3, whose rates rose 3–8% over the tape. Squared returns −64% (p 0.07), jumps −25% (p 0.15).
- **Price drift −446 ticks** (10000 → 9554), z −0.91 vs a martingale, so within noise. Without the jumps, −414 ticks with z −4.3, a negative drift between jumps (r3: z −2.6). It is driven by 500 s blocks such as [1500, 2000) −326 and [3500, 4000) −248; the last 1500 s rise (+330). Not a constant drift to bet on.

## Extra angles (second pass, 2026-11-08)

Scripts: `r4_angles.py` (all sections, rounds 2/3/4 side by side → `logs/r4_angles.log`) and `r4_more.py` (momentum robustness, Hurst → `logs/r4_more.log`). All three (`r4_deep.py` = the first-pass per-split momentum script) are in `strategies/peter/`; run them from the repo root with tape paths as arguments.

### Generator components: what carried over from round 3

| Component | round 2 | round 3 | round 4 | Verdict |
| --- | ---: | ---: | ---: | --- |
| Shocks (maker drought → jump) | 27 | 27 | 28 | **Same schedule**: drought 2.8–2.9 s at ~55 inserts/s, gaps 149–297 s (U(150,300)), jump ≈ 2.9 s after onset. The first comes earlier (167 s). |
| Shock jump size: sd / mean \|J\| | 42 / 31 | 49 / 35 | **64 / 51** | Bigger jumps, no "no-jump" class (0 of 28 with \|J\| < 3) |
| After-shock continuation 1 / 5 s | −0.5 / −0.1 | +0.3 / +0.7 | +1.0 / +1.1 | Shocks still do not trend: **the momentum is not in the shocks** |
| Ladder maker (exact 2.000 ms chains) | 23 | 43,132 | 42,396 | Unchanged, 94% in turbulence; keep the ladder filter |
| Impact per √filled lot, calm / turbulent | 0.0125 / 0.055 | 0.0133 / 0.054 | 0.0127 / 0.060 | Kick law unchanged |
| Spread recovery after a sweep (spread at 50 ms / 1 s) | 1.42 / 1.30 | 1.39 / 1.33 | 1.39 / 1.36 | Liquidity refills as fast as before |

### The momentum mechanism: trend-chasing takers

- **Takers chase the last second's move.** Mean taker sign given the mid move over the previous 1 s:

  | Past 1 s move | ≤ −10 | −10..−3 | −3..−1 | ±0.25 | 1..3 | 3..10 | ≥ 10 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | round 2 | −0.03 | −0.02 | −0.02 | +0.01 | −0.02 | 0.00 | +0.02 |
  | round 3 | −0.12 | −0.05 | −0.03 | −0.01 | +0.02 | +0.09 | +0.14 |
  | **round 4** | **−0.36** | **−0.19** | −0.05 | −0.03 | +0.09 | **+0.24** | **+0.35** |

  After a 10-tick move, about 68% of takers trade in its direction. Their impact (unchanged law) pushes the price further, so the loop feeds itself.
- **The move runs through the takers, not the makers.** In 100 ms bins with takers, the slope of the bin's move on the past 1 s move doubled (0.018 → 0.038). In bins with no takers it barely changed (0.020 → 0.027).
- **The past 1 s move predicts the next 10 s**, beyond the taker flow: slope +0.36 (round 3 −0.04, round 2 −0.06). At 30 s it is gone (+0.06).
- **Order-sign memory is short, not long-memory (no Lillo–Farmer power law):** ACF 0.13 / 0.07 / 0.05 / 0.02 at lags 1 / 5 / 10 / 20 takers, zero by lag 50. Round 3: 0.05 at lag 1. Calm-only lag 1: 0.05 (r3 0.02).
- **Response function** R(τ) = E[sign · (mid(t+τ) − mid before)] per taker, turbulent: 0.8 / 2.5 / **4.2** / 3.6 / 1.6 ticks at 0.1 / 1 / 5 / 10 / 30 s (round 3: 0.6 / 1.4 / 1.3 / 1.1 / 1.0). The price runs for about 5 s, then roughly 60% of it decays by 30 s (the transient impact, T ≈ 15 s). Calm R(5 s) is 0.46 (r3 0.30).
- **Sweeps are a continuation signal.** A taker that widens the spread from 1 to 2+ ticks is followed by +2.6 ticks at 1 s and +5.0 at 5 s in its direction (round 3: +1.1 / +1.0). This happens 95 times per minute.
- **Hurst (DFA, 1 s returns):** 0.81 at 4–32 s (round 3 0.57), 0.63 calm-only (r3 0.57), and 0.44 at 30–600 s (r3 0.43), so momentum at seconds and mild mean reversion at minutes. The tail index is about 2.2 (calm 3.2); calm kurtosis 2.9 vs 1.6.
- **Trend episodes are not drifts.** Turbulent stretches of 10 s or more look slightly more directional (efficiency 1.02 vs 0.79 on r3; a random walk gives about 0.8). But the first 5 s do not predict the rest of the stretch (sign agreement 35%, slope −0.17). The edge lives at 1–10 s, not at the episode scale.

### What it costs a maker

Realized spread per lot of passive fills (mid-based, maker view):

| | half-spread | 0.1 s | 1 s | 5 s | 30 s |
| --- | ---: | ---: | ---: | ---: | ---: |
| round 3, all fills | 0.96 | +0.05 | −0.91 | −0.86 | −0.22 |
| **round 4, all fills** | 1.02 | −0.07 | −1.64 | **−2.81** | −1.06 |
| round 4, calm | 0.68 | +0.32 | −0.16 | −0.25 | +0.06 |
| round 4, turbulent | 1.29 | −0.39 | −2.81 | **−4.84** | −1.95 |
| round 4, fills 1+ ticks behind the touch | 2.06 | −0.47 | −3.30 | **−6.14** | −1.92 |

- **Calm fills** still about break even (r3 calm at 5 s: +0.02, r4: −0.25).
- **Turbulent and deep fills** are three times worse than on round 3. Being passive in turbulence, especially deep in the book where the ladder sits, is what loses.

### Is the momentum tradeable? Naive IOC round trip

**Rule:** if the mid moved ≥ X ticks in the last L s, IOC at the touch in that direction after a latency; after H s, IOC out at the opposite touch. One trade at a time, per lot, no fees, our own impact ignored.

| Rule | n | round 4 ticks/lot (t) | round 3 | round 2 | r4 at 100 ms / 250 ms latency | r4 splits positive (of 16) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L 1 s, X 10, H 1 s | 608 | **+5.0 (t 16.4)** | +1.3 (3.9) | −1.4 | +4.6 / +4.0 | 15 |
| L 1 s, X 10, H 2 s | 430 | **+8.2 (13.0)** | +1.6 | −1.9 | +7.4 / +6.5 | 15 |
| L 0.5 s, X 5, H 1 s | 917 | +4.2 (16.0) | +0.7 | | +3.8 / +3.3 | 15 |
| L 1 s, X 10, H 5 s | 259 | +12.3 (7.5) | +1.7 | −0.7 | | 13 |
| L 1 s, X 10, H 30 s | 82 | −8.5 | −1.5 | −6.9 | | 7 |

- **Not a shock effect.** Excluding [onset − 0.5, onset + 8] s around the 28 shocks gives +5.5 / +8.8. The trades triggered inside shocks lose (−0.7).
- **Symmetric:** up +4.9, down +5.2.
- **Latency-tolerant:** at 250 ms it still makes 80% of the 5 ms edge. On round 3 the same edge is gone by 100 ms.
- **Size:** the median touch quantity at entry is 86–127 lots. At about 80 lots per trade, the L1 X10 H1 rule alone is worth roughly 600 × 5 × 80 ≈ 240k on the public tape, before our own impact (which pushes the price our way on entry and against us on exit).
- **Holding period:** 1–5 s. By 30 s the price has partly reverted and the trade loses.
- **Implications for a round-4 strategy:**
  - Follow strong 0.5–2 s moves with IOCs and exit within a few seconds.
  - Stop quoting passively against the move in turbulence (skew or pull the side about to be run over), and keep away from deep levels in turbulence.
  - Keep the shock sniper and the ladder filter from round 3; those mechanics are unchanged.
