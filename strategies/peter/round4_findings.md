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
