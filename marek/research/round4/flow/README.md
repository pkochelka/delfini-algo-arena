# Round 4 order-arrival processes

Tape: `data/round4.csv` (6000 s, 1,228,523 inserts, 58,445 IOC takers: 56,955 regular + 1,490 TWAP children). Scripts are
in this folder. Run with `uv run --with pyarrow --with numba python marek/research/round4/flow/<script>.py`.
Run order: `prep.py` (writes `cache/`, git-ignored), `bins.py`, `events.py`, `hmm.py`, `regimes_smooth.py`,
`takers_tab.py`, `attach_s.py`, `ladder_build.py`, `dwell.py`, `shocks3.py`, `export.py`, then anything else.
`takers_tab.py` reads novelty's TWAP flags (`../novelty/cache/round4_takers_tw.parquet`) and s_dynamics' segments
(`../s_dynamics/vol_segments.csv`). Everything that uses S reads `../s_dynamics/s_path.parquet` (v0).

## Summary

Arrivals are round 3's: the same calm/volatile chain, the same taker rates, sizes and limit rule, the same shock schedule.
**One rule changed: the taker's side is no longer a coin flip.** Each regular taker buys with probability

    P(buy) = Φ((S − mid) / 4.42)        (≈ logistic(0.370 · (S − mid)))

where S is the current hidden price and mid is the current book mid. The slope is the same in calm and volatile.
Given S − mid, the signs are independent. That one rule produces all of the sign persistence: a buy kicks S up, the book
lags behind S, so S − mid stays positive for a while and the next takers lean to the buy side. The effect is 3× stronger
in volatile only because S − mid is much wider there (sd 1.90 against 0.51 in calm), since kicks are larger and S drifts
(novelty). There is no hidden direction state, no meta-order class (apart from the TWAP agent that novelty and makers
found, which is fixed-side and does not follow this rule), and no chasing of past moves.

| Component | Calm (70 % of time) | Volatile (30 %) | Shock |
| --- | --- | --- | --- |
| Regime dwell | mean 135 s (CI 97–198), exponential (KS p 0.51) | mean 56 s (CI 40–82); not exponential (KS p 0.02, CV 0.60), gamma k 2.7 fits (p 0.97); n = 31 | independent of regime |
| Regular takers per side | 3.94 buy / 4.03 sell /s | 5.93 / 6.00 /s | drought 9.4 /s total (normal); burst ≈ 70 extra over 3.5 s |
| Total rate vs S − mid | flat | flat | — |
| **Side** | **P(buy) = Φ((S − mid)/4.42)**, same law | same | jump head: ≈ 4.5 extra jump-side takers in 0–0.2 s, not from this law |
| Size | rounded LN(4.094, 1.400), all states (r3 4.094, 1.397) | same | same |
| Limit, regular | round(S ± 1): 94.6 % | round(S ± 2.55): 87.2 % | — |
| Limit, aggressive tail | 5.3 %, excess mean 3.2 | 12.4 %, excess mean 6.5 | — |
| TWAP children (novelty/makers) | 160 lots every 0.500 s, one side per episode, 0.28 /s average | 0.23 /s | same limit rule, no tail |
| Maker inserts (1 s HMM) | 187 /s | 222 /s | ≈ 54 /s for 3.0 s |

## Evidence

### 1. Regimes (`hmm.py`, `regimes_smooth.py`, `dwell.py`)
- 1 s Poisson HMM, shock windows −2…+20 s masked. Joint taker + maker, K = 2: takers 8.27 / 12.33 /s, makers
  187.2 / 221.9 /s (r3 7.9 / 11.9 and 188 / 222). As in r3, K = 3–4 only split volatile into 2–4 s ladder clusters.
- The smoothed 1 s segmentation (`regimes_1s.csv`) agrees with s_dynamics' S-based `vol_segments.csv` on 97.7 % of
  seconds. `regimes.csv` uses the s_dynamics switch times: 32 volatile segments, 29.8 % of the time.
- Dwell, uncensored segments (n = 31 each): calm 6.9–446 s, mean 135; volatile 6.0–138.5 s, mean 56. Calm is exponential
  as in r3. Volatile is more regular than exponential (CV 0.60): gamma k 2.7, scale 20.7 fits with KS p 0.97 against
  exponential p 0.02. With 31 segments this is suggestive, not settled. In r3 the volatile dwell was exponential (p 0.77).
- 12 of 32 volatile starts sit within 0.05 s of an integer second (3.2 expected); ends do not (s_dynamics and novelty
  saw it too). Open.

### 2. The side rule (`sign1.py`–`sign7.py`, `sidelaw.py`, `sidelaw2.py`, `sidelaw3.py`, `sidecheck.py`, `ratex.py`, `sidelaw_fy.py`)
**What the persistence looks like.**
- Sign ACF lag 1–5: 0.079, 0.072, 0.064, 0.057, 0.048 (all takers). Calm 0.033 → 0.025; volatile 0.160 → 0.099.
  Removing the TWAP children changes nothing (novelty).
- Pair correlation of signs by time separation (volatile): 0.185 at < 50 ms, 0.097 at 0.25–0.5 s, 0.068 at 0.5–1 s,
  0.034 at 1.5–2 s, 0.013 at 3–4 s, 0 by 5 s, slightly negative (−0.01) at 6–12 s. Calm: 0.03 at < 0.1 s, 0 by 1.5 s.
- The total count is Poisson (volatile DI 0.97 at 1 s, ex-shock), while the buy − sell imbalance has variance
  2.4× (1 s) to 4.7× (5 s) the count. So the split moves, not the rate.
- After a taker in volatile, the same-side rate is ×1.25 and the opposite side ×0.85 for 0–50 ms (relative to 4–8 s
  later); the total is ×1.04. That is a shift in the side probability, not Hawkes excitation or child orders.

**Alternatives tested.**
- *Chasing past moves (trend followers):* rejected. Shock jumps are 20–145-tick S moves that no taker caused. From
  +0.2 s to +5 s after a jump the jump-side share is 0.50–0.52 (`jumpchase.py`). A taker who chased S or mid moves over
  0.25–5 s would be almost all on the jump side. In logits, past S moves over 0.1–20 s add 0–13 nats once S − mid is
  in (volatile: 996 → 1009).
- *Hidden direction state or meta-orders:* rejected. Given S − mid, the residual sign ACF is zero: lags 1–4 calm
  −0.004/−0.002/0.001/−0.001, volatile 0.002/0.003/0.006/0.008. Past signs (EWMA 0.25/1/4 s) add 5–20 nats on top of
  S − mid, against 630–996 for S − mid alone. A latent state would leave residual correlation.
- *Informed by S relative to the book:* accepted. In volatile, corr(sign, S − mid) is 0.30; in calm it is 0.093.
  On r3's final S the same fit gives slope +0.015 (calm) and −0.001 (volatile), so this is new in round 4.
- *Additive "arbitrage" class* (extra takers on the cheap side, with the coin-flip takers unchanged): rejected. In
  regime interiors, with TWAP and shock windows removed, the volatile aligned rate rises from 6.1 to 10.4 /s as |S − mid|
  goes from < 0.25 to > 5. The against rate falls from 5.9 to 1.1 /s, and the total stays at 11.5–12.3 /s. Calm is the
  same (total 7.7–8.8 /s).
- *Look-ahead (takers who know future S):* not supported. S lagged by 0.05 s loses 30 % of the likelihood and S lagged
  by 0.2 s loses all of it, so the rule uses current S. Novelty finds the limits also reference current S with no lead.

**The law.**
- Reference price: the plain mid. Fitting S − bid and ask − S separately gives coefficients 0.207 and −0.163 pooled
  (0.187 / −0.183 in volatile). The microprice and the last trade add nothing (−0.017, +0.007). (S − mid)/spread fits
  worse (1096 against 1133 nats).
- Takers use the full S = F + Y: a transient-impact proxy Y added to the logit gets coefficient 0.000.
- Functional form: logit slope 0.370 calm, 0.370 volatile, 0.369 pooled. The probit (σ = 4.42) is 1.6 nats better
  than the logit. The mixture "π informed takers who trade on sign(S − mid + noise), plus coin-flip takers" goes to
  π = 1, so the probit stands. Calibration by 0.5-tick bin over x = S − mid:

  | x | −5…−3 | −3…−2 | −1…−0.5 | 0…0.5 | 0.5…1 | 1…1.5 | 2…3 | 3…5 | 5…8 |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | observed P(buy) | 0.203 | 0.299 | 0.434 | 0.530 | 0.565 | 0.592 | 0.701 | 0.815 | 0.914 |
  | Φ(x/4.42) | 0.200 | 0.292 | 0.438 | 0.520 | 0.561 | 0.607 | 0.708 | 0.800 | 0.905 |

- The law is stable across subsets. Probit σ: calm 4.32, volatile 4.43; post-jump 3–8 s 4.62; spread 1 → 4.30,
  spread 3–4 → 4.12; size < 30 → 4.19, ≥ 150 → 4.71; 1000 s thirds 4.46 / 4.36 / 4.44.
- It is robust to the S estimate. Using the causal s_filt, or S at the insert before the taker (which excludes the
  taker's own limit as evidence), gives corr 0.31–0.34 in volatile (s_hat 0.30). Novelty's check with s_filt at the
  row before: slope 0.368 / 0.403.
- **The aggressive tail leans harder.** Probit σ is 4.9–5.1 for takers with regular limits and 1.8–2.35 for tail
  limits. P(tail) rises with alignment xa = sign·(S − mid): calm 0.04 → 0.19 and volatile 0.06 → 0.23 from xa < −0.5 to
  xa > 1.5 (logit slope 0.48 per tick in calm, 0.21 in volatile). The tail excess itself does not depend on xa.
  `valuation.py` compares joint (side, tail) models:

  | Model | Params | LL relative to M0 |
  | --- | --- | --- |
  | M0: side law, tail independent | 3 | 0 |
  | M1: "noisy valuation". V = S + N(0, 4.9); side = sign(V − mid); tail iff \|V − mid\| > 9.5 (calm) or 8.1 (volatile) | 3 | +173 |
  | M2: free logit of P(tail) on xa | 5 | +215 |

  So M1 captures most of the effect but not all of it.
- Size does not depend on alignment: mean log size 4.08 aligned against 4.06 opposed in calm, 4.07 against 4.17 in
  volatile.
- It also explains the "unfilled IOC ⇒ next taker on the other side" pattern. In volatile, corr(s_i, s_{i−1}) after an
  unfilled predecessor is −0.54; the law predicts −0.44. In calm it is −0.13 and the law predicts −0.14. An unfilled
  buy means the ask sits above round(S + k), so S is below the mid.
- The TWAP agent's side does not follow the law: the first child agrees with sign(S − mid) in 21 of 37 episodes.

### 3. Taker limits, sizes, rates (`limits.py`, `tail.py`, `takers_r4.py`)
- Limit relative to S (s_sd < 0.3, ex-shock):
  - Calm: best k = 1.00, 94.6 % in band, tail 5.3 % with excess mean 3.2.
  - Volatile: best k = 2.55, 87.2 % in band, tail 12.4 % with excess mean 6.5. The r3 values were 94.5 % / 4.9 % / 3.4
    and 88.9 % / 9.3 % / 6.3. The larger volatile tail share follows from the tail-alignment coupling above, because
    aligned takers are more common in volatile.
  - TWAP children: the same k (99.6 % / 99.2 % in band) and no tail.
- Size: rounded lognormal MLE (4.094, 1.4005) on regular takers, the same in both regimes. 66 regular takers of exactly
  160 lots remain once the TWAP children are removed.
- Rates per side, regular takers, ex-shock: calm 3.94 / 4.03, volatile 5.93 / 6.00. Per 1000 s block the total runs
  7.87–8.17 (calm) and 11.62–12.15 (volatile). Buy share 0.495 / 0.497.

### 4. Shocks (`events.py`, `shocks3.py`, `jumphead.py` → `shock_events.csv`)
- **Count and spacing.** 28 droughts, the first at 167.4 s.
  - Gaps run 148.6–296.9 s (mean 208.6).
  - U(150, 300): KS p 0.07. U(145, 300): p 0.12. Against r3's gaps: KS p 0.10.
  - The law is the same family. Here it sits towards the short end, with 4 gaps of 148.6–150.2 s.
- **Drought.** Lasts 3.02 ± 0.06 s, with makers at 54 ± 5 /s. The taker rate is unchanged (9.35 /s against a 9.4 mix).
- **Jump time.** S jumps at the drought end: t_jump − end = −0.004 ± 0.057 s.
- **Jump size.**
  - J has sd 64.8 (r3 47.2; variance ratio 1.88, p ≈ 0.1) and mean |J| 51. It is normal (KS p 0.84; Laplace p 0.50).
  - The largest jumps are −144 and +137. Only 1 of 28 has |J| < 5, so the r3 "no-jump class" is not visible.
  - |J| is 49 (calm, n = 20) and 57 (volatile, n = 8).
- **No pre-onset tell.** |corr| of J with pre-onset features is at most 0.31. The 2 s pre-onset S move gives 0.31 and
  the 60 s move −0.30, with n = 28.
- **Jump head.**
  - In 0–0.2 s after the jump, 171 of 185 takers are on the jump side. That is ≈ 4.5 extra per shock, and the first one
    comes a median 82 ms after the jump (r3 77 ms).
  - The side law does not explain it. At those takers the book has already re-quoted (mean |S − mid| 0.5–0.9), so the
    law predicts only 0.55 jump-side.
  - This is the same separate component as in r3.
  - From +0.2 s on, the observed jump-side share equals the predicted one (0.505 against 0.515 at 0.2–0.5 s, 0.509
    against 0.509 at 1–2 s).
- **Burst.** 70 ± 15 extra takers in the 3.5 s after the jump (r3 66). It now correlates with |J| (corr 0.44; extra ≈
  62 + 0.17·|J|; volatile 0.87 with n = 8). r3 gave 0.02. Low n, so medium-low confidence.
- `shock_events.csv` columns: onset, gap, drought end/dur/maker rate, t_jump, jump lags, jump, regime at jump, drought
  taker rate, burst count and buy share, first jump-side taker latency, pre-onset features, post-60 s move.

### 5. Ladder table (`ladder_build.py`, `export.py` → `ladder_events.csv`)
- 21,842 multi-rung ladders holding 66,003 inserts (5.6 % of maker inserts).
  - Lengths: 2 → 10.8k, 3 → 4.7k, 4 → 3.2k, 5 → 1.6k, 6 → 1.5k, 7 and longer → 80.
  - Rate: calm 0.87 /s, volatile 10.2 /s. Side split 0.503. Steps are +1 in 96 % of links.
- Makers own the rest: 3 agents instead of 4, and rung size ×4/3.

### 6. Real-time prediction (`predict.py`, `toxicity.py`)
- The setup: a 0.1 s grid, ex-shock, by regime. The features use only causal information: s_filt − mid, taker
  imbalance EWMA (0.25 s and 1 s, non-TWAP), the side of the last unfilled IOC within 1 s, and past mid moves over
  0.5 and 2 s.

  | Target | Regime | Corr with causal S − mid | R², S − mid only | R², tape-only set | R², all |
  | --- | --- | --- | --- | --- | --- |
  | Next-taker sign | calm | 0.08 | — | — | — |
  | Next-taker sign | volatile | 0.29 | — | — | — |
  | Next 1 s mid change | calm | — | 0.20 | 0.08 | 0.21 |
  | Next 1 s mid change | volatile | — | 0.40 | 0.33 | 0.45 |
  | Next 5 s mid change | calm | — | 0.012 | 0.005 | — |
  | Next 5 s mid change | volatile | — | 0.16 | 0.16 | 0.20 |
  | 20 s | either | — | about 0.01–0.02 | — | — |

  For the next-taker sign, the tape-only features reach 0.05 (calm) and 0.22 (volatile).
- Who wins per filled lot. These are S-tick markouts from the taker's side, lot-weighted:

  | Volatile takers by alignment xa | Share of lots | At fill | 1 s | 5 s | 20 s |
  | --- | --- | --- | --- | --- | --- |
  | xa ≥ 2 | 20 % | +1.4 | +11.4 | +21.9 | +11.9 |
  | xa < −1 | 9 % | −2.6 | −2.6 | −7.5 | −5.9 |

  In calm almost all lots have |xa| < 1, and the spread between groups is small (−0.4 to +0.6 at 5 s).

## Validation
- Given S − mid, the residual sign ACF is zero in both regimes, and the calibration table matches from x = −5 to +8.
- The law is regime-free: one σ for both regimes, and the volatile ACF is larger only because S − mid is wider there.
- On round 3 the same fit finds nothing (slope ≤ 0.015), and r3's sign ACF is ≈ 0.
- The flow-only and S-only regime segmentations agree on 97.7 % of seconds.
- Rates, sizes and limits reproduce r3's numbers to the second decimal.

## Confidence
- **Side law:** high. That P(buy) = Φ((S − mid)/σ) with σ ≈ 4.4, regime-free, is the mechanism. The exact form
  (probit or logit, plain mid or a slightly asymmetric bid/ask weighting) is medium. The tail coupling and the M1
  valuation story are medium.
- **Rates, sizes, limits and the shock schedule:** high.
- **Medium to low:**
  - Volatile dwell is non-exponential (n = 31).
  - J sd is larger than r3's (p ≈ 0.1).
  - Burst size scales with |J|.

## Still open
- **A closed-loop simulation.** Side law + impact + book lag + novelty's S velocity should reproduce the observed sign
  ACF (0.079 at lag 1) and the 5 s imbalance-to-move correlation. That needs the makers' book model, and it has not been
  run. An open-loop simulation with the observed S − mid path gives about half of the ACF (0.081 against 0.161 in
  volatile). That is expected, because it removes the feedback from each sign to the next S − mid.
- **The exact tail rule.** How P(tail) depends on alignment, and whether one "valuation" draw sets both side and tail.
- **The S velocity found by novelty.** The side law is stated in terms of S, whatever drives S. This is s_dynamics' job.
- **Integer-second volatile starts**, and the volatile dwell law.

## Trading implications
1. **Takers are now informed relative to the book.** The book they trade against is the market's stale mid, not S.
   - A quote that lags S is hit with probability Φ(lag/4.42) per arriving taker, against 0.5 in r3.
   - Example: in volatile, a bid sitting 3 ticks above S takes about 80 % of the 12 takers/s; the other side sees
     about 2/s.
   - So keep quotes centred on S (ideally on S plus its velocity), not on the book mid. Pull or skew the side that
     the book's lag makes cheap.
2. **Real-time detection without S.** In volatile:
   - The 0.25 s taker imbalance predicts the next sign with corr 0.22.
   - An unfilled IOC means S sits beyond the opposite touch by more than k, so the next takers go the other way
     (corr −0.54).
   - The IOC limit itself pins S: round(S + 1) in calm, round(S + 2.55) in volatile.
   - With an S filter, S − mid predicts the next sign with corr 0.29, the next 1 s mid change with R² 0.40 and the next
     5 s mid change with R² 0.16.
   - In calm everything is about 3× weaker.
3. **Toxicity is concentrated.** In volatile the lots taken on the cheap side (xa ≥ 2, 20 % of lots) gain about 22
   S-ticks per lot by +5 s. Being on the stale side during a volatile move is the main loss; calm flow is close to r3.
4. **Persistence fades after about 5 s and turns slightly negative at 6–20 s** (pair corr −0.01; novelty sees
   reversal of the move). Lean with the flow for a few seconds, not longer.
5. **Shocks:** the r3 playbook carries over, with bigger jumps (sd 65).
   - Pull before onset + 3.0 s.
   - Expect ≈ 4.5 jump-side snipers in the first 0.2 s.
   - After that the burst follows the side law. Once the book has re-quoted it is symmetric, so quote both sides at
     the new S.

## Files
- `regimes.csv`: segments from the s_dynamics switch times (start, end, regime, dur, censored, shocks_inside).
- `regimes_1s.csv`: the 1 s joint HMM segmentation, smoothed.
- `shock_events.csv`: one row per shock (see section 4).
- `ladder_events.csv`: one row per multi-rung ladder.
- `rates_60s.csv`: per-minute rates by stream.
- `cache/takers_s.parquet`: one row per taker. Columns: t, sg, vol, px, filled, book, mid, reg, shock_dt, tw, s_hat,
  s_sd, s_filt.
- `cache/s_series.npz`: S at every insert.
