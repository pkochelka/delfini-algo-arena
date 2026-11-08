# Round 4: what changed vs round 3, and tradable hypotheses

Written 2026-11-08 from the data-explorer exports `data_explorer/round4/` and `data_explorer/compare-round3-round4/`
(whole tapes, 6000 s each), cross-checked against the agents' board `marek/research/round4/NOTES.md` (in progress, so
those numbers may still move). All numbers are in ticks and seconds. "r3" = `data/round3.csv`, "r4" = `data/round4.csv`.

**Bottom line.** The book and the order flow look almost the same as round 3. The price process does not. In round 4 the
mid **trends**: 1 s returns are positively autocorrelated out to about 5 s and give most of it back over the next 25 s.
Takers now earn money against makers, and every maker markout bucket is worse. Round 3's stack assumed the opposite
(S is a random walk plus impact that decays, and future S moves can't be predicted). Two new non-S agents (a 160-lot
TWAP and a 15-lot touch joiner) could also feed the S filter wrong data, which is the same kind of mistake that cost
round 3 40 % of its edge.

## 1. What did not change (round 3's machinery carries over)

| | r3 | r4 |
| --- | ---: | ---: |
| Events / s, cancel / insert | 402, 0.954 | 400, 0.952 |
| Spread mean, share at 1 tick | 1.25, 81.6 % | 1.26, 81.0 % |
| Touch size, insert size, aggressive size | 968, 76.7, 135 | 965, 78.4, 137 |
| Passive fill rate | 15.2 % | 15.5 % |
| Where passive orders go (0 / −1 / −2 ticks from best) | 516k / 361k / 134k | 515k / 351k / 129k |
| Queue imbalance distribution | U-shaped | same |
| Latency: adverse mid move during 5 ms delay | 0.013–0.016 | 0.016–0.018 |
| Latency: fill before a 5 ms cancel lands | 0.26–0.28 % | 0.25–0.27 % |
| Calm minutes: σ 1 s, 10th percentile | 1.13 | 1.17 |
| Maker markout, low-vol regime | −0.23 | −0.33 |

From the board: maker rate 195 /s, taker size lognormal (4.12, 1.39), HMM regime rates (takers 8.1 / 12.2 /s, makers
187 / 222 /s), the 2.000 ms ladder maker, cancel-life quantiles, and the 28 % volatile share are all as in round 3.
**So round 3's calm machinery (calm quoting, ladder filter, regime HMM) should port almost unchanged.**

## 2. What changed

### 2a. The price trends (the main change)

| | r3 | r4 |
| --- | ---: | ---: |
| σ at 0.1 / 1 / 10 / 60 s (ticks/√s) | 4.05 / 4.49 / 4.63 / 4.41 | 4.92 / 6.17 / **8.74** / 7.06 |
| Variance ratio VR(2/5/10/30/60) vs 1 s | 1.11 / 1.19 / 1.13 / 0.91 / 0.86 | 1.29 / 1.74 / **1.98** / 1.64 / 1.41 |
| ACF of 1 s returns, lags 1 / 2 / 3 | 0.11 / 0.02 / −0.02 | **0.29 / 0.15 / 0.10** |
| Sum of 1 s ACF, lags 1–5 / lags 6–30 | 0.09 / −0.20 | **0.64 / −0.52** |
| Sum of 100 ms ACF, lags ≤1 s / 1–5 s / 5–10 s | 0.23 / 0.00 / −0.12 | 0.58 / **0.69** / −0.18 |
| ACF of consecutive mid changes, lag 1 | 0.06 | **0.32** |
| Impact after an aggressive order, 1 / 5 / 60 s | 0.86 / 0.93 (peak 2 s) / 0.65 | 1.58 / **2.56 (peak 5 s)** / 0.96 |
| Taker PnL per lot vs fill price, 1 / 5 / 10 s | +0.10 / +0.06 / −0.09 | **+0.79 / +1.77 / +1.38** |
| OFI → next-bin R², 1 s / 5 s / 30 s | 0.006 / 0.000 / 0.011 | **0.045 / 0.016 / 0.053** |
| Queue imbalance → 1 s mid move, extreme buckets | −0.54 / +0.50 | −0.81 / +0.70 (5 s: −2.1 / +1.3) |
| Share of minutes with σ 1 s > 5 | 27 % | 46 % (median minute 4.3 vs 2.6) |
| Mid range over the tape | 9768–10130 | 9188–10023 (end 9554, −446) |

Reading: if the last 1 s moved r, the next 5 s move averages about **+0.64 r** and the following 25 s about **−0.52 r**.
That is momentum out to roughly 5–10 s, then a partial give-back. Calm minutes look as before (low quantiles match), so
the change sits in the volatile regime. The board says the same: the residual after removing fill impact has VR up to
4.2 at 40 s in volatile (r3 reverted to 0.18), past 1 s mid move → next 5 s corr **+0.37 in volatile** (r3 +0.04), runs
of 5–20 s trends then plateau, |net| per volatile segment 31 ticks (r3 5.5). The fill-impact kernel itself is about the
same as r3. So S has a hidden **velocity/drift state**, and takers lean with it (taker sign ACF lag 1 = 0.13 vs 0.05).

### 2b. Makers lose more, and small takers are now toxic too

| Maker markout (vol-weighted) | r3 | r4 |
| --- | ---: | ---: |
| Spread 1 / 2 / 3–5 / >5 before the fill | −0.70 / −1.58 / −2.16 / −3.85 | −1.48 / −2.07 / −3.11 / **+0.72** |
| Vol regime low / medium / high | −0.23 / −0.83 / −1.71 | −0.33 / **−1.82 / −2.82** |
| Aggressor size 1–5 / 6–20 / 21–100 / >100 | **+0.26 / +0.22** / +0.06 / −1.12 | **−0.38 / −0.61** / −0.69 / −1.84 |
| Maker's share of touch 0–.2 / .4–.6 / .8–1 | −0.97 / −0.70 / −1.12 | −2.18 / −1.15 / −1.67 |

Queue fill within 1 s also drops (1–5 lots: 0.56 → 0.48, 6–20: 0.53 → 0.47): in a trend, the price walks away from a
resting quote before it gets filled, and it fills when the price runs through it.

### 2c. New agents (board, being fitted)

- **TWAP taker:** IOC of exactly **160 lots every exactly 500,000 µs**, one side per episode, 37 episodes of 22–58
  children (10.5–28.5 s), 12 % of time, limit at the touch / touch+1, 96 % filled. In the explorer this shows up as the
  146–174 lot size bin: buy 1026 → 1840, sell 1098 → 1689. It is *not* the source of the sign ACF (the ACF is the same
  without it).
- **15-lot touch joiner:** holds one 15-lot order at the same-side touch and re-posts 1 µs after it is filled or
  cancelled. About 750 orders in about 46 chains, episodes of 1–20 s, often "buy takers while the price falls". This is
  where the new 1 µs taker → insert pairs (382 vs 13) and most of the extra cancel → insert pairs (690 vs 360) come from.
- **Shocks:** 28 droughts (r3 similar), duration 3.02 s, gaps 148.6–296.9 s (mean 209), first onset 167.4 s; the crude
  jump sd is **≈ 66 (r3 ≈ 50)**. Sweep depth tail is heavier (k 1.58 vs 1.74; ≥2 levels 3.8 % vs 2.8 %).
- Volatile *starts* cluster within ±0.03 s of an integer second (ends don't).
- Depth 5–19 ticks behind the best is 15–150 % thicker (e.g. 15 ticks: 1.95 → 4.47). One-sided book share 0.03 % → 0.06 %.

## 3. Hypotheses, ranked by expected value per hour of work

Each has: the claim, why the data supports it, how to test it on `data/round4.csv`, and when to drop it. Baseline first
(step 0). Accept only on the real r4 tape (no synth tapes), split calm / volatile / shock, and also
run one regression on r3 so the change doesn't overfit r4's seed.

### Step 0. Baseline: how much does round 3's best lose on round 4, and where?
Run `strategies/marek/round3/r3_push.py`, `r3_hmm.py` and `r3_dev_best.py` on r4 with the eval3 split (calm / volatile /
shock / breach). Also log the S-filter error against `marek/research/round4/s_dynamics/s_path.parquet` by regime and
inside vs outside TWAP episodes. The rest of the list goes in the order of where that PnL is lost.

### H1. Remove the new non-S agents from the S filter (cheap, possibly large)
- **Claim:** TWAP children and 15-lot joiner orders bias the S filter. Treat them like the r3 ladder rungs.
- **Why:** the r3 filter reads each taker's limit price as round(S ± 1). TWAP limits are touch / touch+1, not S-based,
  and they come in clusters of 22–58. The joiner posts at the touch whatever S is. In r3 one maker the filter didn't
  know about cost 40 % of the edge (ladder fix +280k). s_dynamics v0 already excludes 1 µs reactive makers and found
  volatile taker band coverage 86 % vs r3's 91 %, which looks like the same kind of contamination.
- **Test:** online detectors. TWAP: a 160-lot IOC with a previous 160-lot IOC on the same side exactly 500,000 µs before
  (±1 µs) → drop it, and every later child, from the filter's taker likelihood. Joiner: a 15-lot insert 1 µs after a fill
  or cancel → drop it from the maker bounds. Compare filter error vs s_path and PnL.
- **Drop if:** filter error inside TWAP windows doesn't move, and PnL changes by less than ±5k.

### H2. Give the fair value a velocity term (the main lever)
- **Claim:** F(t) = S_hat + v·h, where v is a filtered S velocity (local-linear-trend Kalman, or an EWMA of S_hat
  changes over 0.5–2 s, by regime), predicts the next 1–5 s better than round 3's S_hat. Round 3's decaying-impact
  model points the other way.
- **Why:** 1 s ACF sum 0.64 over lags 1–5; 100 ms ACF sum 0.69 over 1–5 s; VR 1.98 at 10 s; the board's residual
  momentum (vol beta +0.46 past 2 s → next 2 s, r3 −0.29). In r3 the walk-forward GBM found S moves unpredictable (R² ≤
  0). In r4 they are predictable by construction.
- **Test:** offline first: regress the s_path move over the next 1 / 2 / 5 s on v at several half-lives, by regime, and
  report R² and the stable coefficient. Then plug in as the snipe/quote centre, with separate h_calm and h_vol.
  Expectation from the data: h_calm ≈ 0, h_vol of a few seconds.
- **Drop if:** out-of-sample R² < 0.02 in volatile (split the tape in two halves), or PnL on r3 falls by more than 20k
  (h_vol should come out near 0 on r3).

### H3. Hold the trend side ~5 s, then get out before the give-back
- **Claim:** in volatile, inventory is pinned at the cap most of the time (r3: 90 %). Which side we're pinned on, and for
  how long, decides PnL. Pin with the velocity, keep it about 5–10 s, then flatten or flip.
- **Why:** the impact curve peaks at 5 s (2.56) and falls to 0.96 at 60 s. Taker PnL peaks at 5 s (+1.77/lot) and halves by
  20 s. 1 s ACF lags 6–30 sum to −0.52. In r3 every exit lost because there was no give-back (the carry kept going). r4
  has one, so the "exits are dead" result from round 3 needs re-testing here.
- **Test:** for each volatile entry, the hindsight best hold time as a histogram. Then a rule: exit when |v| decays below a
  threshold or after T_hold ∈ {3, 5, 8, 12} s, reduce-only IOC at edge ≥ 0. Compare against the r3 "never exit" behaviour.
- **Drop if:** every T_hold loses against never exiting, as in r3.

### H4. Take with the momentum (taker-following), limited by room
- **Claim:** after a run of same-side mid changes or a large 1 s move in volatile, an IOC in the move's direction has a
  positive 2–5 s markout even after the 5 ms delay and the spread.
- **Why:** taker PnL vs fill is +0.79 at 1 s and +1.77 at 5 s (r3 +0.10 / +0.06). Tick-by-tick mid-change ACF lag 1 = 0.32.
  The cost: the post-trade touch is already 0.2 ticks worse at 5 ms, and about 0.6 ticks of half-spread.
- **Test:** event study on the tape: after N ≥ 2 same-direction mid changes within 200 ms (or |Δmid 1 s| ≥ k·σ), simulate
  an IOC at the touch +5 ms, mark at +1/2/5/10 s. If it's positive, add it as a take signal that uses room only when the
  snipe signal doesn't (the cap binds, so this is mostly about choosing the side, see H3).
- **Drop if:** the 5 s markout net of entry is under +0.5/lot, or it only wins when the snipe signal already fires.

### H5. Volatile passive quotes: pull or skew the against-trend side
- **Claim:** the r3 volatile passive quotes (vfm20 flatten-only F ± 2) now get run over. Quote only on the with-trend
  side (or widen the against-trend side by about |v|·1 s).
- **Why:** medium/high-vol maker markouts roughly doubled (−1.82 / −2.82). Small takers are toxic now (−0.38 / −0.61 vs
  +0.26 / +0.22). Queue fills only happen when the trend runs through the quote.
- **Test:** the vfm20 component on/off on r4, then a velocity-gated version. Calm quotes: try a light v-skew only if H2 finds
  a nonzero h_calm.
- **Drop if:** gated vfm20 is within ±3k of off.

### H6. Trade the TWAP schedule
- **Claim:** once two 160-lot children 500 ms apart are seen, the next 20–56 children's side, size and time are known
  to the µs. Three uses:
  (a) **Avoid being hit:** don't rest a quote on the side the TWAP hits at t_next (cancel it about 6–10 ms before, given
  5 ms latency) unless the edge vs F covers the child's impact. Re-quote after.
  (b) **Drift:** if the children carry the usual fill impact (board: kernel ≈ r3, G(2 s) vol 0.065, calm 0.017 per
  filled^0.62), 160 lots every 0.5 s is a predictable push on S. Hold the TWAP side while the episode runs.
  (c) **End of episode:** the push stops and half of it decays (r3 p ≈ 0.5, T ≈ 15 s). If that still holds in r4, fade
  after the last child (a missed 500 ms slot means the episode is over).
- **Why:** 37 episodes, 12 % of time, no skipped slots, random phase, both sides.
- **Test:** event study around children k ≥ 3 (S move per child; maker markouts at the touch just before vs not), and
  S over the 30 s after episode end. Only (a) needs no new model.
- **Drop if:** S moves less than 0.1 tick per child and the markout of quotes hit by children is the same as for normal takers.

### H7. Retune the shock sniper for bigger jumps and a known gap law
- **Claim:** shock jumps are about 30 % larger (sd 66 vs 50), so the shock value and the right trigger thresholds are
  larger. Gaps between shocks are 148–297 s, so the drought hazard is near zero for 148 s after a shock.
- **Why:** board shock table; the heavier sweep-depth tail; maker markouts in spreads > 5 flipped from −3.85 to +0.72
  (wide-spread fills right after a shock now pay the maker, so quoting inside a post-shock gap once S is known could pay).
- **Test:** per-shock breakdown of r3_push on r4. Rescale `bm_sw*` / `ss_*` thresholds by 66/50. Add a "no shock possible"
  window [last onset, +148 s] to the drought detector to cut false alarms. Try a post-shock passive quote at S ± 1–2 in
  spreads ≥ 5 after the jump is located.
- **Drop if:** it only moves one or two shocks (that's the shock lottery; judge by the median shock).

### H8. Recheck the calm stack: is the transient-decay assumption still there?
- **Claim:** calm is mostly unchanged, but the board's calm residual VR ≈ 1.0 (r3 0.34) says calm impact may no longer
  revert. Any round 3 code that expects calm reversion (filter decay, take-centre, theta) could be slightly mis-centred.
- **Test:** fit the decay parameters (p, T) on r4 calm with the s_path, and run the r3 calm block with decay off vs on.
- **Drop if:** within ±3k.

### H9. Small tells (do last)
- **Integer-second volatile starts:** raise the calm→volatile switch prior within ±30 ms of an integer second. In r3 an
  exact-regime oracle was worth only +4k, so expect ≤ +5k.
- **Queue imbalance:** its 1–5 s slope is 2–4× r3's, but that is probably the trend showing up in the queue. Use it only
  as an extra input to the H2 velocity, not as its own signal.
- **15-lot joiner as information:** check its fill markouts. If its episodes really line up with "buy takers while the
  price falls", its side may mark the drift's direction. Probably small.
- **Thicker deep book (5–19 ticks):** more exit depth after big moves. Re-test the r3 "oversize outside shocks is dead"
  result only if H3 shows exits are worth more.

## 4. Things not to do

- **Don't bet on the direction.** The −446 tick drift is one random path of a trending S. Round 2 mirrored-price folds gave
  identical PnL, and the strategy should stay symmetric.
- **Don't trust round 3's dead-ends list in volatile.** "Exits always lose", "fading loses", and "future S unpredictable"
  were all measured on a process with no momentum. They still hold for calm unless H8 says otherwise.
- **Don't fit TWAP/joiner agents into the base maker/taker laws.** Exclude them first (H1). The board's fits already do.

## 5. Open questions for the generator agents

1. Is the velocity state a separate hidden process (e.g. an OU drift that switches with the regime), or momentum built
   into the impact law? This decides whether H2 should filter v from S changes or from fills.
2. Do TWAP children carry the same impact per lot as ordinary takers, and does that impact decay?
3. Do volatile segments start with a fresh random drift? If so, the first 1–2 s of each segment hold the velocity
   information. Pair that with the integer-second start tell.
4. Does the jump law for shocks scale with anything (regime, recent drift)?
