# Round 4 makers: the 15-lot agent, the ladder and the base makers

Tape: `data/round4.csv`, 6000 s, 1,170,078 maker inserts (G) and 58,445 IOCs (F).
- **S:** s_dynamics' v0 `s_path.parquet` (round-3 model, ladder and 1 µs reactive inserts excluded). Median posterior
  sd 0.06 tick in calm, 0.12 in vol.
- **States** (`states.py`): calm / vol from the per-insert `vol` flag of S v0; drought = shock onset → jump, post = 3 s
  after the jump (`flow/shock_events.csv`).
- **Per-insert class labels for everyone:** `cache/order_class.parquet` (`labels.py`), same row order as
  `s_path.parquet`. cls 0 base maker 1,103,634 / 1 ladder chain 65,689 / 2 15-lot agent 755 / 3 regular IOC 56,955 /
  4 TWAP child 1,490, plus chain ids.

## In short

Round 4's maker side is **round 3's, with the ladder re-split into 3 agents with bigger rungs, plus one new
participant: a passive "15-lot agent" that works a one-sided order at the touch for a few seconds while S drifts its
way.** I also found the TWAP taker on the way: 160-lot IOCs every 0.500000 s (novelty and flow own it now).

| Part | Round 4 | vs round 3 | Confidence |
| --- | --- | --- | --- |
| Base makers | rate ≈ 184 /s, size round(LN(2.66, 1.80)), calm D = 0.069 + Exp(0.444), vol mean D 1.62, lifetimes calm 0.461 s / vol 0.380 s | unchanged | high |
| Base cancels | decision hazard α 1.8 + β 2.7–3.0 · max(0, mv) /s, delay Exp(b ≈ 61); model-free hazard by mv = r3 bin for bin | unchanged (β lower in the MLE, likely S v0) | high / medium (β) |
| Ladder | 3 agents (≤ 3 live orders per level), rung = B·U(0.75, 1.25), calm B ≈ LN(5.33, 0.33), vol B ≈ LN(3.82, 0.27), band touch−2…−7 | 4 agents → 3, B ×1.35 ≈ 4/3, so the same depth per level | high |
| **15-lot agent (new)** | one live order of 15 at the same-side touch; re-posts 1 µs after its fill or cancel row; cancels when the touch improves past it; one side per instance; 31 episodes, calm only | new | high (rules) / medium (start and stop rules) |
| **S drift during the agent's episodes** | S moves the agent's way at ≈ 0.75–1.5 tick/s from its first order to its last fill, then stops | new | high |

## 1. The 15-lot agent (`r15_*.py`)

### Identification (`r15_chains2.py` → `cache/r15_v2.pkl`)
- **Link rule:** a maker insert exactly 1 µs after the end row (fill or cancel) of an earlier agent order, on the same
  side, with volume 15 if that order was filled, or its unfilled remainder if it was cancelled. Seeds are any 15-lot
  maker order.
- This finds **755 orders in 43 instances**: 364 links after a fill, 348 after a cancel, 43 fresh starts. 749 orders
  are 15 lots; 6 are 6- or 10-lot remainders after a partial fill (4 partial fills in total).
- The agent places no other 15-lot orders. With its 749 removed, the 15-lot base count is 15,711, and the mean of the
  14- and 16-lot counts is 15,705.
- So it accounts for both 1 µs patterns from the lead's recon:
  - **taker → insert 1 µs, 382:** 364 are agent re-posts after a fill; the other ≈ 18 are background (r3 13).
  - **cancel → insert 1 µs, 690:** 348 are agent re-posts after its own cancel. The rest, 342, is ≈ 230/µs of chance
    coincidence (the 2–10 µs bins hold 222–264 in r4 and 221–250 in r3) plus a ≈ 110 excess that r3 also had (360 − 230).
    So nothing else is new. The pairs are "73 % same side, 52 % same volume" because the agent's pairs are 100 % / ~99 %.

### Rules

| Item | Value |
| --- | --- |
| Price | **joins the same-side touch at insert** (754 of 755; one start one tick inside). Never improves, never sits behind. |
| Size | 15. After a partial fill and a cancel it re-posts only the remainder. A new 15 follows only a full fill. |
| Re-post | exactly **1 µs after** its own fill (the taker's row) or cancel row, at the touch as it is then. 364 of 403 fills are followed by a re-post; the 39 that aren't end their instance. |
| Cancel | **only after the same-side touch improves past it** (350 of 352 cancels; the other 2 were at the end of an instance). Delay from the touch change to the cancel row: mean 23.1 ms, median 17 ms, p90 50 ms; MLE Exp(358) + Exp(49) /s. That is the base makers' delay, Exp(470) + Exp(52), mean 21 ms (KS p 0.04). It does not cancel when the touch moves away from it: it can't, since it is at the touch, and a fill triggers a re-post instead. |
| Lifetime | filled orders mean 0.43 s, median 0.35 s; cancelled orders mean 0.32 s, median 0.22 s. 53 % of orders end filled. |
| Queue | it joins at the back: median 327 lots ahead at insert (p90 2,144). The takers that fill it are large: median 149 lots (all takers 63). 36 of 403 fills are TWAP children. |
| Side | one side per instance and per episode, never both sides at once. 12 buy episodes, 19 sell. |
| Instances | 43 instances in **31 episodes**. An episode is a run of same-side instances less than 3 s apart (`r15_episodes.csv`); 22 have one instance, 7 two, 1 three and 1 four. A second instance can start while the first is live, so the agent can show 2 × 15 at the touch. |
| Episode duration | median 8.2 s, p10–p90 5.3–12.9 s, range 2.7–19.3 s. 279 s in total, 6.5 % of calm time. |
| Episode starts | gaps p10/p50/p90 26/123/473 s, mean 195 s (≈ Poisson, about 0.005 /s). **All 43 instance starts are in calm** (S v0 vol flag); P ≈ 0.7^43 if they ignored the regime. No relation to shocks: the median distance to the nearest jump is 55 s, against 54.5 s for random times. |
| Instance end | 39 of 43 end on a full fill with no re-post. 4 end on a cancel with no re-post; these are the 4 fastest drifts (15–30 ticks, S left 2–7 ticks beyond the agent's price), and one runs into a vol start. Instance fill totals are 15 × n with n = 1…22. n is consistent with uniform on 1..18 (KS p 0.59). |

So in code it is something like this:

```python
while active:
    post(side, 15 or remainder, price=touch(side))
    wait until the order is filled, or the same-side touch improves past it, then cancel (usual delay)
    re-post 1 µs after the fill or cancel row
    stop after the n-th full clip (n ≈ U{1..18})
```

The stop rule is the weakest part. "n clips" fits 39 of 43 instances. A price target, or "stop when the drift ends",
cannot be ruled out with 43 instances (see §1.3).

### The S drift (`r15_s.py`, `r15_profile.py`, `r15_kick.py`, `r15_last.py`, `r15_episodes.py`)

S (v0) change signed by the agent's side, so + means S moved the way of its orders (up for a bid):

| aligned at | −2 s | −1 s | −0.5 s | 0 | +0.5 s | +1 s | +2 s | +3 s | +5 s | +8 s | +10 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| episode start, mean | −0.24 | −0.21 | +0.04 | 0 | +0.46 | +0.87 | +1.88 | +3.15 | +5.2 | +9.7 | +11.3 |
| episode start, median | +0.22 | −0.27 | +0.11 | 0 | +0.25 | +0.42 | +1.62 | +2.53 | +4.1 | +6.0 | +4.7 |
| episode end, mean | −3.59 | −1.74 | −0.85 | 0 | −0.42 | −0.02 | −0.36 | −0.18 | −0.68 | −1.9 | −3.25 |
| episode end, median | −1.89 | −0.80 | −0.23 | 0 | −0.48 | −0.87 | −1.56 | −2.74 | −4.2 | −4.3 | −5.3 |

- **Per instance:** S moves +6.9 ticks the agent's way, start to end (sd 6.7, t = 6.7, 42 of 43 positive). The median
  rate is 0.74 tick/s (p10–p90 0.26–1.95). The mid moves +6.4.
- **No warning beforehand.** The drift starts at the agent's first order to within about 0.3 s.
- **It stops at the last fill.** It is then partly given back: median −4 ticks over the next 10 s, mean −3.
- **The drift runs on clock time, not on the agent's fills.** At its fills S moves *against* the agent, by −0.19 over
  ±20 ms; that is the kick of the taker that filled it. Between fills, S drifts +0.2 per 200 ms. Measured over order
  lives, the rate is 1.48 tick/s during middle orders, 0.84 during the first and 0.85 during the last (0.61 when the
  last was filled). So the drift ramps up and down at the edges, or S v0 smooths it.
- **The rest of the market looks normal in these windows.** Regular takers 8.0 /s (other calm 8.2), base makers 185 /s
  (183), ladder 2.0 /s (2.1). Takers lean only 53 % to the agent's side. That fits flow's side law
  P(buy) = Φ((S − mid)/4.42) with S running ahead of the book; the takers are not informed.
- **Calm momentum is wider than the agent.** Novelty's cross-check gives calm residual ACF 0.34 outside the windows, and
  calm 4 s |ΔS| outside the agent and TWAP windows has median 1.7 ticks and 24 % > 3. So the windows are a clean tell,
  not the whole of the calm drift. Lag-1 ACF of 1 s calm ΔS is 0.44 inside the windows and 0.09 outside.
- **Side vs the past (weak, n = 31).** The episode side agrees with the sign of the S move over the previous 90–200 s
  in 65–84 % of episodes (120 s: 26 of 31, but I scanned many horizons), and with the last shock jump in 19 of 28.
  Over the previous 5 s it is *against* the side in 24 of 31 (mean −1.0 tick). So the agent may be a slow trend
  follower that waits for a small counter-move. This is tentative.

**Cause.** Either the agent's activity drives a drift term in S (a meta-order with a "presence" impact that ends when
its quantity is done), or the generator draws one event, "drift in direction d", with an informed passive executor
attached. Both read the same on the tape. The sharp start, the stop at the last fill and the 4 cancel-ended
instances (fastest drifts, where the price ran away from the agent) slightly favour "the agent's quantity sets the
episode". The mechanism belongs to s_dynamics.

### Tradeability (`r15_detect.py`)
- **The first tell** is the first exact-1 µs re-post. It comes a median 0.57 s after the episode start (max 1.4 s),
  leaving a median 8.0 s of episode.
- **From that moment**, side-signed (t over 31 episodes):

  | horizon | mid | S |
  | --- | --- | --- |
  | +1 s | +0.8 | +1.0 |
  | +2 s | +2.0 | +2.2 |
  | +5 s | **+5.2** (t 5.8) | +5.6 |
  | to the episode end | **+8.4** (t 7.7) | +9.0 |

  The minimum to the end is −1 tick.
- **From every re-post** (712): mid +1.3 at 1 s, +2.4 at 2 s, +4.9 at 5 s.

## 2. Ladder (`ladder_check.py`, `chains.py`)

- **Bursts and band:** chains by exact 2000 µs gaps: 10.6k / 4.7k / 3.2k / 1.6k / 1.5k bursts of length 2–6 (r3
  12.1k / 6.3k / 2.8k / 1.5k / 1.2k), plus 78 of length 7–10 (r3 51). The band is unchanged: in vol 99.9 % of rungs
  are at touch−7…−2. Vol start-level shares: −2 62 %, −3 8 %, −4 11 %, −5 9 %, −6 10 % (r3 61/10/9/10/10). Vol lengths
  2/3/4/5/6 = 44/24/17/9/7 % (r3 51/26/12/6/5).
- **At most 3 live ladder orders per price level** (r3: 4). Counting the new order, a ladder insert finds 1/2/3/4
  live orders at its level:

  | regime | 1 | 2 | 3 | 4 |
  | --- | --- | --- | --- | --- |
  | calm | 1981 | 1183 | 558 | 3 |
  | vol | 11419 | 9307 | 5040 | 15 |

  So the ladder is now **3 agents**.
- **Burst size B** is up by ×1.35 in both regimes, i.e. ≈ 4/3:

  | regime | log B | median B | r3 log B | r3 median |
  | --- | --- | --- | --- | --- |
  | calm (L ≥ 3, 735 bursts) | 5.33 ± 0.33 | 217 | 5.02 ± 0.34 | 152 |
  | vol (18k bursts) | 3.82 ± 0.27 | 46 | 3.52 ± 0.31 | 34 |

  With 3 agents instead of 4, the ladder depth per level is about as before. Calm L = 2 chains are polluted by chance
  2000 µs pairs of base orders (log-size skew −3.6), so use L ≥ 3 for calm.
- **Rung noise** is still B·U(0.75, 1.25). Max/min within a burst, quantiles 50/90/99 %:

  | | L = 4 | L = 6 |
  | --- | --- | --- |
  | observed | 1.36/1.56/1.64 (calm), 1.37/1.55/1.63 (vol) | 1.46/1.59/1.65 |
  | simulated U(0.75, 1.25) | | 1.45/1.59/1.65 |
  | simulated U(0.8, 1.2) | | 1.35/1.44/1.48 |

- **Lifetimes** (`lifetimes.py`): calm chain p50 2.6 s, p90 9.9 s, 10 % filled (r3 2.8 / 10.8 / 6 %); vol p50 0.48 s,
  p90 1.4 s, 27 % filled (r3 0.60 / 1.7 / 24 %).
- **Rate** (novelty): bursts 0.98 /s calm, 10.3 /s vol (r3 1.23 / 10.8).

## 3. Base placement (`offset.py` → `offset_by_state.csv`, log `cache/offset.log`)

The censored MLE is round 3's, with S integrated over its posterior and a uniform touch−2…−7 component for unseen
ladder singletons. Base = cls 0, so the ladder chains and the agent are excluded.

| State | n | d0 | scale | mean D | w ladder | at the clip | r3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| calm | 757k | 0.069 | 0.444 | 0.513 | 2.2 % | 0.50 | 0.066 + Exp(0.450), mean 0.516 |
| vol | 326k | 0.197 | 1.421 | 1.62 | 0.3 % | 0.21 | 0.574 + Exp(1.050), mean 1.62 (local optimum 0.175 + Exp(1.46)) |
| drought | 4.4k | 0.58 | 2.04 | 2.62 | 0 | 0.01 | 0.43 + Exp(2.18), mean 2.61 |
| post | 15.6k | 0.076 | 0.723 | 0.80 | 2.1 % | 0.36 | 0.052 + Exp(0.611), mean 0.66 |

- **Calm is the round-1 law again.**
- **Vol has the same mean D.** The fit landed in the shallow 0.2 + Exp(1.4) basin that round 3 also found; the shape
  question (shifted exponential vs Gamma) is the same open item. The one-multiplier model m·D_calm (m 3.08) is only
  +0.002 nats/obs worse.
- **No maker insert crosses.**

## 4. Base cancels and lifetimes (`cancel_mle.py` → `cancel_mle.json`, `lifetimes.py` → `lifetimes.csv`)

This is round 3's direct MLE: decision hazard α + β·max(0, mv) by state at insert, then an Exp(b) delay to the row. The
sample is 60k random base orders on a 2.5 ms grid; log in `cache/cancel_mle.log`.

| S | α calm | β calm | α vol | β vol | α drought | β drought | α post | β post | b (/s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r4, S v0 (`cancel_mle.json`) | **1.78** | **2.73** | 1.85 | 2.98 | 4.85 | 4.2 | 1.65 | 4.3 | 61 |
| r3, S v1 | 1.69 | 3.44 | 1.73 | 3.39 | 5.07 | 3.2 | 1.93 | 3.1 | 64 |

Model-free hazard by mv (/s, age > 50 ms):

| mv | ≤ −0.5 | −0.5–0 | 0–0.1 | 0.1–0.25 | 0.25–0.5 | 0.5–0.75 | 0.75–1 | 1–1.5 | 1.5–2 | 2–3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calm r4 | 1.67 | 1.69 | 1.86 | 2.49 | 3.25 | 4.15 | 4.93 | 5.43 | 5.18 | 3.70 |
| calm r3 | 1.64 | 1.66 | 1.83 | 2.44 | 3.30 | 4.31 | 4.81 | 5.70 | 5.52 | 4.47 |
| vol r4 | 1.81 | 1.90 | 1.86 | 2.54 | 3.36 | 4.03 | 4.82 | 6.30 | 8.12 | 9.12 |
| vol r3 | 1.69 | 1.63 | 1.96 | 2.99 | 3.21 | 4.00 | 5.83 | 7.06 | 8.57 | 10.0 |

- The model-free curves match round 3 bin by bin up to mv ≈ 1.5. The lower MLE β (2.7–3.0 vs 3.4) comes from the
  sparse high-mv bins.
- **I read the cancel law as unchanged.** The β difference is within what S v0 can do: v0 has no drift term, so mv is
  noisier during the drifts.

| Base state | mean life (s) | p10 | p50 | p90 | filled | r3 mean |
| --- | --- | --- | --- | --- | --- | --- |
| calm | 0.461 | 0.078 | 0.327 | 0.996 | 14.7 % | 0.457 |
| vol | 0.380 | 0.065 | 0.236 | 0.870 | 15.5 % | 0.379 |
| drought | 0.208 | 0.044 | 0.148 | 0.424 | 9.3 % | 0.213 |
| post | 0.437 | 0.065 | 0.249 | 0.897 | 38 % | — |

Every order still gets a cancel row. Lifetime does not depend on size (0.44–0.50 s across size bands).

## Validation
- **Agent rules:**
  - every one of the 755 linked orders obeys the price rule but one;
  - 350 of 352 cancels follow a touch improvement;
  - the cancel delay matches the base delay law;
  - instance fill totals are all multiples of 15 once remainder re-posts are linked;
  - there is no excess of 15-lot orders outside the chains.
- **Drift:** the signed S change is positive in 42 of 43 instances. Mid and S agree (+6.4 vs +6.9 per instance). The
  drift is absent before the start and after the end.
- **Ladder cap:** "≤ 3 per level" holds in 30k sampled ladder inserts apart from 21 (0.07 %, pending
  cancels).
- **Base makers:** placement and lifetimes reproduce round 3 to 2–3 digits.
- **Not done:** a simulation of the agent against the tape. No S v1: everything is on S v0, which has no drift term,
  so the drift profile may be slightly smoothed.

## Still open
1. **The agent's stop rule** (n clips ≈ U{1..18}, or a price target, or the drift's own end) and its **start / side
   rule** (Poisson in calm; perhaps in the direction of the 1–3 min trend, after a small counter-move). There are only
   31 episodes.
2. **Is the drift caused by the agent or only announced by it?** That needs a drift term in the S filter
   (s_dynamics).
3. **Whether vol D is 0.2 + Exp(1.42) or 0.57 + Exp(1.05).** Same as round 3, cosmetic.
4. **The ≈ 110 non-agent excess of cancel → insert 1 µs pairs.** It is old (r3 has it too) and not investigated.

## What it means for trading
- **The 15-lot agent is a tell of informed flow, not a liquidity source.** Its 15 lots at the touch are negligible
  liquidity. Its presence means S runs its way at ~1 tick/s for the next ~8 s.
  - **Detecting it live:** a 15-lot maker insert, or the unfilled remainder of an earlier one, at the same-side touch,
    exactly 1 µs after a fill or cancel of a 15-lot order on that side. Confirm with a second such re-post.
  - **What to do:** stop quoting against it (our ask when it bids). Skew or lean the other way, or take. The expected
    move after the first re-post is +5 ticks in 5 s and +8 by the end, worst case −1.
  - **When to stop:** when it doesn't re-post 1 µs after a fill or cancel, the episode is over and S gives back
    ≈ 3–5 ticks over 10 s.
  - **How much it is worth:** about 31 episodes per 6000 s, so it is worth wiring into the strategy's S estimate as a
    drift term: S_pred = S + d · 1 tick/s · min(τ, remaining).
- **Its fills are not toxic for it, but its presence is toxic for us.** Being on the same side as the agent is good;
  being on the opposite side is bad.
- **Ladder:** the same depth per level as round 3, now 3 bigger orders instead of 4, so queue position behind ladder
  size at touch−2…−7 is as bad as before. The round-3 ladder filter (exact 2000 µs chains, hold inserts ≥ 2 behind the
  touch for 2 ms) still applies unchanged.
- **Base makers:** no change. The round-3 S filter's maker likelihood can be reused as is, with the agent and the TWAP
  excluded via `cache/order_class.parquet`.
