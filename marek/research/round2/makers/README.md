# Round 2 makers: quoting and cancellation rules

Tape: `data/round2.csv`, 6000 s, 1,069,859 maker inserts (G) and 53,423 IOCs (F).
- **S:** I took the hidden fair price from the s_dynamics agent's grid HMM, `smooth_v1.npz` (a copy is in
  `cache/s_smooth_v1.npz`). It gives a smoothed posterior mean per insert, with a median posterior sd of 0.09 tick in
  calm and 0.19 in volatile.
- **States:** these come from the flow agent's labels (see `states.py`).
  - **calm** and **vol**: the 2-state regime, from `flow/regimes_envW.csv`.
  - **drought**: the 3.0 s after each of the 27 shock onsets.
  - **post**: the 3 s after the jump.

## The model in one table

| Part | Calm (≈83 % of time) | Volatile (≈15 %) | Drought (3 s before a shock jump) | Confidence |
| --- | --- | --- | --- | --- |
| Reference price | S, hidden fair price | S | S | high (calm), medium (vol) |
| Price rule | bid = min(floor(S − D), ask − 1), ask = max(ceil(S + D), bid + 1) | same | same | high |
| Offset D | 0.05–0.09 + Exp(0.42–0.45), mean 0.51 (the round-1 law) | **≈ 2.6 × the calm D**: 0.23 + Exp(1.10), mean 1.31 | wider, mean ≈ 1.9 (n = 4k, poor fit) | high / medium / low |
| Side, size | 50/50, lognormal(2.657, 1.803), the same in every state | same | same | high |
| Cancel decision hazard | α + β·max(0, mv): **α = 1.73 /s, β = 6.1 /s per tick** | α = 1.9, β = 3.4–4.5 | **α ≈ 6 /s** (3.5×), β ≈ 4 | high / medium / medium |
| Decision → cancel row | dead time ≈ 5 ms, then Exp with rate ≈ 45–55 /s (mean total ≈ 25 ms). The row is written even after a fill. | same | same | medium-high |
| Dependence on distance from S | none (γ = −0.04 on current distance) | none seen | — | high |

`mv` is the adverse move of S since the order was inserted: S down for a bid, S up for an ask. The cancel row is the
row the tape writes, whether or not the order has filled.

**In short:** in calm the maker process is round 1 unchanged. The volatile regime multiplies the quote offset by about
2.6, the same factor as the taker offset (about 2.5–3 per the flow agent). Shocks raise the base cancel rate about 3.7× for
3 s, and the book drains. Nothing is new about who the makers are. The 72k-lot orders are the lognormal tail, and they are
placed and cancelled like every other order.

## 1. Placement (`offset.py`, `offset_by_regime.py`, `offset_by_state.py`, `offset_regime.py`, `touch_vs_s.py`)

**Method.** I fitted the exact censored MLE of D = d0 + Exp(scale) (or Gamma) under the round-1 clip rule. S is
integrated over its posterior (Gaussian, 7-point Gauss–Hermite). An insert at the clip price (ask − 1 for a bid) counts as
censored: P = F(S − clip).

**Results by state** (`offset_by_state.csv`):

| State | n | d0 | scale | mean D | Share at the clip | Gamma k | One-multiplier m (D = m·D_calm) | Δnll/obs, one-mult − free |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calm | 891k | 0.089 | 0.420 | 0.509 | 0.52 | 0.72 | 1 | — |
| vol | 159k | 0.089 | 1.224 | 1.31 | 0.13 | 1.78 | **2.62** | −0.0065 (the one-mult fit is *better*) |
| drought | 4.2k | 0.54 | 1.33 | 1.87 | 0.006 | 1.74 | 3.97 | +0.05 (poor) |
| post | 14k | 0.05 | 0.60 | 0.66 | 0.39 | 0.58 | 1.29 | mixture of states |

**Calm is the round-1 law.** On all calm inserts the free fit is 0.089 + Exp(0.42). The mean, 0.509, matches round 1's
0.50. On the subset with precise S (posterior sd < 0.08) the likelihood prefers the round-1 values, 0.05 + Exp(0.45),
over the free fit. S error is the only reason d0 and scale move slightly. D does not depend on side (scale 0.568 vs
0.567), on size (0.565–0.571 across size bins, 1 to 72k lots) or on clip status.

**Volatility is two regimes, not a continuum.** Rolling 60 s fits (`offset_rolling60.csv`) give scale ≈ 0.40–0.45 in
every calm window and ≈ 1.09–1.16 in fully volatile windows. Mixed windows fall in between. Bins of 3 s realised S vol
show the same thing (`offset_by_regime.csv`): scale is 0.42–0.43 for rv < 0.7 and 1.05–1.23 for every rv > 1.1, with
no trend inside the volatile regime. So D switches with the regime: D_vol ≈ 2.6 · D_calm. It does not follow realised
vol continuously.

**Makers quote off S, not off the touch.** `touch_vs_s.py` replays the price rule with the real S and the real book, then
compares the replay with the observed inserts. In calm it reproduces, for every bin of same-side-touch distance from S,
both the observed gap S − price of unclipped inserts (to ±0.02) and the clip share (to ±0.03):

| Touch − S bin | 0–0.25 | 0.25–0.5 | 0.5–0.75 | 1–1.5 | 1.5–2 |
| --- | --- | --- | --- | --- | --- |
| obs / sim gap (unclipped) | 1.256 / 1.249 | 1.472 / 1.453 | 1.683 / 1.668 | 1.500 / 1.500 | 1.935 / 1.908 |
| obs / sim clip share | 0.133 / 0.158 | 0.500 / 0.471 | 0.716 / 0.699 | 0.766 / 0.762 | 0.930 / 0.925 |

The volatile regime matches less well. Where the touch is more than 1.5 ticks from S, the replay clips 32–59 % of
inserts against 14–42 % observed. Either S is less accurate there (sd 0.19, and S moves ×5 faster), or D_vol is not
exactly a fixed multiple of D_calm.

**Clipping.** No maker insert ever crosses: bid − (ask − 1) is at most 0. Crossing prices are clipped to one tick inside
the opposite touch, never dropped, exactly as in round 1. 45.6 % of all inserts sit at the clip price: 52 % in calm,
13 % in vol.

**Extreme bids.** A few inserts land 30–106 ticks from the touch. These are the post-shock quotes at the new level, with
the stale book still resting.

## 2. Cancellation (`cancel_mle.py`, `delay_event.py`, `scratch_life.py`)

**Lifetimes** (insert → cancel row), all inserts:

| | mean | p10 | p50 | p90 |
| --- | --- | --- | --- | --- |
| all | 0.442 | 0.076 | 0.312 | 0.968 |
| calm | 0.459 | 0.079 | 0.331 | 0.997 |
| vol | 0.357 | 0.066 | 0.235 | 0.795 |
| drought | 0.210 | 0.046 | 0.151 | 0.435 |
| post | 0.372 | 0.066 | 0.247 | 0.829 |

- 99.993 % of makers get a cancel row.
- Every filled order also gets one, as in round 1. Filled orders are 14.07 % of inserts and 14.07 % of cancel rows.
- 2.1 % of inserts are partly filled and then cancelled.
- Lifetime does not depend on size: 0.43–0.45 s from 1 lot to 20k lots. The 29 orders above 20k lots have a mean of
  0.34 s.
- Lifetime is shorter for inserts placed deep behind the touch (0.32 s at −5 ticks). That is a regime effect, because
  deep inserts come mostly from the volatile regime.

**Measured hazard of a live order** (cancel row per order-second, age > 50 ms) by mv and state:

| mv (ticks) | ≤ −0.5 | −0.5–0 | 0–0.1 | 0.1–0.25 | 0.25–0.5 | 0.5–0.75 | 0.75–1 | 1–1.5 | 1.5–2 | 2–3 | > 3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calm | 1.82 | 1.76 | 1.90 | 2.64 | 3.52 | 5.22 | 6.10 | 8.38 | 12.6 | 14.4 | 38 |
| vol | 1.82 | 2.12 | 2.18 | 2.68 | 3.85 | 3.88 | 5.23 | 5.85 | 6.75 | 10.6 | 17.9 |
| drought | 6.7 | 7.2 | 5.4 | 7.1 | 12.6 | | | 12 | | | 16 |

- Favourable moves don't matter: the hazard at mv ≤ −0.5 equals the hazard at mv ≈ 0.
- At |mv| < 0.1 the current distance from S doesn't matter either: calm hazard is 1.8–2.0 in every distance bin from
  0 to 3 ticks.
- Both features are the same as in round 1.

**Direct MLE** (`cancel_mle.json`). I used 20k random makers on a 2.5 ms grid, each with its full S path from insert to
its cancel row; fills are included, because the row comes regardless. The likelihood is
L = Σ_j P(decision in step j) · f_delay(T − s_j). α and β are per state, at insert.

| Fit | α calm | β calm | α vol | β vol | α drought | β drought | α post | β post | delay | nll (n = 20k) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Exp(b) delay | **1.73** | **6.10** | 1.92 | 4.47 | 6.07 | 4.3 | 1.83 | 6.9 | b = 44.9 /s | −71 |
| L0 + Exp(b) | 1.73 | 5.94 | 1.93 | 4.45 | 5.79 | 4.4 | 1.75 | 6.5 | L0 = 3.9 ms, b = 56 | +35 (optimiser did not reach the nested Exp optimum) |
| Gamma(2, b) | 1.64 | 5.57 | 1.84 | 3.41 | 5.18 | 4.7 | 1.76 | 5.3 | mean 1.8 ms | −1004 |
| Exp delay + exp(γ·distance) | 2.11 | 5.73 | 2.92 | 4.59 | 3.0 | 4.1 | 2.75 | 4.7 | γ = −0.09 | worse |

Across delay forms, α_calm stays at 1.64–1.73 and β_calm at 5.6–6.1, which is round 1's (1.72, 5.8). In the volatile
regime α barely rises (1.9) but β falls to 3.4–4.5 per tick. The drought has a 3–3.5× base rate. Distance from S adds
nothing.

**Caveat on the delay shape.** The MLE prefers an almost-zero Gamma delay. That contradicts two things:
- the age profile: the calm hazard is 0.49 /s at 0–10 ms, 1.03 at 10–20, 1.46 at 20–30, 2.0 at 50–100 and 2.5 at
  0.2–1 s;
- the exact-timestamp event study below.

I think the smoothed S, which uses future inserts, gets its timing ahead of the true S, and that rewards short delays.
I trust the event study for the delay and the Exp-delay MLE for α and β. The b values are rough.

**The delay has a dead time** (`delay_event.py`). This event study uses the exact taker timestamps as S jumps. It takes
the orders resting on the side the jump moves against and measures their cancel-row intensity after the taker:

| ms after taker | 0–5 | 5–6 | 6–8 | 8–10 | 10–12 | 12–15 | 15–20 | 20–25 | 25–30 | 30–40 | 40–50 | 50–70 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calm, impact > 0.5 (n = 5045, mean 0.71 tick) | 2.5–2.7 | 2.58 | 2.76 | 3.10 | 3.25 | 3.50 | 3.81 | 4.20 | 4.44 | 4.79 | 4.96 | 5.19 |
| vol, impact > 1.5 (n = 2675, mean 2.3 ticks) | 3.5 | 4.62 | 4.92 | 5.22 | 5.71 | 6.24 | 7.23 | 7.65 | 8.42 | 9.08 | 9.17 | 8.96 |

- Nothing happens for the first ≈ 5 ms. After that the response is exponential with rate ≈ 50–60 /s and reaches its plateau
  by about 50 ms.
- So the delay is about 5 ms + Exp(≈ 55 /s), a mean of ≈ 23 ms, close to round 1's Exp(40) mean of 25 ms.
- The same dead time shows in the lifetime density near 0. Lifetimes under 1 ms occur at about a third of the
  hypoexponential rate, and the minimum lifetime is 0.5 ms.

## 3. Book consequences (`book_stats.py`, `book_vs_s.py` → `book_by_state.csv`, `stale_episodes.csv`)

| State | time (s) | sd(mid − S) | spread 1 / 2 / 3+ | mean spread | touch qty | E[S − bid], E[ask − S] |
| --- | --- | --- | --- | --- | --- | --- |
| calm | 4952 | **0.44** (round 1: 0.41) | 93.9 / 6.0 / 0.1 % | 1.06 | 1268 | 0.52, 0.54 |
| vol | 884 | **1.50** | 36.7 / 42.9 / 20.4 % | 1.88 | 437 | 0.90, 0.98 |
| drought | 80 | 1.53 | 6 / 5 / 89 % | 3.95 | 211 | 1.83, 2.13 |
| post | 80 | 2.85 | 68 / 19 / 13 % | 1.76 | 554 | 0.97, 0.79 |
| all | 5996 | 0.80 | 83.9 / 11.6 / 4.4 % | 1.23 | 1122 | |

- **Gap half-life.** I measured the ACF of mid − S on a 10 ms grid.
  - Calm: 0.97 at 10 ms, 0.86 at 50 ms, 0.74 at 100 ms, **0.50 at 200 ms**, 0.15 at 500 ms.
  - Vol: 0.69 at 50 ms, **0.44 at 100 ms**, 0.15 at 200 ms.
- **Response to an S jump of more than 0.5.**
  - Calm: the gap stays at 0.44 for the first 100 ms and is 0.25 by 300 ms and 0.10 by 500 ms. The dead time plus the
    Exp delay keeps the book stale for about 100 ms.
  - Vol: the gap starts at 1.4 and is 1.0 at 100 ms, 0.37 at 200 ms and about 0 at 500 ms.
- **One-sided book.** One side is empty 0.07 % of the time. That is 274 episodes (0.046 /s), with a median of 9 ms and a
  maximum of 136 ms. They come mostly from droughts and shocks.
- **Spread by 600 s segment.** The one-tick share ranges from 61 % to 93 % (`book_stats.py`), tracking the share of
  volatile time in each segment.

**Stale-quote episodes** (`stale_episodes.csv`). An episode is a time the touch is beyond S by more than the threshold: ask < S − thr or bid > S + thr.
Rates are per side, per second of that state:

| thr | state | starts /s per side | time share | duration p10 / p50 / p90 (ms) | > 5 ms | touch qty p50 | edge at start, max (ticks) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.25 | calm | 0.26–0.29 | 4 % | 14 / 105 / 340 | 96 % | 410 | 0.36, 0.51 |
| 0.5 | calm | **0.10** | 2 % | 22–35 / 150–160 / 350–370 | 97 % | 400–490 | 0.60, 0.80 |
| 1.0 | calm | 0.02 | 0.2 % | 22 / 110–120 / 250 | 95 % | 340–430 | 1.10, 1.23 |
| 0.5 | vol | **1.34–1.44** | 15 % | 12 / 93 / 215 | 95 % | 245 | 1.29, 1.85 |
| 1.0 | vol | 0.98–1.08 | 10 % | 9–12 / 85 / 196 | 95 % | 247 | 1.78, 2.30 |
| 2.0 | vol | 0.50–0.53 | 4 % | 10–12 / 75 / 160 | 94 % | 250–300 | 2.77, 3.21 |
| 1.0 | post | 0.69–0.73 | 4 % | 6–9 / 31–52 / 116–150 | 95 % | 205 | 2.6–5.2, 4.0–6.6 |

- About 60 % of episodes start within one order of an IOC, i.e. at a taker-impact jump. The other 40 % start at a
  posterior S move, which is either exogenous or the HMM's timing.
- Calm is round 1 again: 0.10 /s per side at 0.5, against round 1's 0.057.
- The volatile regime has 13× the episode rate, and about twice the edge per episode, on half the touch quantity.

## 4. Shock jumps: the biggest stale book (`shocks.py` → `shock_stale.csv`)

**Timing.** The jump comes 3.00 s after drought onset. For the large shocks I took the first unclipped maker insert at
the new level as the jump time: medians 3005–3065 ms, range 2953–3069 ms. The rest of the spread comes from the onset
estimate. Small shocks (|J| < 10) are hard to time.

**The stale book is thin.** The drought cancels the book to about 1/8 depth: α ≈ 6 /s, and the makers' D is wider. At
the jump:
- 0–8 orders are resting beyond S_after by more than 0.5 tick, 0–2369 lots per shock, 8183 lots over all 27.
- Their size-weighted edge is about |J|: 8 ticks at J = 11 and 102 ticks at J = 111.

**It goes by cancel, not by fill.** Only 2.5 % of the stale quantity is filled. Share of the stale quantity still resting
after the jump:

| after jump | 5 ms | 10 ms | 20 ms | 50 ms | 100 ms | 200 ms |
| --- | --- | --- | --- | --- | --- | --- |
| still resting | 94 % | 75 % | 36 % | 24 % | 16 % | 8 % |

That matches the cancel model: with mv ≫ 1 the decision is immediate, so the row comes after the 5 ms dead time plus
Exp(≈ 50 /s).

The first market taker in the jump direction arrives a median of 70 ms after the jump, and sometimes 1–2 ms after.
- With 5 ms latency we would see the jump in the new-level inserts and sweep the stale side at +5–10 ms, while 75–94 %
  of it is still there.
- The total stale edge was 320k tick·lots over the tape. That is concentrated: shock #24 alone (+111 ticks) had
  2369 lots × 102 ticks.
- The ±100 position limit caps a single shock at about 100 lots × |J|.

## 5. What is new vs round 1
1. **Regime multiplier on D**, ×2.6 in the volatile regime. The taker offset scales the same way, so a single
   width multiplier is the likely generator knob.
2. **Drought cancel boost**: α rises from 1.73 to about 6 /s for 3.0 s before each shock, and placement widens.
3. **A dead time of about 5 ms** in the cancel response. In round 1 it was folded into Exp(40); round 2 shows it
   clearly. It may have been in round 1 too.
4. **Large orders are nothing special.** Above 5000 lots: 624 observed against about 600 expected from
   lognormal(2.66, 1.80). Their D, lifetime and side are the same as everyone's.
5. **β in the volatile regime is lower**, 3.4–4.5 against 6.1 per tick. Per regime-scaled tick (mv / 2.6) it would be 12. So
   cancels respond to raw ticks, slightly attenuated, possibly by S error in the volatile regime.

## Implications for a market maker
- **Calm (83 %): the round-1 playbook transfers.**
  - The book is S ± (0.05 + Exp(0.45)), clipped to the touch, with sd(mid − S) 0.44 and a gap half-life of 200 ms.
  - Pull or requote within 5 ms of any S jump toward your order. The market's makers need at least 5 ms and typically
    25 ms, so there is about 20 ms of edge, slightly more than in round 1.
  - Queue seniority still pays: every market order dies at about 1.73 /s plus 6/tick of adverse move.
- **Volatile (15 %).**
  - Market makers quote 2.6× wider. The touch is about 0.9–1.0 from S, with half the depth.
  - Stale quotes appear 13× as often (1.4 /s per side at 0.5 tick, about 90 ms median, edge about 1.3–1.9).
  - Price impact is ×4–5, so quoting at the touch carries much more adverse selection. Prefer sniping stale touches
    over making there, or quote at least 2–3 ticks from S.
  - The regime is visible within about 1 s: the spread at insert, or the maker envelope width.
- **Shocks.**
  - Pull everything at drought onset + about 2.9 s. The drought is detectable as maker inserts dropping to 30 %.
  - At the jump, sweep the stale side within 5–10 ms. It is cancelled, not traded, with a half-life of about 15 ms.
  - Then quote into the post-jump taker burst at the new level.
- **Sniping in general.** Stale touches beyond S by more than 0.5 tick arrive at about 0.1 /s per side in calm and
  1.4 /s in vol. 95 % of them outlast 5 ms, and 60 % are triggered by a visible taker.

## Open questions
- Is the decision hazard in the volatile regime α + β·mv with β the same as calm, but attenuated by S error? A synthetic
  check with S error of the right size would settle it.
- The exact delay law: 5 ms + Exp(b), or a gamma? The MLE comparison is in `cancel_mle.json`, and a sharper event study
  would use only the largest taker jumps.
- The drought D law is fitted on n = 4k with a thin book: wider, but the shape is unclear.
- How sensitive all of this is to S. `s_load.with_s(path)` takes a newer smoothed S file; rerun `offset_by_state.py`,
  `cancel_mle.py` and `book_vs_s.py` when s_dynamics publishes its final path.

## Files
| File | What it does |
| --- | --- |
| `prep.py` | order table + quotes + trades → `cache/` |
| `s_load.py` | attaches S (posterior mean, sd, 5–95 %) per insert |
| `states.py` | calm / vol / drought / post labels (flow agent) |
| `regime.py` | own S-vol regime label (rv over 3 s; agrees with the flow labels 96.4 %) |
| `hmm.py`, `fit_dyn.py` | my first grid HMM for S (superseded by the s_dynamics path; can't handle the shock jumps) |
| `offset*.py`, `touch_vs_s.py` | placement fits and the S-vs-touch replay |
| `cancel_mle.py`, `delay_event.py` | hazard tables, MLE, delay event study |
| `book_stats.py`, `book_vs_s.py`, `shocks.py` | book consequences, stale episodes, shock stale book |
| `scratch_*.py` | quick looks (lifetimes, a slice of the tape) |
