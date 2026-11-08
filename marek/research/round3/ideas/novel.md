# novel: structural and out-of-the-box ideas (round 3)

Written 2026-11-08 by the "novel" agent. Baseline `r3_best.py`: round 3 794,988 (calm 238.6k, volatile 421.7k,
shock 134.7k, breach 17.7k), synth seed 1 908,414. Analysis scripts: `marek/research/round3/ideas/novel/*.py` (each has
a usage line; NULL_LOGDIR = a full-log simulator run of a do-nothing strategy, RUN_LOGDIR = a full-log run of
r3_best). Prototypes: `strategies/marek/round3/ideas/novel_ovf.py` (overflow quotes, `ov_*`) and `novel_ri.py`
(reachable imbalance, `ri_*`); both are identical to r3_best with their flags off. Rows in
`strategy/results.jsonl`, tags `novel_*`.

**Bottom line.** I found one structural mechanism the strategy does not fully use, and it does not double PnL:
**overflow quotes at the takers' limit price**. In the volatile regime a resting order exactly at round(S ± 2.55)
would earn +358k at a 1 s markout on round 3 (causal placement, 100 lots per taker, no position limit), but 79 % of
those lots arrive on the side where r3_best is already pinned at ±100. In practice, inside the ±100 cap, it is worth
0 to +4k. Volatile money is all short-horizon momentum: snipes, overflow fills and the inventory r3_best holds all
bet on it and compete for the same 100 lots of room, and nothing exits cheaply in the volatile regime. Every other
structural idea I tested is either already captured or a dead end (numbers below). The simulator has no exploitable
rule.

## 1. Ranked ideas

| # | Idea | Measured upper bound (round 3) | Realised in prototype | Status |
| --- | --- | ---: | ---: | --- |
| 1 | Overflow quotes at the taker limit, volatile | e1 +358k / e30 +125k (no room limit); +3.5k e1 where r3_best has room on the adding side | −3.6k (all), **+3.2k r3 / +24k s1 (`ov_age` 3 s)** | prototyped, marginal |
| 2 | Same, calm | e1 +76k / e30 +50k; r3_best already takes 498k lots at the limit | −132k (strict replacement) | dead as a replacement |
| 3 | Reachable-imbalance drift (book depth inside the takers' reach) | frictionless hold: 206k vol + 92k calm, 1.1M lots turnover | −6k to −74k | prototyped, dead |
| 4 | Delayed impact in volatile (+26 % over 2 s) | — | c_v 0.06: −7.6k; 0.09: −27k | dead, c_v 0.0741 already best |
| 5 | Short-horizon momentum position | frictionless 141–153k vol | already earned: r3_best vol inventory PnL +146k | already captured |
| 6 | Transient impact as a directional hold (5–60 s) | frictionless 24k calm + 11k vol | — | dead (bound too small) |
| 7 | Zombie liquidity (replay artefact) | 19.7k lots, 8.1k edge in r3_best today | — | dead |
| 8 | Pre-jump direction tell | no consistent predictor (r2 vs r3) | — | dead, too risky |
| 9 | Simulator rules (penalty, latency, IOC, potential position, mark-out) | no gain found | — | dead |

### 1. Overflow quotes at the takers' limit (the "quote where takers' limits reach" mechanism)

**Mechanism.** In the generator each IOC taker moves S by c · *filled*^e, and only by what it filled (unfilled IOCs
move nothing; r3_synth models it the same way). A resting order at the **back of the level P = the taker's limit**
fills only when a taker empties every market order up to P, and only with what it has left. Two effects make that
fill cheap:
- The taker cannot go past P, so its impact is capped by the (thin) depth it emptied. It is not c · q^e of a big
  taker that runs on through us. In half of these fills the taker had found nothing at all within reach (median
  emptied depth 0).
- The replay's S path carries only the taker's *original* fill, so our extra fill adds no impact.

One tick inside the limit (o = −1) is the opposite case: takers fill us and go on, with their full impact.

**Evidence** (`novel/overflow.py`, null replay; virtual 100-lot order at the back of P = round(S_pre ± k) + o on
every taker, k = 1 calm / 2.55 volatile, shock windows masked; markouts on exact S):

| placement | calm lots | calm e1 | calm e30 | vol lots | vol e1 | vol e30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| o = −1 (one tick inside the reach) | 1.14M | −667k | −352k | 315k | +143k | +92k |
| **o = 0 (at the limit)**, P from the causal filter s_filt | 141k | +76k | +50k | 158k | **+358k** | +125k |
| o = 0, P from the smoother s_hat (look-ahead, optimistic) | 145k | +97k | +99k | 171k | +390k | +247k |
| o = +1 (beyond the reach; tail takers only) | 5k | +11k | +8k | 16k | +17k | +74k |

For comparison, the tape's makers at distance 2–3 from S in volatile lose −0.7 to −0.9 per lot at 30 s
(`novel/depth_markout.py`). Those fills come from takers that went on past them.

**Why it does not pay inside r3_best.** Split by r3_best's inventory at each taker (same script, second argument):

| volatile, o = 0 | lots | e1 | e30 |
| --- | ---: | ---: | ---: |
| would reduce our position | 28.7k | −17.4k | +65.5k |
| would add, with room | 4.5k | +3.5k | −12.1k |
| **blocked (we are already at ±100 on that side)** | **124.3k** | **+372k** | +71.5k |

The overflow happens on the side S keeps moving towards: the book lags S, so the thin side is the one S is heading
for. That is the same side r3_best is pinned on after sniping. Its alpha also decays: +2.3 per lot at 1 s, +0.8 at
30 s.

**Prototype** `novel_ovf.py`: `ov_on` bit 1 = volatile, bit 2 = calm. In the volatile regime outside shock windows,
it rests one bid at round(F − 2.55) and one ask at round(F + 2.55), rounded outward by `ov_m` 0.1, `ov_q` lots each,
and cancels anything else. `ov_age` posts the reducing side only once the position is that many seconds old.

| tag | set | round 3 | Δ | synth s1 | Δ | note |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| novel_ov1 | ov_on 1 | 791,415 | −3.6k | 904,682 | −3.7k | vol −4.1k r3 / +14.9k s1 |
| novel_ov_age3 | ov_on 1, ov_age 3 | 798,209 | **+3.2k** | 932,533 | **+24.1k** | vol +2.2k / +4.2k; the rest of s1 is shock noise |
| novel_ov_addonly | ov_on 1, ov_age 1e9 | 795,124 | +0.1k | 908,003 | −0.4k | |
| novel_ov1_notake | ov_on 1, take_t 99 | 381,867 | −413k | 443,369 | −465k | overflow alone, without volatile snipes |
| novel_ov_calm | ov_on 2 | 662,914 | −132k | 801,170 | −107k | strict at-limit calm quotes lose queue seniority |

Realised volatile overflow fills in novel_ov1: 38k lots at +2.10 per lot at fill, −0.95 at 1 s, +0.21 at 30 s
(`novel/ov_diag.py`). These are mostly reducing-side fills, the adversely selected case.

**Calm is already doing this.** 498k of r3_best's 666k calm passive lots fill exactly at the taker's limit (o = 0,
e30 +254k). Another 167k fill one tick inside (o = −1, e1 −15k, e30 +3k), so they are worth about zero but take up
room. Removing them by quoting further out (theta 0.5) loses queue position: −9.8k r3, −4.6k s1 (tag novel_th05).

**Estimate:** +0 to +4k on round 3 as it stands. The +372k blocked-side pool is reachable only with more room or a
cheap exit for volatile momentum inventory, and I found neither (IOC unwind in volatile: the volatile agent's vu_u 1.0
lost 133k; calm IOC unwind, tag novel_uw03: −18k r3).

### 3. Reachable imbalance (RI) as a short-horizon S predictor

**Mechanism.** It follows from the same filled-capped impact: the next buy taker's kick depends on the ask depth
inside round(S + k), the next sell taker's on the bid depth inside round(S − k). Let RI = E[min(q, D_ask)^0.61] −
E[min(q, D_bid)^0.61] under the taker size law.

**Evidence** (`novel/reach_imb.py`, null book, 50 ms grid):
- Volatile: corr(RI, ΔS over 0.5 s) +0.17, β 0.097 per unit RI (RI sd 7.4); at 1 s +0.16; at 5 s +0.03.
- Calm: corr +0.10 at 200 ms.
- Frictionless 100·sign(RI) would earn 206k (vol) + 92k (calm) with 1.1M + 2.3M lots of turnover, so it does not
  survive the spread on its own.

**Prototype** `novel_ri.py`: shift F by β·RI.

| tag | set | round 3 | Δ | synth s1 | Δ |
| --- | --- | ---: | ---: | ---: | ---: |
| novel_ri1 | ri_on 1 (vol β 0.1, calm 0.015) | 745,747 | −49k | 837,331 | −71k |
| novel_ri_v03 | ri_on 1, vol β 0.03, calm 0 | 788,837 | −6.2k | 886,838 | −21.6k |
| novel_ri_cq | ri_on 1, calm β 0.05 + passive | 721,398 | −74k (calm −65k) | 829,942 | −78k |

Dead: the drift is real, but the snipes' holding horizon is seconds to minutes, and RI reverses by then.

### 4. Delayed impact in volatile

Event study on exact S (`novel/impact_profile.py`): the volatile response per unit q^0.61 is 0.048 at 20 ms,
0.054 at 200 ms, 0.061 at 2 s and 0.043 at 30 s. That is +26 % after the trade, then reversion. Takers ≥ 600 lots:
+3.3 ticks (0–0.2 s) rising to +4.2 (1–2 s). The S-free check on later takers' limit prices agrees
(`novel/drift_check.py`, ±0.45). Mechanism: after a big up-kick the bids fall out of the sell takers' reach (they fill
nothing and move nothing), while buy takers keep filling. r3_best's c_v 0.0741 already over-kicks, which anticipates
the drift: c_v 0.06 gave 787,424 (−7.6k) / 908,848 (+0.4k), and c_v 0.09 gave 767,730 (−27k) / 892,523 (−16k)
(tags novel_cv06, novel_cv09). Dead.

### 5–6. Directional holding (momentum, transient impact)

From `novel/momentum.py` and `novel/steer_bound.py`, 100 ms grid, shocks masked.
- **Momentum.** Volatile past-0.5 s → next-0.5 s corr +0.10, reversing by 5–10 s (past 5 s → next 10 s: −0.16).
  Frictionless 100·sign(past move): 141–153k vol. r3_best's own volatile inventory PnL (Σ inv · ΔS) is already
  +146k, so this is captured.
- **Transient impact.** The decaying-impact state x(T 15 s) predicts the 10 s move with corr −0.13 (calm) and −0.19
  (vol). The frictionless hold of 100·sign(−x) is only 24k calm + 11k vol. r3_best's calm inventory (mean |inv| 78)
  earns +3k. Dead.

### 7. Zombie liquidity

Replay artefact: when a replayed taker fills our resting order instead of a market order, that market order outlives
its tape life. `novel/zombie_fills.py`: r3_best's IOCs hit such orders for 19.7k lots and 8.1k of edge (median 11 ms
after their original exhaustion). Fully filled orders still get their scheduled cancel row (155,141 of 155,144), so a
zombie dies at its maker's cancel time. Dead.

### 8. Pre-jump direction tell

- **Drought.** Inside the 27 round-3 droughts there are **zero** market cancels (`novel/drought_tell.py`).
- **Sign predictors.** Maker-price offsets and taker sides predict the jump sign only through the smoother.
  drift_drought corr −0.08.
- **Pooled r2 + r3** (`novel/jump_tell.py`): pre60 +0.43 / +0.16, pre30 +0.49 / −0.03, S − S(previous jump)
  +0.39 / +0.25, S − 10000 −0.32 / −0.17. These are inconsistent across rounds, after testing 9 features on n = 27
  each. A wrong bet costs 100 × 47 per shock. Not worth it.
- **Shock inventory** (`novel/shock_inv.py`). r3_best made +54.6k on inventory held across the jump. That is sniper
  timing plus luck, not a tell.
- **Stale removal after the jump** (`novel/jump_timeline.py`): 8.1k stale lots, about 94k of stale value. In the
  first 10 ms only 758 lots are cancelled and 171 traded; by 200 ms, 4,969 are cancelled and 4,171 traded. The
  bigmoves agent owns this.

### 9. Simulator rules checked (no exploit)

- **Breach penalty.** An exact time integral, `_advance_time` → `_accrue_breach_penalty`; no sampling. It is also
  independent of size, which the oversize trick already uses.
- **Zero-time pairs.** IOCs with the same ready time activate in sequence with zero elapsed time. That is legal, but
  the book is never crossed, so a same-instant unwind always pays the spread.
- **Potential position** counts resting same-side quantity at activation, never in-flight IOCs. Each IOC is checked
  against the inventory of the moment.
- **Our orders** activate before any market event at or after their ready time. No public snapshot follows our own
  activations.
- **Self-trade prevention** lets our IOCs skip our own resting orders.
- **Modify** can only shrink an order, and keeps its priority.
- **Mark-out** sweeps market orders only. Penalty accrues until the last pending activation. Nothing to gain at the
  end of the tape.

## 2. Other dead ends (with numbers)

- **Calm IOC unwind** (unwind_u 0.3, pos 60 → 20, tag novel_uw03): 776,662 (−18k), s1 912,922 (+4.5k).
  Calm is pinned at |inv| ≥ 99 for 56 % of calm time. Pinned, passive throughput is 104 lots/s (e30 49/s); flat, it
  is 234 lots/s (e30 127/s) (`novel/calm_inv.py`). But paying the spread to get flat does not pay.
- **Regime-switch lag.** The first 3 s of the 38 volatile starts make +27.8k (in line with the steady rate), so
  there is no detection loss to recover.
- **Volatile passive by distance from S.** The tape's makers lose at every reachable distance (d 0–3: e30 −1.9 to
  −0.7 per lot). Only the exact-limit overflow (idea 1) is positive.

## 3. What would be needed to double

The big pools I can measure are all volatile momentum:
- snipes: 206k lots at e1 +1.7 per lot;
- blocked overflow: 124k lots at e1 +3.0;
- the inventory drift r3_best already earns.

That alpha lives for 1–5 s and reverses by 10–30 s. With ±100 lots of room and a 10k/s breach penalty (100 ticks
per lot per second, against about 2 ticks of alpha), more of it can only come from **turning the room over faster**:
exiting momentum inventory after about 2–5 s at under 1 tick of cost. Every exit I or the other agents tried in
volatile costs more than that: IOC unwind −133k (volatile agent), reducing-side passive (volatile agent's vf_on,
−4k), reducing-side overflow (ov1, −4k). I have no candidate that does it. If another team is far ahead on this kind
of tape, my guess is that they either have a better volatile exit or capture the heavy-tailed shocks better (the
round-2 shock at 5398 s alone carried 1.22M of stale value).
