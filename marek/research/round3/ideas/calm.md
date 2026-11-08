# Calm angle: where calm PnL goes and what moves it

Angle "calm", 2026-11-08. Base: `strategies/marek/round3/r3_best.py` (round 3: 795.0k, calm 238.6k). Calm here is
eval3's calm part (outside volatile segments and shock windows). Measurements are on `data/round3.csv` against the
exact S (`s_dynamics/s_path.parquet`, s_hat). Screens ran on round 3 plus synth `r3_v1_seed1` (baseline 908.4k, calm 253.2k).

Scripts are in `ideas/calm/`: `an1.py` to `an5.py` (analysis) and `screen*.sh` / `final.sh` (eval3 runs, tags `calm_*`).
Prototypes:
- `strategies/marek/round3/ideas/calm_a.py`: r3_best plus every mechanism below behind PARAMS flags. With the flags
  off it is identical to r3_best.
- `strategies/marek/round3/ideas/calm_log.py`: a logging wrapper that gives identical PnL.
- `strategies/marek/round3/ideas/calm_oracle.py`: calm quotes on the exact S (research only).

**Bottom line.**
- Calm cannot double through quoting rules. Every change that puts the room anywhere except the target level loses
  between 33k and 130k of calm PnL. Those changes are: size split, pre-join, queue-depth cap, over-cap queue seniority,
  and the inventory target.
- What does help is a set of small threshold changes, each worth +5k to +8k. Together they give **+18k calm on
  round 3** (`calm_c2`): IOC take 0.15 → 0.10, a reduce-only IOC at edge ≥ 0 + 1·sd, and theta_keep 0.10 → 0.
- A perfect S estimate is worth only +16k in calm.
- The real ceiling is queue priority: about 365k with ±100 if we were always first in the queue (ceiling.md), against
  ≈ 123k now. I found no way to buy that priority with room.

## 1. What limits calm PnL now (measured, `an1.py`, `an2.py`, `an5.py`)

r3_best calm fills, per lot. e0 = S at the fill minus the price (pre-trade S). m_h = S(t + h) minus the price.

| | lots | e0 | m1 s | m5 s | m30 s | m60 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| passive | 601k | +0.74 | **+0.18** | +0.19 | +0.49 | +0.55 |
| IOC take | 142k | +0.56 | **+0.90** | +1.03 | −0.45 | −0.61 |

Total calm taker fill volume on the tape is 1,087 lots/s. We fill ≈ 200 lots/s of it.

Q1: fill rate, edge per fill, adverse selection, or time pinned?
1. **Edge per fill is not the limit.** We earn 0.74 at the fill. Going from 0.74 at the fill to 0.18 a second later is
   the filling taker's own impact on S, which costs 0.56 per lot (≈ 337k over calm passive lots). It grows with the
   taker's size:

   | taker size | share of our passive lots | m1 |
   | --- | ---: | ---: |
   | ≤ 120 | 29 % | +0.46 … +0.53 |
   | 120–250 | 26 % | +0.32 |
   | 250–500 | 23 % | +0.11 |
   | 500–1000 | 14 % | −0.17 |
   | > 1000 | 9 % | −0.46 … −0.63 |

   Queue depth at the time we joined tells the same story:
   - 0 lots ahead: m1 +0.43
   - 1000–3000 ahead: −0.03
   - more than 3000 ahead: −0.06

   By the fill price relative to the touch when we sent the order:
   - Orders inside the spread (rel −1, 103k lots): m1 +0.41.
   - Orders that were one tick behind and got filled after the touch moved (rel +1, 110k lots): −0.04.
2. **Room is binding.** In calm, |pos| = 100 for 55.8 % of the time and ≥ 90 for 61 %. Pinned episodes last a median
   0.63 s (mean 1.13 s). While pinned, the reducing quote sits one tick behind the touch 30 % of the time, because F is
   within theta of the far touch.
3. **But the markout view misleads about which lots to drop.** Fills in which we take 150–200 lots from one taker
   (the "flips" from −100 to +100) are 230k lots with m1 = 0.00. Yet removing the flipping part (`tq_x 0`) **halves**
   calm PnL: 238.6k → 112.3k on r3 and 253.2k → 116.6k on synth. A lot that leaves us at the opposite cap is what the
   next reducing fill and the next IOC need. The value lies in using the room, not in the per-lot markout.
4. **S error is small.**
   - F − S in calm: sd 0.14, mean |·| 0.09.
   - The exact-S oracle (`calm_oracle`, smoothed s_hat, so it uses the future): calm 238.6k → **254.8k** (+16.3k), total
     817.1k.
5. **The IOC side is near its ceiling.** ceiling.md puts calm stale V200 at 104k; r3_best takes 97k of IOC edge.
   Calm IOCs mostly reduce position (118k of 142k lots). Even the lowest-edge bucket (F edge 0.3–0.5 at send) marks
   out m1 +0.55. Calm IOC fills peak at +3–5 s and revert by 30–60 s (transient impact), so they have to be recycled
   quickly, and they are.

## 2. Ideas, ranked (eval3; Δ calm is r3 / synth seed1; Δ total r3 in brackets)

| # | idea (flags in `calm_a.py`) | tag | Δ calm r3 | Δ calm syn1 | status |
| --- | --- | --- | ---: | ---: | --- |
| 1 | **combo: take 0.10 + reduce-only IOC (`tr_take 0`, `tr_take_sd 1`) + `theta_keep 0`** | calm_c2 | **+18.3** (+16.5) | **+10.1** | tested, finalist (see 3) |
| 2 | reduce-only IOC at a lower threshold: when long, sell IOC every bid ≥ F + 0 + 1·sd, up to the position (flat only) | calm_trt0 | +6.7 (+7.0) | +5.2 | tested |
| 3 | theta_keep 0.10 → 0 (keep resting orders down to zero edge: queue priority is worth more than 0.1 tick) | calm_kp0 | +8.4 (+8.7) | +0.2 | tested |
| 4 | calm IOC take 0.15 → 0.10 | calm_tk10 | +5.9 (+5.9) | +1.8 | tested |
| 5 | take_sd 2 → 1 | calm_tsd1 | +4.7 (+3.9) | +3.1 | tested |
| 6 | IOC room ignores our own resting passive orders on that side and cancels them (`oc_on`) | calm_oc | +3.7 (+6.5) | +1.5 | tested; **loses −14.7 calm combined with take 0.10/sd 1** |
| 7 | inventory skew 0.10 (reducing side tighter) | calm_isk10 | +2.8 | 0.0 | tested, noise level |
| 8 | theta 0.25 → 0.20 | calm_th20 | +1.8 | +0.2 | tested, noise level |

The combinations are not additive:

| tag | flags | Δ calm r3 | Δ calm syn1 |
| --- | --- | ---: | ---: |
| calm_c1 | take 0.10 + trt0 | +6.4 | +5.3 |
| calm_c2 | c1 + theta_keep 0 | +18.3 | +10.1 |
| calm_c3 | c2 + inv_skew 0.1 | +11.4 | +7.0 |
| calm_kp0trt0 | theta_keep 0 + trt0 | +15.1 | +8.4 |
| calm_trtm1 | tr_take −0.1 | +7.4 | +3.8 |

The synth seed1 shock part swings ±25k on calm-only changes. Calm keys act during the shock window too, but the swing is
noise. Judge calm changes on the calm part.

Expected gain on round 3 from the calm angle: **+15k to +20k (about 2 %)**. That is far from 2×.

## 3. Finalist `calm_c2` on all synth tapes

`--set take=0.10 tr_take=0.0 tr_take_sd=1.0 theta_keep=0.0` on `calm_a.py`, against r3_best (fin_best_v1 / fin_v2):

| tape | r3_best | calm_c2 | Δ total | Δ calm | Δ vol | Δ shock |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| round3 | 795.0 | 811.5 | +16.5 | +18.3 | −0.6 | −1.2 |
| v1 seed1 | 908.4 | 915.7 | +7.3 | +10.1 | −1.8 | −1.0 |
| v1 seed2 | 968.0 | 965.3 | −2.8 | +4.8 | −0.5 | −7.1 |
| v1 seed3 | 931.3 | 938.4 | +7.1 | +6.2 | +0.3 | +0.5 |
| v1 seed4 | 861.0 | 866.2 | +5.3 | +11.7 | −1.6 | −4.8 |
| v1 seed5 | 998.8 | 1008.3 | +9.5 | +8.2 | +0.2 | +1.2 |
| v1 seed6 | 870.8 | 881.2 | +10.4 | +4.7 | −1.8 | +7.5 |
| v2 seed11 (held out) | 826.4 | 839.6 | +13.2 | +13.8 | 0.0 | −0.7 |
| v2 seed12 | 878.6 | 882.0 | +3.4 | +9.2 | −0.2 | −5.6 |
| v2 seed13 | 853.5 | 883.8 | +30.3 | +13.6 | −0.8 | +17.4 |
| v2 seed14 | 888.4 | 899.6 | +11.2 | +9.4 | +0.8 | +1.0 |
| **mean v1 (6)** | 923.1 | 929.2 | **+6.1** | **+7.6** | −0.9 | −0.6 |
| **mean v2 (4)** | 861.7 | 876.2 | **+14.5** | **+11.5** | 0.0 | +3.1 |

- **Calm is up on 11 of 11 tapes**, by +4.7k to +18.3k.
- The volatile part is unchanged (the keys are calm-only).
- The shock part is noise (−7k to +17k).
- It passes the brief's rule: positive on round 3 and on the synth mean.
- To adopt it, add to r3_best: `PARAMS.update({'take': 0.10, 'theta_keep': 0.0})`, plus the reduce-only IOC block
  (`tr_take` in `calm_a.py`, about 15 lines in `FVMM._quote` before the regular take).

## 4. Dead ends (numbers, do not retry)

| idea | tag | Δ calm r3 | Δ calm syn1 | why |
| --- | --- | ---: | ---: | --- |
| pre-join pj_q 50 / 100 one tick behind (existing flag) | calm_pj50/100 | −62.5 / −68.5 | −61.8 / −69.8 | takes room from the target level |
| touch size = flattening part + x, rest one tick deeper (`tq_x 50, tq_deep 1`) | calm_tq50d | −54.4 | −52.8 | same: the flip lots are the valuable ones |
| `tq_x 0` (no flipping part) / with deep rest | calm_tq0 / tq0d | −126.3 / −81.5 | −136.6 / −75.2 | halves calm |
| queue cap: at most 400 lots of (market ahead + ours) at the level (`qk_max 400`) | calm_qk400 | −32.8 | −41.0 | dropping deep-queue lots wastes room |
| **over-cap queue seniority**: orders left one tick behind the target do not count against the room, each capped at cap − pos (`oj_on`, `oj_q 0`) | calm_oj0b | −50.0 (breach +5.6k) | −66.7 (+11k) | sweeps fill both levels; adverse fills plus breach and exit |
| same, plus a new deep order of 100 at target ± 1 (`oj_q 100`) | calm_oj100b | −46.2 (breach +12.3k) | −57.0 | |
| size-only Y skew: with transient stock \|Y\| > 1, cap the side that builds inventory against the reversion at 0 (`ys_lim 0`) | calm_ys0_1 | −18.9 | −18.8 | see below |
| reach_off 1.0 → 0.5 / 1.5 | calm_ro05/15 | −99.8 / −73.9 | −133.4 / −59.7 | 1.0 is sharp, matches round(S ± 1) takers |
| keep_depth 0 (2 = identical to 1) | calm_kd0 | −2.9 | −19.3 | |
| theta_keep 0.2 | calm_kp2 | −8.3 | −7.1 | |
| take 0.20, take_sd 3 | calm_tk20 / tsd3 | −1.4 / −12.2 | −8.0 / −18.2 | |

Implementation warning (cost me one run). `modify_order` answers with `REJECTED` when the shrink is no longer valid,
for example after a partial fill. The base `on_private_order_update` then drops the order, although it is still
live. Its later fills are ignored, so the position drifts past the cap (−20M penalty). `calm_a.py` counts the pending
modifies per order and treats their `REJECTED` and `MODIFIED` answers as quantity updates. Anyone who uses
`modify_order` on the r3_best base needs this fix.

## 5. Inventory and transient impact (Q3, `an3.py`, `an4.py`)

- The transient stock Y, built from the tape fills with (1 − p)·c·Q^e and T = 16.7 s, predicts calm S drift exactly
  as the model says:

  | horizon | β of ΔS on Y | model −(1 − e^(−h/T)) |
  | --- | ---: | ---: |
  | 1 s | −0.056 | −0.058 |
  | 5 s | −0.255 | −0.259 |
  | 10 s | −0.43 | −0.45 |

  The correlation is only −0.10 at 1 s and −0.26 at 10 s.
- An oracle that holds ±100 against Y through all of calm earns **34k** (38k with the model drift). r3_best's position
  is uncorrelated with Y (corr 0.013), so it captures ≈ 0 of this.
- But holding a target costs fills. The size-only version (`ys_lim 0`, `ys_y0 1`) loses 19k on both tapes. The drift
  is 0.06 tick/s per tick of Y, while a passive round trip earns ≈ 0.2–0.4 per lot within about 1 s. Any rule that
  idles room for tens of seconds to harvest reversion loses.
- Reducing fills are not much better than increasing ones on markout (m1 +0.19–0.20 against +0.14–0.19). What helps is
  reducing faster with IOCs at edge ≥ 0 + 1·sd (idea 2).

## 6. Ladder-specific (Q4)

- In calm, ladder rungs are only 3–5 % of the stale touch quantity (makers README). The IOC take already sums every
  market order through the limit, ladder or not, so stale rungs at the touch are already sniped. The remaining upside
  is bounded by ≈ 4 % of calm IOC edge (< 4k).
- The ladder never quotes at touch − 1, so "joining before the ladder fills a level" means joining touch − 2 or deeper.
  That is the pre-join and over-cap seniority family, which loses (section 4).
- Any order we leave two or more ticks behind the touch sits behind 150-lot rungs that are never cancelled inside the
  band. Fills there come only from sweeps (big takers, m1 < 0).

## 7. What would be needed to double calm

The gap is queue priority: a ceiling of ≈ 365k if first in the queue (ceiling.md) against ≈ 123k of passive edge now.
Our first-in-queue fills (0 ahead, or inside a 2-tick spread) already mark out +0.41–0.43 per lot. Being first more
often needs one of three things:
- speed into a newly opened level, which is impossible at 5 ms latency against makers arriving every ≈ 5 ms;
- resting orders at levels before the price reaches them, which costs room under the cap and loses here;
- tolerated breaches, which lose here because calm sweeps fill both levels.

I see no calm mechanism that reaches 2×. Calm is +2 %. The doubling has to come from volatile, where room is pinned
90 % of the time and V200 is 864k against 251k now (ceiling.md).
