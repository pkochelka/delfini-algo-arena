# volatile: the volatile regime inside the ±100 limit

Angle owner: volatile. Base: `strategies/marek/round3/r3_best.py` (tag `fin_best_r3`: r3 794,988, volatile part 421,723;
synth v1 seed 1 908,414, volatile 448,179). Prototype: `strategies/marek/round3/ideas/volatile_tk.py`. With every flag
off it gives 794,988 on r3, identical (tag `volatile_off`). Analysis scripts are in
`marek/research/round3/ideas/volatile_an/` (see Files). All tags start with `volatile_`. Screening was on r3 plus synth
seed 1, at most 2 simulator processes at a time.

**Summary.**
- **The ±100 room is the only binding constraint in volatile.** r3_best sits at |pos| = 100 for 90 % of volatile time
  and flips between +100 and −100 about 0.8 times per second. Raising the volatile cap gives:
  - cap_t 150: +226k gross on r3 (+212k on s1)
  - cap_t 200: +425k on r3 (+401k on s1)

  These are gross of the 10k/s breach penalty. Edge per lot does not fall with size. The ceiling agent found the same
  +425k.
- **The take rule is at a local optimum in every direction I tried.** Threshold level, a split between reducing and
  adding lots, edge-scaled size, full-room IOC size, r3-fitted volatile filter parameters, a transient-impact shift and
  a momentum shift all came out flat (±5k) or clearly negative.
- **IOC room recycling loses −110k to −133k.** Measured exit curve: an IOC exit right after a volatile snipe loses
  1.1–1.5 ticks per lot, and the exit only breaks even after ≈ 0.25 s. Under a size-independent 10k/s penalty, that
  also kills oversize on ordinary volatile moves.
- **Fading the transient impact in volatile takes is strongly negative** (−42k at 0.1·Y, −236k at 0.3·Y). Over the
  holding time we actually have, the S drift is *momentum*, not reversal:
  - Holding time between flips: median 1.05 s, time-weighted 2.55 s.
  - S drifts toward F − mid with β = +0.57 at 1 s and +0.71 at 2 s.
  - The reversal only shows up by 15 s (β −0.31).
- **The one positive result: a momentum-gated, flatten-only passive quote (`vfm20`).**
  - Round 3: **+12.1k**.
  - Synth v1 mean (6 seeds): **+1.8k**. The volatile part is up on all 6 (+5.8k mean); the shock part is −3.8k mean,
    which is a lottery (see §2).
  - Held-out v2 family (4 seeds): **+2.8k** mean, volatile +8.4k, up on all 4.

  The gain is small because the quote only recycles room cheaply once the momentum carry has run out.
- **Regime detection is already essentially instant.** Entry lag is 40 ms median (p90 0.12 s) and exit lag 40 ms
  (p90 0.18 s). False volatile time is 0.5 % of calm. Faster detection is worth less than 1k.

Doubling volatile PnL needs about +100 lots of room held through the move. The penalty rules that out, and I found no
within-limit mechanism that recovers more than ≈ 1–3 % of it.

## 1. Measurements (r3, r3_best instrumented; scripts `an1`–`an12`)

| Quantity | Value |
| --- | --- |
| Volatile time with \|pos\| = 100 / ≥ 90 | 90 % / 91 % (mean \|pos\| 95.0); calm 56 % / 61 %; shock windows 63 % |
| Volatile IOC sends / qty sent / qty filled | 2,018 / 231,656 / 208,813 (fill ratio 0.90) |
| Takes where visible qty < room | 38 % of volatile takes (77k lots of room left unused at that instant) |
| Holding runs (same sign of pos) | 1,175; median 1.05 s, p25 0.52, p75 2.01, p90 3.03; time-weighted mean 2.55 s |
| Take "episodes" (same side and touch price, < 50 ms apart) | 5,388; 73 % find zero room on that side (= ceiling's 73 %) |
| Visible qty up to our limit per episode | median 308, p90 1,560, p99 5,031 lots; total 3.48M against 209k filled (6 %) |

**Markout of volatile IOC fills** (S = exact s_hat, measured from send time + 20 ms; the s_path row at a taker carries
the pre-kick S, so +0 is biased low):

| horizon | +20 ms | +0.2 s | +1 s | +2 s | +5 s | +15 s | +60 s | +120 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S − price per lot | 1.31 | 1.61 | 1.70 | 2.16 | 2.13 | 2.22 | 1.93 | 1.86 |

Of the ≈ 2.1 ticks per lot, ≈ 0.8 comes after the fill (book-lag momentum, below). 209k lots × 2.1 ≈ 440k, which is
the volatile PnL.

**Edge per lot by F-edge at the fill price.** Most of the volume is low-edge, and it is still positive. Don't raise the
threshold.

| F-edge | [0.5, 1) | [1, 1.5) | [1.5, 2) | [2, 3) | [3, 4) | [4, 6) | ≥ 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lots | 84,229 | 58,774 | 31,427 | 22,555 | 6,467 | 3,761 | 1,600 |
| S-markout at +1 s | 0.80 | 1.20 | 2.66 | 2.67 | 4.49 | 7.70 | 7.59 |

**Bad snipes are rare.** Fills with negative S-edge at +20 ms are 4 % of volatile qty, −13k in total. A plot of F
against s_hat seems to show |F − S| > 1.5 for 21 s of volatile time, but these are single 20 ms samples at kick
instants (the pre-kick row issue). Fills inside them have S-edge +1.12 per lot. Ceiling's exact-S oracle agrees: a
perfect filter adds about 0 in volatile.

**Exit curve after a volatile snipe.** This is what an immediate unwind (or the exit of an oversize excess) gets:
exit-side touch (bid for buys, ask for sells) minus our fill price, per lot, qty-weighted.

| delay | 0 | +20 ms | +50 ms | +0.1 s | +0.2 s | +0.5 s | +1 s | +2 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exit − price | −2.11 | −1.46 | −1.12 | −0.86 | −0.26 | +0.46 | +0.72 | +1.31 |
| share > 0 | 0 % | 0 % | 1 % | 5 % | 25 % | 48 % | 51 % | 53 % |

The opposite touch is pre-kick, so a round trip through the same book costs about the old spread. Holding an excess X
for Δ seconds costs 10k·Δ. At Δ = 0.5 s it breaks even only for X > 10,900 lots, so it can never pay outside shocks.
This matches bigmoves §3.

**S dynamics that matter for holding** (`an5`, `an10`; exact S on a 50 ms grid, volatile interiors, shocks masked):
- Book-lag momentum. Regress S(t+h) − F(t) on g = F − mid:

  | h | 0.05 s | 0.2 s | 0.5 s | 1 s | 2 s | 5 s | 15 s |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | β, volatile | +0.05 | +0.21 | +0.39 | +0.57 | +0.71 | +0.26 | −0.31 |
  | β, calm | 0.00 | +0.05 | +0.14 | +0.24 | +0.44 | +0.63 | +0.15 |

  sd(g) is 1.71 in volatile and 0.55 in calm. Mechanism (ceiling agent): when the book sits below S, buy takers (limit
  round(S + 2.55)) fill more than sell takers, so the kicks are one-sided.
- Past S moves. In volatile, the past 0.5 s predicts the next 0.5–2 s with β +0.09 to +0.13. The past 10 s predicts the
  next 5 / 15 s with β −0.15 / −0.24. A transient-stock proxy (Σ ΔS·e^(−age/13.6)) gives β −0.03 at 1 s, −0.07 at 2 s,
  −0.20 at 5 s and −0.36 at 15 s.
- So over our 1–2.5 s holding time the drift is with the snipe (momentum). The transient reversal (T ≈ 14 s) only
  matters on positions held ≥ 5–15 s, which volatile flips rarely leave in place.

## 2. Ideas, ranked by measured gain

| # | Idea (flags in `volatile_tk.py`) | Evidence / upper bound | r3 Δ (vol Δ) | s1 Δ (vol Δ) | Status |
| --- | --- | --- | ---: | ---: | --- |
| 1 | **Momentum-gated flatten-only passive quote** (`vf_on, vf_th 2.0, vf_keep 1.5, vf_mom −0.5, vf_pos 60`) | cheap room recycling once the carry has run out; ceiling's rp_mom is the same idea | **+12.1k (+9.1k)** | +3.6k (+8.4k) | tested: synth v1 mean +1.8k (vol +5.8k), v2 held-out +2.8k (vol +8.4k); passes the brief's rule, small |
| 1b | same, vf_th 2.5 / keep 2.2 | | +6.0k (+3.6k) | +1.0k (+7.4k) | tested |
| 2 | Split take: reducing lots at threshold −0.2 (`vt_red −0.2`) | flips are the PnL; cheaper flips | +3.7k (+3.5k) | −6.3k (−5.5k) | tested, noise |
| 3 | Edge-scaled adding size (`vt_on, vt_ek 1`) | keep room for the follow-up episodes (73 %) | +1.3k (+0.8k) | −0.2k (+0.4k) | tested, noise |
| 4 | r3-fitted volatile filter (c_v .062, e_v .63, jv_v .0676, d0_v .673, lam_v 1.0, lamj_v .18, bj_v .92, tr_T_v 13.6) | ceiling: exact S adds 0 in volatile | +2.7k (−1.7k) | −3.7k (−4.9k) | tested, noise |
| 5 | IOC for the full room, not just the visible qty (`vt_full`) | 38 % of takes have visible qty < room | −0.5k (−0.1k) | −0.8k (−0.2k) | dead (no extra fills arrive within the 5 ms) |
| 6 | Faster regime detection (ladder bursts, 20–70-lot share) | lag 40 ms median → ≤ 1k | — | — | dead, measured |

**Idea 1 in detail (tag `volatile_vfm20`, `volatile_vfm20_s6`).** While turbulent, outside droughts, and with
|pos| ≥ 60, one order rests on the reducing side at F ± 2.0 (kept while edge ≥ 1.5, reach_t 2.5), at most 100 lots, and
only while (F − mid)·sign(pos) ≤ −0.5. That is, only once the book-lag momentum has turned against the position, so the
exit no longer gives up carry.

| tape | Δ total | Δ volatile | Δ shock | Δ calm |
| --- | ---: | ---: | ---: | ---: |
| round 3 | **+12.1k** | +9.1k | +1.2k | +1.7k |
| s1 | +3.6k | +8.4k | −5.5k | +0.7k |
| s2 | +10.3k | +8.4k | +2.4k | −0.5k |
| s3 | −1.3k | +3.5k | −3.5k | −1.2k |
| s4 | −16.6k | +6.0k | −22.6k | 0.0k |
| s5 | +3.6k | +3.8k | −0.7k | +0.5k |
| s6 | +11.1k | +4.8k | +7.2k | −1.0k |
| **synth v1 mean** | **+1.8k** | **+5.8k** | −3.8k | −0.2k |
| v2 s11 / s12 / s13 / s14 | +12.2k / +0.8k / −3.1k / +1.4k | +6.6k / +7.7k / +8.3k / +10.8k | +6.0k / −4.0k / −10.4k / −9.2k | |
| **synth v2 mean (held out)** | **+2.8k** | **+8.4k** | −4.4k | |

- **The s4 shock loss is not caused by the quotes.** I blocked them for 20 s after every drought detection
  (`vf_shk=1`, tag `volatile_vfm20shk`) and s4 shock was still −19.2k against base. r3 came out +10.2k. The volatile
  changes alter the position entering each shock, and that is a lottery.
- **Variants:**
  - θ 2.5 / keep 2.2: r3 +6.0k, s1 +1.0k.
  - No gate (`vf25`): r3 −4.6k.
  - The gate is what makes it work, the same as ceiling's rp_mom (r3 +10.9k, synth 4-seed mean −2.0k).

Expected gain on r3: about +10k from idea 1. Nothing else in this angle clears noise.

## 3. Dead ends (numbers, so nobody retries them)

All are r3_best + one change. Δ total (volatile part).

| Tag | Change | r3 | s1 | Why |
| --- | --- | ---: | ---: | --- |
| volatile_taket0.6 | take_t 0.3 → 0.6 | −11.2k (−13.0k) | −24.1k (−23.8k) | fewer flips. Flip count, not edge per lot, is the PnL |
| volatile_taket1.0 | take_t → 1.0 | −38.6k (−37.4k) | −65.6k (−63.9k) | same |
| volatile_vt_add03 | adding lots need +0.3 more edge | −0.2k (−3.3k) | +7.3k (−11.2k; shock +18.5k) | reserving room for follow-ups doesn't pay |
| volatile_vt_red-02_add03 | both | +1.0k (−1.2k) | −15.3k (−15.2k) | |
| volatile_yk01 | volatile take reference F − 0.1·Y (fade the transient) | −42.0k (−34.1k) | −81.1k (−80.9k) | blocks the post-kick snipes; the drift over the holding time is momentum |
| volatile_yk03 | F − 0.3·Y | −235.6k (−215.4k) | −307.0k (−319.8k) | same |
| volatile_gk03 | take reference F + 0.3·(F − mid) (ride the momentum) | −10.6k (−9.4k) | −23.3k (−45.9k) | sweeps deeper at worse prices with room that is used up anyway; ceiling's mk_k 0.5 −19.6k |
| volatile_gk06 | F + 0.6·(F − mid) | −30.1k (−21.4k) | −52.3k (−72.3k) | |
| volatile_vu1 | volatile IOC unwind \|pos\| > 50 → 0 at ≥ F − 1 | −133.3k (−109.4k) | −120.5k (−125.2k) | exit costs ≈ 1 tick plus the carry; next move 50/50, so EV ≈ −cost |
| volatile_vf25 | flatten-only passive at F ± 2.5, no momentum gate | −4.6k (−4.2k) | +9.7k (+8.8k) | gives up the carry (= ceiling rp1/rp2, round-2 passive unwind) |
| volatile_vf30 | same at F ± 3.0, reach_t 3.5 | −17.3k (−7.6k) | +16.4k (+11.5k) | r3 and synth disagree (synth has thinner ladder and higher σ) |
| (e10 era) | pv_thr 0.3 / 0.5 / 0.7 | 780.1 / 778.1 / 778.9k | synth mean 894.9 / 897.2 / 903.8k | detection is already instant; 0.7 is in r3_best |
| (e10 era) | passive_t 1, theta_t 2.0 / 2.5 | −38.5k / −13.4k on r3 | −12.9k / −1.7k synth mean | passive in volatile loses; at 90 % pinned this is ≈ flatten-only without a gate |

**Room oracle** (penalty ignored: gross = PnL + breach; `xu_tgt` raised with the cap, else the oversize exit dumps the
excess and gross falls to −258k):

| run | r3 gross | Δ | s1 gross | Δ | volatile IOC lots (r3) | S-markout +5 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r3_best | 812.7k | | 923.5k | | 209k | 2.13 |
| cap_t 150 | 1,038.9k | +226k | 1,135.8k | +212k | 293k | 2.26 |
| cap_t 200 | 1,237.5k | +425k | 1,324.5k | +401k | 371k | 2.37 |

Holding the extra 100 lots would breach for ≈ 1,560 s, a 15.6M penalty.

## 4. Answers to the brief's four questions

1. **Capacity.**
   - Pinned 90 % of volatile time. 73 % of take episodes find zero room.
   - Each 100 lots of cap is worth ≈ 425k gross, but only as held room.
   - IOC recycling costs more than room is worth under the ±100 rule (−110k to −133k).
   - Passive flatten-only quotes:
     - ungated: −4.6k
     - gated on momentum: +12.1k (idea 1)
   - Unwinding into ladder depth: the bid ladder sits 2–7 below the pre-kick bid, so it is a loss. Not run; bigmoves
     measured it.
2. **Take rule.**
   - Thresholds are optimal: take_t 0.3 beats 0.05 / 0.15 / 0.6 / 1.0.
   - Follow-up snipes after a big taker already happen: every event re-evaluates, and the IOC fill ratio is 0.90. They
     are blocked by room, not by the rule.
   - The follow-up episodes have a higher instantaneous edge (2.9 at 0.2 s against 2.1 for the first one). On r3 their
     edge falls to 0.6 by 15 s, against 2.6 for first episodes (noisy, sd ≈ 0.4). Reserving room for them (vt_add,
     vt_ek) is flat.
   - Size per snipe against edge: flat.
   - The taker law (round(S ± 2.55) plus tail) gives passive fills only at x ≤ 2.5, and those are the adverse ones.
3. **Regime detection.** P(volatile) > 0.7 within 40 ms (median) of the S-based switch, both ways. Nothing left to gain.
4. **Transient impact.**
   - Over the 1–2.5 s holding time, S moves *with* the snipe (book-lag momentum β +0.57 at 1 s). The 14 s decay only
     matters beyond 5 s.
   - Choosing the pinned side by Y: −42k at 0.1·Y.
   - Choosing it by the forward value F + k·(F − mid): −10.6k at 0.3.
   - The only use that worked is negative: don't exit while the momentum still favours the position (vf_mom gate).

## 5. Leads for other angles

- **calm:** the same book-lag momentum exists in calm, slower: β 0.24 / 0.44 / 0.63 at 1 / 2 / 5 s, sd(g) 0.55.
  Calm IOC fills peak at +5 s (1.03 per lot) and then revert (−1.06 at 120 s, 142k lots, noisy). Calm limit fills rise
  0.19 → 0.68. Unloading take inventory between about 3 and 10 s might be worth tens of k (ceiling's gap 3).
- **bigmoves:** use the exit curve above. On non-shock volatile moves, breakeven needs > 10k lots of excess at 0.5 s, so
  oversize can't pay there.

## Files

- `strategies/marek/round3/ideas/volatile_tk.py`: prototype. Flags `vt_on/vt_red/vt_add/vt_ek`, `vt_full`, `vt_yk`,
  `vt_gk/vt_gmax`, `vu_u/vu_pos/vu_to`, `vf_on/vf_th/vf_keep/vf_q/vf_mom/vf_pos/vf_shk`. All off = r3_best.
  - Candidate settings:
    `vf_on=1 vf_th=2.0 vf_keep=1.5 vf_mom=-0.5 vf_pos=60`.
  - vf_mom gates on (F − mid)·sign(pos) ≤ vf_mom.
  - Exit quotes are blocked during droughts and below vf_pos.
- `marek/research/round3/ideas/volatile_an/`:
  - `mkinst.py`, `mkinst_set.py`: instrumented copies (fills, sends, takes, 20 ms state samples) → pickle.
  - `an1.py`: pinning and markouts by regime.
  - `an2.py`, `an3.py`: take episodes and edge buckets.
  - `an4.py`: regime-detection lag.
  - `an5.py`, `an10.py`: S predictability.
  - `an6.py`: holding runs.
  - `an7.py`: visible qty against room.
  - `an8.py`, `an9.py`, `an12.py`: filter error and bad snipes.
  - `an11.py`: exit curve.
  - `ev.sh`: eval3 wrapper.

  - `export_s.py`: writes `spath.npz` and `sfilt.npz` (from `s_dynamics/s_path.parquet`) into the cwd; the `an*`
    scripts expect them there. Run them with `uv run --project <repo> python anN.py LOG.pkl`.
