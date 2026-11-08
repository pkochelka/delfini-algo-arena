# bigmoves: sniping stale quotes on big moves, with oversize (> 100 lots)

Angle owner: bigmoves. Base: `strategies/marek/round3/r3_best.py` (tag `fin_best_r3`, r3 795.0k, synth v1 mean 923.1k).
Prototype: `strategies/marek/round3/ideas/bigmoves_trig.py` (all `bm_*` flags off = r3_best).
Analysis scripts (all `uv run --with pyarrow python <script> NAME [SIMLOG]`, NAME = r3 / r2 / s1…):
`marek/research/round3/ideas/bigmoves/`: `ceiling.py` (shock stale book, fate, IOC ceiling), `triggers.py` (signal timing
per shock, value at each), `unwind.py` (r3_best entry/exit/penalty per shock), `nonshock.py` + `ns_mine.py` + `exitliq.py`
(stale value after non-shock takers, what we took, exit liquidity), `droughtbook.py`, `timeline.py`. Outputs `ceiling_*.csv`,
`triggers_*.csv`, `nonshock_*.pkl` sit next to them.

**Summary.** There is no doubling in this family. The shock ceiling on round 3 is about 104k gross stale value for an IOC
that lands 5 ms after the first direction signal. r3_best already takes 58k of it, so at most ≈ 45k is left. Outside shocks
the stale value beyond the 100-lot room is large (≈ 0.5M gross in volatile), but it cannot be exited: the stale quote itself
caps every new opposite-side quote below our purchase price. Two small trigger changes are worth taking: +34k on r3, +14k on the
v1 synth mean, +2k on the held-out v2 mean. The synth tapes overstate shock value about 5×, so shock changes that "win on synth" need care.

## 1. Ranked ideas

| # | Idea | Measured upper bound (r3) | Result | Status |
| --- | --- | ---: | ---: | --- |
| 1 | Early sweep trigger with edge credit (`bm_sw`, `bm_sw_cr`, `bm_sw_big_only`) | 30k (shock 18, 4127 s) | +27.9k r3, −0.5k s1 | prototyped, tested |
| 2 | Clip-trigger edge credit for every clipped insert (`ss_clip_e=2`, existing knob) | 6.7k r3 (shock 10); 86k s1 (shock 10) | **+5.6k r3, +63.4k s1** | tested |
| 1+2 | Combined, with early triggers only for sweeps spanning ≥ 3 ticks = tag `bigmoves_C2` | ≈ 45k left over r3_best | **r3 +34.0k, v1 mean +14.2k, v2 mean +1.8k, r2 +7.1k** | finalist, recommended |
| 3 | `ss_far` 3 → 5 (on top of C1, = C2 on r3/s1) (fewer false "far insert" triggers before the jump) | s1 shock 12: false DOWN trigger 24 ms before a +57 jump | −0.1k r3, +5.3k s1 (on top of C1) | tested, marginal |
| 4 | Invisible-insert trigger (`ss_inv=1, ss_inv_n=1`) | shocks 2, 14: ≈ 11k | **−23.0k r3, −255.8k s1** | dead end |
| 5 | Re-fire on a later clip (`ss_reclip=1`) | — | −27.0k r3, +29.5k s1 (on top of C1) | rejected (r3 negative) |
| 6 | Non-shock oversize (xcap-style) with fast unwind into ladder | 516k gross beyond 100 lots, e ≥ 3, volatile | not run: exit is blocked (see §3) | dead end, measured |
| 7 | Higher cap / wider IOC limit (`xs_cap`, `ss_lim`) at shocks | 0 on r3: cap 950 and lim 8 never bind | — | not worth it |

Expected gain on round 3: ≈ +34k from C2 (+4 %); on synth +2…+14k. Everything else in the family is below +15k.

## 2. Shocks (questions 1, 2, 4)

### 2.1 Stale book at the jump, and where it goes

Definitions: τ = first insert after onset + 2.85 s that reveals the jump direction (a maker bid ≥ S_pre + 1.5 or a far
ask ≥ S_pre + 10, mirror for down; a taker limit ≥ 6 through S_pre). Stale = resting orders on the losing side priced
inside the new S. Value = Σ lots × |S_new − price|. S_new = exact S just after the jump (s_dynamics path); synth: mid + true J.
"cN_950" = what an IOC landing N ms after τ takes: 950 lots, limit stale touch + 8, best edge first.

| | r3 (27 shocks) | r2 (27) | synth s1 (26) |
| --- | ---: | ---: | ---: |
| Stale lots at τ | 4,100 | 7,939 (2,369 at 5398 s) | 18,368 |
| Stale value at τ | 171k | 403k (251k at 5398 s) | 879k |
| … still there at τ + 5 / 10 / 20 / 50 ms | 104k / 61k / 45k / 13k | 356k / 303k / 35k / 0.4k | 841k / 708k / 580k / 465k |
| Removed by market takers / by cancels | 64k / 107k | 11k / 393k | 474k / 405k |
| c5_100 (no oversize) | 56k | 40k | 97k |
| c5_950 / c10_950 / c20_950 | **104k** / 61k / 45k | 208k / 177k / 35k | 592k / 510k / 426k |
| c5 with no cap (limit 8) | 104k | 356k | 841k |
| Captured by r3_best (jump-side fills, valued at S_new) | **58.0k** (58.7k in a 2.3 s window) | — | 231k (`triggers.py` v_fire) |
| Shock-window PnL of r3_best | 135k | 255k (shockgen_best) | 207k |

Read-outs:
- **Round 3 against round 2.** Without the one r2 monster (5398 s, +113 jump: 251k of stale value, 101k of PnL) round 2
  had 153k of stale value and round 3 has 171k. The round-2 → round-3 drop in shock PnL (255k → 135k) is just that one
  shock; the stale book is the same size. Neither trigger latency nor the cap explains it.
- **The cap (950) and the IOC limit (ss_lim 8) never bind on r3.** c5_950 = c5 no-cap = c5 no-limit = 104k. On r2 the
  cap binds only at 5398 s.
- **The stale book disappears faster in r3:** half of it is gone 10 ms after τ (r2: 24 %). This is mostly because in r3
  the first visible signal comes later relative to the true jump, after the β·mv cancels have started.
  Market takers took 38 % of it, nearly all at shock 18.
- **Synth overstates shock value about 5×** (879k vs 171k of stale value): its drought book is twice as thick at the
  touch (median 580 vs 273 lots within 5 ticks at onset + 2.95 s; spread 3 vs 4), and it decays more slowly (84 % left
  at +10 ms vs 50 %). So oversize and shock-trigger changes win much more on synth than on round 3.
  Judging shock changes on synth (the brief's rule) is biased toward oversize.

### 2.2 Where r3_best loses the remaining ≈ 45k (r3)

Per shock, value at τ + 5 ms against what r3_best took (`triggers.py`, `ceiling_r3.csv`):

| Shock (onset) | J | Available (c5_950) | r3_best took | Why |
| --- | ---: | ---: | ---: | --- |
| 18 (4126.9) | +126 | 30.0k | 0 | a 402-lot IOC (then a 63-lot one) swept the stale asks 52 ms before the first new-level maker insert. That is 2.955 s after the onset estimate, before the window opens at `ss_w0` 2.99. r3_best fired 3 ms after the first new bid, by when the asks were gone. **Fixed by idea 1.** |
| 10 (2375.5) | −34 | 6.7k | 0 | clip ask (208 @ bid + 1) at −20 ms, but the excess was gated off (edge 1.5 < xs_edge, no clean cancel) → cap-only, and the position was already near the limit. **Fixed by idea 2.** |
| 14 (3187.2) | +52 | 5.6k | 0.3k | invisible asks at the new level at −4.7 ms. Stale asks were cancelled at +2.8…+12.7 ms; the first clip came at +15.5 ms, too late. Only an invisible-insert trigger could catch it (idea 4, which loses elsewhere). |
| 2 (663.3) | +53 | 11.6k | 6.0k | fired on the clip 1.7 ms after the invisible insert; half the stale asks were cancelled between +5 and +10 ms. Latency-bound. |
| 20 (4559.1) | −22 | 14.9k | 10.8k | a sweep at −39 ms came before the clip at +13 ms. Idea 1 should help here, but it did not change this shock on r3. |
| others | | 35k | 41k | on time |

There were no wrong-direction triggers on r3 (every r3_best shock trigger had the right sign; same on s1). The penalty side:
all 17.7k of r3's breach penalty comes from shocks. It mostly comes from small jumps where the excess was taken anyway,
e.g. shock 4 (J +5.7): 930 lots, entry +10.1k, exit −8.2k, penalty 2.0k, net ≈ 0 (`unwind.py`).

### 2.3 Earlier or more reliable detection (question 2)

- **The jump's first trace is usually invisible.** Makers repost at S_new ± D. For |J| > 20 the far-side repost (asks
  after an up-jump) lands more than 20 ticks from the touch, so the public feed shows nothing. The strategy still gets a
  callback with an unchanged book, but it cannot tell the side. The direction appears 1–5 ms later as a *clipped* insert
  (bid at ask − 1 after an up-jump), which r3_best already uses, or as a sweep. Invisible-insert timestamps (ms
  relative to the s_dynamics jump time): shock 2 −28.6, 10 −18.6, 11 −46.6, 14 −4.7, 15 −25.9, 18 −41.6.
- **Cancel waves are too slow and too noisy to give the direction first.** The stale-side cancel hazard is 1.7 + 3.4·mv,
  then a delay of Exp(470) + Exp(52), mean 21 ms, so the first stale cancel comes 3–20 ms after the jump. Meanwhile
  drought background cancels run at ≈ 5–6 /s per order on both sides. Firing on "invisible + one one-sided cancel"
  (`ss_inv_n=1`) lost 23k on r3 and 256k on s1.
- **Sweeps: one real miss.** In a calm drought (spread ≈ 4, taker limit round(S ± 1)) a market taker almost never
  trades 2+ levels unless S has jumped. The window start `ss_w0` = 2.99 shut out the 4127 s sweep. `bm_sw` accepts a
  ≥ 2-level one-sided sweep from onset + 2.90. `bm_sw_cr` gives it edge credit 10 when its trades span ≥ 3 ticks
  (`bm_sw_span`), so the first shot may carry the excess. Without the credit, F hardly moves on a 402-lot taker (kick
  ≈ 2.7 ticks), the excess stays gated, and only 100 lots would go.
- **Drought clock.** The jump is at drought onset + 3.000 s exactly, but the onset estimate is tens of ms off: the
  4127 s sweep came at flow-onset + 2.955, so that onset estimate was ≥ 45 ms late. The clock alone cannot time a 5 ms
  shot. It is only good as a window.

### 2.4 Unwind, multi-shot, IOC limit (question 4)

- After the excess fills, r3_best's unwind (xu_lim 1.0, then F − 4 after 80 ms) loses little: on r3 the exit fills are
  +4.9k in total against S_new, and the penalty is 17.7k over 1.77 s spent above |100|. The exit works at shocks because
  makers repost at S_new − D once the stale side is gone (20–50 ms). There is nothing to gain here.
- Multi-shot beyond about 1000 lots: MAX_POTENTIAL_POSITION is 1000, and the excess cannot be sold back before the stale
  side is gone, so a second 950-lot shot has nothing left to hit. On r3 the cap does not bind anyway.
- `ss_lim` 8: not binding on r3, r2 or s1 (c5 with limit 8 = c5 with no limit).

## 3. Non-shock big moves (question 3): dead end

`nonshock.py`: at each filled market taker outside shock windows, 5 ms later (our latency), stale lots on the side the
taker moved through, against S. A sniper with unlimited room consumes what it takes, so nothing is counted twice.
Volatile regime, r3:

| Edge vs S | events | stale lots | stale value | first 100 lots/event | first 300 | first 950 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ≥ 2 | 1,180 | 633k | 1.95M | 290k | 640k | 1.20M |
| ≥ 3 | 524 | 278k | 1.17M | 164k | 360k | 680k |
| ≥ 4 | 212 | 137k | 713k | 85k | 188k | 361k |

Calm adds < 0.11M. So the value beyond the 100-lot room is real: ≈ 516k gross (e ≥ 3, lots 100–950). r3_best takes 31k
of the 164k first-100 value at e ≥ 3, because at **91 % of these events it is already pinned at ±100 in the direction of
the move** (`ns_mine.py`).

Why oversize cannot collect it (`exitliq.py`):
- To keep the excess we must sell it back (after an up-move) into bids. But bids cannot sit above the stale ask, so every
  maker bid in the tape is clipped to ≤ stale ask − 1, below our purchase price. A replayed tape never re-prices them.
  Opposite-side depth within 1 / 2 / 4 ticks of S at these events: median 0 / 0 / 0, mean 0 / 0 / 145 lots.
- New opposite-side volume within 1–3 ticks of S over the next 100 ms: median 0, mean 18–44 lots.
- For the 116 biggest events (excess value ≥ 2k, 382k of the 516k), the time until 200 lots of exit volume appear
  within 1.5 ticks of S is 202 / 302 / **453** / 859 / 1840 ms (10th–90th percentile). That is a penalty of ≈ 4.5k per
  event against ≈ 3.3k of gross value, before the S risk on up to 950 lots (σ ≈ 3 ticks over 0.45 s).
- The ladder is not exit liquidity. It anchors to its own side's touch, so after an up-move the bid ladder sits at the
  old bid − 2…−7, i.e. 3–8+ ticks below the purchase price. Its depth is exit for a *loss*.
- At shocks the same clipping happens, but the stale side vanishes in ≈ 20 ms (mv ≈ |J| drives the cancel hazard to
  ≈ 400 /s), and only then can new quotes at S_new − D form the exit. Non-shock moves of 3–5 ticks give a hazard of only
  12–19 /s.

Conclusion: the same as round 2 ("no exit liquidity"), now with the mechanism measured. The room shortage is the real
limit here: 250k of first-100 value at e ≥ 2 arrives while we are pinned. That is a question for inventory management,
not oversize.

## 4. Finalist results

Tag `bigmoves_C2` = `bm_sw=1 bm_sw_cr=1 bm_sw_big_only=1 ss_clip_e=2` on `bigmoves_trig.py` (PnL in k, net of penalty).

| Tape | r3_best | C2 | Δ | Shocks changed (> 1k) |
| --- | ---: | ---: | ---: | --- |
| round3 | 795.0 | **829.0** | **+34.0** | 10 +5.6, 18 +27.9 |
| round2 | 761.0 | 768.1 | +7.1 | 10 +1.5, 19 +5.9 |
| v1 seed 1 | 908.4 | 971.7 | +63.3 | 10 +63.4 |
| v1 seed 2 | 968.0 | 968.7 | +0.7 | 6 +3.1, 15 −2.1 |
| v1 seed 3 | 931.3 | 935.0 | +3.7 | 17 +3.7 |
| v1 seed 4 | 861.0 | 860.3 | −0.7 | — |
| v1 seed 5 | 998.8 | 998.8 | 0.0 | — |
| v1 seed 6 | 870.8 | 889.1 | +18.3 | 6 +19.9, 4 −1.4 |
| **v1 mean (6)** | 923.1 | **937.3** | **+14.2** | |
| v2 seed 11 | 826.4 | 827.2 | +0.8 | 5 +5.8, 4 −4.9 |
| v2 seed 12 | 878.6 | 885.9 | +7.3 | 24 +9.2 |
| v2 seed 13 | 853.5 | 853.8 | +0.3 | — |
| v2 seed 14 | 888.4 | 886.9 | −1.4 | 8 −1.7 |
| **v2 mean (4, held out)** | 861.7 | **863.5** | **+1.8** | |

Recommendation: adopt C2. It is positive on r3, on the v1 mean, on the held-out v2 mean and on r2. Calm and volatile
PnL do not change on any tape; every change is inside shock windows.

**Why `bm_sw_big_only` matters.** The first version (`bigmoves_C1`, any one-sided ≥ 2-level sweep in
[onset + 2.90, onset + 3.10], credit also when the visible side was emptied) also made +34.0k on r3, but lost 67.9k on
v2 seed 11 and 22.3k on v1 seed 4 (v2 mean −21.1k). The cause, at v2 seed 11 shock 13 (J −107): 27 ms *before* the jump,
a 241-lot volatile drought taker (limit S + 2.5) swept two adjacent ask levels and emptied the thin ask side. That fired
UP with edge credit, used up the shock, and the real down-jump was then missed. C2 lets only sweeps spanning ≥ 3 ticks
(`bm_sw_span`) trigger early. The real 4127 s sweep spanned 9945 → 9949; a normal taker limit cannot do that in a
4-tick drought book.


## 5. Dead ends (do not retry)

- Early sweep trigger on any ≥ 2-level sweep (`bigmoves_C1`, without `bm_sw_big_only`): v2 mean −21.1k (seed 11
  −67.0k) from false pre-jump sweeps in volatile droughts.
- `ss_inv=1, ss_inv_n=1`: r3 −23.0k, s1 −255.8k (`bigmoves_inv1`).
- `ss_reclip=1` on top of C1 (≈ C2 on r3/s1): r3 −27.0k, s1 +29.5k (`bigmoves_D2`).
- `xs_edge=0` (excess on every trigger) on top of C1 (≈ C2 on r3/s1): r3 ±0, s1 −3.4k (`bigmoves_C3`).
- Non-shock oversize into the ladder: §3. Exit volume near S is ≈ 0 for 100+ ms because the stale quote clips it,
  the median wait for 200 lots is 453 ms (≈ 4.5k of penalty), and the gross excess per event is ≈ 3.3k.
- Raising `xs_cap` or `ss_lim` at shocks: neither binds on r3.
