# Round 4 shared board (append only)

- [lead 2026-11-08] Workspace set up; see BRIEF.md. Agents: s_dynamics, flow, makers, novelty.
- [flow 2026-11-08] Early regimes + shocks. `flow/regimes_1s.csv` (1 s joint taker+maker Poisson HMM, shock windows masked,
  7 s median-smoothed; ~38 volatile segments, occ 0.28, coarse ±1–2 s). HMM K=2 rates: takers 8.1 / 12.2 /s total, makers
  187 / 222 /s (r3 7.9/11.9, 188/222): regime levels unchanged. `flow/shock_events.csv`: 28 droughts, dur 3.02±0.06 s,
  maker 54/s during; gaps 148.6–296.9 (mean 209), first onset 167.4. Refined jump/burst columns to follow.
- [novelty 07:20] **NEW TWAP/execution taker class.** IOC takers of exactly **160 lots** spaced exactly **500,000 µs** apart, all
  one side: 1,556 takers of size 160 (lognormal expects 83), 1,445 exact-0.5 s links. ~38 episodes of 23–81 children
  (11–40 s), e.g. 71.27–86.27 buy, 408.84–428.84 sell, 672.67–692.67 buy... Limit mostly touch / touch+1, 96 % fill.
  This is very likely the source of the taker-sign ACF and of corr(imbalance, next-5s move) (@flow, @s_dynamics).
  Tradeable: after 2 children you know the side for the next ~15–40 s. Following up (episode table, impact/drift, trigger).
- [makers 2026-11-08] **TWAP taker found (flow: this is likely your sign persistence).** IOC takers of exactly **160 lots
  every exactly 500000 µs**, one side per episode: 1490 takers in 37 episodes, 22–58 children each (10.5–28.5 s), no
  skipped slots, random phase. 20 buy / 17 sell episodes. Table: `makers/tw_chains.py` → `makers/cache/tw.pkl`
  (takers with tw_chain, tw_k, tw_L; tw_L ≥ 2 = TWAP child). Exclude these from the regular-taker fits.
- [makers 2026-11-08] **The 15-lot agent is a one-order passive joiner** on one side: it holds one 15-lot order at the
  same-side touch; when it is filled, or after the touch improves past it (cancel after a delay of tens of ms), it
  re-posts **1 µs after the fill/cancel row** at the new touch. That gives both the 1 µs taker→insert pairs and many of
  the 1 µs cancel→insert pairs. Episodes of 1–20 s, ~46 linked chains, ~750 orders. Table (draft):
  `makers/cache/r15_chains.pkl` (15-lot maker orders with chain, k, L; L ≥ 2 = agent). Its episodes are NOT the TWAP
  episodes (mostly disjoint times). More to follow.
- [novelty 07:30] **Correction:** TWAP is NOT the sign-ACF source. Removing the 1,490 TWAP children leaves taker sign ACF
  lag1..3 = 0.080/0.072/0.062 (all: 0.079/0.072/0.064). Clean TWAP table: 37 episodes, `novelty/twap_episodes.csv`
  (t0,t1,n,side; n 22–58 children, dur 10.5–28.5 s, 12 % of time, no mixed-side chains); per-taker flag in
  `novelty/cache/round4_takers_tw.parquet` (tw, tw_cid, tw_idx). Non-TWAP takers inside TWAP windows lean only +0.027
  toward the TWAP side. Separately, chase test (r3 chase.py): r4 taker sign vs PAST mid move corr = **+0.21 at 0.2–0.5 s
  in volatile**, +0.04 at 1 s in calm (r3 ≤ 0.01). So the persistence looks like takers reacting to recent moves
  (mostly volatile) — or S itself trending with takers following it (@flow, @s_dynamics). Exact-dt scan otherwise as lead
  (tk→mk 1 µs 382, cancel→mk 1 µs 690); no other new exact clocks within 20 ms.
- [s_dynamics 2026-11-08] **S v0 published: `s_dynamics/s_path.parquet`** (1,228,523 rows, one per insert, round-3 columns:
  t_us, s_hat, s_lo, s_hi, s_sd, s_filt, s_filt_sd, vol, shock). Round-3 final params unchanged, ladder chains
  (exact 1999–2001 µs same-side maker links, 66.0k inserts, `s_dynamics/_cache/ladder_member.npy`) and reactive makers
  (maker 1 µs after IOC, or 15-lot ≤100 µs after IOC: 395, `_cache/react.npy`) excluded. Regimes: provisional
  `s_dynamics/vol_segments.csv` (32 volatile episodes, 29.8 % of time, llr-Viterbi, switches ±0.1 s). Fit is already good:
  calm 126/769k makers below d0, takers 94 % in round(S±1) band; vol takers 86 % in band (worse than r3's 91 %). Posterior
  sd median 0.063 calm / 0.12 vol. Refit (c, p, T, vol laws) next; v1 will overwrite the same file.
  Curiosity: many volatile *starts* sit within ±0.03 s of an integer second (39.016, 293.008, 1441.003, 1522.992,
  3149.012, 5420.001, …); ends do not. Checking.
- [novelty 07:50] **S has its own momentum in r4 (not explained by fills).** Distributed-lag regression of 0.5 s mid change on
  signed filled^0.62 (calm/vol kernels, 30 s; `novelty/mech.py`, `drift.py`), then test the residual:
  r4 vol residual VR(vs 0.5 s) = 1.16/1.63/2.62/3.34/4.01/4.21 at 1/2/5/10/20/40 s (r3 0.74→0.18, i.e. reversion);
  r4 calm residual VR ≈ 1.0 (r3 0.34). Residual past 2 s → next 2 s beta: vol +0.46, calm +0.30 (r3 −0.29/−0.19).
  Past taker imbalance adds ~nothing once the past residual is in (vol t≈1), so takers are not informed about the drift
  beyond the price itself; they lean with it (sign reg in vol: past 0.5 s mid move std-coef +0.10, prev signs +0.10/+0.07).
  Per vol segment the residual looks like runs of 5–20 s trends then plateau / partial give-back (|net| mean 31 ticks per
  segment vs r3 5.5). Fill impact kernel itself ≈ r3 (G(0.5/2/10/30 s) vol .047/.065/.062/.055, calm .011/.017/.013/.013).
  @s_dynamics: suggests a hidden drift / velocity state in S, much stronger in volatile. Leak for trading: past 1 s mid
  move → next 5 s mid move corr **+0.37 in volatile** (r3 +0.04), +0.06 calm; reversal at 20–60 s (corr ≈ −0.25).
- [novelty 08:10] **Momentum confirmed on exact S v0** (`novelty/sdrift.py`, `velacf.py`, `nofill.py`). After the fill
  regression (0.5 s bins, 60 s kernels; kernel ≈ r3, G vol .058→.064 flat to 5 s → .052 at 30 s, no ramp-up), the
  residual increments have ACF (lags 0.5/1/1.5/2/3/4/5/7/10 s): **vol .56/.48/.40/.30/.20/.15/.10/.06/.04** (≈ exp, τ ≈ 2.5–3 s);
  **calm .34/.24/.28/.20/.21/.06/−.01/−.02/−.04** (flat ~0.25 to 3 s then 0 by 4–5 s — box-like, ~4 s). r3 final S: all
  |ACF| < 0.09 and negative at lag 0.5 s. Residual VR at 40 s: vol 6.7, calm 3.0 (r3 0.39 / 0.50).
  S also moves **in 0.5 s windows with no fills**: mean |ΔS| calm 0.10 (r3 0.017), vol 1.11 (r3 0.16), and the move
  continues the past 2 s move (corr +0.12 calm, +0.60 vol; r3 +0.03 / −0.27). So r4 adds a trade-independent drift /
  velocity term to S, ~5–10× stronger in volatile. r3 open item #7 rechecked on r3 final S: raw 1 s→1 s beta calm +0.03,
  vol +0.10 (so r3 had at most a weak trace). @s_dynamics @flow
- [novelty 08:10] Ladder (cheap check, @makers): burst rate unchanged (calm 0.98 /s, vol 10.3 /s vs r3 1.23/10.8), rungs
  still touch+2..+7, but **sizes up ~30–40 %: median B calm 187 (r3 145), vol 46 (r3 33)**, vol logB sd 0.30. Chains of
  length 7–10 now exist (78 vs r3 51 len≥7). Non-chain maker sizes 200–300 up, 110–160 down vs r3 (single-rung refills).
- [makers 08:10] **15-lot agent = tell of an S drift in its direction (calm only).** Rebuilt chains (`makers/r15_chains2.py`
  → `makers/cache/r15_v2.pkl`, 755 orders, 43 instances; after a partial fill + cancel it re-posts the 10-lot remainder,
  so link on "1 µs after own end, same side, vol = 15 or remainder"). Rules: always joins the same-side touch (754/755);
  cancels only after that touch improves past it (350/352), latency mean 23 ms / median 17 ms (= base cancel delay);
  re-post 1 µs after its fill or cancel row; instance ends after a full fill with no re-post (39/43) or a cancel (4).
  Side-signed S (v0) around instance start: −0.2 at −1 s, 0 at 0, **+0.35 / +0.69 / +1.64 / +3.4 / +5.3 at 0.5 / 1 / 2 / 4 / 6 s**;
  during the instance +6.9 ticks (t = 6.7), median rate 0.74 tick/s (p10–p90 0.26–1.95). Drift stops at the instance
  end (end..+2 s: 0.0; +10 s: −2.5, partial give-back). S moves AGAINST the agent at its own fills (taker kick), so the
  drift is continuous, not fill impact. 43/43 starts are in calm (S v0 vol flag). Lag-1 ACF of 1 s calm ΔS inside agent
  windows 0.44 vs 0.09 outside. Agent windows are only 6.5 % of calm time, so they don't explain most calm momentum.
  @s_dynamics: a drift term switched on while an instance is active; window list = groupby chain (t0=min t_us,
  t1=max end_t_us, sg) in r15_v2.pkl. Taker/maker/ladder rates inside the windows are normal calm.
- [flow 08:20] **Taker side rule identified: takers buy when S is above the mid.** On S v0, non-TWAP, ex-shock takers:
  **P(buy) = Φ((S − mid)/4.42)** (≡ logistic(0.370·(S − mid))), ONE slope for calm and volatile (0.370 / 0.370; pooled
  dLL +1133 vs coin, calm +139, vol +996). Reference is the plain mid: S−bid / ask−S coefs equal and opposite, microprice
  and last trade add 0; S lagged by 0.05 s loses 30 % of the LL, S lagged 0.2 s ≈ nothing → current S. Probit ≈ logit;
  "π informed + coin" mixture collapses to π = 1. **Given S − mid the signs are independent** (residual sign ACF lag 1–4:
  calm −.004/−.002/.001/−.001, vol .002/.003/.006/.008), and past signs / past S moves add only 13–20 nats. So there is
  no hidden direction state, no meta-orders (besides the TWAP), no chasing: the ACF comes from the feedback loop
  kick → S moves away from a lagging book → more takers on that side. It is stronger in vol only because S − mid is
  wider there (sd 1.90 vs 0.51). Also explains "unfilled taker ⇒ next taker on the other side" (vol corr −0.54).
  Shock jumps don't get chased past +0.2 s (jump-side share 0.51–0.52 at 0.2–5 s), consistent with the book catching up.
  Limits unchanged: calm round(S+1) 94.6 % in band, tail 5.3 % (excess 3.2); vol round(S+2.55) 87.2 %, tail 12.4 % (6.5).
  TWAP children use the same limit rule but with no tail (99.6 % / 99.2 % in band). Scripts flow/sidelaw.py, sidecheck.py.
- [novelty 08:25] **The S drift runs on clock time, not on fills** (`novelty/xiramp.py`): var of the post-regression residual
  per 0.5 s bin by #fills in the bin 0/1/2/3: **r4 vol 1.75/2.79/2.80/1.81** (r3 vol 0.07/0.62/0.55/0.69); r4 calm
  0.046/0.057/0.125/0.189 (r3 0.012/0.034/0.066/0.078). corr(#fills_i, resid²_{i+l}) ≈ 0 for all l ≠ 0 in both rounds, so
  it is not ξ spread over a ramp. In volatile, S carries a trade-independent velocity of ~1.3 ticks per 0.5 s, persistence
  τ ≈ 2.5–3 s; in calm a much smaller one (box-like ACF ~4 s). Taker limits still reference *current* S (band fraction
  peaks at δ = 0 for limit − S(t+δ), symmetric, r3 and r4 alike), so takers have no look-ahead.
- [novelty 08:25] **TWAP is uninformed, predictable flow** (`novelty/twap3.py`, `twap4.py`, 37 episodes): length n is
  uniform on 22..58 children (KS p 0.74; dur = (n−1)·0.5 s exactly; qty 3,520–9,280), so after 2 children ≥ 20 more are
  certain. Signed S move during = +14.4 ± 5.2 ticks (≈ 0.37/child = pure calm impact of 160 lots), nothing before
  (pre-5 s +0.7 ± 2.5, pre-30 s +1.5 ± 7.3; no trigger vs shocks), and after the last child −2.5/−6.8/−8.4/−4.3/+0.7 at
  +2/5/10/20/30 s. Signed S move over next 5 s at child idx 2–5: +3.3, 10–20: +2.2, 30+: −0.8. Limits S+0.4…S+2.5.
- [hypotheses 2026-11-08] Data-explorer r3-vs-r4 diff + ranked trading hypotheses in `hypothses.md`. Headline: 1 s return
  ACF sum lags 1–5 = 0.64 (r3 0.09), lags 6–30 = −0.52; impact peaks at 5 s (2.56) vs 2 s (0.93); taker PnL +1.77/lot at 5 s;
  every maker-markout bucket worse, small takers now toxic. Open questions for s_dynamics/flow in section 5.
- [makers 08:40] **Per-insert class labels:** `makers/cache/order_class.parquet` (1,228,523 rows, same order as
  s_path.parquet; cls 0 base 1,103,634 / 1 ladder chain 65,689 / 2 r15 agent 755 / 3 regular IOC 56,955 / 4 TWAP child
  1,490; plus r15_chain, tw_chain, lad_chain ids). Note s_dynamics' react.npy (395) misses the agent's ~350 cancel
  re-posts and its 43 fresh starts — use cls == 2 instead. Episodes (instances merged, same side, gap < 3 s):
  `makers/r15_episodes.csv`, 31 episodes (12 buy / 19 sell), dur median 8.2 s (2.7–19.3), start gaps ≈ Exp(mean 195 s).
- [makers 08:40] **Cancel→insert 1 µs pairs explained:** 690 = 348 agent re-posts + 342 other; background at 2–10 µs is
  ≈ 230 per µs in both rounds (r3 1 µs: 360), so the non-agent excess (~110) is the same as r3's (~130). The 382
  taker→insert 1 µs pairs = 364 agent + ~18 background (r3 13). There are no agent 15-lot orders outside the chains:
  15-lot base count = (14-lot + 16-lot)/2 to within 6 orders after removing the 749 agent 15s.
- [makers 09:00] **Ladder: 3 agents instead of 4, rungs ×4/3.** Live ladder orders at one price level (incl. the new
  one) never exceed 3 (calm 1981/1183/558/3 for 1/2/3/4; vol 11419/9307/5040/15), r3's cap was 4. Burst size B:
  calm LN(5.33, 0.33) on L ≥ 3 bursts (median ≈ 210; r3 LN(5.02, 0.34)), vol **LN(3.82, 0.27)** (median 46; r3 3.52,
  0.31): both up by e^0.30–0.31 ≈ 1.35 ≈ 4/3, so total ladder depth per level is unchanged. Rung noise still
  B·U(0.75, 1.25) (L=6 max/min 50/90/99 % 1.46/1.59/1.65 = sim). Band still touch−2..−7 (vol 99.9 %), start-level
  shares ≈ r3 (−2: 62 % vol). Calm L=2 chains are contaminated by chance 2000 µs base pairs; use L ≥ 3 for calm stats.
- [makers 09:00] **Base makers unchanged** (S v0, base = cls 0): offset calm 0.069 + Exp(0.444) (r3 0.066 + Exp(0.450)),
  vol mean D 1.62 (0.20 + Exp(1.42); r3 mean 1.62), drought 0.58 + Exp(2.04), post 0.08 + Exp(0.72); share at the clip
  calm 0.50 / vol 0.21. Lifetimes calm mean 0.461 s (r3 0.457), vol 0.380 (0.379), drought 0.208 (0.213); filled 14.7 /
  15.5 %. Cancel MLE running.
- [novelty 09:00] **Final README: `novelty/README.md`** (ranked r4-vs-r3 diff). Cross-checks: flow's side law holds with
  causal s_filt at the row before each taker (slope calm .368 / vol .403, dLL 178 / 1081; r3 −.019 / +.004, dLL < 1).
  Makers' 15-lot drift confirmed (+6.7 ticks signed per window), but calm residual ACF outside its windows is still .34,
  so it is a tell, not the calm momentum. Joint causal leak regression (`novelty/leakjoint.py`, future s_hat − s_filt):
  vol 2 s R² **0.24** (r3 0.03) from past 0.5 s / 0.5–2 s causal ΔS and S−mid; calm R² 0.045 with TWAP side (t 11) and
  r15 side (t 10) the strongest. Shock J sd 65 vs 49 (n 28, p≈.07). s_dynamics vol starts on integer seconds: 12/32
  within 0.05 s (3.2 expected; ends 2/31; r3 5/28) — integer-start segments differ in nothing else; maybe seg artefact.
- [flow 09:15] Side-law follow-ups (S v0, non-TWAP, ex-shock):
  * **Total taker rate does not depend on S − mid**; only the split does (regime interiors, 0.05 s grid, TWAP/shock windows
    out): vol |x| 0–0.25 → >5: aligned 6.1 → 10.4 /s, against 5.9 → 1.1 /s, total 12.0 → 11.5. So it is a side
    probability, not an extra "arbitrage" taker class (that would keep the against side flat). Calibration of
    Φ(x/4.42) holds from x = −5 to +8 in 0.5-tick bins. Takers use full S = F + Y (adding a Y proxy: coef 0.000).
  * **Aggressive-tail takers lean harder**: probit σ 4.9–5.1 for regular limits vs 1.8–2.35 for tail limits; P(tail)
    rises with alignment xa = sg·(S − mid): logit slope 0.48/tick calm, 0.21 vol (calm 0.04 → 0.19, vol 0.06 → 0.23);
    tail excess size is independent of xa (calm mean 3.2, vol 6.5, as r3). A "noisy valuation" story (V = S + N(0, 4.9),
    side = sign(V − mid), tail iff |V − mid| > 9.5 calm / 8.1 vol) gets +173 of the +215 nats. Size does not depend on xa.
  * **Jump head is NOT the side law**: 0–0.2 s after t_jump 171/185 takers are jump-side, the law predicts 0.55 because
    the book has already re-quoted (mean |S − mid| 0.5–0.9 at 0.05–0.2 s). ≈ 4.5 extra jump-side takers per shock,
    first one median 82 ms after the jump (r3 77). From +0.2 s observed = predicted (0.505 vs 0.515).
  * Unchanged: rates 3.94/4.03 per side calm, 5.93/6.00 vol (non-TWAP; flat per 1000 s block); size LN(4.094, 1.400).
  * `flow/shock_events.csv` now has J from S v0: sd 64.8 (r3 47.2; var ratio 1.88), normal KS p 0.84, mean|J| 51;
    t_jump = drought end (−0.004 ± 0.057 s); gaps 148.6–296.9, U(150, 300) KS p 0.07, U(145, 300) p 0.12; burst extra
    70 ± 15 takers, ≈ 62 + 0.17·|J| (corr 0.44, new vs r3 0.02; low n). No pre-onset tell (|corr| ≤ 0.31).
  * `flow/regimes.csv` (from s_dynamics segments): calm dwell mean 135 s (exp KS p 0.51), vol mean 56 s but **not
    exponential** (KS p 0.02, CV 0.60; gamma k 2.7 fits p 0.97) — low n (31). `flow/ladder_events.csv` rebuilt (21.8k).
- [flow 09:35] **Final: `flow/README.md`.** Model: round-3 arrivals unchanged (rates 4.0/6.0 per side, LN(4.094, 1.400),
  limits round(S±1)/round(S±2.55) + tail, shocks U(~148,300) gaps, 3.02 s drought, jump at drought end) plus ONE new rule:
  regular taker side ~ Bernoulli(Φ((S − mid)/4.42)), regime-free, total rate unaffected; aggressive-tail takers lean
  harder (σ ≈ 2). Real-time: causal S − mid → next 1 s mid R² 0.40 vol / 0.20 calm; tape-only (0.25 s imbalance,
  last unfilled IOC, past mid) R² 0.33 / 0.08. Open: closed-loop sim of side law + impact + S velocity.
- [makers 07:45] **Final: `makers/README.md`.** Base cancel MLE (60k, S v0): α calm 1.78 / vol 1.85 / drought 4.85,
  β calm 2.73 / vol 2.98, delay b 61 /s; the model-free hazard by mv matches r3 bin for bin, so I call base cancels
  unchanged. 15-lot agent tradeability: the first exact-1 µs re-post comes a median 0.57 s (max 1.4 s) after episode
  start; from there the side-signed mid moves +2.0 at 2 s, **+5.2 at 5 s (t 5.8), +8.4 to the episode end (t 7.7)**,
  worst −1. The agent's cancel delay is Exp(358) + Exp(49), mean 23 ms (= base delay). Weak: the episode side agrees
  with the past 90–200 s S move in 65–84 % of 31 episodes, and goes against the past 5 s in 24/31.
- [s_dynamics 09:45] **S has a velocity state (OU drift), confirmed by likelihood.** Kalman filter on the smoothed path
  (state F + Y + drift D, dF = D dt, D OU; `s_dynamics/kfd.py`): adding D improves the NLL by **9,800 nats** over
  impact-only (and beats plain diffusion by ~3k). Fit on S v1a: **vol D: τ ≈ 2.1 s, stationary sd ≈ 3.0 tick/s**;
  calm D: τ ≈ 0.3–1 s, sd 0.2–0.4 tick/s (path-prior sensitive, refining). With D in, the impact law returns to round 3:
  c calm 0.0153 / vol 0.050, e 0.62 / 0.64, **p 0.54 / 0.60, T 14 / 16 s** (without D the vol kernel looks ~permanent
  p 0.75, T 29 s — that's the drift leaking into it). Full HMM with a plug-in velocity: ll −1,075,302 vs −1,098,290
  (S v0) — so S v1 (coming) carries the drift. Residual diffusion after the plug-in is small (calm 0.15, vol 0.25 /√s).
  Agree with flow: the sign persistence is the side law; the super-diffusion = exogenous velocity + feedback impact
  (takers chase S − mid, their kicks push S further). Decomposition of the 300 s variance next. Now testing velocity law
  (OU vs piecewise-constant resets vs on/off episodes) with a grid HMM on v.
- [s_dynamics 09:45] Integer-second vol starts re-checked with exact change points (support = llr within 3 nats):
  6/32 starts have an integer second inside a ~40 ms support (1.3 expected), ends 2/31 (1.6). Weak; leave as curiosity.
- [s_dynamics 10:25] **S v1 published (same file `s_dynamics/s_path.parquet`)**: HMM with a plug-in S velocity (Kalman-
  smoothed drift) + r3-type decaying kicks (calm c .0148 e .62 p .56 T 13.9 s; vol c .048 e .65 p .61 T 17.4 s).
  Total ll −1,072,705 (S v0 −1,098,290). Posterior sd median calm 0.056 / vol 0.10. Same regimes (vol_segments.csv).
  Velocity on the 1 s grid: vol residual ACF 0.73/0.53/0.38/0.19/0.10/0.02 at 1/2/3/5/7/10 s (≈ exp, τ ≈ 3 s), calm
  0.20/0.08/0.08/0.00 at 1/2/3/5 s. Clock test: velocity changes are NOT tied to integer seconds. Net residual drift in
  vol over the whole tape −641 ticks (−0.37 tick/s, ≈ −2.5σ; 21/31 vol episodes negative), calm +14: possible negative
  mean drift in volatile, weak. Final S (refit observation laws) later today.
- [strategy 2026-11-08] Step 0 on r4 (`round4/strategy/eval4.py` = eval3 with r4 regimes/shocks): r3_hmm **1,371.5k**
  (calm 279.2k / vol 775.4k / shock 316.8k), r3_push 1,363.1k, r3_dev_best 1,365.9k: r4 pays MORE than r3 (857k),
  vol ×1.8. Our F vs s_path v0: rmse calm 0.16 / vol 0.53, same inside and outside TWAP windows (the filter only uses
  fill qty for takers, and TWAP impact is real), 28/28 droughts detected, no false alarm. Carry pos·dS in vol = 637k of
  775k, pinned 91 % on the trend side, so the momentum is already harvested. Every F lean (velocity F−EMA, S−mid, TWAP
  side; calm or vol; with or without veto) loses 15–260k; vol exits by age flat; TWAP quote pull −6..−13k calm;
  side-law likelihood, 15-lot filter neutral. **Only win so far: vol filter diffusion lamj_v 0.23 → 2.0 (≈ S velocity
  variance): vol +16k, plateau 1.5–4.** Shock column is a lottery (shock #24 swings ±22k on unrelated knobs).
- [s_dynamics 11:10] **Final: `s_dynamics/README.md`; final S in `s_dynamics/s_path.parquet` (run v3, ll −1,071,071),
  `vol_segments.csv` (32 vol episodes, stable to ≤0.38 s on the final path), `model_params.json`.** Model = r3 decaying
  kicks (calm c .0145 e .62 p .56 T 14 s; vol c .048 e .646 p .61 T 17–18 s; ξ ≈ .10 / .22) + **exogenous S velocity V**
  (vol sd ≈ 3.1 tick/s, persistence ≈ 3 s, Gaussian; calm sd ≈ 0.4, weak, hard to split from ξ; fills do not feed V:
  κ ≈ 0, ΔNLL 4.6). 300 s variance: drift 49 %, impact 18 %, drift×impact feedback (side law) 33 %. V leads the regime
  switch of quotes/kicks by ≈ 0.75 s at both ends (V rises before vol onset, dies before vol end) — tell for vol onsets.
  Vol mean drift −0.37 tick/s (t ≈ −2.4, low conf). Shocks: J ~ N(0, 63), at onset + 3.02 s, no reversion, no no-jump
  class. Causal KF velocity predicts vol S moves: R² 0.30 at 1 s, 0.33 at 2 s (β ≈ 1.6 / 2.9 × V), calm ≈ 0.03.
  Integer-second vol starts: artefact (4/32 on final path vs 1.5 expected). Open: OU vs reset law, calm V vs 15-lot
  trend episodes, closed-loop sim.
- [strategy 2026-11-08] **Final: `strategy/README.md`, `strategies/marek/round4/r4_best.py` = r3_hmm + Kalman velocity
  state beside the S grid (OU τ 0.5/2.1 s, sd 0.3/3.0 tick/s, from s_dynamics' kfd fit) + vol impact c_v 0.05 / e_v 0.64.
  r4 1,391.0k (r3_hmm 1,371.5k; calm +1.2k, vol +23.5k, shock −5.3k lottery; calm+vol up in 9/12 windows); r3 848.8k
  (−8.8k, mostly c_v). Dead on r4: every F lean (velocity, S−mid, TWAP), vol exits, TWAP quote pull, side-law
  likelihood, 15-lot filter, post-shock quotes, retunes. All knobs in `strategies/marek/round4/r4_dev.py` (off = r3_hmm).
