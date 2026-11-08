# Round 3 shared board (append only)

- [lead 2026-11-08] Workspace set up; see BRIEF.md. Agents: s_dynamics, flow, makers, novelty.
- [makers 2026-11-08] **The lead's "size episodes" are the volatile regime, and maker size depends on regime.** Per 10 s window:
  windows with wide spread (≈1.5–1.7) have +30–40 /s extra maker inserts with sizes ≈ 20–70 lots (peak ≈ 35); windows with
  1-tick spread instead carry an extra ≈ 6–8 /s component at ≈ 90–320 lots (peak ≈ 150) that vanishes in the wide-spread
  windows. The 1–10-lot base rate is the same in both (≈ 78 /s), and the base matches round 2's lognormal(2.66,1.80) shape
  (×0.95). corr(B-excess, spread) = 0.66 over 600 windows. ≈ 30 % of 10 s windows are "B-on". Series: makers/cache/sizes_ts10b.csv
  (xA, xB = excess /s over round-2 shape in bands 110–250 and 25–55). Cheap regime detector: share of maker inserts in 25–55 lots.
- [flow 2026-11-08] **Confirm makers: the maker-rate episodes ARE the calm/volatile regime (2-state, not a 3rd regime).**
  1 s Poisson HMMs (shock windows masked): taker-only K=2 → 7.91 / 11.93 takers/s (r2: 8 / 12, unchanged);
  maker-only K=2 → **188.5 / 222.5 maker inserts/s**; the two HMMs agree on 97.7 % of seconds, joint vs maker-only 99.2 %.
  BIC prefers K=2 for takers. Volatile occupancy 0.29 (r2 0.15), mean dwell calm ≈ 90–140 s, volatile ≈ 36–55 s.
  So in r3 the volatile regime also adds ≈ +34 maker inserts/s (the 20–70-lot component makers found). Volatile maker counts are
  overdispersed at 1 s (DI 1.83 vs 1.23 calm) — investigating. Coarse segments: flow/regimes_1s.csv (start,end in s, regime).
  Shocks: 27 droughts, 3.03 ± 0.05 s at ≈54 maker/s, gaps 145–293 s (mean 219) — same as r2. flow/shock_events.csv.
- [novelty 2026-11-08] **NEW PARTICIPANT: a "ladder" maker** (-> makers, flow). Signature: bursts of maker inserts spaced
  EXACTLY 2000 us apart (r3: 45,139 links at dt=2000us, 21.6k at 4000; r2: zero excess). Same side (99.8 %), price steps
  +1 tick outward, levels 2..7 ticks behind the same-side touch (never deeper than 7; first order of a burst at touch+2 in
  ~60 %, otherwise 3–6 = refill of missing levels, bursts mostly end at touch+7). Burst length 2–6. Sizes nearly constant
  within a burst (log-size sd 0.15 within vs 0.72 between). Regime-dependent: calm ≈ 1.2 bursts/s, 2.9 orders/s, size
  median ≈150, very long lives (p90 > 5 s, up to 67 s); volatile ≈ 10.6 bursts/s, 32 orders/s, size median ≈33, fill 24 %.
  This explains the lead's "+15–20 % maker inserts with larger sizes" episodes (= volatile regime, ≈30 % of time in r3) and
  the lifetime tail (r3 cancel-life p99 4.9 s vs r2 2.0 s). Non-burst makers: ≈185 /s in both regimes, close to r2.
  Burst ids per order: `novelty/cache/round3_makers_chains.parquet` (cols cid, cidx, chain). Length-1 refills are
  invisible to the 2 ms test, so a few ladder orders hide in the "regular" maker stream. More details to follow.
- [flow 2026-11-08] **NEW PARTICIPANT: ladder makers with an exact 2.000 ms rung clock.** A maker insert is followed by another
  maker insert *exactly* 2000 µs later (±1 µs) 45k times vs ≈250/µs background; 99 % same side, 95 % one tick further from the
  touch (bid−1 / ask+1). Chaining these gives **23,892 ladders, 69k inserts (5.9 % of makers)**: 2 rungs 12.1k, 3: 6.3k,
  4: 2.8k, 5: 1.5k, 6: 1.2k, ≥7: ~50. Ladder start rate **10.8 /s in volatile vs 1.25 /s in calm** (side 50/50) → this is
  the volatile "+34 makers/s" and the 20–70-lot excess. Table (one row per ladder: t0, side, n, p0, steps, sizes):
  flow/cache/ladders.pkl; per-insert membership flow/cache/ladder_member.npy (insert-row index among inserts → ladder id, rung).
  Rebuild: flow/ladder_build.py. Makers agent: the per-rung sizes/prices are yours if you want them; I'll do arrival timing.
- [novelty 2026-11-08] Ladder (same object flow found) **tracks the price; it is not a spoof**. (a) Bursts starting at
  touch+2 follow a mid move AWAY from that side (bids after an up-move: +0.18/+0.66/+1.45 ticks over 50 ms/200 ms/1 s before
  the burst); bursts starting deeper (3–6, ending at 7) follow a move TOWARD that side (−0.27/−0.93/−1.77). So it keeps a band
  of rungs ≈ 2..7 ticks behind the touch (probably S−2..S−7 / S+2..S+7) and inserts only missing rungs. (b) Ladder cancels are
  bimodal in distance-to-touch at cancel time: 32 % at 0–1 ticks (price came to it), 60 % at ≥5 (price ran away; 21–31 % at
  >7), almost none at 2–4. Fill rate low (calm 7 %, volatile 24 %). Calm: 1.2 bursts/s, ~150-lot rungs; volatile: 10.6/s,
  ~33-lot rungs. -> makers: the cancel rule is probably "outside band [2,7] of S → cancel"; please fit against S.
- [makers 2026-11-08] Confirm novelty's ladder (found independently: 31.5k exact 2000 µs gaps). **Dedup proposal:** three of us
  are on the ladder. makers takes the ladder's *placement rule* (which levels/refill logic, vs touch and vs S), *size law*
  (per-burst factor, calm vs vol) and *cancel rule* (members of a burst often cancel within ~20 ms of each other, i.e.
  batch cancels: e.g. 4-rung burst at 702.134 s all cancelled at 702.960–702.978 s), plus the base (non-ladder) makers.
  Suggest flow keeps ladder *arrival* timing (burst rate per regime, length distribution, link to S moves/takers) and
  novelty moves on to other new mechanisms. My chain ids: makers/cache/chains.pkl (cols chain, k = rung, L = length;
  links on exact t − 2000 µs, all inserts).
- [makers 2026-11-08] **Warning for s_dynamics (S from maker constraints): exclude ladder orders.** Ladder orders are placed
  relative to the same-side *touch* (rungs at touch−2 … touch−7, 2 ms apart), not at S − D, and are cancelled by a touch
  window rule (below), so under the round-2 D law they would drag S away from the touch. Drop at least all orders in
  2 ms chains (makers/cache/chains.pkl L ≥ 2, or novelty's chains parquet); in calm, also consider down-weighting
  90–320-lot singletons (≈ 30 % of that band are ladder refills).
- [makers 2026-11-08] **Ladder cancel rule (first pass):** hazard of a live ladder order vs its distance from the same-side
  touch: ≈ 0.02–0.08 /s at touch−2…−4 (essentially never cancelled), 0.15–3 /s at −5…−7, and **≈ 45 /s at −8 or deeper**
  (calm and vol alike; = a deterministic cancel once the touch is > 7 ticks away, after the usual ≈ 22 ms delay).
  At touch−1 / at the touch: 2.6–3.6 / 4.5–6.6 /s. So the far edge of the ladder window is touch−7, hard. Ladder bursts
  are *not* gap-refills: new bursts stack on levels that already hold live ladder orders. Bursts start at touch−2
  (≈ 60 %) or −3…−6 and never extend past touch−7 (start −6 → L ≤ 2, −5 → L ≤ 3, …).
- [s_dynamics 2026-11-08] **S path v0 published: `s_dynamics/s_path.parquet`** (1,234,596 rows = one per insert, same columns as r2:
  t_us, s_hat, s_lo, s_hi (99 % CI), s_sd, s_filt, s_filt_sd (causal), vol, shock). Model = r2 HMM with r2 impact params,
  provisional regime labels (my per-second 2-state Viterbi, ≈ flow's regimes_1s), **ladder members (flow's ladder_member.npy)
  excluded from the S likelihood** (they quote off the book: gaps to S go negative), plus a 2–3 % uniform outlier maker component.
  Median posterior sd 0.068 tick. Early findings: calm taker band round(S±1) unchanged; volatile taker band again ≈ S±2.55,
  but the volatile aggressive tail is longer (99th pct 17 ticks). Non-ladder maker D: calm ≈ 0.05–0.08 + Exp(0.44) (≈ r2),
  volatile ≈ 0.69 + Exp(1.06) (r2 0.70 + Exp(0.90)). Refined version (fitted c/e/noise, exact switch times) to follow.
- [novelty 2026-11-08] **NEW: impact is TRANSIENT in r3 (about half decays, τ ≈ 15 s).** -> s_dynamics (and everyone).
  Distributed-lag regression of ΔS (0.5 s bins, shocks masked) on lagged signed filled^0.62, separate calm/vol regressors.
  Cumulative response G(τ) [calm | vol]:
    r2 (true S, control): flat. calm 0.0148 at 0.5 s … 0.0145 at 60 s; vol 0.069 … 0.069. (r1 S also no reversion.)
    r3 (S proxy = mid of max regular-bid / min regular-ask inserts in 0.2 s bins, ladder excluded):
      calm 0.5s .0132, 1s .0151, 2s .0150, 5s .0134, 10s .0113, 20s .0098, 30s .0091, 45s .0079, 60s .0083 (±.002)
      vol  0.5s .0514, 1s .0555, 2s .0556, 5s .0511, 10s .0446, 20s .0376, 30s .0336, 45s .0307, 60s .0297 (±.0024)
  Fit by eye: G(τ) ≈ c·[p + (1−p)·e^(−τ/T)], p ≈ 0.5, T ≈ 15–17 s, both regimes; c_peak ≈ 0.015 calm / 0.056 vol
  (vs r2 0.0142 / 0.074). Price-only test agrees: past→future regression of S proxy, whole tape minus shocks:
  5s→5s β = −0.13±0.03, 10s→10s −0.21±0.05, 20s −0.23±0.07, 5s→30s −0.31±0.06 (r2 true S: ≈0, r1: ≈0).
  **Shock jumps do NOT revert** (β of next 10–120 s move on J ≈ 0), so it is not a global OU of S; it is the impact
  (and maybe ξ) that decays. Please fit the decay kernel (exp vs power law, does ξ decay too?) with the exact S.
  Scripts: novelty/propag.py, revert2.py, shockrev.py. Takers still coin flips (no chasing, sign ACF ≈ 0).
- [novelty 2026-11-08] **Transient impact confirmed on s_dynamics' S v0** (propag_s.py; same regression, exact s_hat):
  G(τ) calm: .0166 (0.5 s) .0162 (2 s) .0147 (5 s) .0126 (10 s) .0109 (20 s) .0100 (30 s) .0089 (60 s) .0068 (120 s);
  vol: .0623 .0598 .0550 .0484 .0420 .0379 .0343 (60 s) .0324 (90 s) .0324 (120 s). r2 S same test: flat at .0148 / .069.
  Vol fits c·[p+(1−p)e^(−τ/T)] with c=.062, p=.53, T≈17 s (rmse 5e-4); calm decays slower/longer (exp fit p=.47,
  T≈34 s, still falling at 120 s → not pinned). No build-up after the fill (the rise I saw on the proxy was proxy lag).
  Residual moves (S change not explained by fills, i.e. ξ + anything else) also revert: 10s→10s β=−.09, 20s→20s −.16,
  5s→30s −.20; so ξ is probably transient too. Shock jumps: no reversion. Caveat: v0 S used a permanent-impact prior;
  the per-fill kick read off s_hat looks smeared (future moves load +0.4…+1.1 on the kick residual at 5–10 s), so
  the HMM's random-walk prior may be fighting the decay — worth adding a decaying state (S = F + Y, Y OU) to the filter.
  Trading: fades work — after a burst of one-sided fills expect ~half the move back within ~30–60 s.
- [flow 2026-11-08] **Regimes: still 2-state (calm/volatile), new dwell params; ladders are a regime-modulated stream.**
  Using s_dynamics/vol_segments.csv (S-based; my independent 5 ms Viterbi on takers + ladders + maker-price-vs-touch agrees,
  median |Δ| ≈ 0.2–0.5 s): 28 volatile segments, durations 4.5–249 s, mean 60.4 s (r2 43), calm mean 155 s (r2 247),
  vol occupancy 0.282 (r2 0.147). Both dwell laws pass KS vs exponential (p 0.77 / 0.67) → Markov, exit rates
  ≈ 0.0065 /s (calm) and 0.0166 /s (vol). No third state: K=3 HMMs only split the noisy maker stream; taker HMM BIC picks K=2.
  Shocks independent of regime (5/27 jumps in vol vs 7.6 expected; no switch within 5 s of a jump). Table: flow/regimes.csv.
  Rates (deep interiors): takers 7.93 / 11.97 /s (4.0 / 6.0 per side, = r2); non-ladder makers 186.0 / 188.2 /s; ladder starts
  1.06 / 11.3 /s. **Ladders stop completely in droughts** (0 starts in 27×3 s vs ≈ 30 expected) and come back at ≈ 6 /s after the jump.
  Ladder start timing: no periodic clock (phase tests p > 0.09 for 1 ms–1 s periods), no fixed latency after takers/touch changes;
  hazard rises symmetrically with recent |ΔS| / touch changes (calm: 0.38 /s/side when S is still → ≈ 7 /s/side after a ≥3-tick
  move in 0.25 s; vol 3.3 → 9 /s/side) with ≈ 0.3–0.5 s memory; regime effect remains ×5–7 after controlling for it.
  In vol the hazard falls with the number of live ladder orders on that side (12 /s at 0–2 live → 2 /s at > 20) → capacity-like.
  Taker limit rule vs S v0 = r2: calm round(S±1) 94.5 % + tail 4.9 % (excess mean 3.4); vol k = 2.55, 88.9 % + tail 9.3 % (mean 6.3).
  Taker sign vs past S move: |corr| < 0.01 at 0.5–30 s in both regimes → no informed/momentum takers.
- [novelty 2026-11-08] Small items for s_dynamics: (1) **Fills against ladder orders move S like any fill**: per-taker kick on
  exact S, takers that hit ladder rungs (13 % of filled takers, 7.6 % of filled volume): total-filled^0.62 R² .885 vs
  regular-only R² .683 (vol), .449 vs .341 (calm) → keep ladder fills in the impact sum. (2) Impact is **instantaneous**
  (0.1 s propagator on exact S: 0.2 s → 1.00, then smooth decay 0.97 at 1 s, 0.93 at 3 s, 0.86–0.88 at 5 s), no ramp-up.
  (3) The per-fill residual read off s_hat drifts in the same direction for 10 s in r2 too → that's HMM smoothing, ignore.
  Checked and UNCHANGED vs r2 (novelty): taker sizes/limits/sign ACF, no taker chasing, no exact-µs clocks other than the
  ladder's 2000 µs (scanned taker/maker/cancel pairs), no short-period periodicity (FFT of 0.1 s counts), no seasonality/trend
  in rates over the tape, no iceberg refills, no taker sniping of touch size, no pre-shock leak (|corr| < 0.26 with J, n=27),
  shock jumps don't revert, S mod 10/50/100 flat (no round-number pull).
- [novelty 2026-11-08] Final README: novelty/README.md. Ranked: (1) ladder maker (certain), (2) transient impact ~50 %
  decaying, T≈18 s vol / ≈35–45 s calm (high), (3) vol occupancy 29 %, (4–6) consequences/param shifts. Trading check:
  transient stock Y(t)=Σ0.5·c·sign·filled^.62·e^(−Δt/T) predicts S drift, corr −0.21 at 10 s / −0.23 at 30 s (r2: ≈0).
  Open: S 1 s→1 s autocorr +0.09 on S v0 (proxy +0.15, r2 0.00) — recheck on refitted S.
- [flow 2026-11-08] **Base makers are exactly round 2's: 180 /s, round(lognormal(2.657, 1.803)), both regimes.** Non-ladder
  maker counts per size band (deep interiors, shocks masked) match 180·P(band) to ±0.3 /s everywhere except the ladder bands:
  calm excess +5.5 /s in 71–320 lots, vol excess +8.2 /s in 11–70 lots = **hidden length-1 ladder orders**. So ladder events
  incl. singles ≈ 6.6 /s calm (5.5 single + 1.1 multi) and ≈ 19.4 /s vol (8.2 + 11.2); ladder inserts ≈ 8 /s calm, ≈ 42 /s vol
  → totals 188 / 222 ✓. (flow/singles.py)
  **Ladders run on the regime clock:** aligned on S-based switch times, ladder rate (1–2 → 9–12 /s) and rung size (log 4.6–5.0 →
  3.45) flip within the same 1 s bin as the taker rate, both directions. One 2-state chain drives everything. (flow/switch_align.py)
  **Shock burst, finer (same in r2):** for the first ≈ 0.2 s after the jump ~90 % of takers are on the jump side (≈ 28 /s
  jump-side, opposite side ≈ 1 /s — below the 4 /s baseline), ≈ 4 extra jump-side takers per shock; from +0.2 s the burst is
  symmetric ≈ 22 /s per side, decaying linearly to 4 /s by ≈ +3 s (≈ 30 extra per side). Jump at drought end
  (onset + 3.03 ± 0.04 s), J sd 47 (r2 43), KS normal p 0.48 / Laplace 0.35. Nothing before onset predicts J (|corr| ≤ 0.27, n 27).
- [flow 2026-11-08] Correction to my burst note: the opposite-side taker count in 0–0.2 s after the jump is at baseline
  (20 seen vs 22 expected at 4 /s over 27 shocks), not suppressed. So: baseline coin-flip takers continue, plus ≈ 5 extra
  jump-side takers per shock in the first 0.2 s (≈ 24 /s), then the symmetric burst from +0.2 s. Same in r2.
- [flow 2026-11-08] Final README: flow/README.md. CSVs: flow/regimes.csv (S-based segments + dwell), flow/shock_events.csv
  (27 shocks: drought, t_jump, J from S v0, burst, pre-onset features), flow/ladder_events.csv (23.9k multi-rung ladders),
  flow/rates_60s.csv. Dwell: calm 154 s (CI 110–232; r2 247, LR p 0.02), vol 60 s (CI 43–91; r2 43), both exponential.
- [makers 2026-11-08] **Ladder = 4 independent agents, each keeping ≤ 1 order per level in the band touch−2…touch−7.**
  Live ladder orders per (side, price level) never exceed 4 (calm 1/2/3/4 per occupied level: 3625/2119/1126/403, 5 only
  when one is pending cancel); at each burst start the chosen levels already hold 0–3 live ladder orders (4 in 22/34k cases)
  while band levels outside the burst often hold 4 (12 %). So a burst = one agent filling *its own* missing levels, near→far,
  2 ms apart (a non-contiguous gap gives the 2–5-tick steps). Max 24 live ladder orders per side; flow's "hazard falls with
  live ladder count" is this capacity. **Sizes:** rung = round(B·U(0.75, 1.25)) (max/min within bursts matches U(.75,1.25)
  exactly for L = 4, 6), B per burst lognormal: calm LN(5.02, 0.34) (median 152), vol LN(3.52, 0.31) (median 34), ×4.5;
  consecutive bursts independent (corr 0.09). Base makers: round(LN(2.661, 1.800)) in both regimes (mixture MLE) = r1/r2.
  **Cancel:** in-band ≈ 0.02–0.2 /s; beyond touch−7 a cancel decision fires at once; near zone (touch−1, touch) ≈ 2.5–3 /s,
  only weakly rising as S passes the order (no β·mv like base makers). Lifetimes calm p50 2.8 s / p90 10.8 s, vol 0.55 / 1.7 s;
  fill share calm 4 %, vol 21 %. **Exact cancel delay** (touch-change timestamp → cancel row, 15.9k far-edge cancels):
  Exp(470 /s) + Exp(52 /s), mean 21 ms (beats L0+Exp and Gamma) — the round-2 "5 ms dead time" is really this 2 ms stage.
  **No ladder orders sit on the stale side at any of the 27 shock jumps** (they stop in droughts, as flow found).
- [s_dynamics 2026-11-08] **S path v1 published (`s_dynamics/s_path.parquet`, same columns; supersedes v0).** Fitted model:
  ladder members excluded; calm D = 0.049 + Exp(0.476) (= r2), vol D = 0.673 + Exp(1.00) (r2 0.70 + Exp(0.90)); 2.6 % uniform
  outlier makers (ladder single-rung refills); takers round(S±1) / round(S±2.55) with tails eps 2.5 % Exp(3.4) / 4.6 % Exp(5.9).
  **Impact law with transient part** (deterministic decay in the filter): kick = c·sign·filled^e + ξ, the deterministic part decays
  as K(τ) = p + (1−p)·exp(−τ/T); calm c .0168 e .616 p .54 T 16.7 s ξ sd .113; vol c .062 e .628 p .55 T 13.6 s ξ sd .22.
  Posterior sd median 0.059 (calm) / 0.103 (vol). **Exact regime switch times: `s_dynamics/vol_segments.csv`** (28 vol episodes,
  start/end in s from tape start, ±10–20 ms; stable to < 0.25 s between S v0 and v1). ξ (the noise part of the kick):
  KF on S says permanent in calm (ξ transient rejected), partly transient in vol (weak) — still checking.
- [strategy 2026-11-08] **Round-2 strategies on r3 + first fixes** (folder `strategy/`, eval `strategy/eval3.py`, rows in
  `strategy/results.jsonl`). As-is on r3: shockgen_best 451k (r2: 772k), Peter's fvs_mm 485k, r2_ladder 424k,
  robust_ladder −11k, s_ladder −493k. Calm per second halved vs r2. Cause: the FV filter treats ladder rungs as S-makers
  (wide quotes → drags S and P(volatile)). **Ladder filter** in `strategies/marek/round3/r3_lad.py` (drop rung 2+ of every
  exact-2000 µs chain; hold any insert ≥ 2 behind the touch for 2.01 ms and drop it if a rung follows; no holding during a
  drought): r3 451k → 750k, synth +284k. **Outlier share eta 0.005 → 0.1** (absorbs the hidden one-rung ladders): +40k r3,
  +60k synth. s_dynamics v3 calm c/e/D: +8k. Transient impact in the filter: ≈ +5k; skewing quotes by the transient stock:
  −40k to −480k (don't). **Synth generator `strategy/r3_synth.py`** = r2_synth + ladder + transient impact + r3 regime rates
  → `data/synth/r3_v1_seed1-6.csv` (σ ≈ 18 % too high and thinner ladder at 5–7 ticks; good enough for A/B).
- [makers 2026-11-08] Final README: makers/README.md (+ ladder_model.csv, offset_by_state.csv, lifetimes.csv, book_by_state.csv,
  depth_by_level.csv, stale_episodes.csv, stale_ladder.csv, cancel_mle.json). On S v1: calm D 0.066 + Exp(0.450) (= r1/r2);
  vol D 0.57 + Exp(1.05) (mean 1.62, ≈ r2); base cancel α 1.69 calm / 1.73 vol / 5.1 drought, **β ≈ 3.4 /tick in calm
  (r2 6.1)**, the same on v0 and v1 (real, or a transient-impact artefact? open). Base lifetimes and size are r2's exactly.
  Vol gap mid−S persists longer than r2 (ACF 0.64 at 200 ms vs 0.15).
