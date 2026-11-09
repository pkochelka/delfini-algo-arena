# Round 4: the volatility-signature hypothesis

Written 2026-11-08. Tape `data/round4.csv` only, one regression run on `data/round3.csv`. Base `r4_best.py` (1,391.0k).

**Bottom line.** The hump is real but fully explained by the known generator. It rises because of the S velocity V plus
the side-law feedback, and it falls because the transient impact Y decays. Past moves add nothing out of sample once
V, Y and S − mid are known. The 10–60 s reversion cannot be traded: Y fades, age-based flips and wider takes all lose.
One knob gains: a small volatile lean on the filter's own Kalman velocity, F += 0.15·kv. That is a ~0.15 s nowcast past
the filter's lag, so it uses the short end of the signature, not the 10 s hump.
**`strategies/marek/round4/r4_volsig.py`: round 4 1,413.0k (+22.1k; calm +0.6k, volatile +22.7k, shock −1.1k),
round 3 845.4k (−3.4k).**

## Tools

| Path | What |
| --- | --- |
| `tools/grid.py` | 50 ms grid: mid, S (s_path), KF filtered/smoothed (F, Y, V) from `s_dynamics/_cache/kfd_v2b_D_state.npz`, regime, shock mask → `cache/grid.npz` |
| `tools/sig.py` | signature σ(h) by regime from overlapping increments inside pure regime runs; series mid, S, F_kf, Y_kf, S − ∫V, ∫V |
| `tools/pred.py`, `tools/pred2.py` | OOS (half-split) regressions of future S on V, Y, S − mid, past moves; pred2 uses r4_best's own causal state (log from `logf2.py`) |
| `tools/carry2.py` | forward carry of r4_best's position by position age and by move since the sign was set |
| `eval4.py`, `sweep.py`, `tbl.py`, `summ.py` | copies of `../strategy/` tools, writing to this folder's `results.jsonl`; configs in `cfg/` |

## 1. The signature by regime (σ in ticks/√s; shock windows −1..+20 s removed)

| h | mid all (incl. shocks) | mid calm | mid vol | S calm | S vol | S no-shock pooled | r3 S calm | r3 S vol |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 s | 4.72 | 1.04 | 4.06 | 1.07 | 4.84 | 2.79 | 1.16 | 5.13 |
| 1 s | 6.29 | 1.27 | 8.16 | 1.30 | 8.28 | 4.66 | 1.28 | 6.08 |
| 5 s | 8.16 | 1.49 | 12.60 | 1.49 | 12.62 | 7.08 | 1.26 | 6.36 |
| 10 s | 8.68 | 1.56 | 13.95 | 1.56 | 13.96 | 7.79 | 1.18 | 5.88 |
| 30 s | 7.92 | 1.52 | 13.41 | 1.51 | 13.40 | 7.08 | 1.02 | 4.82 |
| 60 s | 7.34 | 1.42 | 9.05 (n≈4) | 1.41 | 9.05 | 6.47 | 0.92 | 4.33 |

- data_explorer's numbers are reproduced (the "all" column). About 17 ticks²/s of the level is the 28 shock jumps
  (sd 63). They add a flat amount at every h, but they make the 30–60 s points very noisy.
- **Volatile:** the variance rate goes up ×8.3 from 50 ms to 10 s. It stays flat to 30 s (13.4, so no real reversion
  inside a volatile episode). The 60 s drop rests on about 4 independent windows and is biased toward episode ends.
  **Calm:** ×2.3 up to 10–20 s, then a mild decline. The pooled hump-then-decline mixes the two.
- Round 3 has the same shape on a small scale: a rise to 2–5 s, then a decline from transient impact decay.
- **Decomposition (vol):** ∫V (the smoothed velocity integral) rises monotonically, 0.69 → 6.3 at 10 s → 7.5 at 60 s,
  with no reversion. The kick path S − ∫V rises 4.7 → 9.3 at 10 s and falls to 8.4 at 30 s and 4.9 at 60 s. That is
  the side-law feedback (persistent kicks) followed by Y decaying, with only p ≈ 0.6 of each kick permanent. KF Y alone
  humps 1.8 → 3.4 → 1.8. **The decline after 10 s is the known transient impact.** V does not mean-revert.
- **Residual structure (`pred.py`, exact S, OOS half-split):** in volatile, [V, Y, S − mid] give R² 0.38 / 0.23 /
  0.09 at 2 / 5 / 10 s and ≈ 0 at 20–30 s. Adding past S moves over 2–30 s lowers OOS R² at every horizon. In-sample
  the past-move betas are negative at 10–30 s, but they do not survive the split. Calm R² ≤ 0.02 everywhere.
  **Nothing beyond the known components.**
- **Our own state (`pred2.py`, r4_best causal F, kv, Y):** target S(t + h) − F(t), volatile. kv's beta is 1.8 / 3.4 /
  5.8 at 1 / 2 / 5 s, F − mid's beta 1.6–3.3, and R² is 0.37 at 1 s. Y's partial beta is −0.5 / −1.0 / −1.25 at
  5 / 10 / 20 s, about 2× the pure decay. Y alone has OOS R² < 0 because it rises with the trend.
- **Carry by position age (`carry2.py`, volatile):** position sign changes are frequent (62 % of pinned time has age
  < 2 s). Forward carry is positive to +5 s for every age. At +10/+20 s it turns negative for positions aged 5–10 s
  (−7.6 / −12.3), but those are 9 % of the time and the strategy usually flips before then.

## 2. Variants (r4_volsig.py knobs, all default off; deltas vs r4_best in k, cv = calm + volatile)

| Knob | Mechanism | Total | calm | vol | cv | shock | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `vs_yv` 0.05 / 0.1 / 0.2 | volatile F −= k·Y (fade the transient, 10–30 s reversion) | 1346 / 1216 / 905 | +1 / −1 / −5 | −33 / −176 / −453 | | | dead, monotone (Y sd ≈ 11 in volatile) |
| `vs_yc` 0.1 / 0.2 | calm F −= k·Y | 1319 / 1251 | −68 / −132 | 0 | | | dead |
| `vs_agt`/`vs_agd` 3/0.5, 5/0.5, 5/1, 8/1 | position age ≥ A s: reducing/flip takes need d less edge (hold ~A s, then exit) | 1367–1386 | 0 | −3.5..−11 | −4..−11 | −1..−16 | dead (as vx exits) |
| `vs_tw` 0.2 (with kv) | wider volatile takes (1–10 s adverse selection) | 1390 | −2 | −3 | −5 (vs kv-only −28) | | dead |
| `vs_tw` −0.1 (with kv) | narrower | 1400 | −1 | +22 | +20 | −11 | no better than kv alone |
| **`vs_kv` 0.1 / 0.15 / 0.2 / 0.25 / 0.3 / 0.5 / 1.0** | **volatile F += k·kv (the filter's Kalman velocity)** | 1417 / **1413** / 1397 / 1399 / 1391 / 1384 / 1372 | 0 / +0.6 / −3 / −1 / −3 / −2 / −3 | +16 / **+23** / +20 / +16 / +19 / +15 / −8 | +16 / **+23** / +17 / +15 / +16 / +12 / −11 | +10 / −1 / −11 / −6 / −16 / −19 / −8 | **kept, 0.15** |
| `vs_kv` + clip `vs_kvmax` 1.5–4 | | 1387–1424 | | +13..+20 | +13..+19 | lottery | no better than unclipped 0.15 |
| `vs_kv` 0.15 + age 8/0.5 | | 1429 | +2 | +22 | +24 | +14 (#24) | +1 cv over kv alone; the age knob alone loses, so it is not kept |

- The kv lean has a plateau: volatile +13..+23k for every setting with k 0.1–0.3. k = 0.15 gives cv +23.2k, up in 9 of
  12 windows, t_cv 3.7. Without its biggest shock delta, the shock column is +15.7k. The total swings ±15k with
  shocks #6 and #24 (lottery, as before).
- Why this works when the old velocity leans (F − EMA F, F − mid) lost: kv·0.15 ≈ 0.3 ticks, which is a nowcast that
  covers the filter's own lag. The old leans were horizon bets of 1–5 s. They shifted F by several ticks and delayed
  the flip at V reversals.

## 3. Final

`strategies/marek/round4/r4_volsig.py` = r4_best + `vs_kv` 0.15 (no numba, same imports). **Round 4 1,413.0k**
(calm 281.0k, volatile 821.6k, shock 310.4k). **Round 3 845.4k** (r4_best 848.8k: calm −0.2k, volatile −3.2k).

**Verdict on the hypothesis:** the 1–30 s shape is not exploitable beyond what the stack already does. The rising
part is V plus the side-law feedback, which the pinned trend-side position already harvests. The falling part is the
known Y decay, and every way of trading it (Y fade, age exits, wider takes) loses. The only gain comes from the
short end, a sub-second velocity nowcast inside the filter's lag.
