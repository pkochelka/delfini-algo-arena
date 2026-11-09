# Round 4: the peaked price impact after takers (hypothesis 2)

Written 2026-11-08. Tape `data/round4.csv` only, plus one regression run on `data/round3.csv`. Baseline
`strategies/marek/round4/r4_best.py` scores 1,391.0k (calm 280.5k, volatile 798.9k, shock 311.5k).

**Verdict: dead.** The 5 s peak and the decay after it are real, but they are moves of S itself. The mid only lags S
for about 2 s. The filter tracks the hump in real time: the mean signed error S − F after a taker is 0.00 ± 0.05 at
every lag. On top of what the filter already knows (S estimate, kv, Y, F − mid), the extra signal is small and unstable
between the two halves of the tape. Every lean, fade, timed exit and post-peak passive quote either lost on calm +
volatile or did nothing. The one positive total (+4.8k) is the shock-#24 lottery. **The final file,
`strategies/marek/round4/r4_impactpeak.py`, has every knob off, so it is identical to r4_best: round 4 1,391.0k, round 3
848.8k.**

## 1. Decomposition (tools/decomp.py, decomp2.py)

Each taker's impact at horizon h splits into three parts:
- A = sg·(S₀ − m₀): the gap between S and the mid at trade time, which is the side law's selection;
- B = sg·(S(t+h) − S₀): the move of S itself;
- C = sg·(mid(t+h) − S(t+h)): how far the book still lags S.

Sample: filled regular takers (no TWAP), shock windows excluded, exact S from `s_dynamics/s_path.parquet`.

| h | volatile impact | A | B | C | calm impact | A | B | C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 s | 2.43 | 0.70 | 2.26 | −0.53 | 0.34 | 0.08 | 0.32 | −0.06 |
| 1 | 3.50 | 0.70 | 3.14 | −0.34 | 0.41 | 0.08 | 0.35 | −0.02 |
| 2 | 4.68 | 0.70 | 4.15 | −0.17 | 0.45 | 0.08 | 0.38 | −0.01 |
| 5 | **6.15** | 0.70 | **5.50** | −0.05 | 0.46 | 0.08 | 0.38 | 0.00 |
| 10 | 5.26 | 0.70 | 4.52 | +0.05 | 0.48 | 0.08 | 0.40 | 0.00 |
| 20 | 3.10 | 0.70 | 2.37 | +0.03 | 0.53 | 0.08 | 0.44 | 0.01 |
| 30 | 2.69 | 0.70 | 1.98 | 0.00 | 0.49 | 0.08 | 0.40 | 0.01 |

- **The peak is a volatile-only effect.** Calm impact is flat at about 0.45 from 1 s on, with no peak. The volatile
  peak (6.2 ticks at 5 s) is about 2.4× the pooled curve. TWAP children show the same shape (2.9 at 5 s, 3.1 at 7 s).
  Inside shock windows the curve is flatter (1.9 at 5 s).
- **The book lag C closes within about 2 s,** so both the peak and the decay are S moving, not the mid catching up.
  The decay from 5 s to 30 s is S coming back, not the mid reverting toward S.
- B splits by generator parts as follows (volatile, 5 s → 30 s). The own kick is 0.87 → 0.66. Kicks of later takers
  are **3.23 → 0.63**. Decay of the transient Y left by earlier takers is −0.24 → −0.82. Drift (V) is 1.61 → 1.47.
  So the hump is mostly later takers following the side law Φ((S − mid)/4.42), plus V. The reversion is the kicks of
  takers after about 5 s, which run net against the original trade (mid then sits about 0.05 above S; summed over
  300 takers, that is about −2 ticks), plus Y decay. V does not revert.
- Larger orders give a higher, longer peak: filled 1–20 lots peak at 4.8, 300+ lots at 10.5 around 7 s.

**Is it in the filter's information set?** Event study against r4_best's own logged filter (`tools/logf.py` →
`cache/r4_best_round4.npz`; `tools/evt.py`, `evt2.py`, `evt3.py`):
- The live filter error sg·(S(t+h) − F(t+h)) is +0.002 at 5 s and ≈ 0 at every lag, in both regimes. F rides the hump
  up and down with S.
- Anchored 5 s after a volatile taker, S over the next 10 s falls 1.5 ± 0.8 ticks below the filter's own forecast
  (F + kv drift − Y decay). Per 500 s window the residual is −1.4, −10.1, −8.6, +4.7, −1.5, −1.7, +1.3, −3.7, −0.9,
  +3.0, −2.7. Two windows carry it.
- Regressing S(t+h) − F on [kv, Y, F − mid] plus age-binned signed flow (0–0.5 … 20–30 s), with a two-halves
  out-of-sample test: the flow bins lower out-of-sample R² at h ≥ 2 s. Every bin coefficient is positive
  (continuation); no aged bin fades. The extra fade sits in Y, which correlates 0.99 with the slow flow: the Y
  coefficient at 10 s is −0.84 against the model's −0.49, but −1.12 in the first half and −0.45 in the second.
- Volatile position carry (`tools/carry.py`), E[sign(pos)·ΔS] while |pos| ≥ 60, in ticks:

  | Position state | +1 s | +2 s | +5 s | +20 s |
  | --- | ---: | ---: | ---: | ---: |
  | All | +3.5 | +5.5 | +8.3 | |
  | Post-peak: Y backs pos ≥ 3, Xf ≤ 1 (5 % of the time) | +1.5 | +2.3 | +4.3 | |
  | Y backs pos ≥ 3 | | | | −4.7 |

  A post-peak position still earns at every horizon the strategy actually holds for. The give-back shows only at
  20 s, and the position has flipped long before then (pinned at the cap, flipping on the filter).

## 2. Variants (`r4_impactpeak.py` knobs, tags t1–t4 in results.jsonl)

Xf and Xs are decayed sums of signed taker kicks (filter kick law) with τ 1.5 s and 12 s. "cv" is calm + volatile.
Deltas are against r4_best, in k.

| Test | Total | Δ calm | Δ vol | Δ shock | Δ cv |
| --- | ---: | ---: | ---: | ---: | ---: |
| Filter: faster volatile decay, `tr_T_v` 15 → 8 / 11 | 1,367.9 / 1,378.7 | −0.2 / −0.1 | +0.5 / −0.6 | −23.4 / −11.5 | +0.3 / −0.7 |
| Filter: bigger volatile transient, `tr_p_v` 0.55 → 0.45 / 0.35 | 1,392.5 / 1,379.2 | 0.0 / −0.7 | −0.1 / −0.3 | +1.7 / −10.7 | −0.1 / −1.0 |
| Filter: `tr_T_v` 11 + `tr_p_v` 0.45 | 1,373.8 | +1.3 | +0.1 | −18.7 | +1.4 |
| Quote around F − y_k·Y, `y_k` ±0.2 (both regimes) | 761.8 / 945.9 | −137 / −149 | −454 / −232 | | −591 / −381 |
| Volatile: follow fresh flow, F += k·Xf, k 0.05 / 0.1 / 0.2 | 1,378.4 / 1,333.1 / 1,306.3 | −2 / −3 / −8 | −21 / −39 / −59 | | −23 / −42 / −67 |
| Volatile: fade old flow, F += k·Xs, k −0.02 / −0.05 | 1,359.5 / 1,186.0 | +0.6 / −1 | −20 / −181 | | −20 / −182 |
| Volatile: fade the transient, F −= 0.05·Y | 1,345.1 | +0.8 | −33.7 | −13.0 | −32.9 |
| Volatile, passive quotes only: Xf 0.1 / Xs −0.03 | 1,385.0 / 1,374.1 | −0.8 / −1.0 | −4.3 / −5.3 | | −5.1 / −6.3 |
| Volatile, takes only: Xf 0.05 / Xs −0.01 / Y 0.02 / Xf 0.05 + Xs −0.01 | 1,370.5 / 1,388.3 / 1,374.7 / 1,365.9 | | −19.0 / −3.2 / −3.6 / −12.1 | | −20.7 / −2.9 / −2.1 / −11.8 |
| Calm: follow fresh flow, Xf 0.1 / 0.2; Xf 0.1 + Xs −0.03; Xs −0.05 | 1,355.2 / 1,325.5 / 1,363.3 / 1,317.8 | −12 / −42 / −7 / −54 | ≈ −1 | | −14 / −43 / −7 / −55 |
| Calm, passive only: Xf 0.05 / 0.1 / Xs −0.02; takes only: Xf 0.1 | 1,359.7 / 1,368.7 / 1,365.4 / 1,369.6 | −0.4 / −6.6 / −6.8 / −4.0 | ≈ 0 | | −1.0 / −7.0 / −8.2 / −4.1 |
| Fade after the peak: unblock the volatile flatten quote when Y backs pos ≥ y and Xf ≤ x; (y, x) = (3, 1) | 1,382.6 | +0.5 | 0.0 | −8.9 | +0.5 |
| Same, (8, 0) | 1,395.7 | +0.6 | 0.0 | +4.1 (#24 +12.5; without it −8.4) | +0.6 |
| Same, (3, 1) with the quote at θ 1.0 | 1,385.7 | −1.7 | −0.1 | −3.5 | −1.8 |
| Same, (15, −2) | 1,391.3 | +0.1 | +0.2 | 0.0 | +0.3 |
| Volatile aged exit (`vx_u` 0, age 5 s), gated on Y ≥ 5 backing pos and Xf ≤ 0 | 1,391.1 | 0 | +0.1 | 0 | +0.1 |
| Calm take unload 3–10 s after the take (`cu_u` 0) | 1,390.9 | −0.1 | 0 | 0 | −0.1 |

- Shock-column swings of −11 to −23k are mostly shock #6 (−17k on almost every change). That shock is a lottery, not
  a signal.
- Every F lean loses, in both directions (follow or fade), in both regimes, and in every form: FV, passive-only or
  take-only. The fade-after-peak and exit gates almost never fire (the post-peak state is 2–5 % of pinned volatile
  time), so they cannot move volatile PnL.

## 3. Why it is dead

1. **The filter already knows it.** The hump is S, and F tracks S with zero mean error after a taker. It gets there
   through the side-law flow (F − mid), kv and the Y decay. The only extra fade signal (excess Y decay over 5–20 s)
   comes from two 500 s windows and flips between the two halves of the tape.
2. **The horizon does not match.** The fade lives at 5–30 s. The strategy's volatile inventory sits at the cap and
   flips on the filter in about 1–2 s. At the horizons we actually hold, carry is positive even after the peak
   (+1.5 / +2.3 / +4.3 at 1 / 2 / 5 s). Exiting or fading there throws away carry for a reversion the filter will
   trade anyway, one flip later.
3. **Calm has no peak at all,** so the calm leak (−63k carry) is not an impact-timing problem.

## Files

| Path | What |
| --- | --- |
| `strategies/marek/round4/r4_impactpeak.py` | r4_best + `ip_*` (Xf/Xs/Y leans, `ip_mode` 0 FV / 1 passive / 2 takes), `fp_*` (fade-after-peak flatten quote), `vx_y`/`vx_xf` (exit gate); all off = r4_best (verified 1,390,969) |
| `tools/mk.py` | builds r4_impactpeak.py from r4_best.py |
| `tools/decomp.py`, `decomp2.py` | impact decomposition (A / B / C; own / later / Y decay / drift) |
| `tools/logf.py` | logs r4_best's filter (F, sd, pv, kv, Y, turb, pos, touch) every 10 ms → `cache/` |
| `tools/evt.py`, `evt2.py`, `evt3.py` | event studies against the filter (live error, anchored forecasts, per window) |
| `tools/resid.py`, `ycoef.py`, `xfit.py` | regressions of S(t+h) − F on the filter state plus flow features (two halves) |
| `tools/carry.py` | volatile position carry by impact-peak state |
| `eval4.py`, `sweep.py`, `summ.py`, `cfg/`, `results.jsonl` | harness copies (results go here), sweep configs, runs (`base4`, t1–t4, `r3final`) |
