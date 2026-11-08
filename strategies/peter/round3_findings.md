# Round 3 (public tape): what changed versus round 2

`data/round3_public.csv`: 6000 s, 2.41M rows, 1.23M orders. Round 3 is **round 2 plus two new mechanics**, a deep-ladder maker and mean-reverting price moves. The shocks, droughts, regimes and size laws are unchanged. Written 2026-11-08.

## How to reproduce

```
uv run python -m analysis.fit data/round3_public.csv
uv run python strategies/peter/split_data.py --backtest data/round3_public.csv --folds 12 --warmup-s 60 --end-frac 0.75 --out-dir data/splits_r3/wf
uv run python strategies/peter/split_data.py --backtest data/round3_public.csv --ratios 3 1 --names pre75 hold25 --out-dir data/splits_r3
uv run python strategies/peter/regime_split.py data/round3_public.csv --out-dir data/splits_r3/regime --test-from 4500
uv run python strategies/peter/r1_model.py data/round3_public.csv --out strategies/peter/tuned/round3_model.json   # fails, as on round 2
PYTHONUTF8=1 uv run data-explorer --data data/round3_public.csv --export                                # -> data_explorer/round3_public/
PYTHONUTF8=1 uv run data-explorer --data data/round2_public.csv data/round3_public.csv --export          # -> data_explorer/compare-round2_public-round3_public/
uv run python strategies/peter/r2_structure.py data/round3_public.csv --folds data/splits_r3/wf/folds.json --hold-parts 4
uv run python strategies/peter/r3_deep.py data/round3_public.csv --folds data/splits_r3/wf/folds.json   # the round-3-specific checks
uv run python strategies/peter/r2_fv_fit.py data/round3_public.csv --windows 0:600,1800:2400,3600:4200 --grid eta=0.005,0.02,0.05,0.1
```

The export needs `PYTHONUTF8=1` on Windows. `PYTHONIOENCODING` alone is not enough, because `README.md` is written with `write_text()` without an encoding. Logs from these runs are in `logs/r3/`.

**Splits:**
- `data/splits_r3/wf/wf01-12`: 370 s eval windows over 60–4500 s, each with a 60 s warmup.
- Holdout: 4500–6000 s, in `hold25.csv`.
- `data/splits_r3/regime`: calm dev 21 stretches (2157 s), calm test 6 (571 s), turbulent dev 24 (2050 s), turbulent test 8 (765 s).

## What stayed the same as round 2

- **Participants and sizes:**
  - Maker inserts at 194/s; round 2 had 180/s.
  - Takers at 4.7/s per side; buy share 0.500, sign ACF 0.002.
  - Taker log-size 4.09 ± 1.40, the same as round 2.
- **Shocks:** 27 maker droughts, the same as round 2.
  - The insert rate drops 194 → 54/s for 3.20 s (sd 0.09).
  - Gaps between droughts are 145–293 s, CV 0.17.
  - The jump lands at onset + **3.00 s** (IQR 2.97–3.02), with sd 50 and mean |J| 35. 21 of 27 shocks jump by 8 ticks or more. The largest are +136 (4127 s), −114 (4298 s) and −96 (2614 s).
- **Regimes:** volatility clusters and turbulence comes in episodes. P(another jump within 10 s | a jump) is 0.77, against 0.44 unconditionally. 1 s σ is 1.69 ticks/√s in calm and 7.3 in turbulence.
- **No periodicity.** The 1500 s and 429 s periodogram peaks are only 4 and 14 cycles of a red process; Fisher's g test assumes white noise, so its p-values don't apply. The turbulence ACF is −0.00 at the 1500 s lag and 0.09 at 429 s. Timestamp phase is uniform: the mod-1 ms KS p of 0.003 in r2_structure is a large-n artifact, and a chi² over 100 bins gives p 0.13. Round 2 shows the same peaks.
- **No directional trend:** P drift −127 ticks, z −0.35.
- **The round-1 generator fails, as on round 2:** 583k maker violations.

## New 1: a deep-ladder maker

In calm periods, a new population of orders sits 4–8 ticks from P. Round 2 has essentially none past 4 ticks.

| Offset from P (calm) | Round 3 rate | Size | Life | Round 2 rate | Size | Life |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 110/s | 14 | 0.31 s | 111/s | 14 | 0.31 s |
| 3 | 5.6/s | 33 | 0.45 s | 3.8/s | 14 | 0.31 s |
| 4 | 1.25/s | 100 | 0.88 s | 0.55/s | 15 | 0.30 s |
| 5–8 | about 1.0/s per level | 135–166 | 1.7–2.6 s | ≤ 0.08/s | — | — |
| 9+ | ≈ 0 (cliff) | | | | | |

Sizes are median lots.

- **Rate and shape:**
  - 5.2 inserts/s in total, symmetric (bid share 0.505).
  - Size is lognormal with log-size 4.67 ± 1.08 (median 141). The regular makers have 2.7 ± 1.7.
  - Lifetime is about 6× longer than the regular makers'.
- **The ladder is stationary:** the calm rate has CV 0.08 across the 16 splits and does not follow the turbulent share (Spearman +0.34, n.s.). It is a separate participant on its own clock, not a regime effect.
- **It is centred on the current P,** not on a slow anchor. Deep bid and ask offsets are the same whatever the sign of P − EMA.
- **Effect on the book:** the mean resting depth at 2–7 ticks behind the touch is 750 → 180 lots, against 290 → 9 on round 2 (`compare-round2_public-round3_public`). The book then drops off at 8 ticks.
- **Turbulence:** quotes are wider than in round 2 (rates at 3–7 ticks about 1.3–3.5× higher, sizes 21–33), so the ladder blends into the volatile maker there.

## New 2: price moves revert (transient impact, mean-reverting jumps)

**Taker impact is now partly transient.** The impact kernel is the mid response per unit of sign · filled^0.62, with shock windows excluded:

| Horizon | 0.1 s | 1 s | 2 s | 5 s | 10 s | 30 s | 60 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Round 3, calm | .007 | .018 | **.019** | .017 | .013 | .010 | **.002** |
| Round 3, turbulent | .028 | .072 | **.079** | .073 | .059 | .051 | .056 |
| Round 2, calm | .007 | .016 | .016 | .017 | .018 | .015 | .016 |
| Round 2, turbulent | .032 | .068 | .068 | .071 | .072 | .082 | .083 |

- **Calm:** impact peaks at 2 s and is almost entirely gone by 60 s.
- **Turbulent:** about a third of the impact reverts.
- **Round 2:** permanent in both regimes.
- **Per split:** the transient share is positive in 16 of 16 splits (sign test p 3e-5). Round 2 has 9 of 16.

**Takers win short and lose long.** Volume-weighted taker PnL per lot against the fill price:

| | 0.1 s | 0.5 s | 1 s | 2 s | 10 s | 30 s | 60 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Round 3 | −0.05 | +0.67 | +0.91 | **+1.10** | +0.46 | +0.22 | **+0.03** |
| Round 2 | −0.12 | +0.38 | +0.40 | +0.40 | +0.53 | +0.46 | +0.49 |

- The volume-weighted maker markout at 1 s is −0.91 per lot; round 2 had −0.41.
- Fills against takers of more than 100 lots mark out −1.12. Small takers give about +0.2.

**Jumps outside the shocks revert.** Signed P move after a jump, as a share of the jump; in round 3 there are 215 such jumps with mean |J| 7:

| | +1 s | +5 s | +10 s | +30 s | +60 s | +120 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Round 3, other jumps | 1.04 | 0.93 | 0.76 | **0.32** | 0.27 | 0.11 |
| Round 2, other jumps | 0.93 | 0.99 | 0.99 | 0.95 | 0.69 | 0.84 |

- **Shock jumps** look mostly permanent: 0.87 at 120 s, but only 21 of them and the path is noisy.
- **Reversion speed:** about τ ≈ 20 s. The mid gives back the whole jump by 60 s.
- **P − EMA(20 s) predicts the move over the next 20 s,** a weak signal:
  - Calm: slope −0.14, corr −0.05.
  - Turbulent: slope −0.25, corr −0.16.
  - Round 2 gives +0.03 and −0.04.
- **Everywhere else on the tape:**
  - The 1 s mid ACF at lag 1 is +0.11: the move runs on for about 2 s, then reverts.
  - The variance ratio of P is 1.09 at 5 s and 0.87 at 60 s.
- **Working model (not yet fitted):** S = L + X. L carries the permanent part of impact and the shock jumps. X is an OU process (τ of roughly 5–20 s) that receives the transient impact and the volatile-regime jumps.

## Anomalies

- **One-sided book:** 118 episodes, half of round 2's 274. Median 9.7 ms, max 148 ms; 84 started with an IOC.
- **Busiest second:** 62 takers (Poisson p 4e-26), in post-jump sweeps. The longest gap with no events is 119 ms, at 5036 s.
- **Largest single orders:** a 111,687-lot maker and a 50,114-lot taker. Both fit the lognormal tails (p 0.10 and 0.04).
- **Spread:** above 100 ticks for a few ms around the large shocks (max 134).

## Per-split stability

- **Stable:**
  - Maker rate: CV 3%.
  - Sizes: 0.4–0.8%.
  - Buy share: chi² p 0.19.
  - Calm volatility: 7%.
  - Deep-ladder rate: 8%; deep size 14%.
  - Shock count: 1–2 per split.
- **Unstable, and driven by the turbulent share** (17–76% per split):
  - Taker rate (chi² p 1e-80), jumps (1 to 58 per split), touch qty (665–1241), 1-tick spread share (0.67–0.93) and 1 s vol (1.7–9.1).
  - Taker PnL at 1 s: Spearman +0.77 with the turbulent share.
- **Groups:** turbulent splits are wf02, wf07, wf10, wf11 and hold4; calm ones are wf01 and hold2. Fold PnL differences mostly reflect this, so compare variants on paired per-fold differences.

## Predictability (leave-one-split-out R² of the mid change)

| Horizon | Public, all | Public, calm | P − mid | FV filter, `eta` 0.005 | FV filter, `eta` 0.05 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.5 s | 0.081 | 0.100 | 0.038 | 0.138 | **0.179** |
| 1 s | 0.060 | 0.037 | 0.027 | | |
| 5 s | 0.006 | ≤ 0 | ≤ 0 | | |

- Insert aggressiveness is the strongest public feature. Imbalance in calm periods gives 0.035 at 0.5 s.
- The FV columns are the in-sample fit from `r2_fv_fit.py`, over 3 windows of 600 s.

## Round-2 strategies on the round-3 folds (12 × 370 s, `fold_eval.py --dir data/splits_r3/wf`)

| Strategy | Mean per fold | Worst fold | Notes |
| --- | ---: | ---: | --- |
| `peter/fvs_mm.py` | +29.8k | +4.4k | 12/12 positive. Round-2 folds: +37.9k |
| `peter/fvs_mm.py` with `eta` 0.05 | **+38.0k** | +16.8k | Wins 12/12 folds against the defaults. With `eta` 0.02: +34.4k |
| `marek/round2/r2_ladder.py` | +30.4k | +8.6k | 12/12 positive |
| `marek/round2/final3.py` | +29.5k | +3.6k | **5876 breach** on round 3: check its room logic before using it |
| `peter/regime_mm.py` | +1.9k | −5.3k | |
| `marek/round1/robust_ladder.py` | −9.4k | −26.8k | |
| `marek/round1/sig_ladder.py` | −34.1k | −96.1k | |

**Regime split, round 3:**

| | Calm, dev | Calm, test | Turbulent, dev | Turbulent, test |
| --- | ---: | ---: | ---: | ---: |
| `fvs_mm` | +71.7k | +23.6k | +278.9k | +87.6k |
| `fvs_mm` with `eta` 0.05 | **+107.5k** | **+31.1k** | **+340.6k** | **+107.9k** |
| `r2_ladder` | +62.1k | +15.4k | +278.6k | +80.8k |

### Why `eta` matters: the deep ladder breaks the FV filter's maker law

- **The FV filter treats a 6-tick-deep insert as evidence that S moved.** Its maker law is D = 0.05 + Exp(0.45), with outlier share `eta` 0.005. A single update then shifts the posterior mean by more than `jump_det`.
- **The filter's "jumps" counter fires about 1.5/s on round 3** (911 per 600 s window), against 4.7 per window on round 2.
- **Raising `eta` to 0.05 absorbs the ladder:**
  - Detections fall to 54 per window.
  - Log-likelihood per insert improves from −1.159 to −1.148.
  - Posterior sd falls from 0.32 to 0.19.
  - 0.5 s R² rises from 0.138 to 0.179.
- **Round 2 is unaffected:** `eta` 0.05 gives 37.7k against 37.9k per fold, which is noise.
- **Not adopted as a default yet.** The cleaner fix is a deep-ladder component in the maker likelihood (offset 4–8, size-dependent), or ignoring inserts with size above 100 placed 4 or more ticks behind.

## Implications for strategy work

1. **Fix the FV maker law first.** It is the cheapest gain so far (+27% per fold, holding on the regime test stretches). Try a mixture component for the ladder, or a size and depth filter, against `eta` 0.05.
2. **Stop treating impact as permanent.** `fv_mm` adds c · filled^0.625 to S for good. In calm periods that overstates S's move after about 10 s, and the takes and quotes lean the wrong way. Add a decaying transient term (two-component impact: L and X), with τ fitted from the kernel.
3. **Fade the volatile-regime jumps after about 5 s and keep the shock jumps.** A jump inside a drought window (onset + 3 s) is permanent. Other jumps give back about 70% in 30 s. The first second is still momentum (ACF +0.11), so a fade has to wait.
4. **The deep ladder is liquidity to sell into, and a stale target.** After a shock, the stale side now has about 150-lot orders 4–8 ticks deep that live for about 2 s, so `fvs_mm`'s sweep to stale touch ± 8 finds more volume. It is also 5/s of large passive size: a toxic counterparty for big takers only.
5. **Passive quoting is harder.** The maker markout has doubled to −0.91 at 1 s, and takers win +1.1 per lot at 2 s. Quote wider for 0.5–2 s after large takers, then lean against the transient part of their impact.
