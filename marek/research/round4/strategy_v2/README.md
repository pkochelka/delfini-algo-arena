# Round 4 strategy v2: the finished generator spec against r4_best

Written 2026-11-08. Tape `data/round4.csv` only, one regression run on `data/round3.csv`. Start: `strategies/marek/round4/r4_best.py`
(1,391.0k; calm 280.5k, volatile 798.9k, shock 311.5k). Spec: `../round4_generator.md` and the area READMEs.

**Bottom line.** Nothing in the finished spec beats r4_best beyond noise. Every new tell (the 15-lot joiner, the side law, the
TWAP clock) is real and measurable, but none of them pays under the 100-lot cap. The one kept change is a correctness fix to the
round-3 ladder filter: it let 14 % (volatile) and 20 % (calm) of ladder inserts into the S likelihood.
**Best file: `strategies/marek/round4/r4_v2.py` = r4_best + `lad_hist 1, lad_step 8`. Round 4 1,403.7k (+12.7k; calm −0.1k,
volatile +2.0k, shock +10.8k, which is lottery). Round 3 848.1k (−0.7k; calm + volatile +3.8k, shock −4.5k).** The calm + volatile
gain is noise-sized on round 4 (+1.9k, 6 of 12 windows up, every window within ±1.1k) and a bit firmer on round 3 (+3.8k, 9 of 12).
I keep it because the leak it closes is measured directly, not because of the PnL.

## Tools (this folder)

| Path | What |
| --- | --- |
| `eval4.py`, `sweep.py`, `summ.py`, `tbl.py` | copies of `../strategy/` writing to `results.jsonl` here. eval4 rows now carry `win_cv`; `summ.py` adds `wcv` (windows where calm + volatile is up) and `--strip` / `--w` for readable set columns |
| `tools/j15log.py`, `det15.py`, `ep15.py` | online 15-lot detections + 10 ms state log; detector quality against `makers/r15_episodes.csv` and order_class; PnL / carry / filter bias inside the episodes |
| `tools/fills.py`, `mk.py` | our fills with context (F, pos, touch, live joiner side, queue ahead, age); markouts against s_path by regime, kind, joiner relation, S − mid, edge, queue |
| `tools/verr.py` | kv against s_dynamics' smoothed V (`_cache/kfd_v2b_D_state.npz`) and F − S, by regime, time since a switch, after a jump |
| `tools/ladchk.py`, `ladan.py`, `ladleak.py` | which inserts reach `FV.maker`, against the order_class labels; what the leaked rungs look like |
| `tools/logf.py`, `carry.py`, `ferr.py` | from `../strategy/tools` (logf also logs kv) |
| `cfg/*.jsonl`, `results.jsonl` | sweep configs and every run (tags below); `cache/` is git-ignored |

Noise: shock #6 (1391 s, J = +111, calm) is worth 92.9k to r4_best, and almost any change loses exactly −16.9k there. Judge by
**cv** (calm + volatile), `wcv` and `tcv`. Single knobs move cv by ±3k from path divergence alone.

## What the spec predicts vs what r4_best does (measured)

- **15-lot joiner detector** (r4_dev's `_j15`): 694 detections, 687 are agent orders (99.0 %). All 31 episodes are caught; the
  first detection comes a median 0.57 s after the start (max 1.6 s). 4 of the 7 false ones are 10-lot, so a fresh episode now
  needs a 15. After a detection, S moves +1.4 / +2.4 / +4.8 ticks its way at 1 / 2 / 5 s. After the last detection it is
  +0.8 at 1 s and −2.4 at 10 s.
- **Our filter inside the episodes has no bias:** (F − S)·side = −0.03. r4_best is already long the agent's way on average
  (+5..+85 lots) and earns 24.3k in the episodes plus 21.1k in the 5 s after; carry is +5.2k.
- **Calm passive markouts** (r4_best fills against s_hat, lots-weighted):
  - edge at the fill is +0.67 and +0.18 by 0.1 s (the taker's own kick);
  - against a live joiner, −1.75 at 5 s (−46k); with it, +1.44 (+28k);
  - by z = side·(F − mid): z < −0.25 gives −71k at 5 s and z > +0.25 gives +146k;
  - behind ≥ 200 lots of queue, −0.05.
- **Velocity:** in volatile, smoothed V regressed on our kv has slope 1.39 (corr 0.87), so kv is too timid. At volatile starts kv
  lags badly: slope 4.2 in the first 0.5 s, 2.9 to 1 s, 1.7 to 2 s. Its prior variance relaxes from the calm level with τ 2.1 s.
  After a vol end kv is too large (slope 0.52). Calm kv ≈ 0 (sd 0.07 against V 0.6).
- **Ladder filter leak** (`ladan.py`): ladder inserts fed to the S likelihood were 19.5 % calm, 14.1 % volatile, 20 % shock,
  including 12 % of rungs 1+. Every leaked rung 1+ is exactly 2000 µs after the previous rung of its chain. The chain was broken
  either by a base insert in between (the filter linked only to the last same-side insert) or by a rung step of 2, 3 or 5 ticks.
  Base makers wrongly dropped: 0.09 %.

## Results

Deltas vs r4_best on round 4, k. cv = calm + volatile; wcv = windows (of 12) with cv up.

| # | Test (knobs in `r4_v2_dev.py`) | Result | calm | vol | cv | shock | wcv | Tags |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | Spec constants one by one: calm c/e, jv, tr_p/T, tr_p_v/T_v, jv_v, d0/lam, d0_v/lam_v, q_cv/q_vc, kv calm sd/τ, kv vol sd/τ, c_v/e_v | flat | −1.1..+1.5 | −2.7..+0.1 | −3.2..+0.9 | #6 lottery | | k1 |
| 1 | all of them together | loses | −4.5 | −1.9 | −6.4 | +2.6 | 5 | k1 |
| 2 | joiner lean: F + jl_k·side while live (`jl_k` 0.5–3, hold 1–2.5 s, min 2 detections) | loses | −1.6..−11.8 | −0.8..+1.8 | −0.9..−10.0 | | 1–4 | j1 |
| 2 | joiner asymmetric: back off the against side (`jl_th` 1/3), harder takes against (`jl_tx`), easier takes with (`jl_tw`), tighter with side (`jl_tj`), pull (`jl_pull`) | loses | −0.6..−4.4 | −1.6..+0.9 | −0.3..−5.2 | | | j2, j3 |
| 2 | joiner fade after the episode (`jl_fk` 0.5/1, 5/10 s), small lean 0.3, cap 5 s | flat / loses | −2.6..+1.9 | −5.9..+0.9 | −8.5..+1.2 | | | j3 |
| 2 | 15-only joiner drop from the likelihood (`j15_on`, `j15_q` (15,)) | neutral | +0.4 | −1.2 | −0.8 | +1.2 | | j3 |
| 3 | calm side-law pull: no bid while F − mid < −g, ask mirrored (`sl_g` 0.25/0.5/0.75/1) | loses badly | −11..−55 | 0 | −12..−55 | | | c1 |
| 3 | the same, never pulling the reducing side (`sl_red`) | loses | −7.0 | −1.1 | −8.1 | | | c1 |
| 3 | no calm post behind > 200 / 500 market lots (`qa_max`) | loses badly | −64 / −40 | 0 | −64 / −41 | | | c1 |
| 3 | theta_keep 0.1 / 0.2 | loses | −2.5 / −5.0 | | −2.7 / −6.1 | | | c1 |
| 4 | velocity gain `kv_g` 1.2 / 1.4 / 1.7 (≡ shift multiplier `kv_m`) | flat | −0.4..+0.9 | +0.1..+1.0 | −0.3..+1.9 | | | v1 |
| 4 | prior variance jumps with P(vol) (`kv_inf` 1), + shrink kv on a fall (`kv_shr`); kv_sd_v 4.5 | flat | +0.2..+1.4 | −0.5..+0.7 | +0.1..+1.0 | | | v1 |
| 5 | **ladder linking to any same-side insert 2000 µs earlier** (`lad_hist`), **1–8 ticks out** (`lad_step` 8) | **leak 14 → 0.7 % vol, 20 → 3.5 % calm** | −0.1 | +2.0 | **+1.9** | +10.8 | 6 | l1, l2 |
| 5 | lad_hist with 1-tick steps only / lad_step 3 / lad_tol 10 / lad_deep 3 | same or less | | | +0.5..+2.3 | | | l2 |
| 5 | lower eta 0.10 / 0.05 with the tighter filter | loses | −0.6 / −7.6 | | +0.5 / −6.2 | | | l2 |
| 6 | shock sniper vs J sd 63: `ss_lim` 16/30, `st_pm` 2/4, `ss_shots` 5, `xu_lim` 2, `xs_cap` 1500 | flat; xs_cap 1500 −154k shock | | | ≈ 0 | −34..+11; −154 | | s1 |
| 7 | taker-limit lower bound on S from each filled IOC's worst fill price (`tl_on`; spec law, softer tails, ±0.5 tick bands) | flat | −0.9..+1.9 | −1.8..+1.6 | −3..+1.6 vs l2 | | | t1 |
| 8 | stacks of the small positives (ladder + kv_g 1.1–1.4 + kv_inf/shr + tl) | noise band | | | +0.8..+7.1 | −18..+8 | 6–11 | m1, m2, f1 |
| — | TWAP children as a fill source (offline): only 9 of our passive fills were TWAP children; they mark out −0.73 at 5 s | dead | | | | | | — |

Why the tells don't pay (within-episode PnL, `ep15.py`, episode + 5 s after):

| Joiner variant | episode PnL | +5 s | carry in episode |
| --- | ---: | ---: | ---: |
| r4_best | 24.3k | 21.1k | 5.2k |
| lean jl_k 2 | 22.2k | 17.5k | 26.6k |
| lean jl_k 0.5 | 27.5k | 18.2k | 19.1k |
| no quote against (jl_th 3) | 22.9k | 21.0k | 21.1k |
| pull against (jl_pull) | 20.9k | 20.9k | 18.2k |
| harder takes against (jl_tx 2) | 24.9k | 20.2k | 5.7k |

Leaning works as intended: carry goes up ×4–5. But the trading edge falls by more, because a pinned book stops making markets.
Calm spread capture runs ≈ 69–85 PnL/s at the cap, about what 100 lots earn from a 0.7–0.85 tick/s drift. The joiner (≈ 1.3
tick/s for ≈ 8 s), the TWAP (≈ 0.74 tick/s) and the calm V (sd 0.4) all sit at or below that break-even. The against-side fills
look toxic in markouts, but they recycle position room for the with-side fills (+1.44 / lot), so pulling them loses both. The
same mechanism kills every calm pull (#3): markouts are not PnL when the fill is what flattens you.

Tracking fixes (#1, #4, #7) move volatile by ±2k. With F − S rmse at 0.41 in volatile and the position at the cap 88 % of the
time, the flip timing is set by stale-quote availability, not by the last 0.1 tick of F.

## Final: `strategies/marek/round4/r4_v2.py`

`r4_v2_dev.py` + `PARAMS.update({'lad_hist': 1, 'lad_step': 8})`. Same imports as r4_best, no file reads, no `modify_order`.
`r4_v2_dev.py` with every v2 knob off reproduces r4_best exactly (1,390,969).

| Run (tags `final`, `r3final`, `r3base`) | Total | calm | volatile | shock | breach |
| --- | ---: | ---: | ---: | ---: | ---: |
| r4_best, round 4 | 1,391.0k | 280.5k | 798.9k | 311.5k | 12.0k |
| **r4_v2, round 4** | **1,403.7k** | **280.4k** | **800.9k** | **322.4k** | 12.7k |
| r4_best, round 3 | 848.8k | 254.2k | 422.3k | 172.2k | 17.6k |
| **r4_v2, round 3** | **848.1k** | **253.9k** | **426.4k** | **167.8k** | 19.0k |

- Round 4 per-window cv deltas: −0.2, +0.3, +0.7, +0.7, +0.5, −0.0, −0.3, +0.1, +1.1, −0.4, −0.3, −0.2k (6 of 12 up). Per-shock:
  +4.8k on #8, +3.6k on #21, +1.7k on #24, the rest within ±0.9k (median 0).
- Round 3: calm + volatile +3.8k, up in 9 of 12 windows; shock −4.5k (−3.0k on #17).
- Leave-one-out is trivial (one change). In the stacks (m2), removing the ladder fix costs 3.7k cv, removing kv_g 4.1k,
  removing kv_inf/shr 4.1k, removing tl 1.1k. Without tl the kv parts drop to +0.8..+2.2k (f1), so only the ladder fix is kept.
- Not kept, neutral-to-positive and ready as knobs: `kv_inf 1, kv_shr 1` (r4 cv +2.2 on top of the ladder fix, r3 +0.7k);
  `kv_g 1.2–1.3`.

## Open

- **The cap is the limit everywhere.** Calm tells are worth less than the spread they cost when pinned. Volatile is at the cap
  88 % of the time. Only a looser position limit would turn the round-4 predictability into PnL, and breaches cost 10k/s.
- The calm ladder leak is still 3.5 %: these are first rungs whose chain never shows (single-rung bursts look like deep base
  quotes). eta 0.15 absorbs them.
- Joiner end detection (agent order filled and no re-post 1 µs later) is not built. The give-back after the end (−2.4 at 10 s)
  is too small to fade (#2 fade rows).
- Shock #6 (J = +111) is a ±17k discrete outcome on almost any change; r4_best and r4_v2 both sit on the lucky side.
