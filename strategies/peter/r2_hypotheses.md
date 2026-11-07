# Round 2: one hypothesis per finding, tested with `r2_mm.py`

[r2_mm.py](r2_mm.py) is a market maker with a simple fair-price model.
- S = U + off. U = 0.037 · Σ sign · √filled, summed over market takers.
- off is bracketed by every maker insert: a bid insert gives off ≥ bid + 0.05 − U, and an ask insert gives off ≤ ask − 0.05 − U.
- An insert outside the bracket resets it to a 1-tick bracket. That is how jumps get picked up.

Every finding in [round2_findings.md](round2_findings.md) became a flag in `PARAMS`. Each flag was tested on its own against the current base.

**Method:**
- Data: the 12 round-2 walk-forward folds, `data/splits_r2/wf`.
- Tool: [paired.py](paired.py). It reports the mean paired per-fold difference, how many folds the variant won, and t.
- [r2_diag.py](r2_diag.py) splits fill markouts by passive or take, calm or turbulent, and time since the last jump.

## Result

| Strategy | Mean per fold (12 × 370 s) | Worst fold | Holdout (4500–6000 s) | Full tape |
| --- | ---: | ---: | ---: | ---: |
| `marek/round1/robust_ladder.py` | +1.1k | −14.3k | | +12.7k (folds total) |
| `peter/s_mm.py` (round-1 S model) | −11.9k | −82.8k | | |
| **`peter/r2_mm.py`** (defaults) | **+16.7k** | +6.4k | +27.6k | +228.0k, no breach, 2 min |
| `peter/regime_mm.py` (parallel work) | +9.1k | +5.6k | +40.6k | |
| `marek/round2/r2_ladder.py` | +24.9k | +4.7k | +87.1k | |
| `peter/fv_mm.py` (parallel work, 21:09 version) | **+36.4k** | +17.0k | **+122.1k** | |

**`fv_mm` dominates `r2_mm`** on every fold and on the holdout.
- `r2_mm` also generalises worse. It makes 45 $/s on the folds but only 18 $/s on the holdout. `fv_mm` makes 98 and 81 $/s.
- The two `r2_mm` wins that could transfer were tested on a frozen copy of `fv_mm`. Neither adds anything there:
  - Spread ≥ 3 as a turbulence trigger: −0.6k (0 of 12 folds won).
  - Widening theta by the uncertainty (`theta_sd` 0.5–1.0): −0.9k to −3.2k (0–1 of 12 folds won).
- The Bayesian grid filter, the regime HMM and the maker-outage detector in `fv_mm` already cover what these flags catch.

Use `fv_mm` as the round-2 base. The value of `r2_mm` is the list below of ideas already tested, so they don't get re-tested.

## Hypotheses, in the order tested

Each line shows the paired difference against the base at that time.

| # | Finding | Hypothesis (flag) | Result | Kept |
| --- | --- | --- | --- | --- |
| H1 | The calm parts follow the round-1 generator | Quote around the bracket S, not the touch (s_mm core) | Calm folds +8–13k, but wf11 −70k without the regime layer | yes |
| H2 | After a jump the stale side is many ticks off | IOC quotes beyond the hard bracket bound (`take`) | +3.0k/fold. Takes are ~60% of lots, +0.27/lot calm, +0.7/lot at 10 s in turbulence | yes |
| H2b | | Also take at S ∓ `take_mid` while the bracket is tight | 0.15: +2.9k; 0.1: −0.8k; 0.3: −1.6k | 0.15 |
| H2c | | Looser hard-bound take (`take_lb` −0.2) | −9.6k (0/12 won) | no |
| H3 | Jumps are visible in the first inserts | On a jump, cancel the stale side or pause passive quotes (`jump_cancel`, `jump_pause`) | ±0. The turbulence gate already covers it | no |
| H4 | Jumps cluster; makers lose 1.3/lot in turbulence | Turbulent if 10 s realised vol of S > 2 (`turb_rv`) → passive cap 0 | −6.5k → +6.3k | yes |
| H4b | | Turbulent → no passive quotes at all, takes only (`turb_off`) | +3.4k on top of H4 (11/12 positive) | yes |
| H4c | | Turbulent → quote far instead (`turb_theta` 1–4) | −1.4k to −5.8k | no |
| H4d | | Hold the flag for 10–60 s after a jump (`turb_hold`) | −0.1k | no |
| H4e | | `turb_rv` 1.5 / 1.7 / 2.5 / 3 | all worse than 2 | 2 |
| H5 | Fills against >100-lot takers mark out −0.54 | Rest at one price level only (`keep_depth` 0) | −0.2k | no |
| H6 | Volatility clusters (ACF of \|r\| 0.21) | theta += vol_k · (σ/σ_ref − 1) | +0.6k early, −0.8 to −1.1k on the final base | no |
| H7 | Touch depth falls in turbulence | Cap ≤ depth_k · touch qty | ±0 | no |
| H8 | The cancel hazard rises with adverse moves | Pull our side on a burst of market cancels at the touch (`cburst_n` 3/5) | −4.0k / −1.7k (t −4.8) | no |
| H9 | Survival first | Equity kill switch (`ks_dd` 2000) | −0.2k (0/12 won). The regime gate does the job | no |
| H10 | | Inventory skew (`inv_skew` 0.2/0.5) | −0.6k / −0.9k | no |
| H11 | The bracket is too wide to quote ~40% of the time | Quote it anyway, with theta += w_k · (width − 0.3) (`max_width` 1.2, `w_k` 1) | +2.8k (9/12 won) | yes |
| H12 | Mean \|pos\| ≈ 83: the cap binds | A lower cap for passive quotes leaves room for takes (`pcap` 60/80) | −1.5k / −1.0k | no |
| H13 | Imbalance has R² 0.05 at 0.5 s in calm | S += imb_k · imbalance (0.1/0.25) | −1.0k / −2.9k | no |
| H14 | Room for the next snipe | Looser limits on inventory-reducing takes (`take_red`) | −7k to −25k (0/12 won): crossing to unwind never pays | no |
| H15 | A side empties 274 times | Quote into an empty side (`empty_quote`) | +0.03k | no |
| H16 | The spread widens in turbulence | Spread ≥ 3 → turbulent (`turb_spread`) | **+1.7k (9/12 won)**. Spread 4 gives the same; 2 gives −0.8k; holding it ≥ 0.5 s loses | yes |
| H17 | The taker rate rises in turbulence | Taker EWMA > 12–45/s → turbulent (`turb_tk`) | ±0 on top of H4 and H16 | no |
| H18 | Impact is ~4× larger in turbulence | c × `c_turb` while turbulent (0.5/2/4) | −1.7k / −5.6k / −37.8k | no |
| H19 | Variance ratio 1.25–1.6: turbulent stretches trend | S += mom_k · (S − S 10 s/60 s ago), ±0.05/0.15 | −9.5k to −25.7k | no |
| — | Periodicity, intraday trend | No usable signal in the findings, so nothing to test | | |

**Fair-price parameters:** c = 0.037, gap 0.05, reset width 1.0 and relax 0 are each locally optimal.
- c 0.03: −1.9k; c 0.045: −5.5k.
- Gap 0: −3.8k; gap 0.1: −2.6k.
- Reset width 0.6: −1.9k; 1.5: −3.5k.
- Relax 0.05: −0.9k.

**Quoting parameters:** theta 0.3, theta_keep 0.1, reach_off 1.0 and size_top 1000 are locally optimal on the final base.
- reach_off 0.5: −5.8k.
- size_top 50: −3.5k.

## What carries over to other strategies

1. **Passive quoting in turbulence loses at any distance and any cap.** Taking stale quotes in turbulence is where the money is: wf11 went from −70k to +36k. `fv_mm` and `regime_mm` found the same thing independently.
2. **Do not cross the spread to unwind (H14).** Inventory carry has zero mean, while the crossing cost is certain.
3. **Any bias added to the fair value hurts:** momentum, imbalance, a regime-dependent c. The take thresholds are very sensitive to bias in S.
