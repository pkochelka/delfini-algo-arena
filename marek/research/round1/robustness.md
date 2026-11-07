# Round 1 robustness: how ladder_mm holds up off the public tape

Question: `strategies/peter/ladder_mm.py` makes 425,639 on the public round1 tape. How much of that carries over to the private tape and to harder rounds? Where does it lose money, and which cheap guards help?

Everything below is reproducible with the scripts in `research/round1/robustness/`:

| file | what it does |
| --- | --- |
| `run_matrix.py` | runs strategies × tapes through `exchange-simulator --no-logs` (3 at a time). `path:{json}` applies a `PARAMS` override. Rows go to `results/*.csv`. |
| `stress_tapes.py` | writes `data/stress/<fold>_<tag>.csv`: every `augment.py` preset plus scenarios it lacks (informed takers `toxicL`, bigger takers `bigtakerK`, trend bursts `burst`, a 300-tick one-way trend `trendup/trenddown`, and a mid-window volatility switch `volreg`) |
| `decompose.py` | splits a full-logs run's PnL into spread capture and inventory carry, and reports maker markouts at 0.1 to 300 s and the linear-drift share |
| `ref/ladder_mm_head.py` | the committed (HEAD) ladder_mm. The working-tree copy was changed while I ran this (see the warning below). |
| `strategies/marek/robust_ladder.py` | ladder_mm plus the guards recommended at the end |

Base tapes for the stress tests are walk-forward folds wf03 and wf07 (1,200 s each). The plain-ladder baseline on them is 42,495 and 44,733.

> **Warning: the working-tree `strategies/peter/ladder_mm.py` currently has `cap: 1000`.** It is an uncommitted edit, made at 15:24 by someone else. With it, ladder_mm builds an inventory of ±513 and pays an 11.0M breach penalty on `wf03_pscale3`/`pscale4`, finishing at −10.9M. Do not submit that file. The committed `cap: 100` version does not breach on any tape I tried.

## 1. Drift dependency: none

Decomposing the full round1 run (`logs/r1_ladder_mm`, identity checked: spread + carry = 425,689 vs a reported 425,639):

| component | PnL | per lot |
| --- | --- | --- |
| spread capture vs the pre-fill mid | +1,336,200 | +0.505 ticks |
| inventory carry ∫ inv · dmid | −910,510 | −0.344 ticks |
| …of which the linear 10,000 → 10,030 drift | **≈ 0** | |
| total | 425,689 | +0.161 ticks |

- The time-average inventory is −0.0. Inventory is at ≥ +90 for 28.1% of the time and at ≤ −90 for 27.7%. It is pinned at \|100\| but on both sides equally, and the carry is the same long (−454k) and short (−456k).
- The carry is adverse selection, not trend: the maker markout is +0.31 ticks/lot at 0.1 s and +0.16 at 1 s, then flat at +0.16 out to 300 s. PnL is earned in the first second after a fill and kept afterwards.
- **Mirror test.** `augment.py mirror` swaps B/A and reflects prices. Each of the 10 round1 folds gives byte-identical PnL on its mirror (e.g. wf03 42,495 = 42,495), because the strategy and the matching are side-symmetric.
- Imposed trends cost little: `drift100` (100 ticks per 1,000 s) loses 9% on wf03 and 11% on wf07. Even `trendup/trenddown` at 300 ticks per fold (~10× round1's total move per fold) keeps 66–79% of the PnL.

## 2. Failure modes (plain ladder_mm, cap 100)

| tape | ladder PnL | vs base | max DD | notes |
| --- | --- | --- | --- | --- |
| round1 full | 425,639 | | 827 | trailing 60 s PnL never below −418; trailing 300 s never below +5,435 |
| **random_stress** (33 min) | **−18,561** | loss | 20,145 | 74% of 60 s windows negative; markout +0.21 at 1 s, −0.16 at 60 s |
| **sample** (60 min) | **−10,399** | loss | 12,318 | informed trending flow: markout −2.8 at 1 s, −7.9 at 10 s |
| **vscale0.3** (thin book, touch ~400) | 6,824 / 6,682 | **−84%** | 2,199 | our 100 lots are a large share of the touch, so sweeps go through us |
| noise3 / all3 | 27–33k | −27 to −31% | ≤1,217 | |
| trendup / trenddown (300 ticks per fold) | 29.7–35.2k | −21 to −34% | ~1,000 | |
| tscale0.25 (events 4× denser vs 5 ms latency) | 32.5k / 36.5k | −18 to −23% | | |
| burst (±40-tick jumps) | 37.6–38.0k | −10 to −16% | | |
| takers, toxic (informed IOC, up to 3/s), shocks, flicker, makers | +0 to +31% | gain | | injected flow on a replayed tape cannot move the price, so extra takers just add fills |
| bigtaker2 / bigtaker4 | 84–88k / 143–145k | +100 to +240% | | |
| pscale2 / 3 / 4 (wider spreads and bigger moves) | 85k / 127k / 170k | scales up | 1.4k / 2.2k / 3.0k | |
| sample_v2 | 263,512 | | 200 | |

- **No breach penalty** on any of ~80 tapes with the committed cap=100.
- **`flat_at/flat_to` is broken.** With `flat_at 100, flat_to 50` it breaches on every tape: 16–17k penalty per round1 fold and 55k on sample_v2. The flatten IOC is left out of the `_room` guard. Never enable it as it stands.
- Other existing knobs on the probe set:
  - `imb_pull 0.2/0.35` and `tight_offset 1` lose 40–70% on round1 folds and do not fix random_stress.
  - A fixed `cap 50` loses ~30% on round1 and sample_v2. It does fix random_stress (−1.5k; `cap 25` gives +1.0k) and doubles the thin-book PnL (13.4k).

In one sentence: the strategy loses money where the tape **trends at the 10–60 s scale** (variance ratio > 1, i.e. maker fills are adversely selected after they are made), and where **its own size is large relative to the touch**. It is not sensitive to the direction of drift, to the tick scale, or to random noise.

## 3. Round1 compared with the other tapes: what the private tape and later rounds may look like

| | round1 | sample_v2 | random_stress | sample |
| --- | --- | --- | --- | --- |
| events / s | 368 | 348 | 100 | 15 |
| spread (time-weighted) | 1.07 (93% at 1 tick) | 1.59 (41% at 1) | 2.7 (46% at 1) | 2.0 |
| touch size (per side) | 1,316 | 818 | 461 | 25 |
| mean IOC size | 142 | 94 | 21.5 | 12.8 |
| σ at 1 s (ticks/√s) | 0.92 | 0.50 | 1.08 | 2.57 |
| variance ratio VR(10 s) / VR(60 s) | 1.00 / 1.04 | 1.11 / 1.19 | 1.62 / 2.19 | 2.44 / 1.55 |
| taker markout at 1 s / 10 s | −0.31 / −0.31 | −0.80 / −0.80 | −1.22 / +0.21 | +2.06 / +8.15 |
| ladder PnL | 425k / 200 min | 263k / 96 min | −18.6k | −10.4k |

- Round1 is an unusually friendly tape for a max-size joiner. It is a pure random walk at every horizon (VR ≈ 1.0), uninformed takers lose 0.31 ticks for good, the book is deep, and the series is flat: every 120 s bin has the same event rate, spread and σ.
- sample_v2 is similar but has mild momentum (VR 1.19 at 60 s), a thinner book and wider spreads. The ladder still makes 2,750/min there.
- The private round1 tape is most likely the same generator with another seed, so expect round1-like numbers. "Rounds get harder" most plausibly means some mix of:
  - informed or trending flow (the random_stress/sample regime)
  - thinner books and smaller touches
  - wider spreads and higher volatility
  - regime switches inside the tape

  The first two are exactly the ladder's failure modes.

## 4. Safety mechanisms tested: `strategies/marek/robust_ladder.py`

The quoting is ladder_mm's (levels=1). Parity check: with `kill=0, depth_k=0` it reproduces ladder_mm exactly (wf03 42,495, random_stress −18,561, sample −10,399). Three guards are added on top:

1. **Equity kill switch.** Equity = cash + pos·mid, sampled each second.
   - Trigger: equity falls more than `ks_dd` = 1,500 below its trailing 60 s peak, or the trailing 300 s PnL is negative.
   - On a trigger, the cap goes to 20 for 120 s.
   - A re-trigger within 900 s doubles the cooldown (max 1,800 s) and halves the defensive cap (20 → 10 → 5 → … → 0).
   - `ks_dd` scales with max(1, σ̂₁ₛ / 0.92), so wider-tick tapes do not trip it on noise. Without that scaling it cost 23% on pscale3 and 64% on pscale4.
   - It is cheap because round1's equity curve never shows a negative 300 s window.
2. **Depth-scaled cap.** cap ≤ max(25, 0.1 × EWMA of the per-side touch quantity, 60 s half-life). On round1 (touch ~1,300) it never binds. At touch ~400 it caps at ~40.
3. **Shrink with `modify_order`.** When the effective cap drops, resting adding-side orders are shrunk instead of cancelled, so they keep queue priority. The guard counts every live order in full (cancel, modify and IOC included) because those can lose the race against a fill.

Also implemented but **off**:
- A variance-ratio regime guard (`vr_max`). It added nothing on top of the kill switch.
- `ks_flatten`, which IOCs the position down to the defensive cap, with the IOC counted in the guard.

Final defaults vs plain ladder (`results/final.csv` vs `results/ladder_stress.csv`):

| tape | ladder | robust_ladder | Δ |
| --- | --- | --- | --- |
| **round1 full** | 425,639 | **425,613** | −0.01% |
| round1 10 folds (sum) | 425,118 | 416,746 | −2.0% (fresh state each fold; wf02 −5%) |
| sample_v2 | 263,512 | 251,312 | −4.6% |
| random_stress | −18,561 | **−111** | +18.5k |
| sample | −10,399 | **−3,213** | +7.2k |
| vscale0.3 wf03 / wf07 | 6,824 / 6,682 | **13,483 / 14,019** | ×2.0 / ×2.1 |
| pscale2 / 3 / 4 (wf03) | 84,990 / 127,428 / 169,904 | 85,450 / 128,118 / 170,824 | +0.5% |
| trendup / trenddown (wf03) | 30,566 / 32,145 | 30,866 / 32,675 | +1–2% |
| noise3, all3, tscale0.25 (wf03, wf07) | 27.1k–36.5k | 27.1k–33.4k | −8.5% to +0.1% (tscale0.25 wf07 is the worst) |
| breach penalty, all tapes | 0 | 0 | |

Across 70 stress tapes built from wf03/wf07 (`results/robust_stress.csv`, run with depth_k 0.1 and without σ-scaling, which only matters for pscale ≥ 3), the robust version is within −4% to +1.6% everywhere except all3/tscale0.25 (−4.5% to −8.5%) and the thin book (+100%).

## 5. Recommendations, ranked

1. **Ship the kill switch and the depth-scaled cap (robust_ladder defaults) instead of the plain ladder.** Measured cost: −0.01% on the full round1 tape, −2% on folds, −4.6% on sample_v2. Measured gains: the −18.6k random_stress loss falls to −0.1k, sample's −10.4k to −3.2k, and thin-book PnL doubles. Since the score is a z-score and negative PnL counts as 0, avoiding a losing round matters more than 2% upside.
2. **Revert `cap` to 100 in `strategies/peter/ladder_mm.py`** (the working tree has 1000) and never enable `flat_at`. Both breach the ±100 limit; the breach costs millions on wider-tick tapes.
3. **Keep every guard threshold scale-free.** The absolute `ks_dd = 1,500` falsely triggered at 3–4× tick volatility until it was scaled by σ̂. For later rounds, put a σ or spread normalisation on any other absolute $ or tick constant as well (e.g. `max_spread = 8`, which already stops quoting about one-sixth of pscale4's time).
4. **Re-tune `depth_k` (0.07–0.15) and `ks_dd` once a round-2 tape exists**, by sweeping on its folds. Both are cheap knobs. `depth_k 0.07` cost 3–5% on round1 and sample_v2 for a slightly better worst case; 0.1 is the round1 sweet spot.
5. **Use mirror and pscale as regression tests, not as scenarios.** The ladder is exactly mirror-invariant and roughly linear in pscale. The informative stress tapes are random_stress, sample, `vscale0.3`, `tscale0.25` and `trend*`.
6. Not recommended: `imb_pull`, `tight_offset`, a fixed lower cap, or the VR guard. Each either costs 30–70% on round1 or adds nothing over the kill switch.

Caveat: replayed tapes cannot react to us. Injected "toxic" takers cannot push prices, which is why `toxic*` raised PnL, so real toxicity is only represented by tapes whose own price path trends after fills (random_stress, sample). The kill switch is the only guard aimed at that case, and its trigger levels are tuned on four tapes.
