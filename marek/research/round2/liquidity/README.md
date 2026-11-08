# Round 2: the liquidity drop at minute 60, and what to do with it

Written 2026-11-07. Tape `data/round2.csv`. Strategy measured: the `cf` config of `strategies/marek/round2/r2_ladder.py`
(`marek/logs/r2/round2/cf/strategy.py`, +398,177 on round2, the best entry in `r2_ladder/results.jsonl`).

## 1. What the drop is

Depth by minute (`depth_by_minute.csv`, from `depth_sim.py`, market-only book sampled every 0.1 s):

| Minutes | Spread | Touch qty (both sides, p50) | Depth within 3 ticks | Depth within 20 ticks |
| --- | ---: | ---: | ---: | ---: |
| Typical calm minute | 1.05–1.1 | 1,900–2,200 | 4,700–5,100 | 4,800–5,300 |
| 60–61 | 1.6–2.1 | 470–1,000 | 1,900–3,150 | 3,400–4,150 |
| 64–66 | 1.85–2.1 | 420–540 | 1,950–2,300 | 3,500–4,000 |
| 34–35, 71, 94 (for comparison) | 1.8–2.1 | 420–620 | 1,900–2,450 | 3,600–4,000 |

**It is the volatile regime, not a new mechanic.** 3600–4040 s is the longest volatile stretch on the tape: segments
3631–3736 and 3816–4033 (`s_dynamics/vol_segments.csv`), with shocks at 3617, 3870 and 4021 inside or next to it.
Minutes 34–35, 71 and 94 show the same footprint at the same depth; they are just shorter. Maker insert rate does not
change (`flow/rates_60s.csv`: about 88/s per side throughout). Depth falls because:

- makers quote 2.6× wider (D = 0.23 + Exp(1.1) instead of 0.05 + Exp(0.45)), so the volume spreads over more levels and
  nothing sits within 0.7 of S;
- S moves about 5× more per taker, so the adverse-move cancel term (β·mv) fires much more often;
- takers arrive 1.5× as often and cross about 2.5 ticks.

Since it is the regime, it can't be predicted ahead (dwell times are geometric). It is detected about 1 s after it starts,
and `vol_on` already does that.

## 2. We already capitalise on it, heavily

| Window | cf PnL | Per second |
| --- | ---: | ---: |
| Whole tape | 398k | 66 |
| Calm (`run.py` split) | 162k | ≈ 35 |
| **3600–3740** | **31.2k** | **223** |
| 3740–3815 (calm gap) | 2.8k | 38 |
| **3815–4040** | **62.2k** | **277** |
| All volatile segments | 183k vol + part of 53k shock | ≈ 280 |

Volatile time earns about 8× calm per second. Minutes 60–67 alone are a quarter of the tape's PnL. Per segment, the
first 2 s cost −6.5k in total and the 5 s after the exit −3.7k. Detection lag is therefore cheap. Long segments earn
270–310/s, so **PnL scales almost linearly with volatile duration**.

Where the volatile PnL comes from (`fills.py`; markouts against the exact S in `s_dynamics/s_path.parquet`):

| Volatile fills | Lots | Edge vs S before the event | Markout +1 s | Markout +30 s |
| --- | ---: | ---: | ---: | ---: |
| **IOC snipes (we take)** | 97.8k | +72.9k | **+192.8k** | +232.8k |
| Passive (we make) | 24.5k | +42.9k | −1.8k | −14.0k |

**The volatile money is almost entirely stale-quote sniping.** Passive quoting in the thin book is about breakeven.

## 3. What limits it: the ±100 cap, not the opportunity

Stale episodes in volatile time (`episodes.py`, threshold 1.0 tick through the true S, fragments within 300 ms merged):

- **850 events** in 793 s of volatile time, about 1.07/s.
- At the peak, the stale quantity has a **median of about 450 lots** at about 1.6 ticks of edge per lot.
- Total stale value: **about 1.8M tick·lots**, against about 190k that we capture.
- **Position is pinned:** |pos| ≥ 90 for 83 % of volatile time, against 56 % in calm.

| Room in the event's direction at its start | Events | Lots captured (mean) |
| --- | ---: | ---: |
| ≤ 5 (pinned the wrong way) | 350 | 0 |
| 5–100 | 196 | 35–71 |
| 150–200 (pinned the right way) | 273 | 129 |

This means three things:

1. **Inventory rebalancing will not help.** The stale quantity (≈ 450) exceeds the total room (200 = buy room + sell
   room), and event direction is a coin flip. Expected capture per event is therefore about 100 lots in every state:
   pinned gives ½ × 200, flat gives 100. Flattening only adds cost. This is round 1's "pinning is the edge, not a
   leak", carried over to sniping.
2. **When pinned the right way we left room unused.** The mean capture is 129 of about 200 lots, because `_snipe` caps each IOC at
   `s_take_max = 100` and waits for the first IOC to come back. **Measured:** `s_take_max = 200` gives

   | Tape | cf (100) | 200 | Δ |
   | --- | ---: | ---: | ---: |
   | round2 | 398,177 | 402,989 | +4.8k |
   | r2_v2_seed1 | 507,941 | 512,921 | +5.0k |
   | r2_v2_seed2 | 512,222 | 516,223 | +4.0k |
   | r2_v2_seed4 | 340,073 | 343,680 | +3.6k |

   All four are positive, with no breaches. It is a one-line change (only the parameter has been varied, never the code).
3. **Misses with room:** 114 of the 470 events where room was ≥ 50 got under 10 lots. Their capturable value is about
   21k. In those events our S is on the wrong side: the median of S_ours − S_true in the event direction is −1.5 ticks,
   against +0.8 in the events we did capture. A quarter of them also fall outside the detected regime.

**The position limit is a hard rule** (`rules.md`). The local simulator enforces it as a 10k/s breach penalty with a
1000-lot potential limit, but do not trade on that.

## 4. Plan of action (ordered by expected value per hour)

| # | Change | Expected | Test |
| --- | --- | --- | --- |
| 1 | **`s_take_max` 100 → 200** (or `= 2·cap`) | **+4–5k per tape, measured** | Done on 4 tapes; add to defaults and re-run the 6-synth + round2 set |
| 2 | **Volatile passive side: flatten-only.** Passive fills with true edge < 1.5 lose about 16k; ≥ 1.5 make about 14k. Try (a) no adding-side passive in volatile, (b) a larger `vol_dist` on the adding side only, (c) a smaller `vol_dist` on the flattening side (it resets room for free). | 0 to +15k | Paired per-segment deltas (`r2_ladder/seg_delta.py`) |
| 3 | **Volatile S estimate.** The miss analysis says S error, not room, decides the remaining misses. Needs ground truth first: have `r2_synth.py` also write S per insert into `.truth.json` (owner: the other session), then fit `s_gap_vol` / `s_c_vol` / `s_exp_vol` against it rather than against PnL. GA `g3_a` (`s_gap_vol = 0.7`) already found +10k this way. | +10–30k | S error sd in volatile on synth truth; then PnL |
| 4 | **Snipe threshold by edge rank.** With room scarce, every lot used at 1.0 edge blocks a later 3.0 edge. Make `vol_take` depend on remaining room (higher when room is small). Snipes at e_now 1.5–2 had −72k at +30 s, so check that band specifically. | 0 to +10k | Grid on synth seeds |
| 5 | Faster regime entry: the insert-offset tell (no maker insert within 0.70 of S) instead of the 1 s spread EMA | ≤ +6k (first 2 s total −6.5k) | Low priority |

**Not worth doing:** inventory targeting or flattening in volatile time (see §3.1), predicting the regime switch
(geometric dwell), and anything that exceeds ±100.

**Risk note for the private tape:** volatile time is random (Markov), and PnL scales with it. The synth seeds range from
340k to 516k with the same strategy. That variance is fine for the z-score, since every tape is positive. Don't tune
anything to the specific minute 60–67 stretch.

## Files

| File | What |
| --- | --- |
| `depth_sim.py` | Replays a tape with a recorder strategy and samples the top-20 book every 0.1 s → `.npy` |
| `depth_by_minute.csv` | Spread and depth per minute for round2 |
| `fills.py` | Splits our fills by regime and role (maker/taker), with markouts against exact S. Needs a full-log run of `cf` in `cache/cf_logs/` |
| `episodes.py` | Runs a strategy and records every stale-quote episode (touch through true S by `thr`) with our position, our S error and detector state. `uv run --with pyarrow python marek/research/round2/liquidity/episodes.py <strategy.py> <out.pkl> 1.0` |
