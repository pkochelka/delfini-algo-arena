# queue_mm hypotheses

Hypotheses for `queue_mm.py` (a copy of `ladder_mm.py` with new flags), with what was tested and the result.
Add new ones under **Open**. Every flag stays in PARAMS (off by default unless marked adopted).

**Data:** round1 only. Folds in `data/splits_r1/wf`. Robustness set: wf03, wf07 and wf11, each under one
`augment.py` attribute change at a time, in `data/splits_r1/aug`:
`orig pscale2 tscale0.25 tscale4 vscale0.3 vscale3 drift30 noise1 takers3 shocks3`.
`mirror` gives exactly the orig PnL (the strategy is symmetric), so it was dropped.

**Score:** mean PnL over the 3 folds per tag. The tables report MEAN over tags and WORST tag.
Each fold is a whole file traded from t=0 (no trade_start).

**Tools:** see [Tools](#tools) at the end.

**Baseline (v1):** ladder_mm tuned (`size_top` 99, `size_deep` 50, `levels` 2, `topup` 6, `skew_lots` 100).
Robustness MEAN +30.6k, WORST +16.1k (vscale0.3). The 12 folds average +27.7k.

## Facts established (round1, wf09, baseline params)

**Markout averages over fills** (ticks/lot): +0.52 at the fill, +0.34 after 0.1 s, +0.18 after 1 s.
Adverse selection eats about 2/3 of the half-spread.

**By market qty left on our side after the fill:**

| Share left | Share of lots | Markout 1 s |
| --- | --- | --- |
| < 5% | 12% | −0.24 |
| > 80% | 20% | +0.50 |

**By queue position before the fill** (market qty ahead of our order):

| Ahead of us | Share of lots | Markout 1 s |
| --- | --- | --- |
| 0 (we're first) | 37% | +0.36 |
| 1–100 | 49% | +0.23 |
| 100–300 | 10% | +0.17 |
| 300–1500 | small | +0.03 … −0.04 |

**By order age at the fill:**

| Age | Markout 1 s |
| --- | --- |
| < 0.1 s | +0.11 |
| 0.1–0.5 s | +0.16 |
| 0.5–2 s | +0.35 |
| 2–10 s | +0.32 |

**Takeaways:**
- The edge comes from **queue seniority**: old orders that market cancels have pushed toward the front.
- PnL is **capacity-bound**: it scales almost linearly with `cap` (60 gives 21.0k, 100 gives 30.6k).
  A cap of 200 breaches, about −3.8M in penalties.

## Tested

| # | Hypothesis | Flag | Result | Verdict |
| --- | --- | --- | --- | --- |
| 1 | Pull a side when its share of the touch qty is small (thin side = toxic fills) | `imb_pull` 0.1 / 0.2 / 0.3 | 12 folds: 27.7k becomes 20.5k / 13.8k / 8.7k | Rejected: cancels destroy queue seniority |
| 2 | Rest deeper (more levels, start behind the touch) to be early in line when the price arrives | `levels` 3–4, `tight_offset` 1, `size_top` 0 | All worse: 8.5k–23.8k vs 27.7k | Rejected: the touch is the place |
| 3 | Cancel only the touch order when its level is about to vanish (market qty < X, other side ≥ ratio·X) | `vanish_qty` 30 / 50 / 150 | MEAN 30.5k / 30.2k / 28.0k vs 30.9k; thin book +0.8k | Rejected: small loss, the level often refills |
| 4 | Step inside a 2-tick spread to be first in a fresh level (thick / thin / reducing side) | `improve` 1 / 2 / 3, `improve_reduce` | MEAN 28.4k / 20.0k / 20.6k / 20.6k; pscale2 down to −34k … −38k | Rejected: being first in a fresh level is the toxic case |
| 5 | Grid search of sizes and topup across tags | `size_top`, `size_deep`, `levels`, `topup` | One level × 200: pscale2 +57k, vscale3 +45k, vscale0.3 +4.9k. Two levels 99/50: thin +16k, pscale2 +45k. `topup` 20 slightly beats 6 (MEAN 30.8k) | No single setting wins every tag |
| 6 | "One level behind" should follow the market's price grid, not 1 tick (on a 2-tick grid a deep order becomes a fresh, toxic level) | `grid_unit` 0 = learn the most common gap between occupied levels | pscale2 45.0k becomes **59.5k**, other tags unchanged. MEAN 32.3k, WORST 16.3k | **Adopted** |
| 7 | Size the touch order from the EWMA market taker size (thick books want more) | `size_k` 0.6 / 0.8 / 1.0 | MEAN 31.7k / 31.2k / 30.2k; vscale3 37.2k becomes 31.8k | Rejected |
| 8 | Drop the deep level when our cap is a small slice of the touch (EWMA touch > 25·cap) | `deep_ratio` 25 | vscale3 37.2k becomes 32.3k (worse); others unchanged | Rejected as is: the thick book wants one level **and** size 200 (`size_top` 200 alone gives vscale3 44.9k but hurts everywhere else) |

| 8b | Regime switch on touch/cap. Thick book (EWMA touch > 25·cap): one level × 200 lots. Otherwise 99/50 on 2 levels | `deep_ratio` 25, `thick_size` 200 | vscale3 **45.3k** (from 37.2k), other tags unchanged. On orig/pscale2/vscale0.3/vscale3/takers3: MEAN 36.6k vs 32.2k without the switch, WORST 16.3k. `deep_ratio` 15 is slightly worse (orig touch ≈ 13·cap, too close) | **Adopted** |

| 9 | Thick-book threshold: where between 1× and 3× depth does one level start winning? (orig touch ≈ 13·cap, 1.7× ≈ 22·cap, 2× ≈ 26·cap) | `deep_ratio` 15 / 18 / 25 / 35; new augments vscale0.6/1.7/2 | vscale1.7: 18 gives **37.6k** vs 33.6k at 25. vscale2: 40.2k vs 39.0k. 12 real folds: 18 is identical to 25 (never switches); 15 switches on round1 (mean 27.5k vs 27.6k) | **Adopted 18** |
| 10 | Thin book (worst tag): more levels, other sizes or topup when the touch is small | `levels` 3, `size_top`/`size_deep` 60/60, 50/100, 40/40×3, `topup` 5/40 on vscale0.3/0.6, pscale2_vscale0.3, vscale0.3_takers3 | All worse than the defaults (MEAN 22.9k vs 18.1–22.7k) | Rejected. The thin book's PnL is low because 0.3× volume means 0.3× flow: per lot of market aggressive volume it earns about 0.060 vs 0.033 on orig |
| 11 | Don't *send* new touch orders on the thin side (cancel nothing, so no seniority is lost) | `gate_share` 0.1 / 0.2 / 0.3, `gate_all` | MEAN 45.5k becomes 36.9k / 34.2k / 31.1k / 34.0k on orig/pscale2/vscale0.3/vscale3/takers3/pscale2_vscale3 | Rejected: a new order on a thin side is near the front of a short queue, so those fills are good |
| 12 | Split the touch order into chunks so a partial fill keeps the senior remainder | (none) | Not run: price-time priority already keeps a partially filled order's remainder at the front, so splitting only adds messages | Dropped (no mechanism) |

**Combined shifts** (new augments, defaults vs the baseline params):

| Tag | Baseline | queue_mm |
| --- | --- | --- |
| pscale2_vscale0.3 | 17.5k | **32.4k** |
| pscale2_vscale3 | 62.4k | **90.4k** |
| vscale0.3_takers3 | 17.6k | 17.8k |
| tscale4_vscale3_drift30 | 36.8k | **44.5k** |

The adaptations add up when shifts combine; nothing broke.

**Pattern across #1, #3, #4, #11:** every attempt to be *selective* about fills (pull, gate, step inside the spread) lost.
In round1-like markets, fill quantity at senior queue positions beats fill selectivity. A new filter idea
should only remove fills, never reset or block queue positions, and should be checked against that.

## Current defaults (adopted: #6, #8b with `deep_ratio` 18, `topup` 20)

**Validation on round1 itself:** this market keeps the same settings, so expect a tie.

| Run | Baseline | queue_mm |
| --- | --- | --- |
| 12 walk-forward folds | mean +27,724 (sd 1,083) | mean +27,630 (sd 1,100) |
| Held-out test 9000–12000 s (whole file, trade_start 9000 s) | +117,290 | +117,072 |

The gains are on the attribute-shifted markets (pscale2 +14.5k, vscale3 +8k vs the baseline).

## Open

- [ ] With `deep_ratio` 18, the 3 robustness folds traded from t=0 show orig 29,716 vs 29,795 at 25. The 12 real folds (traded after a 60 s warm-up) are identical. The touch EWMA may switch early while it is still warming up: require a minimum `n_touch` before switching?
- [ ] Shifts not yet covered by any augment: different aggressive-size distributions (bigger sweeps without a thicker book), a different cancel rate (how fast queues turn over, which drives seniority), and wider spreads with the same tick grid.

## Tools

All in `strategies/peter/`, run from the repo root:

| Script | What it does |
| --- | --- |
| `robust_eval.py strategy.py '{...}' ... [--tags a,b] [--folds wf03,wf07,wf11]` | The robustness table above (tags = augment file suffixes, `orig` = the plain fold) |
| `fold_eval.py strategy.py '{...}' ...` | The 12 walk-forward folds with trade_start |
| `test_eval.py [--strategy path] '{...}' ...` | Held-out test 9000–12000 s (default strategy queue_mm) |
| `fill_probe.py strategy.py data.csv out.npz ['{...}']` + `fill_markout.py out.npz` | Per-fill markouts by queue position / age / share |
| `augment.py ... --only <tags>` | New presets: vscale0.6/1.7/2 and the combined tags |
