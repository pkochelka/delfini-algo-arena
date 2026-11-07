# Round 1: predictive signals and fill selection for ladder_mm

**Result: `strategies/marek/sig_ladder.py` (defaults) = 473,066 on wf01-10 vs ladder_mm 425,118 (+47.9k, +4.8k per fold, paired fold-delta sd 1.5k, 10/10 folds); full tape 473,597 vs 425,639 (+48.0k).** The signal is the touch imbalance, but it only pays when restricted to an *old level being eaten* and acted on by shrinking (modify_order) only the part of the order that would flip the position, while pre-joining the next price level with the rest.

## Setup

- Baseline: `strategies/peter/ladder_mm.py` defaults (one big order per side at the market touch,
  queue priority kept, cap 100, top-up behind). Walk-forward folds `data/splits_round1/wf/wf01-10`:
  total 425,118 (reproduced bit-for-bit by `strategies/marek/sig_ladder.py` with every overlay off).
- Fill-level analysis: the full-tape run `logs/r1_ladder_mm` (425,639, 2,648,147 lots in 51,541 fills).
  `research/round1/signals/build_fills.py` rebuilds the *simulated* market-only book from
  `book_deltas.csv` (our own orders excluded, as the strategy sees the touch) and attaches features
  taken at **t_fill - 5 ms** (the last state a modify/cancel could still act on) plus market-mid
  markouts after the fill. `fill_buckets.py`, `fill_2d.py`, `race.py` print the tables below.
- A/B: `research/round1/signals/sweep_sig.py` (3 workers, PARAMS overrides of `sig_ladder.py`,
  every run appended to `research/round1/signals/results.jsonl`). Deltas are paired per fold vs
  the baseline; "sd" is the sd of the 10 paired fold deltas, so the standard error of a *total*
  delta is about sd * sqrt(10) ~ 6k. With ~30 variants tried, a total delta below ~15k (or fewer
  than 8/10 fold wins) is noise.

Markouts: lot-weighted, ticks, (market mid at t + h - fill price) in our direction. All fills
average +0.33 (100 ms), +0.16 (1 s, 5 s, 30 s, 120 s).

## 1. Which ladder fills lose, and can we see it 5 ms before?

Full-tape run, features at t_fill - 5 ms (`fill_buckets.py`). "share" = market qty at our touch /
(market qty at both touches), i.e. the queue imbalance seen from our side.

| our-side share | lots | % lots | mo 1 s | mo 30 s | PnL @30 s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0-0.05 | 332,629 | 12.6 | -0.156 | -0.208 | -69k |
| 0.05-0.1 | 172,661 | 6.5 | -0.110 | -0.221 | -38k |
| 0.1-0.2 | 277,087 | 10.5 | +0.018 | -0.009 | -3k |
| 0.2-0.3 | 247,365 | 9.3 | +0.073 | +0.129 | +32k |
| 0.3-0.5 | 454,835 | 17.2 | +0.147 | +0.098 | +45k |
| 0.5-0.7 | 437,036 | 16.5 | +0.206 | +0.307 | +134k |
| 0.7-0.9 | 423,698 | 16.0 | +0.351 | +0.337 | +143k |
| 0.9-1.0 | 302,836 | 11.4 | +0.545 | +0.611 | +185k |

The touch imbalance is by far the strongest single predictor: the 19% of lots filled while our side
held < 10% of the touch lose ~107k at 30 s, while the rest earn ~536k. Other features, alone:

- market qty at our touch <= 50 lots (we are nearly alone at the level): 9.4% of lots, -0.25 @30 s.
- level depletion, touch qty now / 200 ms earlier (same price) < 0.25: 6.1% of lots, -0.27 @30 s.
- taker flow against our side over the last 100 ms > 50 lots: 9% of lots, -0.24..-0.45 @30 s; over
  the last 1 s > 0: -0.04..-0.21. Flow in our favour: +0.19..+0.69.
- spread 2 ticks at t-5ms: 3.2% of lots, +0.58 (good fills, nothing to fix).
- microprice - our price (our direction) < 0.25: 34% of lots, -0.10 @30 s (same signal as share).
- short-horizon vol (mid changes in the last 1 s): flat, no information.
- taker size (not known at quote time): > 800 lots -0.31/-0.73 @1 s (see impact_mm docstring).
- age of the touch level (ms since that price became the touch) < 50 ms: 3.5% of lots, -0.29 @30 s.

Lot-weighted OLS of the 1 s markout (`fill_2d.py`, R^2 0.09): share +0.70/unit, flow over 1 s
against us -0.16 per 1000 lots, flow over 100 ms -0.09 per 100 lots, depletion (< 0.5 in 200 ms)
-0.10, spread 2 +0.28; deeper-book share, touch size and vol add nothing once share is in.
At 30 s R^2 is only 0.004 (inventory noise), but the coefficients keep their signs.

Interaction that matters for the strategy (`race.py`): split thin-side fills by how much the
touch has been eaten since its price became the touch (qty / peak qty at that price):

| share | qty / peak | lots | mo 1 s | mo 30 s | PnL @30 s |
| --- | --- | ---: | ---: | ---: | ---: |
| < 0.1 | <= 0.3 (old level being eaten) | 230,978 | -0.263 | -0.332 | -77k |
| < 0.1 | 0.3-0.8 | 73,157 | -0.16 | +0.11 | +8k |
| < 0.1 | > 0.8 (fresh level being built) | 201,155 | +0.006 | -0.195 | -39k |
| 0.1-0.2 | <= 0.3 | 95,725 | -0.096 | -0.238 | -23k |
| >= 0.35 | any | 1,500,481 | +0.22..+0.32 | +0.13..+0.41 | +470k |

A thin side at a *fresh* level (a new price just became the touch and is still filling up) is
where the ladder earns its queue priority: it is first in line at the new level. A thin side at an
*old* level that has been eaten down to < 30-50% of its peak is the level about to break.
That distinction is what makes a guard work (section 3); the plain share guard (imb_pull) fires at
every fresh level and so throws the priority away.

## 2. Does the signal survive 5 ms latency and the cancel race?

Yes. All features above are measured at t - 5 ms. Splitting the thin-side fills (share < 0.1 just
before the fill) by how long the side had already been thin:

| thin for | lots | mo 1 s | mo 30 s |
| --- | ---: | ---: | ---: |
| < 5 ms (cannot be dodged) | 11,813 | +0.04 | -0.27 |
| 5-20 ms | 37,708 | -0.18 | +0.23 |
| 20-100 ms | 183,113 | -0.08 | -0.34 |
| 100-1000 ms | 255,088 | -0.18 | -0.17 |

Only 2% of the thin-side lots arrive within the 5 ms race window; the state is visible tens to
hundreds of ms ahead. The tape-level check (data_explorer round1, "Is the imbalance edge still
there 5 ms later?") agrees: the imbalance -> 100 ms move curve is unchanged when measured from
t + 5 ms (0.04: -0.162 -> -0.160; 0.96: +0.157 -> +0.153). The cost of acting on the signal is
not the race, it is the queue priority given up (section 3).

## 3. Strategy variants (strategies/marek/sig_ladder.py)

`sig_ladder.py` = ladder_mm core (bit-identical with every overlay off: 425,118 on wf01-10) plus a
per-side guard. A side is *unsafe* when

    share = market qty at its touch / market qty at both touches  <  g_share      (spread 1 tick only)
    and  (g_dep > 0)  market qty at its touch  <  g_dep * peak qty at that price since it became the touch

released when share >= g_share + g_hyst (0.05) or the level is no longer depleted, or the touch
price changes (a new level resets the peak, so a fresh level is never unsafe). While unsafe:

- `g_mode` shrink/cancel: modify_order the oldest touch order down to g_keep lots (0 = cancel all),
  no new orders or top-ups on that side; `freeze`: keep orders, no new ones / top-ups.
- `g_keep_flat=1`: keep min(resting, |pos|) on the side that reduces inventory (shrink the
  oldest order with modify_order, so the kept part keeps its queue priority); only the part
  that would flip the position through zero is removed.
- `pre_join=1`: rest the freed room one tick *behind* the touch (the price that becomes the touch if
  the level breaks), so we are early in that queue. It is cancelled when the guard releases
  (`pj_hold=1` keeps it instead: much worse, it blocks the top-up at the touch).

### A/B on wf01-10 (all runs, `summarize.py`; delta = paired vs ladder_mm 425,118; t = mean/(sd/sqrt 10))

Parameter shorthand: s = g_share, dep = g_dep, kf = g_keep_flat, pj = pre_join, h = g_hyst
(default 0.05), flow = flow_k (lots of taker flow against the side over the last 1 s / 1000,
subtracted from share). "shrink" with g_keep 0 is the same as cancel (identical numbers).

| variant | params | total | delta | fold mean +- sd | t | wins | fills |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | - | 425,118 | +0 | +0 +- 0 | +0.0 | 0/10 | 31,327 |
| shrink s.10 | g_share=0.1 | 314,497 | -110,621 | -11,062 +- 2,595 | -13.5 | 0/10 | 16,353 |
| shrink s.05 | g_share=0.05 | 353,534 | -71,584 | -7,158 +- 2,634 | -8.6 | 0/10 | 20,528 |
| shrink s.15 | g_share=0.15 | 270,072 | -155,046 | -15,505 +- 2,013 | -24.4 | 0/10 | 13,364 |
| cancel s.10 | g_share=0.1, g_mode=cancel | 314,497 | -110,621 | -11,062 +- 2,595 | -13.5 | 0/10 | 16,353 |
| freeze s.10 | g_share=0.1, g_mode=freeze | 300,819 | -124,299 | -12,430 +- 1,897 | -20.7 | 0/10 | 21,522 |
| keepflat s.10 | g_share=0.1, g_keep_flat=1 | 329,878 | -95,240 | -9,524 +- 2,248 | -13.4 | 0/10 | 20,795 |
| shrink q50 | g_qty=50 | 376,262 | -48,856 | -4,886 +- 2,036 | -7.6 | 0/10 | 24,121 |
| prejoin s.10 | g_share=0.1, pre_join=1 | 406,689 | -18,429 | -1,843 +- 2,823 | -2.1 | 1/10 | 19,018 |
| s.10 dep.5 | g_share=0.1, g_dep=0.5 | 422,693 | -2,425 | -242 +- 2,561 | -0.3 | 3/10 | 23,564 |
| s.20 dep.5 | g_share=0.2, g_dep=0.5 | 377,618 | -47,500 | -4,750 +- 2,973 | -5.1 | 1/10 | 19,500 |
| s.10 dep.3 | g_share=0.1, g_dep=0.3 | 429,574 | +4,456 | +446 +- 2,035 | +0.7 | 4/10 | 24,290 |
| s.05 dep.3 | g_share=0.05, g_dep=0.3 | 436,623 | +11,505 | +1,150 +- 1,667 | +2.2 | 7/10 | 26,646 |
| keepflat s.10 dep.5 | g_share=0.1, g_dep=0.5, g_keep_flat=1 | 441,015 | +15,897 | +1,590 +- 1,854 | +2.7 | 8/10 | 27,769 |
| prejoin s.10 dep.5 | g_share=0.1, g_dep=0.5, pre_join=1 | 443,020 | +17,902 | +1,790 +- 2,796 | +2.0 | 7/10 | 25,163 |
| freeze s.10 dep.5 | g_share=0.1, g_dep=0.5, g_mode=freeze | 420,388 | -4,730 | -473 +- 620 | -2.4 | 2/10 | 29,529 |
| kfpj s.10 dep.5 | g_share=0.1, g_dep=0.5, g_keep_flat=1, pre_join=1 | 459,792 | +34,674 | +3,467 +- 1,879 | +5.8 | 10/10 | 29,427 |
| kfpj s.05 dep.3 | g_share=0.05, g_dep=0.3, g_keep_flat=1, pre_join=1 | 454,382 | +29,264 | +2,926 +- 1,232 | +7.5 | 10/10 | 30,235 |
| kfpj s.15 dep.5 | g_share=0.15, g_dep=0.5, g_keep_flat=1, pre_join=1 | 453,942 | +28,824 | +2,882 +- 2,103 | +4.3 | 10/10 | 28,444 |
| kfpj hold s.10 dep.5 | g_share=0.1, g_dep=0.5, g_keep_flat=1, pre_join=1, pj_hold=1 | 385,388 | -39,730 | -3,973 +- 3,782 | -3.3 | 1/10 | 24,225 |
| kfpj always (control) | g_share=1.01, g_hyst=0.0, g_keep_flat=1, pre_join=1 | 191,714 | -233,404 | -23,340 +- 2,751 | -26.8 | 0/10 | 6,900 |
| kfpj s.10 nodep | g_share=0.1, g_keep_flat=1, pre_join=1 | 417,829 | -7,289 | -729 +- 2,128 | -1.1 | 2/10 | 24,012 |
| kfpj s.10 dep.7 | g_share=0.1, g_dep=0.7, g_keep_flat=1, pre_join=1 | 458,428 | +33,310 | +3,331 +- 1,818 | +5.8 | 10/10 | 29,554 |
| kfpj s.20 dep.5 | g_share=0.2, g_dep=0.5, g_keep_flat=1, pre_join=1 | 450,517 | +25,399 | +2,540 +- 2,225 | +3.6 | 10/10 | 27,641 |
| kfpj s.10 dep.5 h.10 | g_share=0.1, g_dep=0.5, g_keep_flat=1, pre_join=1, g_hyst=0.1 | 457,412 | +32,294 | +3,229 +- 1,739 | +5.9 | 10/10 | 29,097 |
| kfpj s.10 dep.5 h.0 | g_share=0.1, g_dep=0.5, g_keep_flat=1, pre_join=1, g_hyst=0.0 | 461,422 | +36,304 | +3,630 +- 1,615 | +7.1 | 10/10 | 29,829 |
| kfpj s.10 dep.5 flow.05 | g_share=0.1, g_dep=0.5, g_keep_flat=1, pre_join=1, flow_k=0.05 | 464,216 | +39,098 | +3,910 +- 1,808 | +6.8 | 10/10 | 29,042 |
| kfpj s.10 dep.5 anyspread | g_share=0.1, g_dep=0.5, g_keep_flat=1, pre_join=1, g_spread1=0 | 458,830 | +33,712 | +3,371 +- 1,854 | +5.8 | 10/10 | 29,430 |
| kfpj s.20 dep.7 | g_share=0.2, g_dep=0.7, g_keep_flat=1, pre_join=1 | 444,084 | +18,966 | +1,897 +- 2,315 | +2.6 | 8/10 | 27,469 |
| kfpj s.10 dep.5 h.0 flow.05 | g_share=0.1, g_dep=0.5, g_keep_flat=1, pre_join=1, g_hyst=0.0, flow_k=0.05 | 467,403 | +42,285 | +4,228 +- 1,532 | +8.7 | 10/10 | 29,460 |
| kfpj s.10 dep.5 flow.10 | g_share=0.1, g_dep=0.5, g_keep_flat=1, pre_join=1, flow_k=0.1 | 470,542 | +45,424 | +4,542 +- 1,611 | +8.9 | 10/10 | 28,734 |
| kfpj s.10 dep.5 flow.05 300ms | g_share=0.1, g_dep=0.5, g_keep_flat=1, pre_join=1, flow_k=0.05, flow_ms=300 | 464,916 | +39,798 | +3,980 +- 1,869 | +6.7 | 10/10 | 29,349 |
| kfpj s.10 dep.5 h.0 flow.10 | g_share=0.1, g_dep=0.5, g_keep_flat=1, pre_join=1, g_hyst=0.0, flow_k=0.1 | 473,066 | +47,948 | +4,795 +- 1,478 | +10.3 | 10/10 | 29,087 |
| kfpj s.10 dep.5 flow.20 | g_share=0.1, g_dep=0.5, g_keep_flat=1, pre_join=1, flow_k=0.2 | 471,965 | +46,847 | +4,685 +- 1,650 | +9.0 | 10/10 | 28,204 |
| kfpj s.10 dep.5 flow.40 | g_share=0.1, g_dep=0.5, g_keep_flat=1, pre_join=1, flow_k=0.4 | 472,910 | +47,792 | +4,779 +- 2,034 | +7.4 | 10/10 | 27,373 |

Noise: paired fold-delta sd vs ladder is ~1.5-2.8k, so a total delta has SE ~5-9k; with ~35
variants tried, |delta| < ~15k is noise. Between neighbouring kfpj variants the paired sd is much
smaller (300-900 per fold), so the ordering inside the plateau below is meaningful but small.

Paired vs `kfpj s.10 dep.5` (+34.7k): dep.7 -1.4k, h.0 +1.6k (7/10), anyspread -1.0k, s.05 dep.3
-5.4k, s.15 -5.9k, s.20 -9.3k, flow.05 +4.4k (9/10), flow.10 +10.8k (9/10), h.0 flow.10 +13.3k
(flow.20/.40 equal to flow.10 within 1-2k: plateau from 0.10 up). Flow window 300 ms instead of
1 s: slightly worse (+5.1k vs +4.4k for flow.05, noise).

### What the numbers say

1. **A plain imbalance guard loses badly** (cancel/shrink at share < 0.05/0.10/0.15: -72k/-111k/-155k;
   freeze -124k; alone-at-level q <= 50: -49k). Fills halve. The reason is in section 1: the thin
   side is just as often a *fresh* level, and the ladder earns by being first at fresh levels.
   This is why imb_pull was rejected.
2. **Adding the depletion condition removes the damage**: same guard only while the touch is below
   50% (30%) of its peak at that price: -2.4k (+4.5k / +11.5k at s.05 dep.3) - noise to marginal.
   Without depletion the best combination (kfpj s.10 nodep) is still -7.3k.
3. **How we act matters as much as when.** On the depleted+thin state:
   - cancel everything on the side: about 0;
   - keep the flattening part (modify_order shrink keeps its queue priority, kf): +15.9k;
   - pre-join one tick behind with the freed room (pj): +17.9k;
   - both (kfpj): +34.7k, 10/10 folds, t 5.8; neighbours s.05-0.15 / dep.3-0.7 / h 0-0.10 /
     any spread all +29k..+36k (a plateau, not a lucky point);
   - plus taker flow over the last 1 s in the score (flow_k 0.10) and no hysteresis: **+47.9k**,
     10/10, fold-delta sd 1.5k (t 10). Full tape: 473,597 (+48.0k).
4. Controls: kfpj *always on* (no signal): -233k. Keeping the pre-joined order after the guard
   releases (pj_hold): -39.7k (it blocks the top-up at the touch). So the gain is the signal's
   timing, not the order structure.
5. Fill quality on the full tape, winner vs ladder: lots 2.38M vs 2.65M (-10%), markout +0.20 vs
   +0.16 ticks at 1-30 s; thin-side (share < 0.1) lots 327k vs 505k and their 30 s PnL -46k vs
   -107k. The rest of the book is unchanged.

Why keeping the flattening part works: the ladder sits at |pos| ~ 100 almost all the time
(q50 99), so the reducing side holds ~200 lots = 100 that flatten + 100 that flip. When the level
is about to break we keep the first 100 (at the front, they are mostly filled anyway and only cost
the next-tick move on a flat book) and move the flipping 100 one tick back, where after the break
they are near the front of the new touch at a 1-tick better price; if the level recovers they are
cancelled and topped up at the touch as before.

## 4. Recommendations (ranked)

1. **Adopt `strategies/marek/sig_ladder.py` with its defaults** (= ladder_mm + depletion-gated
   imbalance/flow guard, kf + pj). +47.9k on wf01-10 (10/10 folds, t ~10), +48.0k on the full
   tape, no breaches, same inventory profile. Uses only stdlib (collections.deque) + the allowed
   models; modify_order bookkeeping handles rejected/overlapping modifies (a first version without
   that lost track of an order and breached: if you port this into ladder_mm, keep `mod_n`).
2. If porting piece by piece, the order of value is: depletion gate (g_dep 0.3-0.7) >> keep-flat
   shrink ~ pre-join (each ~+16-18k alone, ~additive) > flow term (+11-13k on top) > hysteresis 0
   (+1.6k, noise-level). Don't use g_share >= 0.2 (-9k vs 0.10) or pj_hold.
3. The thresholds are a broad plateau; no need to tune further on round 1. For later rounds re-check
   the fresh-vs-old split (`race.py`) first: the whole effect rests on "thin + depleted = break".
4. Not worth pursuing (all measured here): cancel/freeze-the-side guards, absolute-qty (alone at
   level) guard, flow-only windows of 300 ms vs 1 s, guarding at 2-tick spreads. Short-horizon
   vol, deeper-book share and touch size carry no extra information at quote time.

## Files

- `strategies/marek/sig_ladder.py` - strategy, all overlays as PARAMS flags (defaults = winner).
- `research/round1/signals/build_fills.py` - sim-book rebuild + per-fill features from a full-log run.
- `research/round1/signals/fill_buckets.py`, `fill_2d.py`, `race.py` - section 1/2 tables.
- `research/round1/signals/sweep_sig.py` - fold A/B (relative to OFF = ladder_mm), `summarize.py`,
  raw results in `results.jsonl`.
- Full-log runs: `logs/r1_ladder_mm` (baseline), `logs/r1_sig_ladder` (winner).
