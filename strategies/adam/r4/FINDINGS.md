# Round 4: momentum and taker-flow overlays (all dead)

Written 2026-11-08. Tape `data/round4_public.csv`, evaluated with `strategies/peter/r4_eval.py`:

- **wf**: the 12 walk-forward folds over the first 75% of the tape. Used for screening and tuning.
- **hold**: the 4 folds after 4500 s. Used only to confirm.

Every file here is a copy of `strategies/marek/round4/r4_best.py` with extra PARAMS. With the new PARAMS at their
defaults, each file is identical to r4_best (wf 1081.7k, hold 278.6k).

**Bottom line:** none of these overlays improves r4_best. The round-4 momentum is already harvested, because in
volatile periods we sit at the +-100 cap on the trend side 91% of the time (Marek's README). Leaning on the momentum,
holding positions longer, flipping less or quoting less all cost more than they gain. Keep r4_best.

## A. Momentum tilt of the IOC take thresholds, turbulent only (`a_tilt.py`, `mom_*`)

The take thresholds `lo_b` and `hi_a` move by `mom_tk_t * m`, where m is clipped to +-`mom_max`. m comes from one of
two sources:

- `mom_src='ema'`: m = F - EMA(F, `mom_hl`).
- `mom_src='kv'`: m = Kalman velocity * `mom_kv_s`.

| Variant (cap 1, hl 1 s) | wf delta | t | won | hold delta |
| --- | ---: | ---: | ---: | ---: |
| ema, gain 0.2 / 0.3 / 0.5 | -15.8k / -16.5k / -25.0k | -1.4 / -1.3 / -1.5 | 6 / 5 / 4 of 12 | +2.6k at gain 0.3 (2/4) |
| kv, gain 0.3, scale 1.0 / 0.5 | +7.7k / +3.8k | +0.6 / +0.4 | 8 / 8 of 12 | +2.5k at scale 1.0 (2/4) |

The same EMA tilt on r3_hmm (`strategies/adam/r4_mom.py`) gave +9.3k on wf and +24.3k on hold. On r4_best it gives
nothing, because the Kalman velocity state in r4_best already captures that signal.

Earlier sweeps on r3_hmm (`r4_mom.py`) also lost:

- Calm tilts, or tilts larger than 1 tick: -100k to -770k.
- The reversed sign (mean reversion): -106k to -419k.
- The asymmetric version (only make counter-trend takes harder): -107k to -151k.

## B. Flip hysteresis in turbulence (`b_flip.py`, `fh_*`)

A turbulent IOC take that reduces or flips the position needs `fh_e` extra ticks of edge, optionally only above
`fh_pos` and scaled by position age. `fh_vf` widens the turbulent flatten-only quote.

| Variant | wf delta | t | won |
| --- | ---: | ---: | ---: |
| fh_e 0.5, pos 50 | -15.7k | -0.6 | 3/12 |
| fh_e -0.3, pos 50 (control) | -6.1k | -0.9 | 5/12 |
| fh_e 2, pos 90 | -159.9k | -4.8 | 0/12 |
| fh_e 0.3, pos 90, age_k 0.2 | -53.8k | -1.8 | 2/12 |
| fh_vf 1.0 | +12.8k | +0.8 | 6/12 (noise; breach 12.7k vs 11.5k) |

Both signs of `fh_e` lose. Reducing and flipping takes in turbulence are worth more than the carry they give up.

## C. Calm guard on the quote facing recent taker flow (`c_guard.py`, `g_*`)

In calm periods only, the passive quote facing the flow is widened by `g_th` ticks or pulled (`g_pull`). Two triggers:

- `g_q` / `g_ms`: any market taker of at least `g_q` lots starts a window of `g_ms` ms.
- `g_flow` / `g_hl`: a signed flow EWMA crosses `g_flow`.

`g_both` widens both sides (a control).

| Variant | wf delta | t | won |
| --- | ---: | ---: | ---: |
| widen 1 tick, any taker, 200 ms / 1000 ms | -113.9k / -117.0k | -4.6 / -4.2 | 0 / 1 of 12 |
| pull after a taker >= 100 lots, 1000 ms | -55.0k | -2.3 | 1/12 |
| widen 1 tick, flow EWMA > 100, hl 2 s | -121.5k | -3.4 | 1/12 |
| control: widen both sides, 200 ms | -151.4k | -4.5 | 1/12 |

Facing-side widening loses 37k less than widening both sides, so the direction effect is real, but it is small next
to the lost spread capture. The calm passive fills are worth more than the -63k calm carry they cause.

## Not tried, with the reason

- **A post-shock second leg:** the excess over the cap costs 10k/s of breach penalty, which is more than the
  +13.9-tick continuation pays.
- **F leans, exits, TWAP signals, vf retunes and filter retunes:** already tested on round 4 and lost
  (`marek/research/round4/strategy/README.md`).
