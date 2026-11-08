# r4_aggro: aggressive / over-cap / wide-spread overlay on r4_mom — dead end

2026-11-08, Adam + Claude. File: `strategies/adam/r4_aggro.py` (r4_mom + new flags, all off by default = r4_mom).
Evaluation: `strategies/peter/r4_eval.py --tier wf` (round 4, 12 walk-forward folds). Base r4_mom = **1067.4k**.

**Result: nothing beats the base. Do not pursue further without a new idea.**

## Over-cap bursts (`xcap`, `xedge`; excess trimmed by the existing `xu_*` unwind)

| Variant | Δ vs base | breach penalty | folds won |
| --- | ---: | ---: | ---: |
| xcap 300, xedge 2 | −1625.6k | 1253k | 0/12 |
| xcap 600, xedge 3 | −1072.5k | 923k | 0/12 |
| xcap 300, xedge 3 | −903.5k | 866k | 0/12 |
| xcap 600, xedge 4 | −544.4k | 599k | 0/12 |
| xcap 950, xedge 5 | −249.7k | 318k | 0/12 |
| xcap 300, xedge 4, forced exit at F−8 after 50 ms (`xu_ts2` 0.05, `xu_lim2` 8) | −380.3k | 132k | 0/12 |
| xcap 200, xedge 5 | −390.3k | 375k | 0/12 |

- Direction was checked: buy over cap only when ask <= F − xedge, sell mirrored; unwind sells longs / buys shorts.
- The default exit (F − 4, `xu_lim2`; note `xu_lim` is overridden by `xu_ts2`/`xu_lim2` in the base) finds no
  liquidity after a big move, so the excess sits over 100 for seconds at 10k/s.
- Even with a fast loose exit the trades themselves lose ~250k: sweeping deep levels then dumping 8 ticks under F
  costs more than the edge. Outside shocks the over-cap round trip does not pay. (Shock-only over-cap, `xs_cap`, is
  already in the base and is fine.)

## Wide / huge spread (`ws_*`, `wst_*`, `wtk_*`)

| Variant | Δ | won |
| --- | ---: | ---: |
| calm penny inside spread >= 2 (ws_edge 0.25) | −5.2k | 6/12 |
| same, ws_edge 0 | −24.4k | 2/12 |
| turbulent inside-spread quote, spread >= 3 | −5.9k | 4/12 |
| all regimes, spread >= 4, edge 1 | −3.3k | 4/12 |
| all regimes, spread >= 5, 300 lots up to pos 300 | −12.7k | 2/12 |
| IOC threshold −0.5 while spread >= 4 | −11.1k | 4/12 |

Round 4 spread is 1 tick 81 % of the time; wide spreads are mostly one-sided sweeps that run over resting quotes.

## Sweep / momentum chaser (`sc_*`, class `AAggro`)

Trigger: mid moved >= sc_x ticks in sc_L s (round4_findings: +5 ticks/lot at 1 s naive). IOC in the move's direction,
hold sc_hold s, IOC out.

| Variant | Δ (base paused while chasing, `sc_block` 1) | Δ (base keeps running, `sc_block` 0) |
| --- | ---: | ---: |
| default (10 ticks / 1 s, 100 lots, hold 1 s) | −115.8k | −10.1k |
| sc_x 15 | −59.9k | −5.3k |
| hold 2 s | −148.3k | −4.2k |
| sc_x 6, sc_L 0.5 | −146.2k | −9.8k |
| F must agree (sc_fg 0 / 1) | −103.9k | −2.8k |
| size 50 / 200 | −112.5k / −355.3k | — |

Pausing the base caused most of the loss; without pausing it is still negative (won <= 1/12). The base already rides
this momentum through F (which leads the book), so the chaser only competes for position room.

## Not used

Round-3 runs from the same session (`r3_aggro_spread.py`, `aggro_sweep.py`) are kept for history only; round 3 is
irrelevant for round 4 decisions.
