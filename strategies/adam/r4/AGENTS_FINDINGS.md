# Round 4: agent overlay and the search for new PnL pots (2026-11-08)

Tape `data/round4_public.csv` only. Base `strategies/marek/round4/r4_best.py`: full tape 1,391.0k
(calm 280.5k, volatile 798.9k, shock 311.5k), wf 1081.7k. Scripts: `d_agents.py` (overlay), `agents_sweep.py`
(full-tape calm / volatile / shock split, rows in `agents_sweep.jsonl`), `agents_attrib.py` (per-episode attribution),
`e_mguard.py` (calm momentum guard). Throwaway analysis scripts were run from the session scratchpad; the numbers are below.

## 1. TWAP and 15-lot agent ride / fade (`d_agents.py`, `ag_*`): dead

Event study (causal detection, paying the spread both ways):

| Signal | Ride | Fade after |
| --- | --- | --- |
| TWAP, 37 episodes, enter at child 2, exit at the missed slot | +12.2 ticks/lot, 78 % positive | calm ~0, volatile +13 at 5 s (12 episodes) |
| 15-lot joiner, 34 episodes, enter at the first 1 us re-post, exit at its stop | +6.5 ticks/lot, 85 % positive (t 6.1) | +5.9 at 10 s after a full-fill stop; after a cancel stop it keeps running (+21) |

Backtests, full tape, deltas vs r4_best (cv = calm + volatile, free of the shock lottery):

| Variant | cv | total |
| --- | ---: | ---: |
| 15-lot ride, quote on the target side kept (`ag_pass` 1) | -1.4k | 1,380.5k |
| 15-lot ride while market making (`ag_pass` 2, band 30) | -2.1k | 1,390.8k |
| 15-lot ride, plain / + fade 10 s / half size / veto only | -4 .. -7k | |
| TWAP ride (any variant, incl. ride while market making) | -18 .. -24k | |
| TWAP fade, volatile only, no ride | -5.2k | 1,372.0k |
| TWAP ride in volatile too + volatile fade | -95.8k | 1,295.9k |

Why: r4_best's Kalman velocity filter already follows both drifts, a TWAP ride stops us selling the spread to its
uninformed 160-lot children, and pausing quotes during episodes costs ~67 $/s of calm market making (TWAP episodes
are 708 s of the tape: ~47k against a 45k ride edge). The wf +18.4k of the 15-lot ride was one episode just before
a +116-tick shock.

## 2. Other pots checked (fail fast, on the tape or the r4_best trace)

| Idea | Test | Result |
| --- | --- | --- |
| Shock snipe beyond what we take | stale-side volume >= 3 ticks through S_new at the jump, best-first up to 1000 lots | 149k total edge if taken instantly at the jump, 96k at +20 ms, 15k at +50 ms (the drought empties the book). We already make 311k in shock windows. Dead |
| Calm inventory skew / unwind (`inv_skew`, `unwind_u`, `th_red`) | wf | inv_skew 0.3 / 0.7: -24.6k / -68.3k; unwind -25.5k; -0.3: +7.5k (t 0.4, noise). Calm seconds pinned at the cap earn 231 $/s vs 118 flat (wf02): pinning pays |
| Maker cancels as an S observation (hazard 1.7 + 3.4 mv) | regression of the next mid move on cancel-side imbalance, raw tape and beyond r4_best's F | raw tape: volatile R2 0.109 -> 0.144 at 0.2 s; beyond F: 0.215 -> 0.215 OOS. The insert filter already has it. Dead |
| Flip / flatten when the position disagrees with the predicted move | offline policy on the trace, half-spread costs | position disagrees with the OOS prediction 17-19 % of turbulent time (move then -1.6 / -3.1 ticks at 1 / 2 s), but the policy loses on the fit half (-1 .. -99k) and gains on the other (+2 .. +25k): spread eats it. Dead |
| Drought liquidity provision | unfilled IOC flow in droughts | 90 % of drought IOC volume (88k lots) goes unfilled, +1.25 / lot at its limit; cap-limited oracle 49k; the base drought agent already fills 18.5k lots at +0.59. <= 10-15k left |
| Passive exit into the post-jump taker burst (`xu_pass`) | round 3 | -62..-66k (marek/research/round3/FINDINGS.md) |

## 3. Where r4_best's PnL comes from (fills on the full tape, mid markouts)

| Bucket | lots | spread edge at fill | 5 s | 30 s |
| --- | ---: | ---: | ---: | ---: |
| turbulent IOC | 169k | -205k | +818k | +776k |
| calm passive | 744k | +446k | +67k | +262k |
| calm IOC | 249k | -146k | +262k | +72k |
| shock IOC (sniper) | 4.8k | -11k | +198k | +196k |

Turbulent PnL is position carry at the cap (5 s markout barely depends on the take's edge vs F), so it is cap-bound.
Calm passive gives back 380k of its 446k spread within 5 s. Calm passive markouts by state at the fill:
edge vs F <= 0.25: -0.91 / lot at 5 s (61k lots); mid moved >= 1 tick against the quote in the last second:
-0.19 .. -0.63 (240k lots); fills > 100 lots: -0.11 (352k lots). -> `e_mguard.py` tests a guard on the quote facing
a recent move, and `theta_keep`.

## 4. Calm momentum guard and theta_keep (`e_mguard.py`, wf vs r4_best 1081.7k): dead

| Variant | wf delta | won |
| --- | ---: | ---: |
| mg_x 1, mg_w 1 / 2 | -56.6k / -46.1k | 1 / 3 of 12 |
| mg_x 2, mg_w 1 / 5 | -14.7k / -10.4k | 3 / 5 of 12 |
| mg_x 1, mg_w 1, mg_win 0.5 | -20.0k | 3/12 |
| theta_keep 0.25 / 0.4 | +0.1k / -7.5k | 7 / 2 of 12 |

Per-fill markouts mislead: passive fills that look toxic on their own also cash out IOC positions.

**Bottom line: none of these ideas beats r4_best; keep r4_best.**
