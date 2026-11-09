# Round 4: capturing more price jumps (generalising r4_volsig)

Written 2026-11-08. Tape `data/round4.csv` only. Base `strategies/marek/round4/r4_volsig.py` (1,413.0k).
New file `strategies/marek/round4/r4_jump.py` = r4_volsig + `jp_*` / `jb_*` knobs, **all off** (no knob survived).

**Bottom line.** r4_volsig's PnL jumps are the shock sniper (plus #6, which went 710 lots oversize). The price jumps it
"misses" fall into two groups, and neither one leaves money that can be taken:
1. **Shocks:** the snipes already take the stale side the tape offers. The visible stale value across all 28 shocks is
   ≈ 296k (3-tick exit cost), and the shock column is 310k. The drought book is 1–5 levels deep, with nothing beyond the
   8-tick `ss_lim`. The losing shocks (#8 −9k, #22 −5k) are the ones where the position before the jump was ±100 on the
   wrong side and there was ≈ 0 stale depth. Jump direction is N(0, 63), so that is variance, not edge.
2. **Non-shock jumps (volatile momentum moves):** the position is already at the cap on the right side for most of
   them. Going past the cap costs 10k/s, and even the best causal predictor only reaches breakeven.

## Where the position is during non-shock jumps (`tools/jumps.py`, `tools/lag.py`, `tools/convex.py`)

| Move | n | aligned pos at onset | share with pos ≥ 80 aligned |
| --- | ---: | ---: | ---: |
| ≥ 10 ticks in 0.5 s | 268 | +82 | 89 % |
| ≥ 15 in 1 s | 207 | +75 | 84 % |
| ≥ 40 in 5 s | 100 | +7 | 47 % (39 % wrong side) |

- For the 52 five-second moves that start unaligned, the position aligns in a median 0.73 s. The mid has moved
  −0.2 ticks by then, so the flip comes before the move.
- The response is concave in the filter state: at |kv| 6–8 or |F − mid| 6–8 the move overshoots the linear fit
  (residual −0.75 to −4.5). The position is already +94 to +100 aligned once |kv| ≥ 4.
- Capture around volatile starts is good: 81 % of the cap ceiling in the first second, against 63 % for other volatile time.

## Burst oversize (non-shock), the only way to take more of a jump than ±100

| Check | Result |
| --- | --- |
| Perfect foresight, 900 lots, sweep in/out, 10k/s (`tools/ceiling.py`) | ≈ 620k from 167 one-second windows. Only 5 % of turbulent moments are net-positive even with the right direction |
| Simple signals kv, F − mid, mix, top 0.5 % (`tools/oracle.py`) | −7k to +8k at best; the two halves of the tape disagree |
| Linear model on 12 causal features, half-split OOS (`tools/clf.py`) | top-20 picks gross 10.1 / 11.4 ticks/lot against 11.1 needed (1 s). Cumulative net top-50: −50k to −70k |
| In the engine, `jb_*` (thr 7–10, tau 0.5–2 s) | **−23k to −349k** (breach 64k–666k) |

## Engine sweep (`cfg/s1.jsonl`, tag s1; deltas in k vs r4_volsig)

| Knob | Total | calm | vol | cv | shock |
| --- | ---: | ---: | ---: | ---: | ---: |
| `jp_kvc` 0.1 / 0.2 / 0.4 (calm kv lean) | 1403.7 / 1407.8 / 1402.8 | −2.2 / −0.8 / −2.2 | −2.5 / −0.4 / −0.8 | −4.7 / −1.1 / −2.9 | −4.7 / −4.1 / −7.3 |
| `jp_k2` 0.1 / 0.25 / 0.5, kv0 3 | 1411.8 / 1415.9 / 1432.6 | | | −2.6 / −0.3 / −1.1 | +1.4 / +3.2 / **+20.6 (#6 +16.9)** |
| `jp_k2` 0.25 / 0.5, kv0 5 | 1413.0 | | | +0.2 | −0.2 |
| `jp_k2` 0.25, kv0 2, both regimes | 1398.0 | −2.1 | +0.4 | −1.7 | −13.3 |
| `jb_on` thr 7 / 8.2 / 10, tau 1 | 1063.7 / 1273.9 / 1371.2 | | −338.5 / −125.0 / −22.6 | | |
| `jb_on` thr 8.2, tau 0.5 / 2; thr 10, tau 2 | 1305.1 / 1222.6 / 1366.4 | | −105.0 / −168.0 / −48.2 | | |

## Tools

| Path | What |
| --- | --- |
| `tools/curves.py` | PnL curves → `cache/curve_*.npz` |
| `tools/logf.py`, `tools/logbook.py` | filter state every 10 ms; state + 20-level market depth every 50 ms → `cache/book_r4_volsig.npz` |
| `tools/shocklog.py`, `tools/shockdiag.py` | tick-level log around each shock; stale depth / fire time / position per shock |
| `tools/jumps.py`, `tools/lag.py`, `tools/convex.py`, `tools/vstart.py` | non-shock jump alignment, flip lag, response convexity, volatile-start capture |
| `tools/oracle.py`, `tools/ceiling.py`, `tools/clf.py` | burst-oversize economics: simple signals, perfect foresight, OOS feature model |
| `eval4.py`, `sweep.py`, `tbl.py`, `summ.py` | harness copies writing to this folder's `results.jsonl` |
