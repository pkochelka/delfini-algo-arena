# Quote width on r3_grandpa (2026-11-08)

Base: `strategies/adam/r3_grandpa.py` (round 3 801.2k: calm 237.6k, volatile 426.0k, shock 137.6k; round 2 769.6k).
Result file: `strategies/adam/r3_grandpa_width.py` = grandpa + `theta 0.2, theta_keep 0.0, take 0.1`:
**round 3 810.7k (+9.5k, calm +12.3k), round 2 777.9k (+8.3k, calm +5.0k)**.
Rows: `marek/research/round3/strategy/results.jsonl`, tags `gpw1` (r3 + 3 synth), `gpw2`, `gpw3`, `gpw_final` (r3/r2).
Driver: `strategy/sweep.py` (many configs in one pool), summary `strategy/summ2.py`.

## Findings

- **Passive width is flat.** theta 0.15 / 0.20 / 0.25 / 0.30 are within ±2k on round 3. 0.40 costs 4k and 0.50 costs 32k.
  theta_sd 0.5 costs 27k of calm.
- **Queue retention beats edge.** theta_keep 0.10 → 0 gives +6k calm. −0.15 loses (−24k). keep_depth 0 loses −6k.
- **IOC take width.** In calm, take 0.10 gives +2.4k calm; 0.05 is about the same; 0.25 loses. take_sd 1.0 helps calm by +4k but
  loses shock #18. The turbulent take_t 0.15 and take_sd_t 1.5 are at their optimum: take_t 0.3 or 0.5 costs 4–9k of volatile,
  and take_t 0 costs −430k.
- **First-in-queue threshold (dev mechanism `th_imp`, now deleted).** Quoting tighter only at a level with no market order
  is calm-neutral, and it loses 35k in drought quoting. Moving the room there (`imp_mv`) loses 13–23k of calm.
  Calm spread is 1 tick 94.6 % of the time, so such levels exist only right after a sweep.
- **Drought width (dr_theta/dr_keep/dr_margin/dr_fu/dr_cap).** Every change moves only the shock part, by ±10–37k. Almost all
  of that is shocks #18 / #11 / #2.
- **Shock part is a lottery.** Shock #18 (+124 jump) is worth about 20k. The drought quoting happened to leave grandpa
  at +100 at the jump, and dr_flat found no bid within 1 tick. Carry into jumps (pos × J) is +35k on round 3, but the sign
  agrees with the jump only 14–17 times against 9–13 (coin flip). Judge changes on calm and volatile, and treat shock
  deltas driven by one shock as noise.
- Calm economics (grandpa, round 3): passive m1 +0.19 per lot on 591k lots, IOC m1 +0.89 on 142k lots. Their sum ≈ eval3
  calm. The taker's own impact (0.56 per lot) is what width cannot fix.
