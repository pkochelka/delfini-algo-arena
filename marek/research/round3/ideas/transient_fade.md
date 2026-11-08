# Transient-impact exploitation on r3_grandpa / r3_delayed_fade (2026-11-08)

Verdict: **dead end.** Nothing beats r3_grandpa on calm+volatile robustly; every aggressive variant loses.
Dev file with all knobs (default off, reproduces r3_grandpa 801,210 with `rv_on=0`): `strategies/adam/r3_delayed_fade_dev.py`.
All runs: full round-3 tape, eval3.py, tags df_base, dfy1-dfy5 in strategy/results.jsonl. Judge on calm+volatile;
the shock part swings +-19k on shock #18 (4130 s, J=126) with the inventory carried in.

## Size of the prize (exact S, true transient stock R from the model)
- mean |R|: calm 1.7 ticks, volatile 6.2. Oracle drift value at a permanent 100-lot fade: 122k (calm 45k, vol 77k).
- R predicts S as the model says (vol slope -0.38 at 5 s, -0.60 at 10 s; corr -0.2..-0.3, sd of dS 14-18 ticks).
- The filter's Y is accurate: corr(Y, R) 0.92 calm / 0.96 vol, slope 0.85 / 1.01.

## Why it cannot be harvested
- Volatile: no passive quotes, the sniper keeps |pos| pinned at the cap (mean 95) and flips sign every 1.3 s.
  Inventory is anti-aligned with the fade (E[pos * sign(-Y)] = -20 lots): the transient decay costs grandpa 19.5k
  in volatile. That is the ceiling; the drift per 1.3 s hold (~0.1 Y) is far below the ~1.5-tick crossing cost.
- Calm: |R| too small; inventory near-neutral to Y (drift +1.1k).
- Post-shock R is large but inconsistent in sign vs J; ~8k ceiling over 27 shocks, inside the shock lottery.

## Results (delta calm+volatile vs grandpa 663.6k unless noted)
| hypothesis | best setting | result |
|---|---|---|
| rv delayed-reversal IOC overlay, calm (original r3_delayed_fade) | as shipped | total 800.5k vs 801.2k, calm -0.6k |
| same overlay also in turbulence (rv_turb) | min_y 6 / 8 | -0.8k / -0.4k |
| take-centre shift F - k Y, volatile (yt_k) | 0.1 / 0.25 / 0.5 | vol -26k / -180k / -309k |
| ... fade side only (y_anti=0) / anti side only (y_fade=0) | k 0.25 | vol -339k / -95k |
| take-centre shift, calm (yc_k) | 0.15 | calm -9k |
| passive reduce-only fade-side order in volatile (yp_*) | th 0.5..1.5, k 0.1..0.25, Y-blind control | vol -32k .. -160k (passive vol fills eat the taker's own kick) |
| calm reduce-side tightening when pos*Y>0 (yq_k) | 0.1 | +5.4k on grandpa, but -3.0k on width params, -3.7k on r2: noise |
