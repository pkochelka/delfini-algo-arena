# HMM regime angle (round 3), 2026-11-08

Goal set by the user: model regime changes with hidden Markov models and push round 3 to 1M. **Not reached.** The best
file is `strategies/marek/round3/r3_hmm.py`, which scores **857.6k** on round 3. That is r3_dev_best (853.7k) with
take_sd 1.0 and take_sd_t 1.0, a noise-level +3.9k.

## 1. HMM of the tape (`hmm_fit.py`, `hmm_fit.json`)

Multi-stream Poisson HMM on 1 s bins. The streams are takers, exact-2 ms ladder links, maker inserts, 20–70-lot maker
inserts and cancels. It is fitted with Baum-Welch for K = 2..5.

| K | BIC | states |
| --- | ---: | --- |
| 2 | 234,709 | calm 71 % / volatile 29 % (90 % agreement with s_dynamics vol_segments) |
| **3** | 217,351 | **calm 70 %, volatile 28 %, drought 1.2 %** (dwell 2.7 s, makers 61 /s, ladder 0, P(drought window) 1.00) |
| 4 | 208,682 | splits by ladder intensity (dwell 1.6–4 s): bursty ladder overdispersion, not a regime |
| 5 | 205,556 | same, finer |

- The tape has exactly the generator's three regimes. The strategy's filter already runs a joint (calm/volatile × S)
  HMM plus a drought detector.
- An earlier oracle test (ceiling.md) found that an exact regime label is worth only +4k. Regime detection is not where
  the money is.

## 2. Taker-limit (residual) quotes (`resid.py`, `lq_value.py`, `fills_vs_lim.py`)

- A taker's unfilled residual has no impact on S. A quote resting exactly at the taker's own limit (round(S ± 2.55) in
  volatile, round(S ± 1) in calm) is reached only by takers that exhaust the book.
  - Volatile: 169k lots (≤ 100 per taker), +1.54/lot at the fill, +2.59 at 1 s.
  - One tick inside the limit: +0.65. Two ticks inside: −0.86.
- But 80 % of that value comes when S is more than 1 tick beyond the mid on the taker's side (book-lag momentum, +3.6/lot
  at 1 s). Those are exactly the moments the strategy is already pinned at ±100 from its IOC takes.
- Our real calm passive fills (r3_dev_best full log):
  - At the taker's limit: 450k lots, +0.30/lot at 1 s.
  - One tick inside: 189k lots, −0.15/lot.

### Strategy tests (`r3_hmm.py` flags, all on round 3, base 853.7k)

| Variant | Round 3 |
| --- | ---: |
| `lq_on`: volatile limit quote, momentum-gated g ≥ 1 / 2 / 0.3 | 851.0k / 853.0k / 842.5k |
| `lq_on` ungated, both sides | 809.5k |
| `lq_tk`: raise the IOC threshold on the gated side to leave room for the limit quote | 763k–845k (all negative) |
| `lq_red`: reduce-only limit quote as the exit after snipes | 800k–835k |
| `kin`: calm, cancel quotes inside the limit / post only at the limit | 763.5k / 771.9k |
| calm theta 0.35 / 0.45 / 0.5 | 852.7k / 849.2k / 845.2k |

- Holding room back for limit fills, or exiting through them, gives up the IOC edge and the volatile momentum carry.
- Calm fills inside the limit look bad per lot, but they are what flips the position. Removing them halves calm cycling.
  This is the same lesson as calm.md.

## 3. Room-allocation ceiling (`dp_ceiling.py`, `dp2.py`, `dp3.py`)

Hindsight DP over the stale events with a ±100 position.

- **Volatile, edge at arrival:**
  - Optimal 345k against greedy 240k.
  - No threshold or split rule beats greedy: the best is 241k at threshold 0.5.
  - The extra 105k needs foresight of which events come next.
- **Volatile, valued at the 1 s markout** (S-path hindsight, so an upper bound only):
  - IOC events only: 718k.
  - IOC events plus residual events: 864k.

## 4. Sensitivity sweep of r3_dev_best (`cd1.jsonl`, tag cd1 in strategy/results.jsonl)

- 63 one-at-a-time changes over calm, volatile, filter and shock knobs. All are within −10k / +2k, except fill_room 0
  (−34k).
- Best combination: take_sd 1.0 with take_sd_t 1.0, **857.6k**. It is now the default in r3_hmm.py.

## Conclusion

- Under the ±100 limit and the 10k/s penalty, this stack sits at a local optimum. Regime modelling (the HMM) is
  already saturated.
- The open room-allocation value (≈ 100k in hindsight) needs an event forecast that no feature we tried provides.
- The round-2 leaderboard (top 1.88M, 10th 1.02M) says much stronger strategies exist. Finding them needs a new
  mechanism, not better regime detection.
