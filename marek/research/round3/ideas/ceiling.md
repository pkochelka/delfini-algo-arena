# Ceiling: where the round-3 money is, and where r3_best leaves it

Angle "ceiling", 2026-11-08. All numbers are on `data/round3.csv` against the exact S (`s_dynamics/s_path.parquet`)
unless marked synth. The scoring is a replay of the tape, so our fills never move S. Regime labels are eval3's: calm
3875 s, volatile 1558 s, shock windows (onset −1 s … +20 s) 567 s.

**Bottom line.**
- **The position limit is the bottleneck.** In volatile, r3_best has |pos| ≥ 95 for 90 % of the time. 73 % of the
  reachable stale-quote events find it with zero room on that side.
- **A better S filter is worth +34k (+4 %).** An exact regime label is worth +4k, and latency costs about 2 %.
- **Most of the money is in volatile stale quotes.** Their value is 3.7M with no size cap, 506k if every event starts
  flat with a 100-lot limit, and 864k if every event starts at the opposite cap (a 200-lot swing). r3_best gets 251k.
- **We can't hold the extra room.** Lifting the volatile cap to 200 (ignoring the penalty) gives +425k gross. But the
  penalty is 10k per second and doesn't depend on size, so a doubling would need cheap turnover of the inventory, not
  bigger positions.
- **Every recycling rule tried so far gives less than +11k.** That includes the one prototype here.

## 1. Opportunity ceilings

### Stale quotes (IOC), per regime
Method (`ceiling/ceiling_book.py`, `ceiling_stale.py`):
1. Replay the market-only book. It reproduces the tape's 7,164,613 filled taker lots exactly.
2. The first time a resting market order is more than 0.25 through S, an oracle sends an IOC.
3. The IOC arrives 5 ms later. It gets the order's remaining quantity at the edge it has then (S at arrival minus price).

An event is all the orders that turn stale at the same instant on one side. V100 takes the best 100 lots of each event
(starting flat), V200 the best 200 (starting at the opposite cap), V1000 the best 1000. The breach penalty and exit
costs are not deducted.

| edge at arrival > 0.5 | calm | volatile | shock windows |
| --- | ---: | ---: | ---: |
| events | 741 | 4,524 | 467 |
| lots reachable | 1.07M | 2.63M | 0.35M |
| value, all lots | 915k | 3,718k | 453k |
| **V100** (flat, ≤ 100 lots per event) | **55k** (14 /s) | **506k** (325 /s) | **58k** |
| **V200** (from the opposite cap) | 104k | **864k** (555 /s) | 98k |
| V1000 (oversize, before penalty and exit) | 354k | 2,354k | 243k |
| value beyond 100 lots | 860k | 3,212k | 395k |
| r3_best IOC edge actually taken | 97k | 251k | 33k take + 17.5k sniper |
| r3_best: value it had room for (own position path) | 64k | **243k** | 42k |
| share of events with zero room on that side | 24 % | **73 %** | 53 % |
| V100 of those zero-room events | 14k | **358k** | 26k |

- **Latency is not the limit.** 97–98 % of the stale lots seen at first sight are still there 5 ms later.
- **Shock jump instants are small.** In the window onset +2.5 … +3.6 s, the total over all 27 shocks is V100 18k and
  V1000 46k. The biggest single one is 9k (shock 12). The drought book is thin and wide.
  - The rest of the shock window (the post-jump burst) holds V100 40k and V200 71k.
- **Volatile event sizes.** Median Q 219 lots, p90 1454. Median best edge 1.1, p90 2.6. Volatile V100 by best edge:
  - < 1 tick: 118k
  - 1–2: 184k
  - 2–4: 151k
  - 4–10: 52k

### Passive capture (`ceiling/ceiling_passive.py`)
For every market taker, assume a quote of ours at P = max(ceil(S + x), bid + 1) (mirrored for bids), sized ≤ 100.
- **Ahead:** we are first in the queue at P.
- **Behind:** we are behind the market orders already at P.

Edge is measured against S after the taker's own kick. "Clip" adds a ±100 position limit and no unwinding.

| x | calm ahead | calm behind | calm ahead, clip | volatile ahead | volatile behind | volatile ahead, clip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 411k | −7k | 365k | −453k | −415k | −225k |
| 0.5 | **458k** | 18k | **364k** | −280k | −269k | −118k |
| 1.0 | 230k | 9k | 164k | −32k | −87k | 38k |
| 1.5 | 10k | 0.5k | 7k | 110k | 10k | 108k |
| 2.0 | 6k | 0.5k | 4k | **164k** | 63k | 118k |

- **Calm passive is queue priority.** At x = 0.25 the ceiling is 411k if we are first in the queue and −7k if we are
  behind. r3_best's calm passive edge is 123k on 596k lots (0.21 per lot).
- **In volatile, only quotes at about S ± 2 earn.** The ceiling there is ≤ 164k, and that is before latency and filter
  error.

## 2. Gap decomposition of r3_best (795.0k)

Method (`ceiling/ceiling_decomp.py`):
- The run is instrumented by `strategies/marek/round3/ideas/ceiling_oracle.py` with `instr_out`, which gives identical
  PnL. Each order is labelled by the code line that sent it.
- Fill edge = signed lots × (S at fill − price).
- Carry = Σ pos × ΔS between inserts.
- Check: fill edge + carry − penalty adds up to 787.6k, against the actual 795.0k.

| regime | type | lots | fill edge | per lot | markout 10 s | markout 60 s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| calm | passive | 596k | 123.1k | 0.21 | 0.21 /lot | 0.59 |
| calm | IOC take | 145k | 97.0k | 0.67 | 0.69 | **−0.70** |
| volatile | IOC take | 206k | 251.2k | 1.22 | **2.32** | 2.02 |
| volatile | passive | 5k | 0.3k | 0.06 | −2.76 | −2.39 |
| shock | passive | 101k | 27.2k | 0.27 | 0.47 | |
| shock | IOC take | 36k | 32.6k | 0.91 | 1.61 | |
| shock | sniper | 4.3k | 17.5k | 4.09 | 8.83 | |
| shock | oversize exit (xunwind) | 3.3k | −7.9k | −2.42 | | |

Carry by regime:
- calm +18.5k
- **volatile +165.2k**
- shock +80.7k. Of that, +33.7k is the position held into the jump: |pos| is 93 on average at the jump, and the jump
  is unpredictable, so this part has zero expected value. Another +22.0k is post-jump carry.

Breach penalty: −17.7k.

Time pinned (|pos| ≥ 95) and mean |pos| by regime:

| regime | pinned | mean \|pos\| |
| --- | ---: | ---: |
| calm | 59 % | 78.5 |
| volatile | **90 %** | 94.9 |
| shock | 65 % | 83.3 |

- **Volatile S has momentum, driven by the book lagging S.** Per lot, S − price for volatile takes is 1.22 at the
  fill, 1.73 at +1 s, 2.23 at +3 s and 2.32 at +10 s. The causal s_filt gives the same, so this is not a smoothing
  artifact. On a 0.1 s grid:
  - E[S(t+1 s) − S(t)] = 0.60 × (S − mid), and the same 0.58 at 3 s. It is about 0 by 10 s.
  - If S − mid is in (2, 3], S moves +1.32 over the next second; if S − mid > 3, +2.48.
  - Calm: 0.09 at 1 s and 0.25 at 3 s.

  The mechanism: when the book sits below S, buy takers (limit round(S + 2.55)) fill more than sell takers, so the
  kicks are one-sided. About 40 % of r3_best's volatile PnL (+165k) is this carry. Holding the position is part of the
  edge.
- **Calm takes revert.** Per lot: 0.67 at the fill, a peak of 0.95 at 3 s, −0.70 at 60 s. This is the transient
  impact. The take inventory is netted against passive fills, whose markout rises from 0.21 to 0.59 over 60 s. Net
  calm carry is only +18.5k.
- **Calm passive is limited by both queue and size** (`ceiling/ceiling_queue.py`). On takers that filled us, market
  orders at our price were filled ahead of us for 328k lots (queue). Another 809k lots at our price were filled after
  our order ran out (size or room).

## 3. Filter quality and the oracle

Online F against exact S, time-weighted (`ceiling/ceiling_filter.py`):

| regime | RMSE | mean \|e\| | P(\|e\| > 0.5) | rmse / filter sd |
| --- | ---: | ---: | ---: | ---: |
| calm | 0.125 | 0.082 | 0.6 % | 0.66 |
| volatile | 0.336 | 0.159 | 4.2 % | 1.27 (overconfident) |
| shock | 1.21 | 0.21 | 8.2 % | 2.68 |

- **Regime detection is essentially exact.** turb is on 99.7 % of volatile time and 0.3 % of calm time.
- **Shock jumps.** Median max |e| in the jump window is 15 ticks. It is back to 0.25 by +3.5 s.

Oracle runs (`ceiling_oracle.py`). The exact S replaces F in every trading decision; the filter keeps running for its
regime and jump state.

| run | PnL | Δ | calm | volatile | shock |
| --- | ---: | ---: | ---: | ---: | ---: |
| r3_best | 795.0k | | 238.6k | 421.7k | 134.7k |
| exact S, filter sd in thresholds | **829.4k** | **+34.4k** | +38.6k | −2.1k | −2.1k |
| exact S, sd = 0.05 | 816.0k | +21.0k | +42.1k | +7.5k | −28.6k |
| exact regime only | 799.0k | +4.0k | +1.0k | +1.6k | +1.4k |

**The S filter is not the bottleneck.** A perfect filter adds at most about 40k (5 %), almost all of it in calm
passive. In volatile it adds nothing, because the strategy is pinned there whatever F says.

Counterfactual larger caps (penalty ignored: gross = PnL + breach; `xu_tgt` raised with the cap so the oversize exit
doesn't dump the extra):

| run | gross |
| --- | ---: |
| r3_best | 812.7k |
| cap_t 200 | **1,237k (+425k)** |
| cap_t 300 | 1,601k (+788k) |

Room in volatile is worth about 4k per lot of cap. But it is held room: it pays through carry and repeated takes, and
it would breach about 1,600 s.

## 4. Ranked gaps

1. **Volatile position room and turnover.**
   - Size of the gap: 358k of V100 in zero-room events, and +425k gross per extra 100 lots of cap.
   - Recoverable under the penalty: only through cheaper turnover. My estimate is 10–50k unless someone finds an exit
     channel that costs less than the 1.2–1.7 ticks of the touch.
   - Constraints:
     - An IOC exit costs about the take edge (1.22) and gives up the momentum carry (+0.5 to 1.1 per lot).
     - A passive exit only fills at x ≤ 2.5 (volatile taker limits are round(S ± 2.55)).
     - The volatile passive ceiling is ≤ 164k even with perfect queue and S.
   - Ideas, all untested except the first:
     - (a) Recycle only when the momentum has run out: the prototype below, +10.9k on r3.
     - (b) Exit into the new maker bids that arrive once our sweep has removed the stale asks. While stale asks rest,
       makers clip bids at ask − 1. This needs a fill-level study of how fast those bids arrive after a sweep.
     - (c) Choose the side by forward value F + 0.6 (F − mid) when pinned. Tested as a take threshold: −19.6k.
2. **Calm passive queue and size.**
   - Size of the gap: ceiling 364k (±100 clip, at x = 0.25–0.5, first in queue) against 123k realised. 328k lots were
     lost to the queue and 809k lots were over our size.
   - Recoverable: +30–100k is plausible if quotes keep their queue seniority. Ideas: pre-join one tick behind (`pj_q`,
     off in r3_best), keep_depth, less re-quoting on small F moves. Untested here; the "calm" angle covers it.
3. **Calm take inventory reversion.** Take markout peaks at 3 s (0.95) and falls to −0.70 by 60 s, on 145k lots. Skewing
   the passive quotes to unload take inventory within about 3–10 s is worth at most a few tens of k. Untested; the
   existing `inv_skew` and `y_k` knobs are the tools.
4. **S filter.** Oracle +34k, mostly calm. A better calm filter (or a tighter sd in calm) might recover 10–20k.
5. **Shocks.**
   - Jump-instant ceiling: V1000 46k against the sniper's 17.5k edge. The post-jump window holds V200 98k against about
     60k taken.
   - At most +30–50k in total. Judge it on synth only: the +33.7k of pre-jump inventory carry on r3 is luck.
   - Flattening before the jump lowers variance, not the mean.

Taken together, no single gap can double PnL under the ±100 limit and a penalty that does not depend on size. The only
pool large enough is the volatile stale-quote value beyond 100 lots (3.2M). Doubling would need a strategy that
recycles inventory about 2× faster than now at a cost below 0.5 tick per lot.

## 5. Prototypes and tests

`strategies/marek/round3/ideas/ceiling_rp.py` is a copy of r3_best; with every flag off it gives 794,988, identical.

**Reducing-only passive quote** (`rp_on`). While turbulent, outside shock windows, and with |pos| ≥ rp_pos (60), it
rests one order on the side that reduces the position, at F ± rp_x. `rp_mom` additionally requires the book-lag
signal not to favour the position: (F − mid) · sign(pos) ≤ rp_mom.

| tag | settings | r3 PnL | Δ | volatile Δ |
| --- | --- | ---: | ---: | ---: |
| ceiling_rp1 | rp_x 2.0, keep 1.5 | 787.4k | −7.6k | −12.1k |
| ceiling_rp2 | rp_x 1.5, keep 1.0 | 775.5k | −19.5k | −19.9k |
| ceiling_rpmom0 | + rp_mom 0 | 804.9k | +9.9k | +8.8k |
| ceiling_rpmomm05 | + rp_mom −0.5 | **805.9k** | **+10.9k** | +8.3k |

Synth v1, rp_mom −0.5:

| seed | Δ total | Δ volatile |
| --- | ---: | ---: |
| 1 | +28.3k | +7.8k |
| 2 | +10.6k | +6.2k |
| 3 | −33.4k | +1.0k |
| 4 | −13.3k | +7.8k |
| mean | **−2.0k** | **+5.7k** |

The shock part swings by ±20–34k because the position entering each shock changes. Seeds 5–6 were not run.

Status: positive in volatile everywhere, but within noise in total. **Not accepted** under the brief's rule. If someone
picks it up, it should be combined with flattening into droughts, so the shock lottery doesn't hide it.

## 6. Dead ends (numbers)

- **Unconditional reducing passive in volatile** (rp1, rp2): −7.6k / −19.5k. It gives up the momentum carry. This
  matches the r2 "passive unwind" and "passive in volatile" results.
- **Momentum-adjusted take threshold** F + 0.5 · clip(F − mid, ±3) in turbulent regimes (`mk_k`): 775.4k (−19.6k), with
  volatile −14.4k.
- **Non-shock oversize takes** (`xcap` 400, existing r3_best mechanism, exit through xunwind at F ∓ 4):
  - xedge 2: PnL −953k, breach 1.30M, gross 351k.
  - xedge 4: PnL 282k, breach 519k, gross 801k, which is below the baseline gross of 813k even before the penalty.

  Stale quotes in volatile come back event after event, so the excess is re-taken and the position stays over 100 for
  52–130 s.
- **Raising cap_t without raising xu_tgt** (the counterfactual): gross −633k. The oversize exit dumps everything above
  100 at F − 1.

## Files

Prototype strategies, in `strategies/marek/round3/ideas/`:
- `ceiling_oracle.py`: instrumentation plus the exact-S and exact-regime oracle. Research only, round 3 only.
- `ceiling_rp.py`: the reducing quote and the momentum-adjusted take.

Analysis scripts, in `marek/research/round3/ideas/ceiling/`:
- `ceiling_mk_oracle.py`: builds `data/.cache/round3_S_oracle.npz`.
- `ceiling_book.py`: book replay against S. Writes `_cache/ceil_stale.parquet` and `_cache/ceil_takers.parquet`.
- `ceiling_stale.py`: stale-quote ceilings.
- `ceiling_room.py`: room-blocked share.
- `ceiling_passive.py`: passive ceiling.
- `ceiling_decomp.py`: fill and carry decomposition.
- `ceiling_queue.py`: queue position of our passive fills.
- `ceiling_filter.py`: F against S.
- `ceiling_momentum.py`: the book-lag momentum.

Run them with `uv run --with pyarrow [--with sortedcontainers] python ...` from the repo root. The full simulator logs
of r3_best (trades.csv, events) are in the scratchpad and can be regenerated with `--log-dir`.
