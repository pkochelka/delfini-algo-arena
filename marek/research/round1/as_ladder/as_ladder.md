# as_ladder: Avellaneda-Stoikov inventory control on top of sig_ladder

Strategy: `strategies/marek/as_ladder.py`. Scripts: `sweep_as.py` (round1 folds, paired against sig_ladder; rows in `results.jsonl`), `stress_as.py` (trending and stress tapes).

The overlay computes the A-S reservation-price offset `skew = q·γ·σ²·τ` in ticks. σ is an online EWMA, τ is a constant horizon (the stationary form), γ = 1e-4 and τ = 100 s, which gives a skew of about 0.85 ticks at q = 100. The book is 1 tick wide, so the skew drives discrete actions instead of continuous quotes:

| Switch | Mechanic |
| --- | --- |
| `as_size_k` (+ `as_shrink`) | Adding-side size = cap·exp(−k·\|skew\|) (Guéant/Fushimi). Shrink also modifies resting adding orders down to that size. |
| `as_back` | Adding side sits floor(\|skew\|/as_back) ticks behind the touch, with hysteresis |
| `as_improve` | Reducing side steps inside a spread of 2+ ticks by floor(\|skew\|) |
| `as_take` | IOC-unwinds at the far touch once \|skew\| ≥ as_take |

All switches off reproduces sig_ladder exactly: 473,066 on the folds.

## Round1 folds (Δ vs sig_ladder, 10 folds)

| Variant | Δ | t | wins |
| --- | ---: | ---: | ---: |
| size k1 | −0.1k | −1.4 | 3 |
| size k2 | −6.8k | −9.3 | 0 |
| size k4 | −14.7k | −10.0 | 0 |
| size k2 shrink | −7.8k | −7.6 | 0 |
| size k4 shrink | −36.0k | −15.9 | 0 |
| back 0.8 | −0.3k | −1.8 | 1 |
| back 0.5 | −15.1k | −11.3 | 0 |
| improve (γ 2e-4) | −96.8k | −17.6 | 0 |
| take 0.8, 20 lots | −38.9k | −9.4 | 0 |
| take 0.8, 50 lots | −96.3k | −11.7 | 0 |
| take 0.6 | −181.5k | −31.7 | 0 |

Size at `size_top·exp(…)` had no effect at all: the cap room (cap − |q| − resting) always binds first. That is why the base is now `as_size_base` = cap.

## Stress and trending tapes (PnL)

| Variant | random_stress | sample | sample_v2 | wf03_trendup | wf03_toxic10 | wf07_vscale0.3 | wf07_drift100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | −15,251 | −10,399 | 277,704 | 35,633 | 52,359 | 14,024 | 43,819 |
| size k2 | −14,560 | −10,600 | 277,590 | 34,846 | 51,609 | 14,125 | 43,495 |
| size k4 shrink | −10,828 | −7,572 | 277,277 | 32,669 | 48,047 | 14,400 | 41,239 |
| back 0.5 | −8,987 | −9,070 | 277,506 | 34,924 | 51,355 | 13,588 | 42,605 |
| back 0.8 | −12,206 | −5,411 | 277,704 | 35,528 | 52,308 | 14,009 | 43,793 |
| improve | −20,796 | −89,048 | 276,661 | 23,580 | 38,676 | 9,653 | 34,249 |
| take 0.8 | −22,319 | −10,546 | 277,631 | 19,866 | 40,172 | 12,897 | 27,185 |
| take 0.8 l20 | −18,909 | −10,745 | 277,724 | 28,581 | 47,377 | 13,606 | 37,042 |

## Takeaways

- **Active unwinding (IOC) and quoting inside the spread on the reducing side lose on every tape.** Crossing pays the spread on fills that the next big taker would have flattened for free. Don't redo this without a new reason.
- **The passive A-S switches** (back-off, exponential sizing with shrink) cut the losses on the two tapes where the ladder loses by 3–6k each. They cost 15–36k on round1 and a few k on the profitable stress tapes.
- **They are not worth enabling unconditionally.** They could fit a regime-gated overlay, switched on only while our trailing PnL or the variance ratio says the tape trends, like robust_ladder's kill switch or inv_mm's `adapt_*`. `back 0.8` costs almost nothing on round1 (−0.3k), so it is the cheapest insurance to test under such a gate.
