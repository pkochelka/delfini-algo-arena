# Recipe: from a new round's data to a submission

This is what worked in round 1, put in order so the next round goes faster. Round 1 numbers are given as reference points. Details are in [FINDINGS.md](FINDINGS.md) and [strategies/peter/round1_generator.md](strategies/peter/round1_generator.md).

**The main lesson.** The biggest gain did not come from tuning. It came from understanding how the tape was generated:

| Strategy | Round 1 full tape |
| --- | ---: |
| `ladder_mm`, reverse-engineered from the leaderboard | 425.6k |
| `sig_ladder`, the ladder after hypothesis testing and autoresearch | 489.6k |
| `s_mm`, which quotes around the reconstructed fair price S | **727.0k** (`logs/s_mm_full`) |

So fit the generator model on day one (step 3). Don't leave it until after the tuning runs out.

---

## 0. Set up the data (about 10 min)

```bash
uv run python -m gzip -d data/roundN.csv.gz
# walk-forward folds over the first 75%; the last 25% stays untouched until the end
uv run python strategies/peter/split_data.py --backtest data/roundN.csv --folds 12 --warmup-s 60 --end-frac 0.75 --out-dir data/splits_rN/wf
uv run data-explorer --data data/roundN.csv --export      # tape statistics -> data_explorer/roundN/README.md
```

Look at the result in [data/splits_r2/](data/splits_r2/), which was already made for round 2. Each round gets its own split. Never look at `hold25` while developing.

## 1. Reverse-engineer the previous round's leaderboard (about 1 h)

1. Download the leaderboard of the round that just closed: PnL, fills, volume and inventory, as far as they are shown.
2. Ask, for the top entries, what kind of strategy produces those numbers. Examples: PnL per fill, volume per hour, whether inventory sits at the cap, whether they are passive or aggressive.
3. Write the simplest strategy that reproduces those numbers. In round 1 that gave `ladder_mm`: large passive orders at the touch, keeping queue priority, with cap 100.
4. Score that strategy and **all of our own strategies** on the same folds with the same harness, and start from the best one:

```bash
uv run python strategies/peter/fold_eval.py strategies/peter/ladder_mm.py '{}' --dir data/splits_rN/wf
uv run walk-forward --strategy <best>.py --backtest data/roundN.csv --folds 10 --warmup-s 60
```

## 2. Check that the previous best still holds on the new tape

Rerun the previous winner on the new folds before building anything new. If its PnL collapses, the tape regime changed (thinner book, trend, wider spread). Find out why first (step 4), then reuse ideas from the previous round.

## 3. Fit the generator model (do this early, about 1 to 2 h)

```bash
uv run python strategies/peter/r1_model.py data/roundN.csv --out strategies/peter/tuned/roundN_model.json
```

- **The script reports violations of the price rules.** "0 violations" with a single-point `c` means the tape is the same kind of synthetic tape as round 1. In that case S is exactly computable and `s_mm` / `s_mm2` apply directly once you plug in the new `c`, `gap` and taker-size parameters.
- **Many violations mean the model broke.** Find out which part: impact exponent, maker offset D, cancel hazard, extra diffusion (`s_sigma`), or informed takers. Extend the model, or fall back to the touch-based ladder family.
- **Validate the fit** by generating a synthetic tape with [r1_synth.py](strategies/peter/r1_synth.py) and comparing book statistics (sd(mid − S), spread shares, touch size, lifetimes). Also compare one strategy's PnL on real versus synthetic folds. In round 1 they matched to within 2%; see the table in round1_generator.md.

## 4. Structured data analysis, to get motivated hypotheses (about 2 h)

Read `data_explorer/roundN/README.md` and run an audit like [r1_structure.py](strategies/peter/r1_structure.py). Copy it and point it at the new tape. Ask these questions every round:

1. **Weird events and anomalies.** One-sided books, bursts of events, quotes far from the fair price, oversized orders. Is each one an edge, or only a bug risk for our code? In round 1 the empty side of the book caused 5,000-tick mid jumps, so every strategy keeps the `None` touch guard.
2. **Periodicity and trends.** Run a periodogram of event counts and squared returns, check timestamp phase uniformity, check for a drift in rates or in price, and run a variance ratio test (momentum versus mean reversion).
3. **What predicts the mid?** Compare past returns, book imbalance, maker-insert aggressiveness, flow, and the model's S − mid. Use leave-one-split-out R² at 0.5, 1 and 5 s.
4. **Does it change across splits?** Compute every statistic per split (the walk-forward windows plus the holdout cut into parts) and test homogeneity. If all splits look the same, fold-to-fold PnL differences are luck, and parameters tuned on one split will carry over.
5. **Where does our PnL come from?** Split it into spread capture and inventory carry, and compute markouts by queue position, by our side's share of the touch, and by fill age. Tools: [fill_probe.py](strategies/peter/fill_probe.py), [fill_markout.py](strategies/peter/fill_markout.py) and `marek/research/round1/robustness/decompose.py`.

Write each finding as a hypothesis with a flag in `PARAMS`, off by default, the way [queue_hypotheses.md](strategies/peter/queue_hypotheses.md) does. Test each hypothesis on the paired per-fold difference: report the mean, how many folds it wins, and t. Keep it only if it wins most folds **and** neighbouring parameter values also gain. A plateau counts; a single good point does not. Record every rejected idea in FINDINGS.md so nobody tests it again.

## 5. Autoresearch, i.e. agents running hypotheses in parallel (in the background)

In round 1 this gave +64k on the ladder (the depletion guard and the flow term in `sig_ladder`). It did not find the S-based approach, because it searched near the ladder.

- **Give the agents the generator model and the S − mid result as context,** not only the best strategy. Otherwise they only tune what already exists.
- **Point at least one agent at a different starting family:** an S-based or model-based market maker, or one that takes stale quotes.
- Every agent reports on the same folds, with paired differences, plus a check on the stress tapes (step 7).

## 6. Port what made `s_mm` beat the ladder

| | `ladder_mm` / `sig_ladder` | `s_mm` |
| --- | --- | --- |
| Reference price | public touch | the hidden fair price S, rebuilt from taker fills: S = S0 + c·Σ sign·√filled |
| Calibrating S0 | none | learned online from bounds set by makers' touch inserts (bid ≤ S − 0.05 ≤ S + 0.05 ≤ ask), slowly relaxed |
| Where to quote | at the touch, always | the best price with an edge of at least `theta` (0.30) from S: improve, join, or sit behind the touch |
| When to cancel | almost never (queue priority comes first) | when the edge drops below `theta_keep` (0.10). In effect it cancels only after S has moved against the order, which is when the market's own makers also cancel. |
| Taking liquidity | never | IOC on stale quotes: ask ≤ S − `take` (0.15) |
| Turning the edge off | — | trade only while the S bounds are tight (`max_width` 0.3) |

The general idea is to quote against a fair-value estimate, not against the visible book. If the next round is not exactly synthetic, the same structure still works with a *learned* fair-value estimate in place of S, for example a regression on imbalance, insert aggressiveness and flow from step 4. Try that before giving up on the idea.

`s_mm2` takes this one step further. It computes the expected value per taker from the taker-size distribution and the lots queued ahead of our order. Benchmark both.

## 7. Check for overfitting (required before every submission)

Round 1 did not overfit. Keep that by running all of these every time:

1. **Walk-forward folds.** Look at the per-fold paired difference against the baseline, not at the total. With `tune.py walkforward`, check that the tuning *procedure* beats the defaults out of sample. If it doesn't, don't trust `tune.py search` either.
2. **Synthetic tapes from the fitted model,** using new seeds and shifted parameters (taker rate, `c`, book depth, cancel speed, toxicity):

   ```bash
   uv run python strategies/peter/r1_synth.py --params strategies/peter/tuned/roundN_model.json --seeds 1 2 3
   uv run python strategies/peter/r1_synth.py --seed 4 --set taker_rate=6 c=0.035 --duration 3000
   ```

   Round 1 used a set of 10 scenarios in `data/synth/div/` (base, calm_thick, busy_flow, vol_drift, wide, fast_turn, slow_turn, toxic, big_sweeps, thin_small). Regenerate that set around the new fit.
3. **Augmentations of real folds** (mirror, pscale, tscale, vscale, drift, noise, extra takers, shocks):

   ```bash
   uv run python strategies/peter/augment.py data/splits_rN/wf/wf03.csv data/splits_rN/wf/wf07.csv data/splits_rN/wf/wf11.csv
   uv run python strategies/peter/robust_eval.py <strategy>.py '{}' '{"flag":1}'
   ```

   Look at both MEAN and WORST. `mirror` must give the same PnL for a symmetric strategy.
4. **Stress tapes:** `marek/research/round1/robustness/stress_tapes.py` and `run_matrix.py`. These cover informed takers, trends and volatility regime switches. Also run on the other tapes we have (sample, sample_v2, random_stress), which trend.
5. **Holdout, once at the end:** `test_eval.py` or the `hold25.csv` file. If the result disappoints, don't tune on it afterwards.

A negative round scores 0 in the z-score. Prefer a small loss of PnL on the main tape over a losing stress tape. The kill switch and depth-scaled cap from `robust_ladder` are cheap insurance.

## 8. Submission checklist

- [ ] `cap` is at most 100 in the submitted file. The working copy has had `cap: 1000` before, which scored about −10M.
- [ ] No breaches on any fold, stress tape or synthetic tape.
- [ ] Guard for a one-sided book: do nothing when either touch is `None`.
- [ ] `modify_order` / reject bookkeeping is intact. One buggy port breached and scored −10M.
- [ ] Only allowed imports (see README). The file passes the static check: no `os`, `getattr`, `str.format` and the like.
- [ ] Round-specific constants (`c`, `gap`, taker-size μ/σ) come from this round's `r1_model.py` fit.
- [ ] The kill switch is on, unless it costs more than about 2% on the folds.
- [ ] FINDINGS.md is updated with the scoreboard and the rejected ideas.

## Suggested time budget for a round

| Step | When |
| --- | --- |
| 0 to 3: data, leaderboard reverse-engineering, baseline check, model fit | first half day |
| 4: analysis and hypotheses | in parallel with 5 |
| 5: autoresearch agents | start in the background as soon as step 3 is done |
| 6: model-based strategy | as soon as step 3 says the model holds |
| 7 to 8: robustness and submission | keep the last 20% of the time for these |
