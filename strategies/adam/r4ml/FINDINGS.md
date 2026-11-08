# Round 4: ML feasibility screen (dead end), 2026-11-08

Question: can small ML models predict the next move of the hidden price better than `strategies/marek/round4/r4_best.py`'s
own filter? If not, no ML overlay can help, and nothing is worth backtesting.

Tape `data/round4_public.csv` only.

## Method

- `log_state.py` replays r4_best (fastsim) and logs its filter state every 20 ms: F, sd, P(volatile), Kalman velocity
  kv, transient Y, turbulent flag, position. It also logs the touch prices, sizes at 3 levels per side, and cumulative
  taker flow, taker counts, big takers and inserts. The hook does not change behaviour: PnL 1,390,968, identical to
  r4_best. Output: `logs/r4ml/state_r4.npz`.
- `feasibility.py` samples a regular grid (0.05 to 0.2 s) and predicts two targets: F(t+h) − F(t) and mid(t+h) − F(t),
  for h = 0.2 to 5 s. Rows with |target| ≥ 10 ticks (the shock jumps) are dropped. Each regime (pv > 0.5) is fitted
  separately. Validation is out-of-sample: 4 contiguous blocks, leave one block out, with a 10 s purge.
- Model ladder:
  - `base`: kv·h, which r4_best already uses.
  - `fstate`: ridge on the 7 filter-state features.
  - `all_lin`: ridge on all 39 features (lagged F and mid moves over 0.1 to 15 s, flow, volume and big-taker windows,
    imbalance at 1 and 3 levels, depth, spread, insert imbalance).
  - `gbm`: HistGradientBoosting on all 39 features.

## Results (out-of-sample R², F(t+h) − F(t))

| h | calm fstate | calm all_lin | calm gbm* | vol base | vol fstate | vol all_lin | vol gbm* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 s | 0.009 | 0.012 | 0.018 | 0.202 | 0.282 | 0.283 | 0.281 |
| 1 s | 0.012 | 0.020 | 0.018 | 0.141 | 0.188 | 0.182 | 0.175 |
| 2 s | 0.017 | 0.027 | 0.021 | 0.061 | 0.068 | 0.035 | 0.031 |

\* Regularized GBM (depth 3, min leaf 1000, 150 iterations), with 0.05 s rows. The first GBM (31 leaves, min leaf 200)
lost to ridge in every block, horizon and regime (for example, volatile 1 s: 0.107 vs 0.184).

- **Raw book and flow features add nothing in volatile** beyond the filter state, linear or GBM, in any block. In calm
  they add +0.01 R² at most, on a total of about 0.02.
- The only sizeable gap is fstate over base in volatile (0.28 vs 0.20 at 0.5 s). Its coefficients (two halves agree):
  - The gap coefficient on F − mid is +0.6 to +0.7.
  - kv·h is scaled by ×1.4 to 1.6.

  So F lags. That is the F−mid lean and velocity-gain family that `marek/research/round4/strategy/README.md` H2
  already backtested (`gm_kv`, `vel_kv`, kv gain and sd retunes). All of them lost, because volatile PnL is position
  carry at the ±100 cap (91 % of volatile time), so a better forecast has no room left to monetize.

## Verdict

Dead end, not backtested further (fail fast). A learned S/mid forecaster cannot beat r4_best's Bayesian filter with
public features. The extra predictability left is the filter lag in volatile, which is already known to be
non-monetizable. If ML is revisited, the target should be a *decision* (for example, fill-level markout given queue
state), not a price forecast. Note that the calm guards (`c_guard.py`, `e_mguard.py`) already suggest little is there
either.

# Part 2: decision ML, a fill-quality gate (dead end), 2026-11-08

Question: can a model tell, at decision time, which r4_best fills lose money, so the strategy can skip them?

## Method

- `log_state.py` now also logs every fill: side, quantity, price, IOC flag, position before the fill, order age, filter
  state and touch. There are 24.5k fill events, and PnL is unchanged.
- `fill_screen.py` uses the per-lot markout side · (mid(t+h) − price) as the target, for h = 1 s and 5 s, in 4 groups
  (calm/turbulent × passive/IOC). It fits ridge and regularized GBM on 34 fill-time features, with 4 blocks, leave one
  out.
- It reports the markout of the predicted-worst 10 % / 20 % of lots and the naive skip value, −Σ qty · markout over
  fills predicted < 0.

## Screen (out-of-sample)

| Group | Lots | Mean markout | Hand rule edgeF worst 10 % | Model worst 10 % | Naive skip value (blocks) |
| --- | ---: | ---: | ---: | ---: | --- |
| calm passive, 1 s | 760k | +0.15 | −0.37 | −0.46 (ridge) | +61k (+10, +13, +18, +21) |
| calm passive, 5 s | 760k | +0.09 | −0.90 | −1.07 (gbm) | +121k (−13, +31, +74, +28) |
| calm IOC, 1 / 5 s | 250k | +0.72 / +0.92 | | | ≈ 0 / negative |
| turbulent IOC, 5 s | 176k | +2.98 | +1.07 | −3.87 (gbm) | +136k to +152k (all 4 blocks positive) |

The flagged takes are the ones against the last second's move. For turbulent IOC, the mid moved 5.5 ticks against the
take (1.6 for the rest), and kv points against the take. 71 % of them reduce or flip the position. For calm passive,
the flagged fills come after the mid moved against the quote, with less edge vs F. In both cases these are sharper
versions of hand rules that lost before (`b_flip.py` and the asymmetric tilt; `e_mguard.py`).

## Backtest: turbulent IOC gate (`ml_gate.py` = r4_best + class `FVAGate`, `mlg_*` params)

- `fit_gate.py` fits a ridge on 16 live features (the filter state plus the mid and F moves over 0.5 / 1 / 2 / 5 s,
  signed by the take side) to the 5 s markout. It fits two models, one per tape half (split at 3000 s). In the
  backtest, the strategy uses the model fitted on the other half, so every second is out-of-sample. On the other half,
  the naive save is +46k to +68k.
- The gate skips a turbulent IOC take (outside shock windows) when the score is below `mlg_thr`. The fitted weights
  are in `logs/r4ml/gate_turb.json` (git-ignored, so rebuild them with `fit_gate.py`).
- With `mlg_on 0`, the file is identical to r4_best (wf 1081.7k, hold 278.6k).

| mlg_thr | wf delta | t | won | hold delta | won |
| --- | ---: | ---: | ---: | ---: | ---: |
| +1 | −18.0k | −0.7 | 4/12 | | |
| 0 | −13.2k | −0.6 | 4/12 | −5.8k | 1/4 |
| −1 | −15.3k | −0.7 | 2/12 | | |
| −3 | −13.0k | −1.2 | 2/12 | | |
| −6 | +15.2k | +0.8 | 3/12 | −0.1k | 2/4 |

The naive +50k turns into −13k to −18k. A skipped take also leaves the old position in place, and in turbulence that
position's carry is worth more than the take's bad markout. At thr −6 almost nothing is gated, and the +15k comes from
a few folds (3/12 won), so it is noise; hold confirms it at −0.1k.

## Verdict

Dead end. Per-fill markouts overstate the value of skipping a fill, by about 65k here, because every fill also changes
the position. The calm-passive gate was not backtested: it has the same counterfactual problem, and the calm guards
(`c_guard.py`, `e_mguard.py`) already lost. Any future decision model must be trained on the *counterfactual* PnL of
the decision (for example, replaying with and without the action), not on the fill's own markout. Keep r4_best.
