# LOB strategies for Algo Trade Arena: reading list

Written 2026-10-11. Every link below was opened and checked to point at the paper named.

## How this fits what we have

The `ata` simulator already gives us the things these strategies care about: three fill models (`trade_through`, `queue`, `touch`), order latency, maker/taker fees, markouts at 1s and 10s (an adverse-selection gauge), and a per-step time limit in the Arena. The starter Strategies already cover the basics: `fixed_spread` (quote around the mid), `inventory_skew` (Avellaneda-Stoikov style), `imbalance_lean` (move quotes towards where the book imbalance points) and `imbalance_taker` (cross the spread when the book is heavily one-sided).

Most of the literature splits into two questions, and a good entry answers both:

1. **Where is the fair price right now, and where is it going in the next few seconds?** (signals from the book)
2. **Given that, where do I quote, how big, and how much inventory do I tolerate?** (the market-making control problem, which is what Avellaneda-Stoikov answers)

Avellaneda-Stoikov on its own uses the plain mid and assumes no short-term predictability. The cheapest big improvement is usually to swap the mid for a better fair price and add a signal-driven skew. That is the thread running through sections 1 to 3.

## Suggested reading order (if you only read six)

1. Stoikov, *The micro-price* (section 1). Short, practical, has code.
2. Cont, Kukanov, Stoikov, *The price impact of order book events* (section 2). Defines Order Flow Imbalance (OFI).
3. Guéant, Lehalle, Fernandez-Tapia, *Dealing with the inventory risk* (section 3), read alongside the hftbacktest GLFT tutorial.
4. Cartea, Donnelly, Jaimungal, *Enhancing trading strategies with order book signals* (section 3). Imbalance signal inside a market maker.
5. Moallemi, Yuan, *A model for queue position valuation* (section 4). Why our `queue` fill model matters.
6. Kolm, Turiel, Westray, *Deep order flow imbalance* (section 6). What actually works if we try ML.

---

## 1. Better fair price: microprice and queue imbalance

Use the top-of-book sizes to estimate where the next mid will be. Simplest form, with bid/ask prices `Pb, Pa` and sizes `Qb, Qa`:

- imbalance `I = Qb / (Qb + Qa)`
- weighted mid `= Pa * I + Pb * (1 - I)`

The weighted mid is noisy; Stoikov's microprice fixes this by estimating the expected future mid from data as a function of imbalance and spread. Plug it in wherever a strategy uses the mid.

- **Stoikov (2018), The micro-price: a high-frequency estimator of future prices.** Quantitative Finance 18(12).
  Paper (paywalled record): https://ideas.repec.org/a/taf/quantf/v18y2018i12p1959-1966.html
  Code, sample data and a link to the free SSRN preprint: https://github.com/sstoikov/microprice
- **Gould & Bonart (2015), Queue imbalance as a one-tick-ahead price predictor in a limit order book.** Shows plain top-of-book imbalance predicts the direction of the next mid move, with logistic regression. Good baseline to beat. https://arxiv.org/abs/1512.03492
- **Lipton, Pesavento & Sotiropoulos (2013), Trade arrival dynamics and quote imbalance in a limit order book.** Links imbalance to which side gets hit next. https://arxiv.org/abs/1312.0514

## 2. Order Flow Imbalance (OFI)

Instead of the book's state, use its changes: how much size was added or removed at the best bid and ask between updates. OFI over a short window explains short-horizon price changes roughly linearly (`dP ≈ beta * OFI / depth`), and it is cheap to compute on every update.

- **Cont, Kukanov & Stoikov (2010), The price impact of order book events.** The original OFI definition and the linear price-impact result. Start here. https://arxiv.org/abs/1011.6402
- **Xu, Gould & Howison (2019), Multi-level order-flow imbalance in a limit order book.** Extends OFI to deeper levels; helps if we get more than level 1. https://arxiv.org/abs/1907.06230
- **Cont, Cucuringu & Zhang (2021), Cross-impact of order flow imbalance in equity markets.** Only relevant if the Arena gives us several related instruments. https://arxiv.org/abs/2112.13213

## 3. Inventory-aware market making beyond Avellaneda-Stoikov

These keep the A-S idea (quote around a reservation price, skew with inventory) but are easier to use or add signals.

- **Avellaneda & Stoikov (2008), High-frequency trading in a limit order book.** The baseline, for reference. https://www.math.nyu.edu/~avellane/HighFrequencyTrading.pdf
- **Guéant, Lehalle & Fernandez-Tapia (2011), Dealing with the inventory risk: a solution to the market making problem.** "GLFT". Adds an inventory limit and gives a closed-form approximation: half-spread and per-unit inventory skew depend only on risk aversion, volatility and the fill-intensity curve `lambda(delta) = A * exp(-k * delta)`. No terminal-time dependence, so it suits a continuous session better than A-S. https://arxiv.org/abs/1105.3115
- **hftbacktest tutorial: GLFT market making model and grid trading.** The most practical resource on this list: code that fits `A` and `k` from trade arrivals vs distance from mid, recalibrates every few seconds, and quotes a grid. Translates almost directly into an `ata` strategy. https://hftbacktest.readthedocs.io/en/latest/tutorials/GLFT%20Market%20Making%20Model%20and%20Grid%20Trading.html
  Simpler, volatility-only variant: https://hftbacktest.readthedocs.io/en/latest/tutorials/High-Frequency%20Grid%20Trading%20-%20Simplified%20from%20GLFT.html
- **Cartea, Donnelly & Jaimungal (2018), Enhancing trading strategies with order book signals.** Applied Mathematical Finance 25(1). Builds a volume-imbalance signal, puts it into the market maker's optimal quotes, and shows out-of-sample gains from lower adverse selection on 11 Nasdaq stocks. Closest paper to "A-S plus imbalance", i.e. a principled version of our `imbalance_lean`. https://ideas.repec.org/a/taf/apmtfi/v25y2018i1p1-35.html
- **Cartea & Wang (2020), Market making with alpha signals.** IJTAF 23(3). Same idea with a short-term momentum signal from market-order flow; the market maker trades directionally when the signal is strong. https://ideas.repec.org/a/wsi/ijtafx/v23y2020i03ns0219024920500168.html
- **Cartea, Jaimungal & Ricci (2014), Buy low, sell high: a high frequency trading perspective.** SIAM J. Financial Math. Market making with order-flow-driven price jumps and adverse selection. https://epubs.siam.org/doi/10.1137/130911196

## 4. Queue position, adverse selection, and when to pull quotes

Under the `queue` fill model, being early in the queue is worth money and being filled at the back of a thinning queue usually means the price is about to move against us. Our `markout_1s` / `markout_10s` metrics measure exactly this.

- **Moallemi & Yuan (2016/2017), A model for queue position valuation in a limit order book.** Values a resting order by its queue position, splitting it into spread capture minus adverse selection. Explains why cancelling and re-joining at the back is costly. https://moallemi.com/ciamac/papers/queue-value-2016.pdf
- **Lehalle & Mounjid (2016), Limit order strategic placement with adverse selection risk and the role of latency.** When to keep or cancel a resting order given imbalance, and how latency changes that. Directly relevant to our `--latency` setting. https://arxiv.org/abs/1610.00261
- **Easley, López de Prado & O'Hara (2012), Flow toxicity and liquidity in a high frequency world.** The VPIN toxicity measure: when order flow is one-sided, widen or step back. https://www.stern.nyu.edu/sites/default/files/assets/documents/con_035928.pdf

## 5. Short-horizon directional trading on signals

Take liquidity only when the predicted move beats the half-spread plus taker fee; otherwise just skew passive quotes.

- **Lehalle & Neuman (2017), Incorporating signals into optimal trading.** Closed-form trading rate with a mean-reverting signal, checked on European order-book data. https://arxiv.org/abs/1704.00847
- **Cont & de Larrard (2011), Price dynamics in a Markovian limit order market.** Probability the next mid move is up, from queue sizes. Theory behind section 1. https://arxiv.org/abs/1104.4596

## 6. Machine learning on the book

Worth a look, but mind the Arena's per-step time limit and overfitting on a few days of data. Kolm et al. find simple models on order flow (OFI-like features) beat deep models on raw book snapshots, which suggests: hand-built OFI and imbalance features plus a small linear or gradient-boosted model.

- **Kolm, Turiel & Westray (2023), Deep order flow imbalance: extracting alpha at multiple horizons from the limit order book.** Mathematical Finance 33(4). https://ideas.repec.org/a/bla/mathfi/v33y2023i4p1044-1081.html
- **Zhang, Zohren & Roberts (2018), DeepLOB: deep convolutional neural networks for limit order books.** The standard deep-learning baseline. https://arxiv.org/abs/1808.03668
- **Ntakaris et al. (2017), Benchmark dataset for mid-price forecasting of limit order book data (FI-2010).** The usual public benchmark, if we want extra data to test features on. https://arxiv.org/abs/1705.03233

## 7. Reinforcement learning for market making

Interesting, but training something robust in a hackathon is hard. A cheaper middle ground: tune a GLFT-style strategy's few parameters with `ata sweep` / `ata cv`.

- **Spooner, Fearnley, Savani & Koukorinis (2018), Market making via reinforcement learning.** https://arxiv.org/abs/1804.04216
- **Gašperov, Begušić, Posedel Šimović & Kostanjčar (2021), Reinforcement learning approaches to optimal market making.** Survey; open access. https://www.mdpi.com/2227-7390/9/21/2689

## 8. Simulating order books

- **Huang, Lehalle & Rosenbaum (2013), Simulating and analyzing order book data: the queue-reactive model.** Order arrival rates depend on queue sizes. Could back more realistic generated Datasets than shuffles. https://arxiv.org/abs/1312.0563

## Books

- **Cartea, Jaimungal & Penalva (2015), Algorithmic and High-Frequency Trading.** Cambridge. Chapter on market making covers A-S, GLFT-style models and adding signals. Book page with sample code and data: https://sebastian.statistics.utoronto.ca/?p=310
- **Bouchaud, Bonart, Donier & Gould (2018), Trades, Quotes and Prices: Financial Markets Under the Microscope.** Cambridge. The empirical side: order-flow persistence, impact, queue dynamics. https://doi.org/10.1017/9781316659335

## A possible build order for us

1. Replace the mid with a microprice (or weighted mid) in `fixed_spread` and `imbalance_lean`, and check `markout_1s` improves.
2. Add an OFI feature over a short window; measure how well it predicts the next mid change on our Datasets.
3. Write a GLFT strategy: fit `A`, `k` and volatility on a rolling window, quote half-spread plus inventory skew around the microprice, shifted by the OFI/imbalance signal.
4. Add a quote-pulling rule when the signal is strongly against a resting side (section 4).
5. Tune with `ata sweep` and validate with `ata cv` under all three fill models, since queue-position effects are where fill models disagree most.

---

## Implemented in this repo

Each idea above that can run as a Strategy now has one in `strategies/`. Shared pieces (Order Flow Imbalance, online ridge regression, a lag buffer, quoting around a fair price) live in `src/ata/core/signals.py`, so they ship with a Submission. Learned signals are fitted online from data the Strategy has already seen during the Run, except in `deeplob_mm` and `rl_mm`, which load weights trained offline (see below).

| Strategy | Section | What it does | Candidate |
|---|---|---|---|
| `microprice_mm` | 1 | Quotes around a micro-price learned online: the mean mid move over the next 20 steps per imbalance x spread bucket (size-weighted mid until a bucket has data). | `micro-mm` |
| `ofi_lean` | 2 | Leans quotes on an EW sum of multi-level OFI, in units of average queue size. | no |
| `glft` | 3, 4 | GLFT spread and inventory skew from intensity (A, k) and volatility fitted online; optional imbalance/OFI lean, quote pulling and requote hysteresis. | `glft-mm` |
| `vpin_mm` | 4 | Microprice quotes; drops price improvement when VPIN is high and pulls the side one-sided flow is hitting. | no |
| `signal_taker` | 5 | Crosses the spread only when an online forecast (imbalance, OFI, last move) beats spread + 2 taker fees. | `signal-taker` |
| `lead_lag_mm` | 2 (cross-impact) | Leans each instrument on every instrument's recent moves via online regression. | `lead-lag-mm` |
| `linear_alpha_mm` | 6 | Online ridge on 9 order-flow features (imbalances, microprice, OFI fast/slow, trade flow, spread); fair = mid + predicted move. | `linear-alpha-mm` |
| `avellaneda_stoikov` | 3 | The paper's reservation price and spread (eqs. 29-30) with volatility and k fitted online; a rolling 1 s horizon by default, or a finite session. | no |
| `order_flow_mm` | 3 (Cartea & Wang; Cartea, Jaimungal & Ricci) | Shifts quotes by a market-order-flow alpha whose slope it learns online, and switches the alpha off where flow has not led the mid. | `order-flow-mm` |
| `queue_mm` | 4 | `microprice_mm`'s fair price; keeps, joins, improves or pulls each quote by the value of its queue place (Moallemi-Yuan, Lehalle-Mounjid), and learns from its own fills whether queue priority pays. | `queue-mm` |
| `queue_race` | 5 (Cont & de Larrard) | Probability that the next mid move is up from the level-1 queues, solved for arrival and departure rates fitted online; scaled to the 20-step move; takes only when that beats 1.5x the crossing cost. | `queue-race-mm` |
| `deeplob_mm` | 6 | A small DeepLOB network (trained offline in numpy on generated data) forecasts the 20-step mid move; quotes lean on it, with its scale re-fitted online. | `deeplob-mm` |
| `rl_mm` | 7 | SARSA over tile-coded state (Spooner et al.), trained offline on generated data; acts greedily, with 9 quote pairs and a market order that clears inventory. | `rl-mm` |

Not implemented: the queue-reactive model (section 8) is a way to generate Datasets, not a Strategy.

**Trained Strategies.** `deeplob_mm` and `rl_mm` load a weights file from `models/`, listed under their Candidate's `files:` so it ships with the Submission (it also fits a `--single-file` build: 24 KB and 183 KB). The training scripts are in `src/ata/train/` and skip the comparison seeds 2 and 3:

```bash
uv run python -m ata.train.deeplob --out models/deeplob.npz   # ~8 min on one core; --datasets r1 ... for Round data
uv run python -m ata.train.rl_mm --out models/rl_mm.npz       # ~13 min on one core; --datasets r1 ... too
```

The DeepLOB network is trained in numpy with hand-written backprop: torch's CPU wheels could not be downloaded here, and the default PyPI wheel brings several GB of CUDA packages. It is scaled down (8 filters, a 16-unit LSTM) to train in minutes and run in about 1 ms per step. Both are trained on generated data only, so they learn its planted signals; retrain them on Round data before trusting them.

### Results on generated Datasets

Mean PnL per scenario, 2 seeds x 0.5 h each, default parameters. These Datasets have Planted signals, so they show whether a Strategy finds a signal that is there and stays safe where there is none; they say little about the real Arena.

`trade_through` fill model (pessimistic):

| Strategy | calm | volatile | imbalance | toxic | pair |
|---|---|---|---|---|---|
| fixed_spread (starter) | 20.1 | 23.2 | 4.6 | -4.7 | -43.4 |
| imbalance_lean (starter) | 5.0 | 18.0 | 124.6 | 4.1 | -6.1 |
| imbalance_taker (starter) | -102.7 | -103.6 | -25.4 | -107.7 | -177.9 |
| microprice_mm | 5.9 | 22.2 | 269.2 | 10.9 | 1.3 |
| ofi_lean | 5.0 | 23.0 | 39.4 | 11.4 | -9.8 |
| glft | 3.6 | 25.4 | -1.3 | 8.7 | 1.9 |
| glft (imb_lean 1.5, skew_mult 0.3) | 5.3 | 18.9 | 129.2 | 2.6 | -6.5 |
| vpin_mm | 4.2 | 14.4 | 69.8 | 8.7 | 0.6 |
| signal_taker | 0.0 | -0.1 | 266.9 | 0.0 | 0.0 |
| lead_lag_mm | 5.3 | 22.6 | 46.3 | 7.3 | 51.7 |
| linear_alpha_mm | 5.2 | 20.0 | 286.8 | 9.9 | 4.2 |
| avellaneda_stoikov | 3.7 | 23.8 | -1.6 | 9.7 | 0.8 |
| order_flow_mm | 3.8 | 23.3 | -1.6 | 7.8 | 3.4 |
| queue_mm | 21.4 | 37.1 | 286.5 | -0.2 | 3.7 |
| queue_race | 3.4 | 21.8 | **335.2** | 8.4 | 1.6 |
| deeplob_mm | 2.4 | 20.5 | 284.8 | 8.0 | 2.9 |
| rl_mm | 2.2 | 31.9 | 83.1 | 3.4 | 26.5 |

`queue` fill model:

| Strategy | calm | volatile | imbalance | toxic | pair |
|---|---|---|---|---|---|
| fixed_spread (starter) | -0.7 | 22.3 | -4.5 | 11.8 | -36.7 |
| imbalance_lean (starter) | 7.7 | 28.8 | 134.7 | 4.0 | -1.2 |
| microprice_mm | 12.9 | 34.4 | 278.6 | 16.2 | 10.3 |
| ofi_lean | 8.6 | 35.2 | 46.0 | 18.0 | -1.8 |
| glft | 7.4 | 34.8 | -4.2 | 14.7 | 8.0 |
| vpin_mm | 7.3 | 21.8 | 80.7 | 11.4 | 5.4 |
| signal_taker | 0.0 | -0.1 | 266.9 | 0.0 | 0.0 |
| lead_lag_mm | 9.7 | 35.5 | 51.0 | 14.3 | 61.9 |
| linear_alpha_mm | 9.8 | 32.6 | 300.8 | 17.0 | 6.5 |
| avellaneda_stoikov | 7.4 | 37.4 | -5.3 | 16.8 | 7.4 |
| order_flow_mm | 7.2 | 37.0 | -6.3 | 16.0 | 7.6 |
| queue_mm | **48.3** | **66.7** | 300.1 | **33.1** | 32.8 |
| queue_race | 7.6 | 34.9 | **341.5** | 16.1 | 7.8 |
| deeplob_mm | 7.1 | 33.5 | 295.6 | 13.8 | 8.0 |
| rl_mm | 3.8 | 36.3 | 90.8 | 5.8 | 31.4 |

What we learned from building them:

- **Microprice first.** `microprice_mm` is the best all-rounder: never negative, and it doubles `imbalance_lean` where imbalance predicts. `linear_alpha_mm` edges it on imbalance and can pick up other features on real data.
- **OFI is weak on generated data** because the generator redraws book sizes every update, so size changes carry no information. Re-measure on Round data before trusting `ofi_lean` or the OFI features.
- **Generated toxic flow cannot be dodged.** Informed trades arrive in the same step as the move, so `vpin_mm` and the trade-flow feature have nothing to react to; real markets may differ.
- **GLFT has no alpha.** On generated data its fitted half spread falls below the 1-tick floor, so it behaves like `fixed_spread` with a fitted inventory skew. Its value is that the spread adapts if Round data is wider or more volatile; check the fitted A and k on real data.
- **Only `lead_lag_mm` pays when instruments are related.** If a Round has several instruments, measure lead-lag first.
- **Learners need a warm-up** (500 to 1000 steps) and do not yet pre-train on `obs.history`.
- **Queue management pays only if fills follow price-time priority.** `queue_mm` makes two to three times `microprice_mm` under the `queue` Fill model on every scenario, but loses on toxic under `trade_through` and trails on imbalance and pair under `touch`. Run `ata run --all-fill-models` on Round data and pick by whichever Fill model matches the Arena's fills.
- **`queue_race` is the best on imbalance, but not because of the queue race.** Swapping its p_up for the plain size ratio costs only about 11 PnL: the gain over `microprice_mm` comes from fitting a continuous scale to the 20-step move instead of buckets. The generator has no real queue dynamics (sizes are redrawn every update), so the race model is untested until Round data; check its fitted `ratio` and `unit` there.
- **Trained Strategies learn only what the generator planted.** `deeplob_mm` matches `linear_alpha_mm` on imbalance at three times the cost per step. `rl_mm` is the only Strategy positive in all ten cells and the best on volatile and pair, but it makes under a third of the others on imbalance. Both need retraining on Round data.
- **Avellaneda-Stoikov behaves like GLFT here**: both fitted spreads fall below the 1-tick floor. With a finite `session`, gamma must be far smaller (about 5e-5 for 30 minutes) or the inventory term swamps the quotes.
- **Market-order flow does not lead the mid in any generated scenario**, so `order_flow_mm` switches its alpha off (check `strategy.gate` on Round data). On a test Dataset where trades lead the move by 3 to 5 updates it gains 20 to 35 over flow-off.
