# Qminers trading hackathon starter

```
strategies/   template_strategy.py, an empty strategy to copy and fill in; trend_follower.py, a simple working one
data/         sample.csv.gz, a synthetic hour of trending prices to start on; the rounds downloaded from the website go here too (git-ignored)
logs/         simulator output (git-ignored)
simulator/    the exchange simulator package
explorer/     the backtest explorer package
```

## Setup

[uv](https://docs.astral.sh/uv/) is a fast Python package and project manager; install it with `curl -LsSf https://astral.sh/uv/install.sh | sh` (or `pip install uv`).

```bash
uv sync                                        # Python 3.13 and everything else, into .venv
uv run python -m gzip -d data/sample.csv.gz    # the simulator reads plain CSV only; one file at a time, the same for each downloaded round
```

`data/sample.csv.gz` is synthetic and its price trends: one resting order per side, and every few seconds an aggressive order takes part of the touch, after which the quotes keep moving in that direction tick by tick for a few seconds and then give part of the move back. `strategies/trend_follower.py` is a simple trend-following strategy: it sends a one-lot IOC order at the touch in the direction of the trades it has just seen. The rounds are recorded market data and look nothing like the sample.

## Writing a strategy

A strategy is one `.py` file defining one subclass of `BaseStrategy` (with several, the alphabetically first is used), constructible with no arguments:

```python
from simulator.models import OrderBookSnapshot, OrderRequest, OrderType, PrivateOrderEvent, Side, Trade
from simulator.strategy import BaseStrategy


class MyStrategy(BaseStrategy):
    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        """After every market event: the public book and all trades since the previous call."""

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        """Every state change of your own orders."""
```

| Call                                                   | Effect                                                         |
| ------------------------------------------------------ | -------------------------------------------------------------- |
| `self.send_order(OrderRequest(side, price, quantity, order_type, client_order_id=None))` | submit; returns the order id (`u000000001`, ...) immediately |
| `self.cancel_order(order_id)`                          | cancel your resting order                                      |
| `self.modify_order(order_id, new_quantity)`            | shrink your resting order                                      |
| `self.now_us()`                                        | simulation time in µs                                          |

`Side.BUY` / `Side.SELL`, `OrderType.LIMIT` / `OrderType.IOC`; prices and quantities are positive integers (prices in ticks).

Submissions run on Python 3.13 with numpy 2.5.4, pandas 3.0.6, scipy 1.18.1 and scikit-learn 1.9.1 available, and are statically checked first: besides the standard library and those packages, import only `simulator.models` and `simulator.strategy.BaseStrategy`. Modules giving system, network, file, threading or introspection access (`os`, `sys`, `io`, `socket`, `subprocess`, `threading`, `importlib`, `inspect`, `gc`, ...), builtins like `open`/`eval`/`exec`/`getattr`/`type`, dunder attributes and `str.format` are rejected.

## Running a backtest

```bash
uv run exchange-simulator --strategy strategies/trend_follower.py --backtest data/sample.csv --log-dir logs/sample --progress
```

`logs/sample/result.json` holds the outcome (`pnl`, `cash`, `inventory`, `mark_price`, `breach_penalty`, `traded_volume`, `fills`, `pnl_se`, time-weighted quantiles of `|inventory|`) and `pnl.csv` the PnL sampled every second. Without `--no-logs` the directory also gets `run.jsonl`, `events.csv`, `book_deltas.csv` and `trades.csv`, several hundred MB per full round. Use `--no-logs` when you only need the result and the PnL curve; it is much faster. `--public-depth N` (default 20) sets how many ticks of the book you see.

## Walk-forward evaluation

```bash
uv run walk-forward --strategy strategies/peter/arb_mm.py --backtest data/round1.csv --folds 10 --warmup-s 60
uv run walk-forward-report --run-dir logs/wf/arb_mm-round1 --open    # rebuild / open the HTML report
```

The tape is cut into `--folds` consecutive eval windows (`--end-frac 0.75` keeps the last quarter held out). Each fold is a fresh, independent replay (in parallel, `--jobs`) of a file starting `--warmup-s` before its window, with the resting book seeded; fold files are cached in `data/.cache/walkforward/`. A fold scores the PnL earned inside its window. Strategies with a `trade_start_us` key in `PARAMS` are told not to trade during the warmup; others trade through it and carry that inventory in (shown as `warmup_pnl`). `--params tuned.json` applies `PARAMS` overrides (flat, or `tune.py`'s `{"params": ...}`).

`logs/wf/<strategy>-<tape>/` (or `--out-dir`) gets `config.json`, `summary.json` (total/mean/median/std of fold PnL, fold Sharpe, t-stat, win rate, worst fold, PnL/hour, stitched max drawdown, breaches, inventory), `folds.csv`, `pnl/<fold>.csv` and `report.html`: KPI tiles, the stitched out-of-sample equity curve, PnL / drawdown / inventory quantiles / fills per fold, each fold's PnL curve, and the fold table.

## Exploring a backtest

```bash
uv run backtest-explorer --log-dir logs/sample     # http://127.0.0.1:8050, --port to change
```

A Dash app over a full-logs run (not `--no-logs`): best bid/ask with your fills marked, your PnL over time, the reconstructed order book at any clicked time, and the event log.

## Strategy results report

```bash
uv run results-explorer logs/r1_ladder_mm --open                       # -> logs/r1_ladder_mm/report.html
uv run results-explorer logs/a logs/b --out logs/a-vs-b.html --open    # comparison tab + one tab per run
uv run results-explorer --all                                          # every run under logs/ -> logs/results-report.html
```

A self-contained interactive HTML report per run, plus `results_summary.json` (every metric) in each run directory. It reads only what the simulator wrote, plus the cached tape replay from `analysis/` for the market mid.

- **Tiles:** PnL, PnL/hour, max drawdown, per-minute Sharpe and t-stat, winning minutes; inventory (mean |inv|, q95, time at limit / flat, breach time); execution (fills, maker share, edge and markouts per unit, PnL/unit); orders (fill ratios, orders per fill, cancels that lost the latency race, order lifetime, two-sided presence, time at the touch). Hover a tile for its definition.
- **Timeline** (one shared zoom): PnL split into *spread edge* (side·(mid just before the fill − price)·qty) and *inventory PnL* (mark-to-mid minus edge, which is where adverse selection shows up), inventory against ±limit, market bid/ask with your fills and quotes, and drawdown.
- **Charts:** PnL per bucket, per-minute PnL distribution, markout curve by horizon (maker/taker/buys/sells), time-weighted inventory distribution, fill edge and sizes, volume bought/sold, order outcomes, lifetimes and placement, and quote distance from the touch.

Runs made with `--no-logs` get only the PnL-based metrics and charts; re-run without it to get the rest.

## Exploring the market data

```bash
uv run data-explorer --data data/round1.csv [data/round2.csv ...]   # http://127.0.0.1:8051, --port to change
uv run data-explorer --data data/round1.csv --export                 # every tab to data_explorer/round1/, no server
uv run python -m analysis.fit data/round1.csv [--window 0 1800]      # fitted strategy parameters, also written to data/.cache/
uv run pytest                                                        # replay parity with the simulator
```

`--export` (optionally `--window T0 T1`, `--dt 0.1`) writes `data_explorer/<tape>/README.md` with every tile, fitted parameter, table, note and a numeric digest of each chart, `summary.json`, the data behind each chart in `csv/<tab>/`, and the interactive `report.html` (git-ignored); with several tapes also `data_explorer/compare-<a>-<b>/`. Re-run it after adding a chart or a round, and read the README there before re-deriving a statistic.

Unlike `backtest-explorer`, this reads the raw tapes, not a simulator run. `analysis/` replays a tape with the simulator's matching rules (inferring the trades, which the CSV doesn't contain) and caches the result in `data/.cache/`, so only the first load of a round is slow. Drag across the timeline to analyse a window; `?tab=<name>` links a tab.

| Tab | What it answers |
| --- | --- |
| Overview | Headline stats, fitted A-S parameters ready to paste (`SIGMA_INIT`, `K_INIT`, A, OFI beta), activity per bin |
| Price & spread | Bid/ask/microprice with aggressive orders, spread over time and distribution, depth heatmap |
| Volatility | Rolling σ, volatility signature, return distribution, return ACF, variance ratio, mid-change timing |
| Liquidity & book | Depth profile, touch sizes, queue imbalance, order sizes, order lifetimes, where orders are placed |
| Trades & flow | Aggressive order rate, inter-arrival times, sizes, sweep depth (k), fill intensity λ(δ) = A·e^(−kδ), net flow, sign ACF |
| Predictability | Imbalance → future mid, the same with 5 ms latency, post-trade markouts (adverse selection), OFI regressions |
| Queue fills | Historical passive-order partial/full fill rates by volume ahead in the FIFO queue, with horizon, side, size and placement filters |
| Maker markouts | Volume-weighted maker markouts and outcome percentiles, conditional on pre-fill spread, maker-side imbalance, trailing volatility and aggressor size |
| Latency & cancels | Adverse midpoint moves and stale quote marketability during a delay; fills before hypothetical cancel activation at a selected order age |
| Compare tapes | Every loaded tape side by side |

The same functions are importable for notebooks and fitting scripts: `from analysis import load_tape, features`.

Execution diagnostics are also importable from `analysis.execution`. The new tabs use the
selected timeline window and expose additional controls above the charts. Queue fills
are observed outcomes of actual historical orders, including actual cancellations;
they are not counterfactual fill forecasts for our own quotes. Maker conditions use
the book before each specific taker event, and markouts are signed for the maker
(positive is favorable), weighted by executed quantity. Outcome percentiles are not
confidence intervals. Volatility regimes are descriptive cutoffs within the selected
window, using trailing rather than future price changes.

Orders/fills without enough follow-up before the window ends are excluded. Cancel
experiments count subsequent actual maker fills strictly before a hypothetical
cancel takes effect; a fill at exactly that time is excluded, matching simulator
ordering. They preserve existing historical cancellations and are separate per-order
experiments, not a simultaneous strategy backtest. Price-delay charts use the sampling
selector (capped at 50,000 decision times), independently of order size/placement
filters. Raw tapes do not contain our own cancel requests. Replay cache version 3
adds starting queue volume and resting quantities; old caches rebuild automatically.

## How the exchange works

**One instrument, replayed.** Each CSV row is a market participant's action:

```
Time,Operation,OrderId,Side,Volume,Price,Lifespan
0.003823,Insert,1000001,B,43,9999,G
```

`Time` in seconds; `Operation` is `Insert` or `Cancel` (of `OrderId`); `Side` `B`/`A`; `Lifespan` `G` = resting limit order, `F` = immediate-or-cancel. The tape is fixed and does not react to you; your orders only add to and take from the book it builds.

**Matching.** Price-time priority. An incoming order (market or yours) matches against the opposite side while prices cross, each fill at the resting (maker) order's price; a LIMIT residual rests at the back of its price level, an IOC residual is cancelled. Market orders can trade with yours; your orders never trade with each other. No fees.

**Latency.** Every `send_order`, `cancel_order` and `modify_order` takes effect at $t + 5\text{ms}$, where $t$ is the simulation time at which you call it: it is applied just before the first market event with timestamp $\ge t + 5\text{ms}$, so the book may have moved by then. An order you place is therefore never filled within 5 ms, and a cancel can lose the race against a fill.

**Feeds.**
- `on_order_book_change` fires after every market event, with a snapshot of every individual order (id, side, price, quantity, `owner` `"market"`/`"user"`) within `public_depth` ticks of the best price on each side, and the list of trades (market-vs-market included) since the previous call. Fills caused by your own order activating appear in the next call's `trades`.
- `on_private_order_update` fires synchronously for your orders: `ACCEPTED`, then one `PARTIALLY_FILLED`/`FILLED` per match (`fill_price`, `filled_quantity`, `remaining_quantity`), `CANCELED` (your cancel, or an IOC residual), `MODIFIED`, or `REJECTED`. Cancelling anything but your own resting order, or a modify that doesn't shrink it, is `REJECTED`; a modify of an unknown id is silently ignored.

**Limits and scoring.** With inventory $I$ (signed), resting same-side quantity $R$ and order quantity $q$ ($s = \pm 1$ for buy/sell):

$$|I + s (R + q)| > 1000 \Rightarrow \text{order REJECTED at acceptance}$$

$$\text{penalty} = 10{,}000 \cdot \int \mathbb{1}\big[|I(t)| > 100\big] dt \quad (\$ \text{ per second})$$

$$\text{PnL} = \text{cash} + \text{markout}(I) - \text{penalty}$$

$\text{markout}(I)$ is the cash from liquidating $I$ against the resting *market* orders on the opposite side, best price outwards (volume beyond the book at the last swept price, or at the last traded price if that side is empty). It is computed at the end and every second for `pnl.csv` and `pnl_se`.
