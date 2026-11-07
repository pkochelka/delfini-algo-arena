# Four basic market-making strategies

These are standalone learning baselines, not optimized competition submissions.
Only strategy Python files live in `delfini-algo-arena/strategies_a/`.
This guide, test runner, fixtures, and local results live in `strategies/adam/`.

## Common behavior

- At most one outstanding order per side, counting submissions and cancellations in flight.
- Internal position cap is 80 lots on either side, below the competition's 100-lot limit.
- A buy reserves capacity immediately on submission, until filled, canceled, or rejected.
  A cancellation request does not release capacity. The strategy waits for confirmation.
- Private fill events update inventory and remaining reserved quantities exactly once.
- Cancel rejections for already-filled orders are harmless and ignored once the order is gone.
- Quote prices use only market-owned orders, excluding our own liquidity from the signal.
- Prices never deliberately cross the current market. A LIMIT order can nevertheless
  execute aggressively if prices move during the exchange's 5 ms latency.
- Decisions occur on market events, at most once per `REFRESH_US`; this is not a timer.
- Existing quotes keep their queue position if the target price has not moved enough.
  Size parameters apply to NEW orders: these strategies do not continually resize resting orders.
- If either side of the visible market book disappears, accepted quotes are canceled.
- No automatic end-of-tape liquidation: final inventory is valued by the simulator's
  liquidation-based markout. The strategies do not know a round's final timestamp.

The inventory reservation invariant is:

```
inventory + outstanding_buy_quantity <= POSITION_LIMIT
inventory - outstanding_sell_quantity >= -POSITION_LIMIT
```

This assumes the simulator's documented complete, ordered private-event feed.

## How they differ

| File | Logic | Intended role | Main weakness |
| --- | --- | --- | --- |
| `basic_touch.py` | Fixed 5-lot quotes at the best market bid and ask; stops adding exposure at its cap. | Simple reference for spread capture. | Accumulates inventory under directional flow; quotes can be picked off. |
| `basic_inventory.py` | Moves the quote center away from accumulated inventory and changes new quote sizes to favor reducing it. | Study inventory control independently of predictive signals. | Reducing exposure may sacrifice good trades; no explicit trend or toxicity model. |
| `basic_imbalance.py` | Adds a top-of-book volume imbalance shift to the inventory strategy. | Test whether displayed liquidity predicts short-term price direction. | Displayed depth can cancel or be misleading, and the relationship can change across rounds. |
| `basic_volatility.py` | Adds volatility-dependent quote width to the inventory strategy. | Reduce exposure to volatile periods. | Can become almost inactive; lagging volatility estimates do not identify all adverse selection. |

### 1. Fixed-size touch quoting

With defaults, buy at the best bid and sell at the best ask, five lots per new order.
The bid and ask are replenished after terminal events when a later market event permits it.
Inventory has no effect on pricing or size until the capacity cap binds.

Tune `BASE_SIZE` first. Larger size increases potential spread income and losses together;
actual fills also depend on queue position and incoming trade sizes. Tune `REFRESH_US`
and `REQUOTE_TICKS` to balance stale prices against losing queue priority.

### 2. Inventory-based quoting and sizing

Let `I` be inventory and `L = POSITION_LIMIT`:

```
center = mid - INVENTORY_SKEW_TICKS * I / L
buy_size  = round(BASE_SIZE * clamp(1 - SIZE_SKEW * I/L, 0.1, 2))
sell_size = round(BASE_SIZE * clamp(1 + SIZE_SKEW * I/L, 0.1, 2))
```

New quantities are additionally clipped to reserved capacity. With the defaults,
`I=40`, `L=80`, and `BASE_SIZE=5`, the center shifts down one tick; new buy/sell
sizes become approximately 2 and 8 lots (Python rounds ties to even).
The desired bid/ask are rounded to integer ticks and clamped to remain passive,
so a shifted center often makes one side less aggressive while leaving the other at the touch.

Increase `INVENTORY_SKEW_TICKS` for stronger price-based rebalancing.
Increase `SIZE_SKEW` for stronger size asymmetry. Set either to zero to disable
that mechanism and compare it with the same strategy otherwise unchanged.

### 3. Imbalance-aware quoting

Aggregate market volume at the best bid (`B`) and ask (`A`), then compute:

```
imbalance = (B - A) / (B + A)
center = mid + IMBALANCE_TICKS * imbalance
             - INVENTORY_SKEW_TICKS * I / L
```

Large bid depth raises the center; large ask depth lowers it. This is a heuristic,
not a fitted probability model or the full microprice formula. It uses only the touch,
not all visible depth, and does not use the public trade list.

Set `IMBALANCE_TICKS=0` to recover the inventory variant. Increase it only if
chronological validation shows imbalance predicts future prices after 5 ms latency
and after accounting for the prices at which you actually get filled.
The data explorer's Predictability tab is useful for investigating this.

### 4. Volatility-adaptive quoting

Estimate a time-decayed price variance rate from successive midpoint changes:

```
weight = 1 - exp(-dt / VOL_TAU_S)
variance_rate = (1-weight)*variance_rate + weight*(mid_change**2 / dt)
half_width = clamp(VOL_MULT * sqrt(variance_rate * RISK_HORIZON_S),
                   MIN_HALF_SPREAD, MAX_HALF_SPREAD)
```

`dt` is in seconds; the variance rate is ticks squared per second. Equal-timestamp
updates skip the variance update. This estimator is a basic approximation and is
sensitive to sampling, midpoint jumps, and microstructure noise. It is not the
Avellaneda-Stoikov optimal spread formula or a calibrated fill-intensity model.

Reduce `VOL_MULT` if quotes sit too far away and fills disappear. Increase it if
fast moves dominate losses. Smaller `VOL_TAU_S` reacts faster but is noisier;
larger values react more slowly. `RISK_HORIZON_S` is a modeling assumption about
inventory exposure duration, not a hard holding-period limit.

## Parameter ranges for initial experiments

These are candidate grids, not evidence-backed optimal values.

| Parameter | Default | Initial candidates | What changes |
| --- | --- | --- | --- |
| `BASE_SIZE` | 5 | 1, 3, 5, 10 | New-order size, capacity usage, and risk. |
| `POSITION_LIMIT` | 80 | 20, 40, 60, 80 | Internal exposure ceiling; retain a buffer below 100. |
| `REFRESH_US` | 100000 | 20000, 100000, 500000 | How often market events can trigger quote decisions. |
| `REQUOTE_TICKS` | 1 | 1, 2, 3 | How much target-price movement justifies cancellation. |
| `INVENTORY_SKEW_TICKS` | 2 except touch | 0, 1, 2, 4 | Price response at the internal cap. |
| `SIZE_SKEW` | 1 except touch | 0, 0.5, 1 | Inventory response through new-order size. |
| `IMBALANCE_TICKS` | 1.5 in imbalance | 0, 0.5, 1, 1.5 | Strength of the liquidity signal. |
| `VOL_MULT` | 1 in adaptive | 0.1, 0.25, 0.5, 1 | Width response to estimated volatility. |
| `VOL_TAU_S` | 10 | 1, 5, 10, 30 | Volatility memory in seconds. |

Change one parameter at a time first. Compare PnL, traded volume, maximum absolute
inventory, end inventory, and penalties together. A larger size can produce a larger
PnL simply by taking more risk. Very few fills can make a PnL comparison uninformative.

Use real public round data when available. Develop on earlier chronological periods,
then evaluate unchanged parameters on later periods or separate tapes. Replaying
isolated CSV slices without rebuilding the initial book is not a valid substitute.
Avoid selecting parameters from the synthetic sample or the private leaderboard.

## Local smoke-test results

One full replay of the supplied synthetic `data/sample.csv`, using unmodified
simulator defaults and the strategy defaults above:

| Strategy | PnL | Traded volume | Maximum absolute inventory | Final inventory | Breach penalty |
| --- | ---: | ---: | ---: | ---: | ---: |
| Touch | 1675 | 465 | 35 | 5 | 0 |
| Inventory | 462 | 472 | 24 | -12 | 0 |
| Imbalance | 67 | 423 | 19 | -7 | 0 |
| Volatility | -90 | 10 | 5 | 0 | 0 |

These results verify operation and illustrate different behavior; they do not
establish expected profitability or a ranking for competition rounds. The sample
is intentionally synthetic/trending. The adaptive variant's defaults were nearly
inactive; its width multiplier is an obvious first diagnostic, not a promised fix.

## Commands

Open PowerShell and change to the repository:

```powershell
cd "C:\Users\adam\Documents\Codex\2026-11-07\new-chat-2\delfini-algo-arena"
```

Run the local checker and compare every baseline on the sample:

```powershell
.\.venv\Scripts\python.exe -B strategies\adam\check_basic_strategies.py
```

It checks import/source restrictions conservatively, one class per file, syntax,
submission size, partial fills, cancellation races, rejected submissions, missing
market sides, pending-order duplication, and forced one-sided fills to both internal
caps. It then runs the simulator on an alternating-flow fixture and the sample,
checking inventory and zero penalties. It is not the organizers' sandbox checker.
Machine-readable results and PnL series go to `strategies/adam/basic_strategy_checks/`.

To compare real round tapes instead:

```powershell
.\.venv\Scripts\python.exe -B strategies\adam\check_basic_strategies.py data/round1.csv
```

Backtest an individual strategy with results only:

```powershell
.\.venv\Scripts\python.exe -B -m simulator.main --strategy strategies_a/basic_inventory.py --backtest data/sample.csv --log-dir strategies/adam/inventory_sample --no-logs
```

For visual inspection, use full logs and the original backtest explorer:

```powershell
.\.venv\Scripts\python.exe -B -m simulator.main --strategy strategies_a/basic_inventory.py --backtest data/sample.csv --log-dir strategies/adam/inventory_sample_full
.\.venv\Scripts\python.exe -B -m explorer.main --log-dir strategies/adam/inventory_sample_full
```

Open http://127.0.0.1:8050. Keep PowerShell running; Ctrl+C stops the explorer.

To submit, upload only your chosen `.py` file from `strategies_a`, never the checker
or this guide. Each strategy is self-contained. Sandbox acceptance and the 15-minute
budget on a full competition tape still need verification on the organizer platform.
Keep local checks distinct from the limited official submission budget.

Preserve experiments with descriptive filenames or parameters and meaningful commits.
The competition requires regular commits, including failed experiments, without
squashing, rebasing, or force-pushing. Nothing in this work was uploaded or submitted.
