# Peter LadderMM comparison

The original strategies/peter/ladder_mm.py was left unchanged. Overrides were applied to an isolated module for each run.

## Identical evaluation protocol

The exact seeded CSV windows produced using Peter's splitter for adaptive_spread were reused. Six chronological folds use the first 75% of round1; each begins with 60 seconds without trading. All 16 configurations were screened on wf01/wf03, the top three completed wf01–wf04, and wf05/wf06 selected the winner. The selection was saved before any ladder test run.

Selection objective: mean fold PnL − 0.5 × population standard deviation of fold PnL − 5 × mean fold 95th-percentile absolute inventory.

The held-out replay is 9000.001434–11999.999025 seconds; trading starts at 9060.001434 seconds (about 49 minutes). Fresh cash/inventory start at zero. All runs use the official simulator, 5 ms latency, the same price-time matching, liquidation-book marking, and penalty settings. No fees are charged.

The test is excluded from ladder parameter selection, but was already examined for the earlier strategy and descriptive data analysis. It is not a new blind sample or organizer-private data.

## Search space and winner

16 configurations: levels {1,2} × tight_offset {0,1} × topup {0,5} × skew_lots {0,40}. Each uses cap 80, size_top 10, size_deep 10 for two levels (0 for one), max_spread 8, improve_reduce 0, and no timed flattening. This matches the earlier search budget, though strategy-specific parameters differ.

Winner: two levels, tight_offset 0, topup 0, skew_lots 40. The ladder keeps orders at the touch and one tick deeper, shifts the inventory-increasing side away by one tick per 40 units, and disables partial-order topups: an existing wanted order retains its queue position until filled or moved. Multiple levels can coexist, unlike the adaptive strategy's one order per side. Matching cap and individual order size therefore does not make their actual exposure identical.

| Strategy | Test PnL | Volume | Max absolute inventory | 95% absolute inventory | Cap | Max order size |
|---|---:|---:|---:|---:|---:|---:|
| Adaptive spread | 8,999 | 35,892 | 43 | 17 | 80 | 10 |
| Fixed half-spread 1 tick (post-selection diagnostic) | 9,692 | 42,845 | 44 | 18 | 80 | 10 |
| Peter default logic, matched sizing | 9,560 | 86,272 | 80 | 80 | 80 | 10 |
| Peter tuned ladder, matched sizing | 22,254 | 99,558 | 78 | 49 | 80 | 10 |
| Peter original defaults | 107,532 | 648,516 | 100 | 100 | 100 | 200 |

All runs incurred zero inventory penalties. The fixed one-tick row is an earlier post-selection diagnostic, not a model chosen by the original validation search.

The tuned ladder exceeds adaptive PnL by 13,255 (+147.3%), with materially higher realized inventory (95th percentile 49 vs 17). Peter's native defaults use much larger orders and stay near the inventory limit (median absolute inventory 99); their 107,532 PnL is a separate sizing/exposure comparison. One held-out interval does not establish general superiority.

## PnL reconciliation

Tuned ladder: cash −158,268 + final inventory 18 × liquidation mark 10,029 − penalty 0 = 22,254. Maximum recorded PnL drawdown: 480.

Original defaults: cash −895,368 + inventory 100 × mark 10,029 = 107,532. Maximum recorded drawdown: 777.

Matched default logic: cash 109,860 − inventory liability 10 × ask-side liquidation mark 10,030 = 9,560. Maximum recorded drawdown: 1,200.

End marking includes pending user actions processed after the last source event. Inventory is valued by sweeping market-owned opposite-side depth; no liquidation order is submitted.

## Validation and reproducibility

Warmup gate passed; four forced one-sided fill cases respected the cap including pending resting orders; sample simulation matched the unmodified simulator exactly. A second full held-out run through the unmodified ExchangeSimulator matched every result field exactly. Source SHA256 remained unchanged. Every matched run was audited for order size <=10, inventory <=80, zero penalties, and no early trading.

See protocol.json, locked_selection.json, verification.json, official_reproduction.json, outcome.json and comparison.csv. Full Backtest Explorer logs exist in full_logs_L2_o0_t0_s40 and full_logs_peter_original.

From the repository, reproduce the complete study with:
```powershell
.\.venv\Scripts\python.exe -B strategies\adam\ladder_research\research.py --jobs 2
.\.venv\Scripts\python.exe -B strategies\adam\ladder_research\verify.py
.\.venv\Scripts\python.exe -B strategies\adam\ladder_research\report.py
```

Existing matching results are resumed. Do not edit the source or dataset mid-study. Timestamps and file hashes are recorded in protocol.json.
