## What is included

| Location | Contents and purpose |
|---|---|
| `models/` | Unmodified snapshots of the five strategies in `strategies_a`: four basic market makers and `adaptive_spread.py`. The original strategy files remain in `strategies_a`. Packaged basic/spread research uses these snapshots. |
| `basic_strategies_guide.md`, `check_basic_strategies.py` | How the baseline strategies differ, parameter tuning, and simulator checks. |
| `poisson_round1.py`, `summarize_poisson_round1.py`, `poisson_round1_results/` | Arrival rates, Poisson/homogeneous-intensity diagnostics, cancellation links and count correlation; code, reports and numeric results. |
| `aggressive_size_fits.py`, `summarize_aggressive_sizes.py`, `test_size_fit_methods.py`, `aggressive_size_results/` | Aggressive-order size distributions, lognormal/Pareto comparisons, tail thresholds and fitting checks. |
| `market_reconstruction/` | FIFO research book, calibration, fresh-data generation, comparisons against the real tape, reports and compact results. Reports explicitly document the generator's volatility/markout mismatch. |
| `spread_research/` | Streaming wrapper around Peter's actual splitter, warmup gates, parameter search, validation/test locks, strategy tests and saved results. |
| `ladder_research/` | Matched-risk ladder tuning and native-default comparisons; official-simulator checks; source/version metadata and results. |
| `ladder_research/offset_order_experiment/` | Failed opposite-order workaround: distant unfilled orders do not offset filled inventory or avoid penalties. Includes both-direction controlled tapes and results. |
| `ladder_research/generalization/` | Frozen ladder source, 51-run temporal/parameter/cross-tape/stress audit, generator limitations, reports and all compact per-run results. |
| `strategy_runs/`, `basic_strategy_checks/` and other compact run summaries | Existing baseline PnL/result evidence; large execution logs omitted. |
| `strategy_dashboard.py`, `check_execution_explorer_live.py` | Earlier baseline dashboard and live callback smoke test for the execution-analysis tabs. Prefer the repository's `results_explorer`, `explorer`, and `data_explorer` for normal use. |

`MIGRATION_MANIFEST.json` is the exact per-file inventory: source, destination, size, copy/exclusion reason, original SHA256, copied SHA256, and whether paths/content were adapted. An `adapted` flag indicates relocation changes, not a new trading experiment. Historical hashes and numbers inside research results remain historical evidence; they are not reinterpreted as results from a later source version.

## What was left in the original folder

Large generated CSV tapes/splits and replay logs, binary calibration/measurement arrays, cached replays, generated HTML with embedded Plotly, temporary test outputs, console/server logs, process IDs and machine-specific service state. All originals are preserved. These derived outputs can be regenerated; their exclusion avoids committing large duplicate data to the team repository. See the manifest for every excluded file.

## Running the packaged work

Run the following commands in PowerShell from the **repository root**, the directory containing `pyproject.toml`. Use the repository's Python 3.13 environment. Put the organizer's `round1.csv` in `data/`; it is not copied into this package.

```powershell
# Arrival-process and aggressive-size research
.\.venv\Scripts\python.exe -X utf8 -B strategies\adam\poisson_round1.py
.\.venv\Scripts\python.exe -X utf8 -B strategies\adam\summarize_poisson_round1.py
.\.venv\Scripts\python.exe -X utf8 -B strategies\adam\aggressive_size_fits.py
.\.venv\Scripts\python.exe -X utf8 -B strategies\adam\summarize_aggressive_sizes.py

# Recreate the omitted chronological CSV splits before spread/ladder replay
.\.venv\Scripts\python.exe -X utf8 -B strategies\adam\spread_research\split_round1.py

# Spread parameter search and locked final test
.\.venv\Scripts\python.exe -X utf8 -B strategies\adam\spread_research\research.py

# Ladder tuning, risk checks and failed offset-order experiment
.\.venv\Scripts\python.exe -X utf8 -B strategies\adam\ladder_research\research.py --jobs 2
.\.venv\Scripts\python.exe -X utf8 -B strategies\adam\ladder_research\verify.py
.\.venv\Scripts\python.exe -X utf8 -B strategies\adam\ladder_research\check_offset_orders.py

# Recreate binary calibration artifacts before generating synthetic tapes
.\.venv\Scripts\python.exe -X utf8 -B strategies\adam\market_reconstruction\calibrate.py
.\.venv\Scripts\python.exe -X utf8 -B strategies\adam\market_reconstruction\compare.py

# Generalization audit and reports
.\.venv\Scripts\python.exe -X utf8 -B strategies\adam\ladder_research\generalization.py
.\.venv\Scripts\python.exe -X utf8 -B strategies\adam\ladder_research\generalization_context.py
.\.venv\Scripts\python.exe -X utf8 -B strategies\adam\ladder_research\generalization_report.py
```

Scripts may resume existing matching results. To perform a wholly fresh experiment, first archive its result directory rather than mixing source versions. The generalization audit deliberately stops if Peter's current source differs from its frozen snapshot. The earlier exact-reproduction report also checks source hashes; inspect saved Markdown reports when looking up historical conclusions. Re-running studies can update records and produce different results if Peter's code, the data or dependencies changed.

The initial native ladder result used size 200 (PnL 107,532); the later version uses size 400 and an inactive imbalance filter (PnL 107,559). The tuned matched-risk ladder earned 22,254; adaptive spread earned 8,999. The public chronological holdout has been examined repeatedly and is not a new blind dataset.

## Visualizations

Regenerate spread full logs with the spread research command above, then:

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m results_explorer.report strategies\adam\spread_research\results\test_full_logs --out strategies\adam\spread_research\results\adaptive_spread_results.html
.\.venv\Scripts\python.exe -B -m http.server 8075 --bind 127.0.0.1 --directory strategies\adam
```

Open `http://127.0.0.1:8075/spread_research/results/adaptive_spread_results.html` to view the page. The generalization report is at `/ladder_research/generalization/report.html` after running its report script. Generated HTML is intentionally ignored by Git.

The optional live explorer callback check takes `--url http://127.0.0.1:8062`; start the repository data explorer first and pass its actual port.

## Relocation changes

Python scripts now locate the repository by finding the ancestor containing `pyproject.toml` and `simulator/`. References to the old research root and reproduction commands were rebased to `strategies/adam`. The packaged baseline/spread scripts use `models/`. The live callback checker takes a URL instead of reading an obsolete process-state file. No original `myshit` files, Peter strategy, simulator or `strategies_a` files were modified.

Saved JSON paths are rebased to this machine's new location. On another machine, regenerate the splits/calibration metadata before running studies; historical JSONs are evidence rather than portable configuration files.
