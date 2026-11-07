# Round 2 research

Start with [REPORT.md](REPORT.md). It contains the findings, model scores, actual baseline PnL, methodology, limitations and recommended next experiments. `report.html` adds nine interactive figures. Serve this directory over HTTP to view it as a page.

## Reproduce

From the repository root, with Python 3.13 and repository dependencies installed:

```powershell
& .\strategies\adam\round2_research\run_research.ps1
```

The runner stops immediately on a failed stage. Research scripts are **not submission files**: they use filesystem access, offline replay and training libraries. No new standalone trading strategy was created by this analysis.

| Stage | What it does |
|---|---|
| `extract.py` | Reconstruct the market; record the data hash and split protocol; build 48 causal book/print features and delayed price targets; historical passive-order labels; fit-period queue, toxicity, latency, distribution and activity tables. |
| `models.py` | Compare mean/microprice/ridge/nonlinear directional forecasts, historical fill classifiers and conditional first-fill markout models. Lock validation selections before test predictions. |
| `risk_structure.py` | Conditional 90th-percentile move forecasts, endpoint-jump classifiers, volatility/liquidity clusters, return tails and source-event attribution. Fixed supplemental study, no retuning of directional models. |
| `baselines.py` | Seed chronological books with Peter's splitter and replay three unchanged Round1 strategies over validation/test using the actual simulator. Save full logs and source snapshots. |
| `interpret.py` | Prior-risk maker cohorts, dependence-aware block bootstrap, coefficient tables, feature/target indexing checks, cancellation-message linkage and equal-exposure count correlations. |
| `flow_models.py` | Training activity spectrum and lagged/phase/nonlinear Poisson forecasts of next-second executed-aggressor arrivals. |
| `diagnostics.py` | Data integrity, aggressor types, complete-bin Fano checks. |
| `report.py` | Assemble the exact tables, readable assessment, interactive figures and completion checks; verify the source-data hash. |

## Outputs

- `results/`: compact CSV and JSON evidence, protocols, fitted-model selections and audits. These can be committed.
- `generated/`: feature arrays, trusted locally fitted pickle estimators, seeded CSV slices, frozen baseline source snapshots and full simulator logs. Derived files are ignored by Git.
- `REPORT.md`: full assessment.
- `report.html`: generated interactive report, ignored by Git.

Model inputs use only publicly observable book/print information. Submitted IOC sizes and raw off-screen messages are diagnostic information, not predictors. Full feature definitions and policy-dependence caveats are in the report and schema files.

The final block is a public chronological fitting holdout. The initial selections use validation only; subsequent exploratory studies reuse that final block. There is no new private or genuinely blind dataset here. Model accuracy and selected historical cohorts do not establish strategy profitability.

## View reports

```powershell
.\.venv\Scripts\python.exe -B -m http.server 8076 --bind 127.0.0.1 --directory strategies\adam\round2_research
```

Open `http://127.0.0.1:8076/report.html`.

For actual baseline results in the repository's Results Explorer:

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m results_explorer.report strategies\adam\round2_research\generated\ladder_native_test_logs strategies\adam\round2_research\generated\adaptive_spread_test_logs --out strategies\adam\round2_research\baseline_comparison.html
```

Then view `/baseline_comparison.html` on the same server. Existing raw-data explorer remains available separately for `round2_public.csv`.
