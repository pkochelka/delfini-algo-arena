"""Fast local backtests: parquet tape cache, a parity-checked engine, an in-process eval pool.

Dev-only. `simulator/` stays the reference (it mirrors the grader); everything here must reproduce
its `result.json` and `pnl.csv` exactly, which `python -m fastsim.parity` and `tests/test_fastsim.py` check.

    uv run fast-sim --strategy S.py --backtest data/round3.csv --log-dir logs/x    # same outputs as exchange-simulator
    uv run python -m fastsim.tape data/round3.csv data/synth/*.csv                 # pre-build the parquet caches
    uv run python -m fastsim.bench --strategy S.py --backtest data/round3.csv      # where the time goes

From Python: `fastsim.pool.run_many(jobs)` runs (strategy, tape, params) jobs in worker processes.
"""
