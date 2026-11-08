# GA mutation agent brief (round 2)

You are a mutation operator in a genetic algorithm that evolves a market-making strategy for the round 2
tape. Goal: maximise mean PnL. You get a PARENT file (and maybe a second parent) and a MUTATION task. You
produce ONE child file, evaluate it, and report.

## Steps
1. `cp marek/research/round2/ga/pop/<PARENT>.py marek/research/round2/ga/pop/<CHILD_ID>.py`
   (all paths relative to repo root `/Users/marek/Documents/delfini-algo-arena`).
2. Apply the mutation to the CHILD file only. Edit nothing else in the repo.
   - Parameter changes: edit the `PARAMS.update({...})  # GA ...` line just above `BUY, SELL = Side.BUY, Side.SELL`
     (add/change keys there; replace the `# GA ...` comment with `# GA <CHILD_ID>`).
   - Code changes: keep them small and local. Put every new mechanic behind a new PARAMS key (add its default to
     the PARAMS dict and turn it on in the update line), so the flag off == the parent.
3. Evaluate (takes ~3–8 min; use a 1200000 ms Bash timeout):
   `uv run python marek/research/round2/ga/eval.py marek/research/round2/ga/pop/<CHILD_ID>.py --id <CHILD_ID> --parent <PARENT> --desc "<one line: what changed>"`
   It prints JSON with `fitness` (mean PnL over 5 tapes: round2 + 4 synthetic), `min`, and per-tape `pnl`.
4. If it reports `error` (import failure, crash, breach), fix the bug ONCE and re-run with the same id.
   If a run is long don't kill it; the simulator is just slow.
5. Final answer (short): child id, what you changed (1–3 lines), fitness, min, parent fitness if given, and
   whether it beats the parent. No other prose.

## Hard constraints (the submission sandbox enforces these; violations disqualify)
- Imports only: stdlib modules like `math`, `collections` + `simulator.models` + `simulator.strategy.BaseStrategy`.
- Banned: `os`, `sys`, `subprocess`, `io`, `pathlib`, `open`, `getattr`, `type`, dunder attributes (`__x__`),
  `str.format`, `print` to files. f-strings are OK.
- Position limit ±100 is hard: never exceed `cap` 100; count in-flight orders toward room (use `self._room`).
- Do not run more than one eval at a time yourself. Do not modify eval.py, other pop files, or the ledger.

## The strategy (r2_gen lineage, ~760 lines)
Market maker that tracks a hidden fair price S (S moves on each filled taker by c*filled^e; makers quote
around S). Key methods: `_update_s`/`_bound_s` (S estimate), `on_order_book_change` (main loop),
`_quote_gen` (calm quoting), `_shock_check` (drought detector → pull quotes before a scheduled price jump),
`_jump_sweep` (IOC the stale book after the jump, flag `js_on`), `_vol_update`/`_vol_act` (volatile-regime
detector and actions, `vol_mode`), `_snipe` (IOC quotes that are through S by `s_take`).

Round 2 facts (full details: marek/research/round2/round2_generator.md, ROUND2_STRAT.md):
- Calm 85% of time: impact 0.0142*filled^0.625, maker offsets 0.05+Exp(0.45), spread ~1.07.
- Volatile 15% (mean dwell 43 s): impact ~5x, takers cross ~2.55 ticks through S, makers never quote within
  0.70 of S, spread ~1.9, taker rate 6/s vs 4/s.
- 27 shocks: maker inserts drop to 30% for exactly 3.0 s, then S jumps N(0,43) ticks. Stale book left behind;
  first taker in jump direction arrives ~70 ms after. Then a symmetric taker burst (+35/s for 1.25 s, fading by +3 s).
- Takers are coin flips: no directional signal. PnL is roughly linear in position cap utilisation.
