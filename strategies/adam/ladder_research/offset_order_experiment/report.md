# Ladder: non-executing offset-order experiment

Result: the proposed offset does not avoid either exchange check. Peter's source and the simulator are unchanged. All 14 controlled runs used the actual LadderMM with isolated cap/size overrides, 5 ms latency, limit 100, potential limit 1000 and penalty 10000 per second.

The market forces a one-sided fill at 1 second. At 2 seconds the probe cancels all resting strategy orders (effective at 2.005 seconds). The replay ends at 4 seconds. The far opposite order uses price 1 for bids or 1,000,000,000 for asks; it is accepted, remains unfilled, and is canceled successfully.

| Direction | Cap | Distant offset | Executed hedge at cleanup | Final inventory | Penalty |
|---|---:|---|---|---:|---:|
| long | 100 | False | False | 100 | 0 |
| long | 100 | True | False | 100 | 0 |
| long | 200 | False | False | 200 | 30,000 |
| long | 200 | True | False | 200 | 30,000 |
| long | 200 | True | True | 0 | 10,050 |
| long | 1100 | False | False | 0 | 0 |
| long | 1100 | True | False | 0 | 0 |
| short | 100 | False | False | -100 | 0 |
| short | 100 | True | False | -100 | 0 |
| short | 200 | False | False | -200 | 30,000 |
| short | 200 | True | False | -200 | 30,000 |
| short | 200 | True | True | 0 | 10,050 |
| short | 1100 | False | False | 0 | 0 |
| short | 1100 | True | False | 0 | 0 |

With cap 200, the penalty is 30,000 with or without the distant order: three seconds outside ±100. Canceling unfilled orders leaves the filled inventory unchanged. A genuine executed hedge removes inventory at 2.005 seconds, but still incurs 10,050 for the earlier 1.005 seconds outside the limit.

With cap 1100, the original ladder orders and the distant order are rejected independently against the potential-position cap 1000. Opposite-side orders do not net against this check.

Code: simulator/simulator.py:_exceeds_position_limit uses inventory plus SAME-side resting quantity and the new order; _accrue_breach_penalty uses abs(filled inventory)>100 only. LadderMM._room likewise guards each side separately.

## Original-default inventory distribution

Reconstructed from actual user fills over the same 49-minute trading interval: 25.34% of time at −100; 25.23% at +100; 50.57% at either exact boundary; 51.15% at |inventory|≥99. Mean inventory -0.749.

The limit is binding. A non-executing order is not a hedge of filled inventory. The available strategy changes are to increase turnover within the cap, reduce inventory through executed trades, or adjust quoting before reaching the boundary. No submission-ready bypass strategy was created because the proposed accounting mechanism fails.

Reproduce from the repository:
```powershell
.\.venv\Scripts\python.exe -B strategies\adam\ladder_research\check_offset_orders.py
```
