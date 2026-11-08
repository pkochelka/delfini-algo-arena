# overfit: is final3 overfit to the r2 shock at 5398 s?

One-at-a-time sweep on `strategies/marek/round2/hyp/overfit.py` (= final3), all 8 tapes, vs `overfit_base` (= BASE_final3:
r2 763,078, ex5398 662,138, 5398 window 100,940, synth mean 713,454). The 5398-free objective is d(r2_ex5398) + 7*d(synth mean),
i.e. the change in total PnL on r2-without-5398 plus s1..s7. "up/dn" counts synth tapes. Rows: `../results.jsonl`, tags `overfit_*`.

## Verdict
- The **shock oversize itself is not overfit**. With xs_cap=0 (no oversize) we lose 89.5k in the 5398 window, but also 17.8k on
  the other r2 shocks and 24.8k per synth tape (-174k over 7 tapes: -65k on the 35 |J|>=60 truth shocks, -104k on the 147 smaller ones).
  Roughly 1/3 of the mechanism's total value (89.5k of 89.5k+17.8k+174k = 281k) comes from 5398. It pays on ~180 synth shocks.
- xs_edge=4, xs_reval=1, xu_lim=1, xu_ts2=0.08, dr_on=1, ss_shots=3 all hold up on the 5398-free objective (each alternative tested loses).
- **ps_on (post-shock burst quoting) is the one mechanism that loses without 5398**: switching it off gains +6.3k on every one of the
  7 synth tapes, +1.3k on r2_ex5398 (+4.0k on the other r2 shocks), win_t 3.86. Frozen as `strategies/marek/round2/overfit_best.py`.
- **Rankings do flip**: ss_lim=5 ranks 13th of 14 by full r2 (-83k, all of it in the 5398 window: the 111-tick jump needs a deep IOC limit)
  but 2nd by the 5398-free objective (+1.9k ex5398, +1.2k synth, 4/1 tapes up). ss_lim=8 was chosen for 5398 alone.
  Combined, ps_on=0 + ss_lim=5 is the strict 5398-free winner (+3.9k ex5398, +7.8k synth, 7/0) but costs 85k at 5398;
  saved as `strategies/marek/round2/overfit_best_strict.py`. Recommended: `overfit_best.py` (ps_on=0 only) unless we believe
  >100-tick jumps with a deep stale book will not recur (synth big shocks are flat under ss_lim=5: -0.3k).
  xs_edge=3.0 ranks 1st by full r2 (+7.7k, all on other r2 shocks) but is flat on synth (-56, 1 up/4 down), so it isn't adopted.

## Table
sorted by 5398-free objective (d ex5398 + 7*d synth_mean = d total pnl on r2-ex + s1..s7):
| ps_on=0 ss_lim=5 | 681,962 | -81,116 | +3,893 | -85,009 | +7,798 | 7/0 | +58,482 | -0.31 |
| ps_on=0 | 764,614 | +1,536 | +1,321 | +215 | +6,321 | 7/0 | +45,570 | 3.86 |
| ss_lim=5 | 679,730 | -83,348 | +1,876 | -85,224 | +1,206 | 4/1 | +10,317 | -0.88 |
| xs_edge=3.0 | 770,815 | +7,737 | +7,737 | +0 | -56 | 1/4 | +7,347 | 1.15 |
| xu_lim2=6.0 | 762,260 | -818 | -818 | +0 | -648 | 3/3 | -5,357 | -1.12 |
| ss_shots=1 | 766,303 | +3,225 | +3,225 | +0 | -1,946 | 2/5 | -10,398 | -0.49 |
| xu_ts2=0.2 | 758,701 | -4,377 | -4,133 | -244 | -1,635 | 2/5 | -15,575 | -2.03 |
| xs_reval=0 | 760,739 | -2,338 | -2,338 | +0 | -3,686 | 1/5 | -28,142 | -1.5 |
| xs_cap=400 | 697,226 | -65,852 | -7,552 | -58,300 | -6,037 | 1/4 | -49,813 | -1.73 |
| dr_on=0 | 739,663 | -23,415 | -9,772 | -13,643 | -10,242 | 2/5 | -81,466 | -1.61 |
| xu_lim=2.0 | 756,323 | -6,755 | -5,790 | -965 | -11,684 | 2/5 | -87,575 | -1.82 |
| xu_ts2=0.04 | 755,237 | -7,841 | -5,748 | -2,093 | -12,319 | 3/4 | -91,979 | -1.88 |
| xs_edge=None | 762,804 | -274 | -274 | +0 | -15,925 | 1/6 | -111,748 | -3.47 |
| xs_edge=6.0 | 656,732 | -106,346 | -16,815 | -89,531 | -16,142 | 1/6 | -129,806 | -2.18 |
| xs_cap=0 | 655,783 | -107,295 | -17,764 | -89,531 | -24,812 | 1/6 | -191,450 | -2.74 |

## Per-shock attribution (delta vs base, PnL in [onset-0.5, onset+6] s)
Synth: truth onsets, big = |J|>=60 (35 shocks), small = the other 147; other = outside shock windows. r2: shock list from
../shocktime/join_base.txt.
```
PARAMS.update({'dr_on': 0}) {'big': -46244, 'small': -21738, 'other': -3712} n_big 35 n_small 147
PARAMS.update({'xs_edge': 3.0}) {'big': 1170, 'small': -2247, 'other': 687} n_big 35 n_small 147
PARAMS.update({'ss_lim': 5}) {'big': -331, 'small': 9717, 'other': -945} n_big 35 n_small 147
PARAMS.update({'xu_lim': 2.0}) {'big': -66552, 'small': -19209, 'other': 3976} n_big 35 n_small 147
PARAMS.update({'xs_edge': None}) {'big': -13441, 'small': -93273, 'other': -4760} n_big 35 n_small 147
PARAMS.update({'xs_reval': 0}) {'big': 1314, 'small': -23153, 'other': -3965} n_big 35 n_small 147
PARAMS.update({'xu_ts2': 0.04}) {'big': -66761, 'small': -20592, 'other': 1122} n_big 35 n_small 147
PARAMS.update({'xs_cap': 400}) {'big': -20022, 'small': -21212, 'other': -1027} n_big 35 n_small 147
PARAMS.update({'ps_on': 0}) {'big': 3549, 'small': 25576, 'other': 15124} n_big 35 n_small 147
PARAMS.update({'xs_cap': 0}) {'big': -65454, 'small': -104093, 'other': -4139} n_big 35 n_small 147
PARAMS.update({'xs_edge': 6.0}) {'big': -50764, 'small': -60370, 'other': -1857} n_big 35 n_small 147
PARAMS.update({'ss_shots': 1}) {'big': 1988, 'small': -12166, 'other': -3445} n_big 35 n_small 147
PARAMS.update({'xu_ts2': 0.2}) {'big': -759, 'small': -9985, 'other': -699} n_big 35 n_small 147
PARAMS.update({'xu_lim2': 6.0}) {'big': 33, 'small': -5864, 'other': 1292} n_big 35 n_small 147
r2 shocks 28
r2 PARAMS.update({'dr_on': 0}) {'5398': -13643, 'other_shocks': -2912, 'nonshock': -6860}
r2 PARAMS.update({'xs_edge': 3.0}) {'5398': 0, 'other_shocks': 8780, 'nonshock': -1043}
r2 PARAMS.update({'ss_lim': 5}) {'5398': -85224, 'other_shocks': 2105, 'nonshock': -229}
r2 PARAMS.update({'xu_lim': 2.0}) {'5398': -965, 'other_shocks': -5790, 'nonshock': 0}
r2 PARAMS.update({'xs_edge': None}) {'5398': 0, 'other_shocks': -91, 'nonshock': -183}
r2 PARAMS.update({'xs_reval': 0}) {'5398': 0, 'other_shocks': -1879, 'nonshock': -459}
r2 PARAMS.update({'xu_ts2': 0.04}) {'5398': -2093, 'other_shocks': -5910, 'nonshock': 162}
r2 PARAMS.update({'xs_cap': 400}) {'5398': -58300, 'other_shocks': -7635, 'nonshock': 83}
r2 PARAMS.update({'ps_on': 0}) {'5398': -258, 'other_shocks': 4002, 'nonshock': -2208}
r2 PARAMS.update({'xs_cap': 0}) {'5398': -89531, 'other_shocks': -17009, 'nonshock': -755}
r2 PARAMS.update({'xs_edge': 6.0}) {'5398': -89531, 'other_shocks': -16339, 'nonshock': -476}
r2 PARAMS.update({'ss_shots': 1}) {'5398': 0, 'other_shocks': 3801, 'nonshock': -576}
r2 PARAMS.update({'xu_ts2': 0.2}) {'5398': -244, 'other_shocks': -2267, 'nonshock': -1866}
r2 PARAMS.update({'xu_lim2': 6.0}) {'5398': 0, 'other_shocks': -356, 'nonshock': -462}
```

Scripts: `sweep.sh` (runs), `table.py` (table), `pershock.py` (attribution; reads the eval tmp dirs).
