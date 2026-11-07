# Data explorer: sample_v2 vs sample

Whole tapes side by side, generated 2026-11-07 12:49.

### Side by side (whole tapes)

|  | sample_v2 | sample |
| --- | --- | --- |
| Duration (s) | 5,750 | 3,599 |
| Events / s | 348 | 14.9 |
| Cancel / insert | 0.873 | 0.978 |
| Aggressive orders / min | 352 | 10.1 |
| Traded volume | 3,177,589 | 7,706 |
| Buyer-initiated share | 0.498 | 0.502 |
| Mid start | 10,000 | 9,999 |
| Mid end | 10,003 | 10,037 |
| Spread mean (ticks) | 1.59 | 2 |
| Spread min | 1 | 2 |
| One-sided share | 9.03e-06 | 0 |
| σ 100 ms | 0.554 | 1.64 |
| σ 1 s | 0.497 | 2.57 |
| σ 10 s | 0.517 | 3.99 |
| Touch size | 818 | 24.7 |
| Insert size | 32.5 | 24.7 |
| Aggressive order size | 94.2 | 12.8 |
| Passive fill rate | 0.109 | 0.0226 |
| k (sweep depth) | 2.93 | - |
| A (intensity) | 259 | - |
| k (intensity) | 4.94 | - |
| OFI next-bin R² 1 s | 0.054 | 0.341 |
| Taker markout 1 s | -0.805 | 2.06 |
| Taker markout 10 s | -0.804 | 8.15 |

#### Volatility signature

data: [`csv/compare/01-volatility-signature.csv`](csv/compare/01-volatility-signature.csv)

| sampling interval (s) | sample_v2 | sample |
| --- | --- | --- |
| 0.05 | 0.5799 | 1.643 |
| 0.1 | 0.5539 | 1.637 |
| 0.2 | 0.5274 | 1.648 |
| 0.5 | 0.5028 | 2.032 |
| 1 | 0.4967 | 2.565 |
| 2 | 0.5065 | 3.222 |
| 5 | 0.5065 | 3.899 |
| 10 | 0.5175 | 3.994 |
| 30 | 0.5293 | 3.701 |
| 60 | 0.5564 | 3.47 |

y: σ (ticks / √s)

#### Spread distribution

data: [`csv/compare/02-spread-distribution.csv`](csv/compare/02-spread-distribution.csv)

| spread (ticks) | sample_v2 | sample |
| --- | --- | --- |
| 1 | 0.4142 | - |
| 2 | 0.5839 | 1 |
| 3 | 0.001932 | - |
| 4 | 3.408e-06 | - |

y: share of time

#### 1 s returns

data: [`csv/compare/03-1-s-returns.csv`](csv/compare/03-1-s-returns.csv)

| Δmid (ticks) | sample_v2 | sample |
| --- | --- | --- |
| -2 | 0.004697 | 0.07838 |
| -1 | 0.0515 | 0.1548 |
| 0 | 0.8833 | 0.2707 |
| 1 | 0.0555 | 0.1459 |
| 2 | 0.005045 | 0.08366 |
| -12 | - | 0.0002779 |
| -11 | - | 0.0002779 |
| -10 | - | 0.0005559 |
| -9 | - | 0.001946 |
| -8 | - | 0.003057 |
| -7 | - | 0.005837 |
| -6 | - | 0.00806 |
| -5 | - | 0.02418 |
| -4 | - | 0.03585 |
| -3 | - | 0.05447 |
| 3 | - | 0.05003 |
| 4 | - | 0.03669 |
| 5 | - | 0.01695 |
| 6 | - | 0.0139 |
| 7 | - | 0.007504 |
| 8 | - | 0.004169 |
| 9 | - | 0.001112 |
| 10 | - | 0.0008338 |
| 12 | - | 0.0005559 |
| 14 | - | 0.0002779 |

y: share of intervals

> Log scale: fat tails stand out.

#### Price impact after aggressive orders

data: [`csv/compare/04-price-impact-after-aggressive-orders.csv`](csv/compare/04-price-impact-after-aggressive-orders.csv)

| seconds after (log) | sample_v2 | sample |
| --- | --- | --- |
| 0.005 | 0.0471 | -0.001656 |
| 0.05 | 0.03701 | -0.009934 |
| 0.1 | 0.03414 | -0.008278 |
| 0.25 | 0.02595 | 0.293 |
| 0.5 | 0.01941 | 1.293 |
| 1 | 0.01542 | 3.056 |
| 2 | 0.008749 | 6.652 |
| 5 | 0.01876 | 10.05 |
| 10 | 0.01662 | 9.149 |
| 20 | 0.02118 | 7.265 |
| 30 | 0.02393 | 6.513 |
| 60 | 0.02287 | 6.152 |

y: ticks

#### Depth profile (bid + ask) / 2

data: [`csv/compare/05-depth-profile-bid-ask-2.csv`](csv/compare/05-depth-profile-bid-ask-2.csv)

| ticks behind best | sample_v2 | sample |
| --- | --- | --- |
| 0 | 817.9 | 24.73 |
| 1 | 984.4 | 0 |
| 2 | 121.8 | 0 |
| 3 | 8.99 | 0 |
| 4 | 0.5723 | 0 |
| 5 | 0.04295 | 0 |
| 6 | 0 | 0 |
| 7 | 0 | 0 |
| 8 | 0 | 0 |
| 9 | 0 | 0 |
| 10 | 0 | 0 |
| 11 | 0 | 0 |
| 12 | 0 | 0 |
| 13 | 0 | 0 |
| 14 | 0 | 0 |
| 15 | 0 | 0 |
| 16 | 0 | 0 |
| 17 | 0 | 0 |
| 18 | 0 | 0 |
| 19 | 0 | 0 |

y: mean qty
