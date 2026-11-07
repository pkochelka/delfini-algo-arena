# Data explorer: random_stress

Data explorer export of `data/random_stress.csv`, window 0.0–2,005.9 s (33.4 min), sampling selector 1 s (charts that use it: return histogram and ACF, rolling σ, variance ratio, OFI scatter).
Generated 2026-11-07 12:41 by `uv run data-explorer --data … --export` (see data_explorer/export.py). Numbers are in ticks and seconds. The data behind each chart is in csv/, machine-readable headline stats in summary.json, interactive charts in report.html.

Contents: [Overview](#overview) · [Price & spread](#price--spread) · [Volatility](#volatility) · [Liquidity & book](#liquidity--book) · [Trades & flow](#trades--flow) · [Predictability](#predictability)

## Overview

### Tape fingerprint

| Statistic | Value | Detail |
| --- | --- | --- |
| Window | 33.4 min | 200,000 events |
| Events / s | 99.7 | cancel/insert 0.54 |
| Aggressive orders / min | 1,331 | 99,517 fills |
| Traded volume | 955,622 | 59% buyer-initiated |
| Mid | 9,998 → 10,031 | range 9,936–10,031 |
| Spread (time-weighted) | 2.7 | 46% of time at 1 tick |
| One-sided book | 0.0% | of time a side is empty |
| σ at 100 ms | 1.26 | ticks / √s |
| σ at 1 s | 1.08 | 10 s: 1.45 |
| Touch size | 461 | mean qty at best, per side |
| Order / fill size | 18.8 / 21.5 | mean insert / aggressive order |
| Passive fill rate | 68.2% | resting orders that got any fill |

### Fitted parameters (selected window)

```python
# random_stress, window 0-2006 s  (python -m analysis.fit)
SIGMA_INIT = 1.263   # ticks/sqrt(s) at 100 ms; 1 s: 1.08
K_INIT = 2.902       # ln(1 + 1/mean sweep depth), depth 0.0581
# intensity fit: A = 10.28 /s, k = 0.534 /tick (22 points)
# OFI beta at 1 s = 0.0009664 ticks/lot (R² 0.174), next-bin R² 0.0636
# taker markout vs fill: 1 s -1.22, 10 s 0.214 ticks
IMBALANCE_DMID_1S = {"0.04": 0.0144, "0.14": 0.2393, "0.25": 0.2564, "0.35": 0.2445, "0.45": 0.0188, "0.55": -0.1112, "0.65": -0.2126, "0.76": -0.2477, "0.86": -0.2088, "0.96": -0.0074}
```

> Copy into a strategy: submissions cannot read files.

### Activity per 30 s

#### Events / Aggressive orders / Spread (ticks) / σ 1 s (ticks/√s)

data: [`csv/overview/01-events-aggressive-orders-spread-ticks-1-s-ticks-s.csv`](csv/overview/01-events-aggressive-orders-spread-ticks-1-s-ticks-s.csv)

| x | Events | Aggressive orders | Spread (ticks) | σ 1 s (ticks/√s) |
| --- | --- | --- | --- | --- |
| 0 | 3068 | 388 | 1.793 | 2.2 |
| 30 | 3023 | 575 | 1.7 | 1.252 |
| 60 | 3074 | 554 | 1.733 | 1.068 |
| 90 | 3072 | 420 | 1.5 | 1.549 |
| 120 | 3018 | 515 | 2.033 | 1.176 |
| 150 | 3064 | 850 | 2.367 | 1.211 |
| 180 | 2949 | 552 | 2.067 | 1.384 |
| 210 | 3073 | 551 | 1.8 | 1.291 |
| 240 | 2938 | 434 | 1.367 | 1.32 |
| 270 | 2980 | 694 | 2.267 | 0.8466 |
| 300 | 3029 | 689 | 3 | 1.393 |
| 330 | 3037 | 372 | 1.533 | 1.133 |
| 360 | 3021 | 904 | 4.4 | 1.278 |
| 390 | 2904 | 927 | 3.6 | 0.5774 |
| 420 | 3049 | 470 | 1.867 | 1.653 |
| 450 | 2986 | 382 | 1.7 | 1.408 |
| 480 | 2982 | 542 | 2.633 | 1.478 |
| 510 | 2962 | 809 | 1.967 | 0.5164 |
| 540 | 3042 | 623 | 2.167 | 1.366 |
| 570 | 2966 | 599 | 2.1 | 0.9083 |
| 600 | 3029 | 887 | 6.367 | 1.768 |
| 630 | 2961 | 493 | 1.7 | 1.375 |
| 660 | 3031 | 738 | 1.333 | 0.7012 |
| 690 | 2906 | 736 | 1.467 | 0.3873 |
| 720 | 2934 | 479 | 1.767 | 0.9487 |
| 750 | 2968 | 413 | 1.6 | 1.041 |
| 780 | 3002 | 808 | 1.2 | 0.4472 |
| 810 | 3057 | 659 | 1.3 | 0.5401 |
| 840 | 2965 | 800 | 1.8 | 0.8991 |
| 870 | 2948 | 732 | 3.5 | 0.7416 |
| 900 | 3036 | 759 | 2.033 | 0.9747 |
| 930 | 2964 | 766 | 1.8 | 0.2887 |
| 960 | 2910 | 938 | 2.7 | 0.5627 |
| 990 | 2963 | 887 | 3.267 | 0.8317 |
| 1020 | 3080 | 538 | 1.6 | 1.313 |
| 1050 | 2941 | 540 | 1.6 | 1.533 |
| 1080 | 2895 | 714 | 6.033 | 1.703 |
| 1110 | 3071 | 452 | 1.433 | 1.31 |
| 1140 | 3045 | 512 | 1.267 | 1.065 |
| 1170 | 3015 | 550 | 1.6 | 0.9958 |
| 1200 | 2952 | 880 | 2.933 | 0.9174 |
| 1230 | 2949 | 557 | 1.3 | 0.5323 |
| 1260 | 2951 | 598 | 1.333 | 0.5323 |
| 1290 | 2937 | 753 | 1.467 | 0.3979 |
| 1320 | 3049 | 384 | 1.5 | 1.144 |
| 1350 | 2968 | 885 | 4.333 | 0.2887 |
| 1380 | 3009 | 517 | 2.3 | 1.461 |
| 1410 | 3088 | 592 | 1.6 | 1.111 |
| 1440 | 2990 | 528 | 1.367 | 1.44 |
| 1470 | 2981 | 511 | 1.267 | 0.9129 |
| 1500 | 3081 | 376 | 1.933 | 1.076 |
| 1530 | 2973 | 677 | 1.433 | 0.736 |
| 1560 | 2988 | 401 | 1.367 | 0.9618 |
| 1590 | 2983 | 580 | 1.3 | 0.9309 |
| 1620 | 2942 | 772 | 2.633 | 1.129 |
| 1650 | 2976 | 956 | 9.533 | 0.2236 |
| 1680 | 2891 | 927 | 3.633 | 1.021 |
| 1710 | 3009 | 805 | 2 | 0.6055 |
| 1740 | 3005 | 930 | 1.767 | 0.6124 |
| 1770 | 2986 | 965 | 4.233 | 0.2041 |
| 1800 | 2995 | 998 | 9.733 | 0.2236 |
| 1830 | 2972 | 946 | 16.37 | 0.2415 |
| 1860 | 2953 | 921 | 6.667 | 1.945 |
| 1890 | 2880 | 759 | 1.733 | 0.8612 |
| 1920 | 2931 | 508 | 1.467 | 1.147 |
| 1950 | 3010 | 682 | 1.7 | 0.4282 |
| 1980 | 2593 | 848 | 4.808 | 0.1698 |

y: y

> Regime changes show up here first: bursts of activity, spread widening, volatility clustering.

## Price & spread

#### Best bid / ask, microprice and aggressive orders

data: [`csv/price/01-best-bid-ask-microprice-and-aggressive-orders.csv`](csv/price/01-best-bid-ask-microprice-and-aggressive-orders.csv)

| series | points | time (s) range | price (ticks) min | mean | median | max | last |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Best bid | 5,000 | 0..2006 | 9935 | 9975 | 9972 | 10028 | 10028 |
| Best ask | 5,000 | 0..2006 | 9936 | 9978 | 9975 | 10034 | 10034 |
| Microprice | 5,000 | 0..2006 | 9936 | 9976 | 9973 | 1.003e+04 | 1.003e+04 |
| Aggressive buy | 17,646 | 0.1476..2006 | 9936 | 9985 | 9981 | 10034 | 10034 |
| Aggressive sell | 12,354 | 0.3434..1967 | 9935 | 9968 | 9966 | 10028 | 10028 |

> Zoom the timeline above to see every quote change. Marker size grows with order quantity; it sits at the worst price the order reached.

#### Spread over time

data: [`csv/price/02-spread-over-time.csv`](csv/price/02-spread-over-time.csv)

| series | points | time (s) range | ticks min | mean | median | max | last |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Spread | 5,000 | 0..2006 | 1 | 2.697 | 2 | 22 | 6 |

#### Spread distribution (share of time)

data: [`csv/price/03-spread-distribution-share-of-time.csv`](csv/price/03-spread-distribution-share-of-time.csv)

| spread (ticks) | share of time |
| --- | --- |
| 1 | 0.4618 |
| 2 | 0.2542 |
| 3 | 0.09133 |
| 4 | 0.04688 |
| 5 | 0.03162 |
| 6 | 0.02641 |
| 7 | 0.01513 |
| 8 | 0.01449 |
| 9 | 0.01131 |
| 10 | 0.009416 |
| 11 | 0.009735 |
| 12 | 0.006771 |
| 13 | 0.003859 |
| 14 | 0.002588 |
| 15 | 0.002629 |
| 16 | 0.002623 |
| 17 | 0.002077 |
| 18 | 0.002127 |
| 19 | 0.001873 |
| 20 | 0.001507 |
| 21 | 0.001207 |
| 22 | 0.0005189 |

y: share of time

> A market maker quoting inside the spread needs it wider than 1 tick most of the time.

#### Resting depth by price (bids +, asks -)

data: [`csv/price/04-resting-depth-by-price-bids-asks.csv`](csv/price/04-resting-depth-by-price-bids-asks.csv)

Heatmap, 24,041 non-zero cells (missing cells in the CSV are 0); x (time (s)) 0.09292..2003, y (price (ticks)) 9916..10053; z mean -170.9, max 4552.

> Snapshots every 100 ms. Look for walls (persistent large levels) and how deep the book is.

## Volatility

#### Rolling σ over 30 s

data: [`csv/vol/01-rolling-over-30-s.csv`](csv/vol/01-rolling-over-30-s.csv)

| series | points | time (s) range | ticks / √s min | mean | median | max | last |
| --- | --- | --- | --- | --- | --- | --- | --- |
| σ | 2,006 | 0..2005 | 0.1826 | 0.9903 | 1.021 | 2.238 | 0.1826 |

> Clustering (calm and wild stretches) means a fixed σ under- and over-quotes in turn: estimate it online.

#### Volatility signature

data: [`csv/vol/02-volatility-signature.csv`](csv/vol/02-volatility-signature.csv)

| sampling interval (s) | σ (ticks / √s) |
| --- | --- |
| 0.05 | 1.319 |
| 0.1 | 1.263 |
| 0.2 | 1.192 |
| 0.5 | 1.086 |
| 1 | 1.08 |
| 2 | 1.107 |
| 5 | 1.275 |
| 10 | 1.446 |
| 30 | 1.569 |
| 60 | 1.705 |

y: σ (ticks / √s)

> Flat: random walk. Rising with the interval: momentum. Falling: bid-ask bounce or mean reversion. Pick σ at the horizon you hold inventory over.

#### 1 s mid returns

data: [`csv/vol/03-1-s-mid-returns.csv`](csv/vol/03-1-s-mid-returns.csv)

| Δmid (ticks) | observed |
| --- | --- |
| -3 | 0.008982 |
| -2.5 | 0.01297 |
| -2 | 0.02595 |
| -1.5 | 0.02695 |
| -1 | 0.07285 |
| -0.5 | 0.1257 |
| 0 | 0.4167 |
| 0.5 | 0.1522 |
| 1 | 0.07236 |
| 1.5 | 0.03244 |
| 2 | 0.02146 |
| 2.5 | 0.01198 |
| 3 | 0.006986 |
| 3.5 | 0.001497 |
| 4 | 0.002994 |

y: share of intervals

| series | points | Δmid (ticks) range | share of intervals min | mean | median | max | last |
| --- | --- | --- | --- | --- | --- | --- | --- |
| normal, same σ | 200 | -3..4 | 0.0002055 | 0.07089 | 0.04905 | 0.1847 | 0.0002055 |

> excess kurtosis 10.74, skew 0.37, 42% of intervals unchanged, n = 2,004

#### Autocorrelation of 1 s returns

data: [`csv/vol/04-autocorrelation-of-1-s-returns.csv`](csv/vol/04-autocorrelation-of-1-s-returns.csv)

| lag (intervals) | correlation |
| --- | --- |
| 1 | 0.08712 |
| 2 | 0.1096 |
| 3 | 0.07303 |
| 4 | 0.0339 |
| 5 | 0.05164 |
| 6 | 0.06716 |
| 7 | 0.03134 |
| 8 | 0.04098 |
| 9 | 0.01177 |
| 10 | 0.03563 |
| 11 | 0.01372 |
| 12 | 0.02369 |
| 13 | -0.01071 |
| 14 | 0.04726 |
| 15 | 0.01025 |
| 16 | 0.04575 |
| 17 | -0.03949 |
| 18 | -0.001733 |
| 19 | -0.01133 |
| 20 | -0.002671 |
| 21 | 0.0213 |
| 22 | 0.01726 |
| 23 | 0.007112 |
| 24 | 0.004846 |
| 25 | 0.008474 |
| 26 | -0.01256 |
| 27 | -0.006269 |
| 28 | 0.002068 |
| 29 | 0.03381 |
| 30 | 0.02847 |

y: correlation

Marked: shaded band y ∈ [-0.04378, 0.04378].

> Bars outside the gray band are significant. Positive at short lags: trends (take); negative: reversion (make).

#### Variance ratio

data: [`csv/vol/05-variance-ratio.csv`](csv/vol/05-variance-ratio.csv)

| horizon (s) | VR(q) |
| --- | --- |
| 2 | 1.087 |
| 5 | 1.345 |
| 10 | 1.623 |
| 30 | 1.987 |
| 60 | 2.194 |

y: VR(q)

Marked: horizontal line y = 1.

> Above 1: moves persist over that horizon. Below 1: they revert.

#### Autocorrelation of 10 ms returns by lag

data: [`csv/vol/06-autocorrelation-of-10-ms-returns-by-lag.csv`](csv/vol/06-autocorrelation-of-10-ms-returns-by-lag.csv)

| lag (s, log) | ACF |
| --- | --- |
| 0.01 | -0.01176 |
| 0.02 | -0.02502 |
| 0.03 | -0.02476 |
| 0.04 | -0.01976 |
| 0.05 | -0.01391 |
| 0.06 | -0.01872 |
| 0.07 | -0.00741 |
| 0.09 | -0.01105 |
| 0.12 | -0.009425 |
| 0.15 | 0.006563 |
| 0.18 | -0.0009763 |
| 0.23 | -0.001561 |
| 0.28 | 0.001493 |
| 0.35 | 0.0007784 |
| 0.44 | 0.0009083 |
| 0.55 | 0.0002586 |
| 0.69 | 0.003443 |
| 0.87 | -0.00663 |
| 1.08 | 0.001363 |
| 1.35 | 0.0006484 |
| 1.69 | -0.003056 |
| 2.11 | -0.0007164 |
| 2.64 | 0.004028 |
| 3.3 | 0.003118 |
| 4.13 | -0.001691 |
| 5.16 | 0.002793 |
| 6.45 | -0.001626 |
| 8.06 | 0.001428 |
| 10.07 | -0.0008461 |
| 12.59 | 0.0007788 |
| 15.74 | -0.00221 |
| 19.67 | 0.0007142 |
| 24.58 | 0.0003246 |
| 30.73 | 6.445e-05 |
| 38.41 | 0.001754 |
| 48 | -0.001821 |
| 60 | 0.002468 |

y: correlation

Marked: shaded band y ∈ [-0.004376, 0.004376]; horizontal line y = 0; vertical line x = 0.01.

> One axis from 10 ms to 60 s, so it does not depend on the sampling selector. Negative at the shortest lags: bid-ask bounce or quote flicker. Positive at second-scale lags: momentum. Dashed line: our 5 ms latency. Values look small because most 10 ms returns are zero: compare them with the gray band, not with 1.

#### Autocorrelation of 100 ms returns, every lag to 10 s

data: [`csv/vol/07-autocorrelation-of-100-ms-returns-every-lag-to-10-s.csv`](csv/vol/07-autocorrelation-of-100-ms-returns-every-lag-to-10-s.csv)

| lag (s) | correlation |
| --- | --- |
| 0.1 | -0.1082 |
| 0.2 | -0.04868 |
| 0.3 | -0.01127 |
| 0.4 | 0.01474 |
| 0.5 | -0.01244 |
| 0.6 | 0.01303 |
| 0.7 | 0.01467 |
| 0.8 | 0.008106 |
| 0.9 | 0.01092 |
| 1 | -0.01119 |
| 1.1 | 0.0149 |
| 1.2 | 0.03927 |
| 1.3 | 0.006856 |
| 1.4 | 0.006934 |
| 1.5 | -0.001112 |
| 1.6 | 0.00998 |
| 1.7 | -0.00119 |
| 1.8 | 0.01334 |
| 1.9 | 0.005762 |
| 2 | 0.01357 |
| 2.1 | 0.01146 |
| 2.2 | 0.001544 |
| 2.3 | 0.007715 |
| 2.4 | 0.01061 |
| 2.5 | -0.004549 |
| 2.6 | 0.0002953 |
| 2.7 | 0.02045 |
| 2.8 | -0.001736 |
| 2.9 | -0.005174 |
| 3 | 0.01232 |
| 3.1 | 0.00131 |
| 3.2 | 0.003966 |
| 3.3 | 0.003498 |
| 3.4 | 0.01685 |
| 3.5 | -0.006111 |
| 3.6 | 0.005919 |
| 3.7 | 0.006778 |
| 3.8 | -0.007751 |
| 3.9 | 0.01045 |
| 4 | 0.001076 |
| 4.1 | 0.004904 |
| 4.2 | 0.007638 |
| 4.3 | 0.009826 |
| 4.4 | 0.005451 |
| 4.5 | -0.01041 |
| 4.6 | 0.003419 |
| 4.7 | 0.01396 |
| 4.8 | 0.01943 |
| 4.9 | -0.01291 |
| 5 | 0.0124 |
| 5.1 | -0.007359 |
| 5.2 | 0.01178 |
| 5.3 | 0.004592 |
| 5.4 | -0.01541 |
| 5.5 | 0.01561 |
| 5.6 | 0.01818 |
| 5.7 | -0.01603 |
| 5.8 | -0.004861 |
| 5.9 | 0.01561 |
| 6 | 0.006622 |
| 6.1 | -0.02009 |
| 6.2 | 0.02521 |
| 6.3 | 0.004123 |
| 6.4 | 0.007639 |
| 6.5 | 0.01014 |
| 6.6 | 0.002249 |
| 6.7 | 0.006858 |
| 6.8 | 0.0004525 |
| 6.9 | 0.00553 |
| 7 | 0.003265 |
| 7.1 | -1.615e-05 |
| 7.2 | -0.001813 |
| 7.3 | 0.009045 |
| 7.4 | 0.007639 |
| 7.5 | -0.003532 |
| 7.6 | -0.004 |
| 7.7 | 0.01233 |
| 7.8 | -0.002047 |
| 7.9 | 0.004905 |
| 8 | 0.004827 |
| 8.1 | 0.002015 |
| 8.2 | -0.007749 |
| 8.3 | -0.003921 |
| 8.4 | 0.003266 |
| 8.5 | 0.01022 |
| 8.6 | 0.008422 |
| 8.7 | -0.004702 |
| 8.8 | 0.004047 |
| 8.9 | -0.003608 |
| 9 | 0.008187 |
| 9.1 | 0.002719 |
| 9.2 | 0.006937 |
| 9.3 | -0.003062 |
| 9.4 | -0.001813 |
| 9.5 | -0.003453 |
| 9.6 | 0.009436 |
| 9.7 | 0.001546 |
| 9.8 | -0.006656 |
| 9.9 | 0.0009996 |
| 10 | -0.004625 |

y: correlation

Marked: shaded band y ∈ [-0.01384, 0.01384].

> 20,058 returns. Between the two charts above: the lag at which bars turn from blue to orange is how long a move keeps going before it starts to revert, which is a natural holding time or signal horizon.

#### Autocorrelation of consecutive mid changes (tick by tick)

data: [`csv/vol/08-autocorrelation-of-consecutive-mid-changes-tick-by-tick.csv`](csv/vol/08-autocorrelation-of-consecutive-mid-changes-tick-by-tick.csv)

| lag (mid changes) | correlation |
| --- | --- |
| 1 | -0.2276 |
| 2 | 0.07862 |
| 3 | 0.0338 |
| 4 | 0.0603 |
| 5 | 0.06119 |
| 6 | 0.03983 |
| 7 | 0.07159 |
| 8 | 0.01567 |
| 9 | 0.05358 |
| 10 | 0.04312 |
| 11 | 0.01795 |
| 12 | 0.04781 |
| 13 | 0.01827 |
| 14 | 0.02151 |
| 15 | 0.01719 |
| 16 | 0.03393 |
| 17 | -0.001699 |
| 18 | 0.02131 |
| 19 | 0.02017 |
| 20 | -0.0004305 |
| 21 | 0.0113 |
| 22 | 0.01275 |
| 23 | -0.00271 |
| 24 | 0.01231 |
| 25 | -0.001509 |
| 26 | 0.02841 |
| 27 | -0.01292 |
| 28 | 0.0139 |
| 29 | -0.006009 |
| 30 | 0.004769 |
| 31 | 0.009519 |
| 32 | 0.0005837 |
| 33 | -0.01564 |
| 34 | 0.01878 |
| 35 | 0.002739 |
| 36 | 0.01764 |
| 37 | 0.006229 |
| 38 | -0.01196 |
| 39 | 0.002174 |
| 40 | -0.002707 |
| 41 | 0.02297 |
| 42 | -0.01988 |
| 43 | 0.02024 |
| 44 | 0.007313 |
| 45 | -0.004164 |
| 46 | -0.007903 |
| 47 | 0.003761 |
| 48 | -0.007396 |
| 49 | -0.01164 |
| 50 | 0.01283 |

y: correlation

Marked: shaded band y ∈ [-0.02702, 0.02702].

> 5,260 mid changes. Lag 1 strongly negative: moves get undone (fade them). Positive: moves continue (follow them).

#### Time between mid changes

data: [`csv/vol/09-time-between-mid-changes.csv`](csv/vol/09-time-between-mid-changes.csv)

| seconds (log) | mid changes |
| --- | --- |
| 4e-06 | 1 |
| 5.789e-06 | 1 |
| 8.379e-06 | 1 |
| 1.213e-05 | 0 |
| 1.755e-05 | 0 |
| 2.541e-05 | 0 |
| 3.677e-05 | 0 |
| 5.322e-05 | 0 |
| 7.703e-05 | 0 |
| 0.0001115 | 2 |
| 0.0001614 | 1 |
| 0.0002335 | 4 |
| 0.000338 | 4 |
| 0.0004892 | 1 |
| 0.0007081 | 4 |
| 0.001025 | 9 |
| 0.001483 | 7 |
| 0.002147 | 13 |
| 0.003107 | 33 |
| 0.004497 | 44 |
| 0.006509 | 62 |
| 0.009421 | 85 |
| 0.01363 | 100 |
| 0.01973 | 171 |
| 0.02856 | 247 |
| 0.04134 | 311 |
| 0.05983 | 411 |
| 0.0866 | 448 |
| 0.1253 | 589 |
| 0.1814 | 592 |
| 0.2626 | 546 |
| 0.38 | 444 |
| 0.55 | 336 |
| 0.7961 | 251 |
| 1.152 | 159 |
| 1.668 | 119 |
| 2.414 | 89 |
| 3.493 | 90 |
| 5.056 | 49 |
| 7.318 | 30 |
| 10.59 | 6 |

y: count

> 5,260 mid changes. Changes are the events a quoting strategy has to react to.

#### Size of mid changes

data: [`csv/vol/10-size-of-mid-changes.csv`](csv/vol/10-size-of-mid-changes.csv)

| Δmid (ticks) | count |
| --- | --- |
| -7 | 1 |
| -5 | 1 |
| -4.5 | 4 |
| -4 | 1 |
| -3 | 13 |
| -2.5 | 15 |
| -2 | 63 |
| -1.5 | 146 |
| -1 | 506 |
| -0.5 | 1832 |
| 0.5 | 1944 |
| 1 | 517 |
| 1.5 | 123 |
| 2 | 54 |
| 2.5 | 23 |
| 3 | 8 |
| 3.5 | 1 |
| 4 | 3 |
| 4.5 | 1 |
| 5.5 | 1 |
| 6 | 1 |
| 7 | 1 |
| 10.5 | 1 |

y: count

> Mostly ±0.5 / ±1 means the price moves tick by tick; large jumps mean gaps in the book.

## Liquidity & book

#### Mean resting quantity by distance from the touch

data: [`csv/liq/01-mean-resting-quantity-by-distance-from-the-touch.csv`](csv/liq/01-mean-resting-quantity-by-distance-from-the-touch.csv)

| ticks behind the best price | Bid | Ask |
| --- | --- | --- |
| 0 | 332.1 | 588.5 |
| 1 | 664 | 1195 |
| 2 | 728.8 | 1275 |
| 3 | 793.6 | 1327 |
| 4 | 828 | 1356 |
| 5 | 863.3 | 1373 |
| 6 | 891.3 | 1391 |
| 7 | 898.5 | 1389 |
| 8 | 984.7 | 1409 |
| 9 | 950.6 | 1357 |
| 10 | 940.5 | 1289 |
| 11 | 918.9 | 1219 |
| 12 | 874.2 | 1166 |
| 13 | 793.9 | 1119 |
| 14 | 766.1 | 1068 |
| 15 | 728.8 | 1030 |
| 16 | 685.2 | 982.5 |
| 17 | 652.3 | 922.3 |
| 18 | 616 | 875.4 |
| 19 | 611.1 | 866.6 |

y: mean qty

> How much volume sits behind the touch: thin books mean large orders sweep several levels.

#### Quantity at the best price (share of time)

data: [`csv/liq/02-quantity-at-the-best-price-share-of-time.csv`](csv/liq/02-quantity-at-the-best-price-share-of-time.csv)

| series | points | qty range | share of time min | mean | median | max | last |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Bid | 2,851 | 1..3803 | 1.496e-08 | 0.0003508 | 6.762e-05 | 0.04961 | 1.182e-05 |
| Ask | 4,202 | 1..6201 | 4.487e-09 | 0.000238 | 6.88e-05 | 0.01906 | 4.834e-06 |

> Small, frequently refreshed touch sizes mean our queue position matters less; large ones mean we wait behind them.

#### Queue imbalance  Qbid / (Qbid + Qask)

data: [`csv/liq/03-queue-imbalance-qbid-qbid-qask.csv`](csv/liq/03-queue-imbalance-qbid-qbid-qask.csv)

| imbalance | share of samples |
| --- | --- |
| 0.025 | 0.2005 |
| 0.075 | 0.1207 |
| 0.125 | 0.06933 |
| 0.175 | 0.04788 |
| 0.225 | 0.02793 |
| 0.275 | 0.02544 |
| 0.325 | 0.01945 |
| 0.375 | 0.02045 |
| 0.425 | 0.01496 |
| 0.475 | 0.01247 |
| 0.525 | 0.01097 |
| 0.575 | 0.01247 |
| 0.625 | 0.01646 |
| 0.675 | 0.01796 |
| 0.725 | 0.01646 |
| 0.775 | 0.02793 |
| 0.825 | 0.02943 |
| 0.875 | 0.04339 |
| 0.925 | 0.09576 |
| 0.975 | 0.1701 |

y: share of samples

> Mass near 0 and 1 means one side is often nearly empty: the strongest imbalance signals.

#### Order sizes at insert

data: [`csv/liq/04-order-sizes-at-insert.csv`](csv/liq/04-order-sizes-at-insert.csv)

| quantity | passive (85,212) | aggressive (44,507) |
| --- | --- | --- |
| 1 | 2838 | 1262 |
| 2 | 2695 | 1314 |
| 3 | 2776 | 1311 |
| 4 | 2744 | 1274 |
| 5 | 2738 | 1236 |
| 6 | 2838 | 1289 |
| 7 | 2750 | 1239 |
| 8 | 2826 | 1277 |
| 9 | 2784 | 1235 |
| 10 | 2689 | 1326 |
| 11 | 2773 | 1245 |
| 12 | 2755 | 1289 |
| 13 | 2823 | 1218 |
| 14 | 2836 | 1246 |
| 15 | 2710 | 1281 |
| 16 | 2745 | 1277 |
| 17 | 2742 | 1346 |
| 18 | 2752 | 1221 |
| 19 | 2826 | 1267 |
| 20 | 2867 | 1307 |
| 21 | 2777 | 1252 |
| 22 | 2842 | 1240 |
| 23 | 2759 | 1243 |
| 24 | 2772 | 1222 |
| 25 | 2729 | 1311 |
| 26 | 2775 | 1223 |
| 27 | 2832 | 1194 |
| 28 | 2793 | 1300 |
| 29 | 2783 | 1254 |
| 30 | 97 | 145 |
| 31 | 85 | 174 |
| 32 | 88 | 141 |
| 33 | 94 | 169 |
| 34 | 84 | 170 |
| 35 | 101 | 154 |
| 36 | 96 | 163 |
| 37 | 94 | 164 |
| 38 | 89 | 138 |
| 39 | 77 | 162 |
| 40 | 102 | 148 |
| 41 | 90 | 136 |
| 42 | 89 | 128 |
| 43 | 88 | 139 |
| 44 | 97 | 180 |
| 45 | 83 | 169 |
| 46 | 95 | 144 |
| 47 | 109 | 155 |
| 48 | 88 | 158 |
| 49 | 85 | 164 |
| 50 | 103 | 167 |
| 51 | 92 | 177 |
| 52 | 97 | 175 |
| 53 | 93 | 157 |
| 54 | 105 | 164 |
| 55 | 76 | 150 |
| 56 | 94 | 134 |
| 57 | 104 | 145 |
| 58 | 102 | 171 |
| 59 | 83 | 167 |
| 60 | 119 | 152 |
| 61 | 88 | 171 |
| 62 | 100 | 153 |
| 63 | 101 | 144 |
| 64 | 95 | 168 |
| 65 | 95 | 161 |
| 66 | 97 | 157 |
| 67 | 91 | 155 |
| 68 | 91 | 145 |
| 69 | 91 | 144 |
| 70 | 93 | 160 |
| 71 | 94 | 142 |
| 72 | 87 | 150 |
| 73 | 80 | 161 |
| 74 | 110 | 151 |
| 75 | 76 | 168 |
| 76 | 91 | 174 |
| 77 | 88 | 141 |

y: count

> Compare with our ORDER_SIZE: quoting much larger than the crowd sizes is unusual for this tape.

#### Lifetime of resting orders

data: [`csv/liq/05-lifetime-of-resting-orders.csv`](csv/liq/05-lifetime-of-resting-orders.csv)

| seconds (log) | cancelled | filled |
| --- | --- | --- |
| 9.4e-05 | 0 | 1 |
| 0.0001447 | 0 | 1 |
| 0.0002227 | 0 | 2 |
| 0.0003427 | 0 | 0 |
| 0.0005274 | 1 | 0 |
| 0.0008117 | 0 | 4 |
| 0.001249 | 0 | 6 |
| 0.001923 | 0 | 7 |
| 0.002959 | 0 | 16 |
| 0.004554 | 0 | 28 |
| 0.007009 | 4 | 28 |
| 0.01079 | 14 | 43 |
| 0.0166 | 77 | 61 |
| 0.02555 | 239 | 99 |
| 0.03932 | 682 | 159 |
| 0.06052 | 1059 | 222 |
| 0.09315 | 688 | 332 |
| 0.1434 | 142 | 495 |
| 0.2206 | 25 | 570 |
| 0.3396 | 35 | 777 |
| 0.5226 | 49 | 996 |
| 0.8043 | 73 | 1106 |
| 1.238 | 119 | 1182 |
| 1.905 | 162 | 1516 |
| 2.932 | 247 | 1872 |
| 4.513 | 308 | 2345 |
| 6.945 | 433 | 2995 |
| 10.69 | 512 | 3450 |
| 16.45 | 686 | 3872 |
| 25.32 | 807 | 4704 |
| 38.97 | 978 | 5498 |
| 59.97 | 1174 | 5847 |
| 92.3 | 1277 | 5882 |
| 142.1 | 1351 | 5144 |
| 218.6 | 1396 | 3810 |
| 336.5 | 1266 | 2574 |
| 517.9 | 1119 | 1332 |
| 797 | 845 | 443 |
| 1227 | 409 | 670 |
| 1888 | 38 | 435 |

y: count

> median 6.32s; 34% live under 10 ms (fleeting quotes).

#### Where passive orders are placed

data: [`csv/liq/06-where-passive-orders-are-placed.csv`](csv/liq/06-where-passive-orders-are-placed.csv)

| ticks from the same-side best (+ inside the spread, − behind) | count |
| --- | --- |
| -39 | 79 |
| -38 | 104 |
| -37 | 143 |
| -36 | 137 |
| -35 | 162 |
| -34 | 181 |
| -33 | 214 |
| -32 | 197 |
| -31 | 236 |
| -30 | 239 |
| -29 | 315 |
| -28 | 374 |
| -27 | 405 |
| -26 | 462 |
| -25 | 488 |
| -24 | 524 |
| -23 | 631 |
| -22 | 713 |
| -21 | 768 |
| -20 | 887 |
| -19 | 998 |
| -18 | 1187 |
| -17 | 1347 |
| -16 | 1565 |
| -15 | 1719 |
| -14 | 1941 |
| -13 | 2284 |
| -12 | 2538 |
| -11 | 2916 |
| -10 | 3185 |
| -9 | 3535 |
| -8 | 4072 |
| -7 | 4767 |
| -6 | 5526 |
| -5 | 5916 |
| -4 | 6360 |
| -3 | 6494 |
| -2 | 6544 |
| -1 | 6384 |
| 0 | 5761 |
| 1 | 1892 |
| 2 | 453 |

y: count

> 0 joins the best price, +1 improves it by a tick, −1 sits one tick behind.

## Trades & flow

#### Aggressive orders per second (17 s bins)

data: [`csv/flow/01-aggressive-orders-per-second-17-s-bins.csv`](csv/flow/01-aggressive-orders-per-second-17-s-bins.csv)

| time (s) | Buy | Sell |
| --- | --- | --- |
| 0 | 8.529 | 3.294 |
| 17 | 8.647 | 5.294 |
| 34 | 1.882 | 20.59 |
| 51 | 4.059 | 11.65 |
| 68 | 2.824 | 14.41 |
| 85 | 4.176 | 17.59 |
| 102 | 5.824 | 4.706 |
| 119 | 4.529 | 7.647 |
| 136 | 15.59 | 5.647 |
| 153 | 0.2941 | 32.47 |
| 170 | 0.2941 | 26.24 |
| 187 | 13.76 | 2.765 |
| 204 | 6.647 | 7.294 |
| 221 | 6.118 | 13 |
| 238 | 6.471 | 10.29 |
| 255 | 11.71 | 3 |
| 272 | 17.53 | 1.882 |
| 289 | 0.1765 | 29.82 |
| 306 | 4.353 | 14.53 |
| 323 | 4.412 | 13.41 |
| 340 | 7.412 | 4.588 |
| 357 | 25.94 | 0.8824 |
| 374 | 30.41 | 0.05882 |
| 391 | 29.76 | 0 |
| 408 | 24.94 | 2.647 |
| 425 | 8.941 | 9.353 |
| 442 | 7.059 | 4.588 |
| 459 | 4.941 | 8.412 |
| 476 | 4.588 | 6.706 |
| 493 | 21.59 | 1.412 |
| 510 | 22.71 | 0.8235 |
| 527 | 31.76 | 0 |
| 544 | 9.059 | 7.471 |
| 561 | 3.059 | 13.94 |
| 578 | 1.471 | 19.59 |
| 595 | 0 | 34.59 |
| 612 | 3.176 | 23.35 |
| 629 | 10.06 | 4.824 |
| 646 | 6.529 | 11.88 |
| 663 | 0.2941 | 27.29 |
| 680 | 1.294 | 20.47 |
| 697 | 0.3529 | 27.47 |
| 714 | 7.529 | 7.176 |
| 731 | 15.41 | 2.588 |
| 748 | 5.471 | 8 |
| 765 | 12.35 | 3.294 |
| 782 | 29.82 | 0.1176 |
| 799 | 16.18 | 1.706 |
| 816 | 25.12 | 1 |
| 833 | 24.94 | 0.4118 |
| 850 | 7 | 18.12 |
| 867 | 2.529 | 15.82 |
| 884 | 0 | 32.18 |
| 901 | 0.2353 | 27.82 |
| 918 | 1.294 | 23.35 |
| 935 | 1.647 | 18.53 |
| 952 | 0 | 33.18 |
| 969 | 0 | 30.06 |
| 986 | 0.05882 | 31.71 |
| 1003 | 2.353 | 24.82 |
| 1020 | 7.118 | 5.647 |
| 1037 | 17.65 | 4.824 |
| 1054 | 5.412 | 8.824 |
| 1071 | 28.71 | 0.1176 |
| 1088 | 23.18 | 1.588 |
| 1105 | 2.294 | 10.53 |
| 1122 | 5.882 | 9.235 |
| 1139 | 2.588 | 15.18 |
| 1156 | 8.353 | 7.235 |
| 1173 | 7.706 | 5.588 |
| 1190 | 30.35 | 0.05882 |
| 1207 | 29.82 | 0.2353 |
| 1224 | 14.65 | 2.471 |
| 1241 | 19.82 | 1.353 |
| 1258 | 27.12 | 0.1176 |
| 1275 | 7.471 | 4.765 |
| 1292 | 27.41 | 0.2353 |
| 1309 | 15.71 | 3.471 |
| 1326 | 4.176 | 8.588 |
| 1343 | 2.294 | 17.94 |
| 1360 | 0 | 31.94 |
| 1377 | 6.059 | 12.88 |
| 1394 | 16.29 | 1.765 |
| 1411 | 12.18 | 5.118 |
| 1428 | 7.824 | 13.41 |
| 1445 | 5.471 | 12.47 |
| 1462 | 11.65 | 5.353 |
| 1479 | 8 | 9.706 |
| 1496 | 3.765 | 10.76 |
| 1513 | 5.353 | 5.882 |
| 1530 | 2.471 | 12.53 |
| 1547 | 0.8824 | 28.35 |
| 1564 | 4 | 9.471 |
| 1581 | 7.412 | 5.824 |
| 1598 | 15.71 | 5.647 |
| 1615 | 11.88 | 6.412 |
| 1632 | 30.47 | 0 |
| 1649 | 31.65 | 0 |
| 1666 | 31.94 | 0 |
| 1683 | 31.12 | 0 |
| 1700 | 31.53 | 0 |
| 1717 | 27.24 | 0.5294 |
| 1734 | 24.94 | 1.059 |
| 1751 | 31.06 | 0 |
| 1768 | 31.59 | 0 |
| 1785 | 32.88 | 0 |
| 1802 | 32.82 | 0 |
| 1819 | 34.06 | 0 |
| 1836 | 31.94 | 0 |
| 1853 | 30.76 | 0 |
| 1870 | 28.65 | 0 |
| 1887 | 26.76 | 0.4706 |
| 1904 | 23.29 | 0.9412 |
| 1921 | 6.765 | 7.706 |
| 1938 | 18.47 | 2 |
| 1955 | 16.24 | 2.529 |
| 1972 | 33 | 0 |
| 1989 | 32.59 | 0 |

y: orders / s

> 44,507 aggressive orders, 955,622 lots. Fills for a passive quote only come from these.

#### Time between aggressive orders

data: [`csv/flow/02-time-between-aggressive-orders.csv`](csv/flow/02-time-between-aggressive-orders.csv)

| seconds | observed | Poisson, same rate |
| --- | --- | --- |
| 0.002741 | 5884 | 5098 |
| 0.008224 | 5106 | 4514 |
| 0.01371 | 4395 | 3997 |
| 0.01919 | 3653 | 3539 |
| 0.02467 | 3190 | 3134 |
| 0.03015 | 2692 | 2775 |
| 0.03564 | 2400 | 2457 |
| 0.04112 | 2023 | 2175 |
| 0.0466 | 1700 | 1926 |
| 0.05208 | 1547 | 1706 |
| 0.05756 | 1383 | 1510 |
| 0.06305 | 1162 | 1337 |
| 0.06853 | 1037 | 1184 |
| 0.07401 | 839 | 1048 |
| 0.07949 | 762 | 928.4 |
| 0.08498 | 669 | 822.1 |
| 0.09046 | 594 | 727.9 |
| 0.09594 | 497 | 644.5 |
| 0.1014 | 480 | 570.7 |
| 0.1069 | 418 | 505.3 |
| 0.1124 | 352 | 447.5 |
| 0.1179 | 349 | 396.2 |
| 0.1234 | 279 | 350.8 |
| 0.1288 | 281 | 310.6 |
| 0.1343 | 237 | 275.1 |
| 0.1398 | 220 | 243.5 |
| 0.1453 | 187 | 215.7 |
| 0.1508 | 168 | 191 |
| 0.1562 | 166 | 169.1 |
| 0.1617 | 140 | 149.7 |
| 0.1672 | 134 | 132.6 |
| 0.1727 | 121 | 117.4 |
| 0.1782 | 105 | 103.9 |
| 0.1837 | 98 | 92.03 |
| 0.1891 | 93 | 81.49 |
| 0.1946 | 82 | 72.16 |
| 0.2001 | 72 | 63.89 |
| 0.2056 | 73 | 56.57 |
| 0.2111 | 71 | 50.09 |
| 0.2166 | 53 | 44.35 |
| 0.222 | 37 | 39.27 |
| 0.2275 | 46 | 34.78 |
| 0.233 | 42 | 30.79 |
| 0.2385 | 44 | 27.27 |
| 0.244 | 29 | 24.14 |
| 0.2494 | 21 | 21.38 |
| 0.2549 | 29 | 18.93 |
| 0.2604 | 39 | 16.76 |
| 0.2659 | 30 | 14.84 |
| 0.2714 | 31 | 13.14 |

y: count (log)

> Coefficient of variation 1.23 (1 for Poisson; above 1 means clustered arrivals, below 1 means regular ones).

#### Aggressive order size

data: [`csv/flow/03-aggressive-order-size.csv`](csv/flow/03-aggressive-order-size.csv)

| quantity | Buy | Sell |
| --- | --- | --- |
| 1 | 798 | 496 |
| 2 | 822 | 521 |
| 3 | 793 | 548 |
| 4 | 786 | 509 |
| 5 | 778 | 471 |
| 6 | 791 | 518 |
| 7 | 768 | 492 |
| 8 | 780 | 520 |
| 9 | 779 | 484 |
| 10 | 814 | 532 |
| 11 | 758 | 511 |
| 12 | 770 | 521 |
| 13 | 728 | 506 |
| 14 | 765 | 490 |
| 15 | 779 | 524 |
| 16 | 793 | 494 |
| 17 | 788 | 558 |
| 18 | 716 | 507 |
| 19 | 782 | 496 |
| 20 | 783 | 530 |
| 21 | 779 | 477 |
| 22 | 744 | 497 |
| 23 | 752 | 489 |
| 24 | 750 | 474 |
| 25 | 766 | 537 |
| 26 | 733 | 490 |
| 27 | 709 | 479 |
| 28 | 767 | 530 |
| 29 | 748 | 495 |
| 30 | 83 | 64 |
| 31 | 104 | 66 |
| 32 | 87 | 57 |
| 33 | 86 | 79 |
| 34 | 91 | 78 |
| 35 | 84 | 69 |
| 36 | 100 | 61 |
| 37 | 88 | 75 |
| 38 | 79 | 61 |
| 39 | 97 | 59 |
| 40 | 78 | 65 |
| 41 | 71 | 64 |
| 42 | 72 | 55 |
| 43 | 83 | 57 |
| 44 | 85 | 94 |
| 45 | 94 | 69 |
| 46 | 90 | 51 |
| 47 | 82 | 70 |
| 48 | 97 | 59 |
| 49 | 78 | 79 |
| 50 | 85 | 78 |
| 51 | 96 | 74 |
| 52 | 91 | 77 |
| 53 | 82 | 70 |
| 54 | 90 | 65 |
| 55 | 86 | 56 |
| 56 | 67 | 61 |
| 57 | 77 | 61 |
| 58 | 91 | 75 |
| 59 | 79 | 77 |
| 60 | 77 | 66 |
| 61 | 90 | 70 |
| 62 | 85 | 58 |
| 63 | 85 | 45 |
| 64 | 92 | 67 |
| 65 | 72 | 76 |
| 66 | 91 | 62 |
| 67 | 79 | 66 |
| 68 | 68 | 66 |
| 69 | 85 | 54 |
| 70 | 87 | 56 |
| 71 | 82 | 53 |
| 72 | 81 | 62 |
| 73 | 90 | 65 |
| 74 | 84 | 59 |
| 75 | 83 | 67 |
| 76 | 94 | 67 |
| 77 | 81 | 48 |
| 78 | 79 | 64 |

y: count

> Large aggressive orders are the ones that sweep through our quote and leave us adversely selected.

#### Levels swept past the touch

data: [`csv/flow/04-levels-swept-past-the-touch.csv`](csv/flow/04-levels-swept-past-the-touch.csv)

| ticks past the best opposite price | observed | geometric, k = 2.90 |
| --- | --- | --- |
| 0 | 0.9514 | 0.9451 |
| 1 | 0.04078 | 0.05192 |
| 2 | 0.006359 | 0.002852 |
| 3 | 0.001213 | 0.0001567 |
| 4 | 0.0001348 | 8.606e-06 |
| 5 | 8.987e-05 | 4.727e-07 |

y: share of orders

> Mean depth 0.0581 → k = ln(1 + 1/mean) = 2.9: the estimator avellaneda_stoikov.py uses online.

#### Fill intensity λ(δ) for a quote δ ticks from the mid

data: [`csv/flow/05-fill-intensity-for-a-quote-ticks-from-the-mid.csv`](csv/flow/05-fill-intensity-for-a-quote-ticks-from-the-mid.csv)

| δ (ticks from mid) | Ask side hit by buys | Bid side hit by sells | A·e^(−kδ): A=10.3, k=0.534 |
| --- | --- | --- | --- |
| 0 | 13.23 | 8.954 | 10.28 |
| 0.5 | 13.23 | 8.954 | 7.871 |
| 1 | 8.54 | 5.157 | 6.027 |
| 1.5 | 5.636 | 3.099 | 4.614 |
| 2 | 4.292 | 2.075 | 3.533 |
| 2.5 | 3.469 | 1.462 | 2.705 |
| 3 | 2.779 | 1.003 | 2.071 |
| 3.5 | 2.197 | 0.67 | 1.586 |
| 4 | 1.866 | 0.4701 | 1.214 |
| 4.5 | 1.562 | 0.3106 | 0.9297 |
| 5 | 1.346 | 0.1575 | 0.7119 |
| 5.5 | 1.122 | 0.05932 | 0.5451 |
| 6 | 0.85 | 0.02792 | 0.4173 |
| 6.5 | 0.6581 | 0.01047 | 0.3195 |
| 7 | 0.5389 | 0.0004985 | 0.2447 |
| 7.5 | 0.4467 | - | 0.1873 |
| 8 | 0.3604 | - | 0.1434 |
| 8.5 | 0.2767 | - | 0.1098 |
| 9 | 0.2129 | - | 0.08409 |
| 9.5 | 0.1506 | - | 0.06438 |
| 10 | 0.09771 | - | 0.0493 |
| 10.5 | 0.05683 | - | 0.03774 |
| 11 | 0.01994 | - | 0.0289 |
| 11.5 | 0.0004985 | - | 0.02213 |
| 12 | - | - | 0.01694 |
| 12.5 | - | - | 0.01297 |
| 13 | - | - | 0.009932 |
| 13.5 | - | - | 0.007605 |
| 14 | - | - | 0.005823 |
| 14.5 | - | - | 0.004458 |
| 15 | - | - | 0.003414 |

y: hits per second (log)

> The Avellaneda-Stoikov arrival model. A straight line on this log axis means the exponential model fits; a kink means it does not.

#### Cumulative signed aggressive volume / Mid

data: [`csv/flow/06-cumulative-signed-aggressive-volume-mid.csv`](csv/flow/06-cumulative-signed-aggressive-volume-mid.csv)

| series | points | x range | y min | mean | median | max | last |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Net buy volume | 44,507 | 0.1476..2006 | -71600 | -994.6 | -14971 | 163674 | 163674 |
| Mid | 2,001 | 0..2006 | 9936 | 9977 | 9973 | 10031 | 10031 |

> If price follows net flow closely, aggressive orders carry information.

#### Autocorrelation of aggressive order signs

data: [`csv/flow/07-autocorrelation-of-aggressive-order-signs.csv`](csv/flow/07-autocorrelation-of-aggressive-order-signs.csv)

| lag (orders) | correlation |
| --- | --- |
| 1 | 0.8265 |
| 2 | 0.8236 |
| 3 | 0.8238 |
| 4 | 0.8172 |
| 5 | 0.8132 |
| 6 | 0.81 |
| 7 | 0.8069 |
| 8 | 0.8045 |
| 9 | 0.803 |
| 10 | 0.7951 |
| 11 | 0.7948 |
| 12 | 0.7908 |
| 13 | 0.79 |
| 14 | 0.789 |
| 15 | 0.7832 |
| 16 | 0.7798 |
| 17 | 0.777 |
| 18 | 0.7716 |
| 19 | 0.7721 |
| 20 | 0.7687 |
| 21 | 0.7644 |
| 22 | 0.7667 |
| 23 | 0.7603 |
| 24 | 0.7582 |
| 25 | 0.7586 |
| 26 | 0.759 |
| 27 | 0.7536 |
| 28 | 0.7506 |
| 29 | 0.7506 |
| 30 | 0.7458 |

y: correlation

Marked: shaded band y ∈ [-0.009291, 0.009291].

> Positive: buys follow buys (split metaorders). Their signs then predict the next order.

## Predictability

#### Future mid move by queue imbalance

data: [`csv/predict/01-future-mid-move-by-queue-imbalance.csv`](csv/predict/01-future-mid-move-by-queue-imbalance.csv)

| imbalance  Qbid / (Qbid + Qask) | 0.1 s | 1 s | 5 s | 30 s |
| --- | --- | --- | --- | --- |
| 0.04264 | -0.03443 ± 0.01622 | 0.01435 ± 0.04171 | 0.5234 ± 0.1066 | 1.432 ± 0.3663 |
| 0.1428 | -0.01455 ± 0.03165 | 0.2393 ± 0.09694 | 0.8964 ± 0.22 | 3.118 ± 0.642 |
| 0.2454 | -0.01284 ± 0.05328 | 0.2564 ± 0.1365 | 0.7083 ± 0.3705 | 2.072 ± 1.06 |
| 0.3472 | 0.002755 ± 0.06496 | 0.2445 ± 0.182 | 0.5441 ± 0.4682 | 2.05 ± 1.218 |
| 0.4451 | -0.03667 ± 0.08591 | 0.01878 ± 0.2414 | -0.06708 ± 0.5604 | 0.6914 ± 1.444 |
| 0.5514 | 0.01034 ± 0.08514 | -0.1112 ± 0.2354 | -0.06897 ± 0.5437 | 0.7629 ± 1.45 |
| 0.6534 | 0.02135 ± 0.07582 | -0.2126 ± 0.2241 | -0.3276 ± 0.5348 | -0.4836 ± 1.431 |
| 0.755 | 0.001748 ± 0.06187 | -0.2477 ± 0.1749 | -0.6649 ± 0.4443 | -0.7325 ± 1.2 |
| 0.857 | 0.0301 ± 0.04233 | -0.2088 ± 0.1189 | -0.8191 ± 0.2945 | -1.349 ± 0.7944 |
| 0.9589 | 0.04984 ± 0.01965 | -0.00737 ± 0.04902 | -0.5506 ± 0.1365 | -2.271 ± 0.4048 |

y: mean Δmid (ticks)  (± is the 95% interval)

Marked: horizontal line y = 0.

> Buckets with under 30 samples hidden. A monotone S-curve means imbalance predicts the next move: use the microprice instead of the mid, or lean quotes with it.

#### Is the imbalance edge still there 5 ms later?

data: [`csv/predict/02-is-the-imbalance-edge-still-there-5-ms-later.csv`](csv/predict/02-is-the-imbalance-edge-still-there-5-ms-later.csv)

| imbalance | measured from now | measured from now + 5 ms |
| --- | --- | --- |
| 0.04264 | -0.03443 | -0.0313 |
| 0.1428 | -0.01455 | -0.01541 |
| 0.2454 | -0.01284 | -0.01606 |
| 0.3472 | 0.002755 | 0.01102 |
| 0.4451 | -0.03667 | -0.02683 |
| 0.5514 | 0.01034 | 0.007759 |
| 0.6534 | 0.02135 | 0.0197 |
| 0.755 | 0.001748 | 0.002914 |
| 0.857 | 0.0301 | 0.02733 |
| 0.9589 | 0.04984 | 0.04587 |

y: mean Δmid over the next 100 ms (ticks)

Marked: horizontal line y = 0.

> Our orders land 5 ms after we decide. If the two curves match, the signal is still worth trading on.

#### After an aggressive order: mid move in its direction

data: [`csv/predict/03-after-an-aggressive-order-mid-move-in-its-direction.csv`](csv/predict/03-after-an-aggressive-order-mid-move-in-its-direction.csv)

| seconds after (log) | price impact (vs pre-trade mid) | taker PnL per lot (vs fill) = −maker PnL |
| --- | --- | --- |
| 0.005 | 0.05176 ± 0.004288 | -1.569 |
| 0.05 | 0.06952 ± 0.005917 | -1.552 |
| 0.1 | 0.0899 ± 0.007387 | -1.531 |
| 0.25 | 0.1454 ± 0.00977 | -1.476 |
| 0.5 | 0.2365 ± 0.01302 | -1.385 |
| 1 | 0.3965 ± 0.01744 | -1.225 |
| 2 | 0.6608 ± 0.02501 | -0.9603 |
| 5 | 1.227 ± 0.04339 | -0.3939 |
| 10 | 1.835 ± 0.06853 | 0.2142 |
| 20 | 2.636 ± 0.1086 | 1.015 |
| 30 | 2.972 ± 0.1398 | 1.351 |
| 60 | 3.65 ± 0.2097 | 2.029 |

y: ticks  (± is the 95% interval)

Marked: horizontal line y = 0; vertical line x = 0.005.

> Positive orange line: passive fills lose money at that horizon (adverse selection). The quoted half-spread must cover it.

#### Order flow imbalance → mid move (R²)

data: [`csv/predict/04-order-flow-imbalance-mid-move-r.csv`](csv/predict/04-order-flow-imbalance-mid-move-r.csv)

| bin size (s) | same bin (impact) | next bin (prediction) |
| --- | --- | --- |
| 0.1 | 0.05702 | 0.003484 |
| 0.5 | 0.1152 | 0.03377 |
| 1 | 0.1737 | 0.06357 |
| 5 | 0.3689 | 0.07906 |
| 30 | 0.5961 | 0.01463 |

y: R²

> Same-bin R² is price impact (descriptive). Next-bin R² is real predictability.

#### OFI vs Δmid, 1 s bins

data: [`csv/predict/05-ofi-vs-mid-1-s-bins.csv`](csv/predict/05-ofi-vs-mid-1-s-bins.csv)

| OFI (lots) | fit |
| --- | --- |
| -996 | -0.9625 |
| 1159 | 1.12 |

y: Δmid (ticks)

| series | points | OFI (lots) range | Δmid (ticks) min | mean | median | max | last |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scatter 1 | 2,003 | -996..1159 | -9 | 0.01648 | 0 | 10.5 | 0 |

> Change the bin size with the sampling selector at the top.

### OFI regression  Δmid = β·OFI

| bin | β (ticks/lot) | R² same | β next | R² next | n |
| --- | --- | --- | --- | --- | --- |
| 0.1 s | 0.00155 | 0.057 | 0.000383 | 0.00348 | 20,058 |
| 0.5 s | 0.00107 | 0.115 | 0.00058 | 0.0338 | 4,010 |
| 1 s | 0.000966 | 0.174 | 0.000585 | 0.0636 | 2,004 |
| 5 s | 0.000803 | 0.369 | 0.000372 | 0.0791 | 400 |
| 30 s | 0.000602 | 0.596 | 9.49e-05 | 0.0146 | 65 |
