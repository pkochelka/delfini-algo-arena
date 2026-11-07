# Data explorer: sample

Data explorer export of `data/sample.csv`, window 0.0–3,599.0 s (60.0 min), sampling selector 1 s (charts that use it: return histogram and ACF, rolling σ, variance ratio, OFI scatter).
Generated 2026-11-07 12:49 by `uv run data-explorer --data … --export` (see data_explorer/export.py). Numbers are in ticks and seconds. The data behind each chart is in csv/, machine-readable headline stats in summary.json, interactive charts in report.html.

Contents: [Overview](#overview) · [Price & spread](#price--spread) · [Volatility](#volatility) · [Liquidity & book](#liquidity--book) · [Trades & flow](#trades--flow) · [Predictability](#predictability)

## Overview

### Tape fingerprint

| Statistic | Value | Detail |
| --- | --- | --- |
| Window | 60.0 min | 53,758 events |
| Events / s | 14.9 | cancel/insert 0.98 |
| Aggressive orders / min | 10.1 | 604 fills |
| Traded volume | 7,706 | 50% buyer-initiated |
| Mid | 9,999 → 10,037 | range 9,867–10,072 |
| Spread (time-weighted) | 2 | min 2 ticks |
| One-sided book | 0.0% | of time a side is empty |
| σ at 100 ms | 1.64 | ticks / √s |
| σ at 1 s | 2.57 | 10 s: 3.99 |
| Touch size | 24.7 | mean qty at best, per side |
| Order / fill size | 24.7 / 12.8 | mean insert / aggressive order |
| Passive fill rate | 2.3% | resting orders that got any fill |

### Fitted parameters (selected window)

```python
# sample, window 0-3599 s  (python -m analysis.fit)
SIGMA_INIT = 1.637   # ticks/sqrt(s) at 100 ms; 1 s: 2.565
K_INIT = -       # ln(1 + 1/mean sweep depth), depth 0
# intensity fit: A = - /s, k = - /tick (1 points)
# OFI beta at 1 s = 0.01944 ticks/lot (R² 0.968), next-bin R² 0.341
# taker markout vs fill: 1 s 2.06, 10 s 8.15 ticks
IMBALANCE_DMID_1S = {"0.09": -2.75, "0.16": -3.5696, "0.26": -0.1819, "0.35": -0.0522, "0.45": -0.0842, "0.54": 0.0892, "0.65": 0.0242, "0.74": 0.3945, "0.82": 1.5809, "0.90": 1.0}
```

> Copy into a strategy: submissions cannot read files.

### Activity per 30 s

#### Events / Aggressive orders / Spread (ticks) / σ 1 s (ticks/√s)

data: [`csv/overview/01-events-aggressive-orders-spread-ticks-1-s-ticks-s.csv`](csv/overview/01-events-aggressive-orders-spread-ticks-1-s-ticks-s.csv)

| x | Events | Aggressive orders | Spread (ticks) | σ 1 s (ticks/√s) |
| --- | --- | --- | --- | --- |
| 0 | 363 | 5 | 2 | 2.486 |
| 30 | 661 | 9 | 2 | 3.768 |
| 60 | 358 | 2 | 2 | 2.236 |
| 90 | 393 | 5 | 2 | 1.966 |
| 120 | 233 | 1 | 2 | 1.354 |
| 150 | 442 | 6 | 2 | 1.975 |
| 180 | 583 | 11 | 2 | 2.904 |
| 210 | 533 | 5 | 2 | 2.799 |
| 240 | 424 | 4 | 2 | 1.871 |
| 270 | 551 | 7 | 2 | 3.088 |
| 300 | 534 | 6 | 2 | 3.271 |
| 330 | 428 | 4 | 2 | 2.074 |
| 360 | 279 | 3 | 2 | 1.354 |
| 390 | 353 | 5 | 2 | 1.835 |
| 420 | 528 | 4 | 2 | 4.324 |
| 450 | 368 | 4 | 2 | 1.722 |
| 480 | 441 | 5 | 2 | 2.443 |
| 510 | 388 | 4 | 2 | 2.288 |
| 540 | 360 | 4 | 2 | 2.017 |
| 570 | 506 | 6 | 2 | 2.121 |
| 600 | 449 | 5 | 2 | 2.025 |
| 630 | 335 | 3 | 2 | 1.693 |
| 660 | 571 | 7 | 2 | 2.978 |
| 690 | 588 | 8 | 2 | 4.082 |
| 720 | 642 | 6 | 2 | 3.141 |
| 750 | 351 | 3 | 2 | 1.941 |
| 780 | 364 | 4 | 2 | 2.176 |
| 810 | 343 | 3 | 2 | 2.098 |
| 840 | 339 | 3 | 2 | 2.331 |
| 870 | 262 | 2 | 2 | 1.304 |
| 900 | 348 | 4 | 2 | 2.191 |
| 930 | 616 | 8 | 2 | 3.812 |
| 960 | 440 | 4 | 2 | 2.16 |
| 990 | 523 | 7 | 2 | 2.309 |
| 1020 | 391 | 3 | 2 | 1.941 |
| 1050 | 352 | 4 | 2 | 1.932 |
| 1080 | 408 | 4 | 2 | 2.008 |
| 1110 | 351 | 3 | 2 | 2.671 |
| 1140 | 365 | 5 | 2 | 1.329 |
| 1170 | 619 | 7 | 2 | 3.22 |
| 1200 | 765 | 9 | 2 | 3.958 |
| 1230 | 442 | 6 | 2 | 1.975 |
| 1260 | 480 | 4 | 2 | 2.944 |
| 1290 | 294 | 2 | 2 | 1.506 |
| 1320 | 336 | 4 | 2 | 1.761 |
| 1350 | 584 | 8 | 2 | 2.671 |
| 1380 | 505 | 5 | 2 | 2.84 |
| 1410 | 481 | 5 | 2 | 2.751 |
| 1440 | 557 | 9 | 2 | 3.531 |
| 1470 | 376 | 4 | 2 | 1.817 |
| 1500 | 323 | 3 | 2 | 1.932 |
| 1530 | 351 | 3 | 2 | 1.751 |
| 1560 | 373 | 5 | 2 | 2.288 |
| 1590 | 322 | 2 | 2 | 1.992 |
| 1620 | 473 | 5 | 2 | 2.733 |
| 1650 | 620 | 8 | 2 | 2.864 |
| 1680 | 335 | 3 | 2 | 2.198 |
| 1710 | 473 | 5 | 2 | 2.966 |
| 1740 | 509 | 5 | 2 | 2.915 |
| 1770 | 540 | 8 | 2 | 2.858 |
| 1800 | 485 | 5 | 2 | 2.955 |
| 1830 | 373 | 5 | 2 | 2.523 |
| 1860 | 454 | 6 | 2 | 3.104 |
| 1890 | 411 | 3 | 2 | 2.221 |
| 1920 | 559 | 11 | 2 | 3.017 |
| 1950 | 605 | 5 | 2 | 2.875 |
| 1980 | 599 | 7 | 2 | 3.199 |
| 2010 | 380 | 4 | 2 | 2.244 |
| 2040 | 372 | 4 | 2 | 2.781 |
| 2070 | 314 | 2 | 2 | 1.924 |
| 2100 | 323 | 3 | 2 | 2.168 |
| 2130 | 274 | 2 | 2 | 1.449 |
| 2160 | 314 | 2 | 2 | 1.798 |
| 2190 | 357 | 5 | 2 | 1.871 |
| 2220 | 464 | 4 | 2 | 2.852 |
| 2250 | 299 | 3 | 2 | 1.581 |
| 2280 | 489 | 5 | 2 | 2.994 |
| 2310 | 466 | 6 | 2 | 2.614 |
| 2340 | 385 | 5 | 2 | 2.53 |
| 2370 | 778 | 10 | 2 | 4.34 |
| 2400 | 660 | 8 | 2 | 2.324 |
| 2430 | 808 | 12 | 2 | 4.461 |
| 2460 | 419 | 3 | 2 | 2.927 |
| 2490 | 432 | 4 | 2 | 2.497 |
| 2520 | 409 | 5 | 2 | 2.787 |
| 2550 | 282 | 2 | 2 | 1.494 |
| 2580 | 460 | 4 | 2 | 2.352 |
| 2610 | 331 | 3 | 2 | 2.614 |
| 2640 | 483 | 7 | 2 | 2.543 |
| 2670 | 595 | 7 | 2 | 3.22 |
| 2700 | 404 | 4 | 2 | 1.703 |
| 2730 | 398 | 6 | 2 | 1.826 |
| 2760 | 335 | 3 | 2 | 2.401 |
| 2790 | 519 | 7 | 2 | 2.834 |
| 2820 | 388 | 4 | 2 | 1.915 |
| 2850 | 310 | 2 | 2 | 1.906 |
| 2880 | 461 | 5 | 2 | 2.106 |
| 2910 | 530 | 6 | 2 | 2.106 |
| 2940 | 705 | 13 | 2 | 3.189 |
| 2970 | 572 | 4 | 2 | 2.266 |
| 3000 | 410 | 6 | 2 | 2.978 |
| 3030 | 416 | 4 | 2 | 2.206 |
| 3060 | 403 | 7 | 2 | 1.826 |
| 3090 | 676 | 8 | 2 | 2.811 |
| 3120 | 546 | 6 | 2 | 2.769 |
| 3150 | 385 | 5 | 2 | 2.938 |
| 3180 | 608 | 8 | 2 | 3.545 |
| 3210 | 493 | 5 | 2 | 2.745 |
| 3240 | 378 | 2 | 2 | 3.266 |
| 3270 | 352 | 4 | 2 | 1.958 |
| 3300 | 307 | 3 | 2 | 1.844 |
| 3330 | 649 | 9 | 2 | 2.787 |
| 3360 | 505 | 5 | 2 | 2.665 |
| 3390 | 301 | 1 | 2 | 1.862 |
| 3420 | 605 | 9 | 2 | 3.114 |
| 3450 | 338 | 2 | 2 | 1.581 |
| 3480 | 417 | 5 | 2 | 2.683 |
| 3510 | 456 | 4 | 2 | 2.898 |
| 3540 | 351 | 3 | 2 | 1.932 |
| 3570 | 438 | 6 | 2 | 2.852 |

y: y

> Regime changes show up here first: bursts of activity, spread widening, volatility clustering.

## Price & spread

#### Best bid / ask, microprice and aggressive orders

data: not saved (42,283 points; `--export --all-csv` writes it)

| series | points | time (s) range | price (ticks) min | mean | median | max | last |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Best bid | 13,893 | 1e-06..3599 | 9866 | 9961 | 9959 | 10072 | 10036 |
| Best ask | 13,893 | 1e-06..3599 | 9868 | 9963 | 9961 | 10074 | 10038 |
| Microprice | 13,893 | 1e-06..3599 | 9866 | 9962 | 9960 | 1.007e+04 | 1.004e+04 |
| Aggressive buy | 307 | 24.36..3581 | 9871 | 9959 | 9959 | 10052 | 10035 |
| Aggressive sell | 297 | 6.863..3590 | 9880 | 9963 | 9962 | 10054 | 10051 |

> Every quote change in the window is drawn. Marker size grows with order quantity; it sits at the worst price the order reached.

#### Spread over time

data: not saved (13,893 points; `--export --all-csv` writes it)

| series | points | time (s) range | ticks min | mean | median | max | last |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Spread | 13,893 | 1e-06..3599 | 2 | 2 | 2 | 2 | 2 |

#### Spread distribution (share of time)

data: [`csv/price/03-spread-distribution-share-of-time.csv`](csv/price/03-spread-distribution-share-of-time.csv)

| spread (ticks) | share of time |
| --- | --- |
| 2 | 1 |

y: share of time

> A market maker quoting inside the spread needs it wider than 1 tick most of the time.

#### Resting depth by price (bids +, asks -)

data: [`csv/price/04-resting-depth-by-price-bids-asks.csv`](csv/price/04-resting-depth-by-price-bids-asks.csv)

Heatmap, 1,220 non-zero cells (missing cells in the CSV are 0); x (time (s)) 0.1..3593, y (price (ticks)) 9869..10072; z mean 0.2533, max 40.

> Snapshots every 100 ms. Look for walls (persistent large levels) and how deep the book is.

## Volatility

#### Rolling σ over 60 s

data: not saved (3,600 points; `--export --all-csv` writes it)

| series | points | time (s) range | ticks / √s min | mean | median | max | last |
| --- | --- | --- | --- | --- | --- | --- | --- |
| σ | 3,600 | 0..3599 | 1.483 | 2.518 | 2.487 | 3.894 | 2.436 |

> Clustering (calm and wild stretches) means a fixed σ under- and over-quotes in turn: estimate it online.

#### Volatility signature

data: [`csv/vol/02-volatility-signature.csv`](csv/vol/02-volatility-signature.csv)

| sampling interval (s) | σ (ticks / √s) |
| --- | --- |
| 0.05 | 1.643 |
| 0.1 | 1.637 |
| 0.2 | 1.648 |
| 0.5 | 2.032 |
| 1 | 2.565 |
| 2 | 3.222 |
| 5 | 3.899 |
| 10 | 3.994 |
| 30 | 3.701 |
| 60 | 3.47 |

y: σ (ticks / √s)

> Flat: random walk. Rising with the interval: momentum. Falling: bid-ask bounce or mean reversion. Pick σ at the horizon you hold inventory over.

#### 1 s mid returns

data: [`csv/vol/03-1-s-mid-returns.csv`](csv/vol/03-1-s-mid-returns.csv)

| Δmid (ticks) | observed |
| --- | --- |
| -8 | 0.003057 |
| -7.5 | 0 |
| -7 | 0.005837 |
| -6.5 | 0 |
| -6 | 0.00806 |
| -5.5 | 0 |
| -5 | 0.02418 |
| -4.5 | 0 |
| -4 | 0.03585 |
| -3.5 | 0 |
| -3 | 0.05447 |
| -2.5 | 0 |
| -2 | 0.07838 |
| -1.5 | 0 |
| -1 | 0.1548 |
| -0.5 | 0 |
| 0 | 0.2707 |
| 0.5 | 0 |
| 1 | 0.1459 |
| 1.5 | 0 |
| 2 | 0.08366 |
| 2.5 | 0 |
| 3 | 0.05003 |
| 3.5 | 0 |
| 4 | 0.03669 |
| 4.5 | 0 |
| 5 | 0.01695 |
| 5.5 | 0 |
| 6 | 0.0139 |
| 6.5 | 0 |
| 7 | 0.007504 |
| 7.5 | 0 |
| 8 | 0.004169 |

y: share of intervals

| series | points | Δmid (ticks) range | share of intervals min | mean | median | max | last |
| --- | --- | --- | --- | --- | --- | --- | --- |
| normal, same σ | 200 | -8..8 | 0.0005929 | 0.03104 | 0.02277 | 0.07776 | 0.0006084 |

> excess kurtosis 1.89, skew 0.11, 27% of intervals unchanged, n = 3,598

#### Autocorrelation of 1 s returns

data: [`csv/vol/04-autocorrelation-of-1-s-returns.csv`](csv/vol/04-autocorrelation-of-1-s-returns.csv)

| lag (intervals) | correlation |
| --- | --- |
| 1 | 0.5851 |
| 2 | 0.2592 |
| 3 | 0.05823 |
| 4 | 0.0003636 |
| 5 | -0.03005 |
| 6 | -0.04074 |
| 7 | -0.04222 |
| 8 | -0.04463 |
| 9 | -0.06068 |
| 10 | -0.03711 |
| 11 | -0.0639 |
| 12 | -0.04594 |
| 13 | -0.02681 |
| 14 | -0.03644 |
| 15 | -0.03868 |
| 16 | -0.04928 |
| 17 | -0.05274 |
| 18 | -0.032 |
| 19 | -0.0005297 |
| 20 | -0.04138 |
| 21 | -0.0587 |
| 22 | -0.04358 |
| 23 | -0.003278 |
| 24 | 0.003946 |
| 25 | 0.006271 |
| 26 | 0.01476 |
| 27 | -0.00821 |
| 28 | -0.02806 |
| 29 | -0.03942 |
| 30 | -0.01627 |

y: correlation

Marked: shaded band y ∈ [-0.03268, 0.03268].

> Bars outside the gray band are significant. Positive at short lags: trends (take); negative: reversion (make).

#### Variance ratio

data: [`csv/vol/05-variance-ratio.csv`](csv/vol/05-variance-ratio.csv)

| horizon (s) | VR(q) |
| --- | --- |
| 2 | 1.586 |
| 5 | 2.296 |
| 10 | 2.436 |
| 30 | 1.9 |
| 60 | 1.554 |

y: VR(q)

Marked: horizontal line y = 1.

> Above 1: moves persist over that horizon. Below 1: they revert.

#### Autocorrelation of 10 ms returns by lag

data: [`csv/vol/06-autocorrelation-of-10-ms-returns-by-lag.csv`](csv/vol/06-autocorrelation-of-10-ms-returns-by-lag.csv)

| lag (s, log) | ACF |
| --- | --- |
| 0.01 | -0.001236 |
| 0.02 | -0.001853 |
| 0.03 | -0.001133 |
| 0.04 | -0.000618 |
| 0.05 | -0.000618 |
| 0.06 | -0.001956 |
| 0.07 | -0.0001033 |
| 0.09 | -0.0035 |
| 0.12 | 0.002985 |
| 0.15 | 0.009161 |
| 0.18 | 0.02265 |
| 0.23 | 0.02584 |
| 0.28 | 0.03181 |
| 0.35 | 0.02347 |
| 0.44 | 0.02038 |
| 0.55 | 0.01873 |
| 0.69 | 0.0176 |
| 0.87 | 0.01276 |
| 1.08 | 0.01492 |
| 1.35 | 0.01122 |
| 1.69 | 0.01019 |
| 2.11 | 0.003396 |
| 2.64 | 0.004014 |
| 3.3 | -0.001853 |
| 4.13 | -0.001339 |
| 5.16 | -0.0004121 |
| 6.45 | -0.001339 |
| 8.06 | -0.004015 |
| 10.07 | -0.001339 |
| 12.59 | -0.001442 |
| 15.74 | -0.0006182 |
| 19.67 | 0.002367 |
| 24.58 | -0.001236 |
| 30.73 | 0.001441 |
| 38.41 | 0.003087 |
| 48 | 0.001646 |
| 60 | -0.002265 |

y: correlation

Marked: shaded band y ∈ [-0.003267, 0.003267]; horizontal line y = 0; vertical line x = 0.01.

> One axis from 10 ms to 60 s, so it does not depend on the sampling selector. Negative at the shortest lags: bid-ask bounce or quote flicker. Positive at second-scale lags: momentum. Dashed line: our 5 ms latency. Values look small because most 10 ms returns are zero: compare them with the gray band, not with 1.

#### Autocorrelation of 100 ms returns, every lag to 10 s

data: [`csv/vol/07-autocorrelation-of-100-ms-returns-every-lag-to-10-s.csv`](csv/vol/07-autocorrelation-of-100-ms-returns-every-lag-to-10-s.csv)

| lag (s) | correlation |
| --- | --- |
| 0.1 | 0.01471 |
| 0.2 | 0.1982 |
| 0.3 | 0.2495 |
| 0.4 | 0.1883 |
| 0.5 | 0.1846 |
| 0.6 | 0.1877 |
| 0.7 | 0.1742 |
| 0.8 | 0.1531 |
| 0.9 | 0.1672 |
| 1 | 0.1428 |
| 1.1 | 0.132 |
| 1.2 | 0.1308 |
| 1.3 | 0.1178 |
| 1.4 | 0.1232 |
| 1.5 | 0.08405 |
| 1.6 | 0.102 |
| 1.7 | 0.08 |
| 1.8 | 0.08467 |
| 1.9 | 0.06622 |
| 2 | 0.05575 |
| 2.1 | 0.05327 |
| 2.2 | 0.05233 |
| 2.3 | 0.03005 |
| 2.4 | 0.03772 |
| 2.5 | 0.02653 |
| 2.6 | 0.01606 |
| 2.7 | 0.02445 |
| 2.8 | 0.02311 |
| 2.9 | 0.005903 |
| 3 | 0.007354 |
| 3.1 | 0.01471 |
| 3.2 | 0.007872 |
| 3.3 | 0.0229 |
| 3.4 | 0.01378 |
| 3.5 | -0.001455 |
| 3.6 | 0.001136 |
| 3.7 | -0.008502 |
| 3.8 | -0.009746 |
| 3.9 | -0.004979 |
| 4 | 0.03658 |
| 4.1 | -0.005912 |
| 4.2 | -0.001662 |
| 4.3 | -0.02011 |
| 4.4 | -0.008295 |
| 4.5 | -0.009435 |
| 4.6 | -0.01482 |
| 4.7 | -0.007673 |
| 4.8 | -0.01089 |
| 4.9 | -0.009124 |
| 5 | 0.03544 |
| 5.1 | -0.01555 |
| 5.2 | -0.0169 |
| 5.3 | -0.02063 |
| 5.4 | -0.007881 |
| 5.5 | -0.01172 |
| 5.6 | -0.01482 |
| 5.7 | -0.007052 |
| 5.8 | -0.01068 |
| 5.9 | -0.02166 |
| 6 | -0.003113 |
| 6.1 | -0.01824 |
| 6.2 | -0.01337 |
| 6.3 | -0.01078 |
| 6.4 | -0.009125 |
| 6.5 | -0.01089 |
| 6.6 | -0.0007298 |
| 6.7 | 0.01813 |
| 6.8 | -0.003943 |
| 6.9 | -0.01317 |
| 7 | -0.01742 |
| 7.1 | -0.01638 |
| 7.2 | -0.005705 |
| 7.3 | -0.01617 |
| 7.4 | -0.01026 |
| 7.5 | -0.01109 |
| 7.6 | -0.014 |
| 7.7 | -0.002078 |
| 7.8 | -0.02519 |
| 7.9 | -0.01286 |
| 8 | 0.0145 |
| 8.1 | -0.01669 |
| 8.2 | -0.01545 |
| 8.3 | -0.02488 |
| 8.4 | -0.01742 |
| 8.5 | -0.01897 |
| 8.6 | -0.02229 |
| 8.7 | -0.0113 |
| 8.8 | -0.02187 |
| 8.9 | -0.006639 |
| 9 | -0.01472 |
| 9.1 | -0.009023 |
| 9.2 | -0.0198 |
| 9.3 | -0.01192 |
| 9.4 | -0.008505 |
| 9.5 | -0.01338 |
| 9.6 | -0.0112 |
| 9.7 | -0.01348 |
| 9.8 | -0.004981 |
| 9.9 | -0.0112 |
| 10 | 0.03088 |

y: correlation

Marked: shaded band y ∈ [-0.01033, 0.01033].

> 35,989 returns. Between the two charts above: the lag at which bars turn from blue to orange is how long a move keeps going before it starts to revert, which is a natural holding time or signal horizon.

#### Autocorrelation of consecutive mid changes (tick by tick)

data: [`csv/vol/08-autocorrelation-of-consecutive-mid-changes-tick-by-tick.csv`](csv/vol/08-autocorrelation-of-consecutive-mid-changes-tick-by-tick.csv)

| lag (mid changes) | correlation |
| --- | --- |
| 1 | 0.2954 |
| 2 | 0.3469 |
| 3 | 0.3045 |
| 4 | 0.2796 |
| 5 | 0.2679 |
| 6 | 0.2195 |
| 7 | 0.2095 |
| 8 | 0.1535 |
| 9 | 0.1467 |
| 10 | 0.1264 |
| 11 | 0.1065 |
| 12 | 0.0841 |
| 13 | 0.06831 |
| 14 | 0.05995 |
| 15 | 0.03032 |
| 16 | 0.02816 |
| 17 | 0.0136 |
| 18 | 0.01845 |
| 19 | -0.001055 |
| 20 | -0.007971 |
| 21 | -0.008075 |
| 22 | 0.0004912 |
| 23 | -0.02934 |
| 24 | -0.02325 |
| 25 | -0.02129 |
| 26 | -0.03791 |
| 27 | -0.02748 |
| 28 | -0.03233 |
| 29 | -0.02893 |
| 30 | -0.04141 |
| 31 | -0.03718 |
| 32 | -0.03729 |
| 33 | -0.03574 |
| 34 | -0.03501 |
| 35 | -0.03739 |
| 36 | -0.04368 |
| 37 | -0.03635 |
| 38 | -0.03728 |
| 39 | -0.025 |
| 40 | -0.03129 |
| 41 | -0.0314 |
| 42 | -0.03645 |
| 43 | -0.03656 |
| 44 | -0.04533 |
| 45 | -0.03243 |
| 46 | -0.03418 |
| 47 | -0.03841 |
| 48 | -0.03273 |
| 49 | -0.03201 |
| 50 | -0.03872 |

y: correlation

Marked: shaded band y ∈ [-0.01991, 0.01991].

> 9,689 mid changes. Lag 1 strongly negative: moves get undone (fade them). Positive: moves continue (follow them).

#### Time between mid changes

data: [`csv/vol/09-time-between-mid-changes.csv`](csv/vol/09-time-between-mid-changes.csv)

| seconds (log) | mid changes |
| --- | --- |
| 4e-06 | 1 |
| 5.737e-06 | 1 |
| 8.229e-06 | 0 |
| 1.18e-05 | 0 |
| 1.693e-05 | 0 |
| 2.428e-05 | 1 |
| 3.483e-05 | 0 |
| 4.996e-05 | 0 |
| 7.166e-05 | 0 |
| 0.0001028 | 0 |
| 0.0001474 | 1 |
| 0.0002115 | 3 |
| 0.0003033 | 4 |
| 0.0004351 | 5 |
| 0.000624 | 4 |
| 0.0008951 | 13 |
| 0.001284 | 15 |
| 0.001842 | 8 |
| 0.002641 | 21 |
| 0.003789 | 31 |
| 0.005434 | 55 |
| 0.007795 | 45 |
| 0.01118 | 96 |
| 0.01604 | 111 |
| 0.023 | 156 |
| 0.03299 | 215 |
| 0.04732 | 353 |
| 0.06788 | 464 |
| 0.09736 | 614 |
| 0.1396 | 752 |
| 0.2003 | 1474 |
| 0.2873 | 1972 |
| 0.4121 | 1317 |
| 0.5911 | 558 |
| 0.8478 | 397 |
| 1.216 | 369 |
| 1.744 | 294 |
| 2.502 | 200 |
| 3.588 | 105 |
| 5.147 | 29 |
| 7.383 | 5 |

y: count

> 9,689 mid changes. Changes are the events a quoting strategy has to react to.

#### Size of mid changes

data: [`csv/vol/10-size-of-mid-changes.csv`](csv/vol/10-size-of-mid-changes.csv)

| Δmid (ticks) | count |
| --- | --- |
| -1 | 4826 |
| 1 | 4863 |

y: count

> Mostly ±0.5 / ±1 means the price moves tick by tick; large jumps mean gaps in the book.

## Liquidity & book

#### Mean resting quantity by distance from the touch

data: [`csv/liq/01-mean-resting-quantity-by-distance-from-the-touch.csv`](csv/liq/01-mean-resting-quantity-by-distance-from-the-touch.csv)

| ticks behind the best price | Bid | Ask |
| --- | --- | --- |
| 0 | 24.71 | 24.76 |
| 1 | 0 | 0 |
| 2 | 0 | 0 |
| 3 | 0 | 0 |
| 4 | 0 | 0 |
| 5 | 0 | 0 |
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

> How much volume sits behind the touch: thin books mean large orders sweep several levels.

#### Quantity at the best price (share of time)

data: [`csv/liq/02-quantity-at-the-best-price-share-of-time.csv`](csv/liq/02-quantity-at-the-best-price-share-of-time.csv)

| qty | Bid | Ask |
| --- | --- | --- |
| 3 | 8.124e-05 | 4.838e-05 |
| 4 | 0.000386 | 0.0003772 |
| 5 | 0.0007218 | 0.0008399 |
| 6 | 0.001139 | 0.000573 |
| 7 | 0.001115 | 0.001437 |
| 8 | 0.001284 | 0.0006966 |
| 9 | 0.0009027 | 0.0009922 |
| 10 | 0.03496 | 0.033 |
| 11 | 0.02804 | 0.03228 |
| 12 | 0.03502 | 0.03138 |
| 13 | 0.0299 | 0.0339 |
| 14 | 0.03235 | 0.03307 |
| 15 | 0.0338 | 0.03021 |
| 16 | 0.03398 | 0.03752 |
| 17 | 0.0344 | 0.03255 |
| 18 | 0.0329 | 0.03004 |
| 19 | 0.0305 | 0.0344 |
| 20 | 0.02938 | 0.03411 |
| 21 | 0.03556 | 0.03207 |
| 22 | 0.03289 | 0.03739 |
| 23 | 0.03289 | 0.03011 |
| 24 | 0.03364 | 0.0317 |
| 25 | 0.03682 | 0.03328 |
| 26 | 0.03034 | 0.03154 |
| 27 | 0.0306 | 0.02835 |
| 28 | 0.03047 | 0.0297 |
| 29 | 0.03069 | 0.02788 |
| 30 | 0.03331 | 0.02991 |
| 31 | 0.03468 | 0.03036 |
| 32 | 0.02948 | 0.03396 |
| 33 | 0.03294 | 0.0324 |
| 34 | 0.03138 | 0.03052 |
| 35 | 0.03157 | 0.03156 |
| 36 | 0.02652 | 0.03512 |
| 37 | 0.03248 | 0.03183 |
| 38 | 0.03084 | 0.0311 |
| 39 | 0.03317 | 0.03134 |
| 40 | 0.02889 | 0.03245 |

y: share of time

> Small, frequently refreshed touch sizes mean our queue position matters less; large ones mean we wait behind them.

#### Queue imbalance  Qbid / (Qbid + Qask)

data: [`csv/liq/03-queue-imbalance-qbid-qbid-qask.csv`](csv/liq/03-queue-imbalance-qbid-qbid-qask.csv)

| imbalance | share of samples |
| --- | --- |
| 0.025 | 0 |
| 0.075 | 0.0002779 |
| 0.125 | 0.0002779 |
| 0.175 | 0.001667 |
| 0.225 | 0.02334 |
| 0.275 | 0.05696 |
| 0.325 | 0.06946 |
| 0.375 | 0.08725 |
| 0.425 | 0.1186 |
| 0.475 | 0.1273 |
| 0.525 | 0.155 |
| 0.575 | 0.1167 |
| 0.625 | 0.09114 |
| 0.675 | 0.07585 |
| 0.725 | 0.0464 |
| 0.775 | 0.02612 |
| 0.825 | 0.001945 |
| 0.875 | 0.001667 |
| 0.925 | 0 |
| 0.975 | 0 |

y: share of samples

> Mass near 0 and 1 means one side is often nearly empty: the strongest imbalance signals.

#### Order sizes at insert

data: [`csv/liq/04-order-sizes-at-insert.csv`](csv/liq/04-order-sizes-at-insert.csv)

| quantity | passive (26,578) | aggressive (604) |
| --- | --- | --- |
| 1 | 0 | 0 |
| 2 | 0 | 0 |
| 3 | 0 | 4 |
| 4 | 0 | 18 |
| 5 | 0 | 26 |
| 6 | 0 | 38 |
| 7 | 0 | 38 |
| 8 | 0 | 35 |
| 9 | 0 | 42 |
| 10 | 872 | 38 |
| 11 | 824 | 45 |
| 12 | 888 | 33 |
| 13 | 841 | 42 |
| 14 | 848 | 29 |
| 15 | 829 | 29 |
| 16 | 908 | 28 |
| 17 | 843 | 24 |
| 18 | 840 | 24 |
| 19 | 879 | 24 |
| 20 | 825 | 21 |
| 21 | 895 | 18 |
| 22 | 915 | 18 |
| 23 | 830 | 7 |
| 24 | 888 | 8 |
| 25 | 911 | 5 |
| 26 | 843 | 6 |
| 27 | 833 | 2 |
| 28 | 827 | 2 |
| 29 | 779 | 0 |
| 30 | 848 | 0 |
| 31 | 862 | 0 |
| 32 | 841 | 0 |
| 33 | 861 | 0 |
| 34 | 847 | 0 |
| 35 | 878 | 0 |
| 36 | 866 | 0 |
| 37 | 876 | 0 |
| 38 | 856 | 0 |
| 39 | 877 | 0 |
| 40 | 848 | 0 |

y: count

> Compare with our ORDER_SIZE: quoting much larger than the crowd sizes is unusual for this tape.

#### Lifetime of resting orders

data: [`csv/liq/05-lifetime-of-resting-orders.csv`](csv/liq/05-lifetime-of-resting-orders.csv)

| seconds (log) | cancelled |
| --- | --- |
| 2e-06 | 0 |
| 2.8e-06 | 0 |
| 3.92e-06 | 0 |
| 5.488e-06 | 0 |
| 7.683e-06 | 0 |
| 1.076e-05 | 0 |
| 1.506e-05 | 0 |
| 2.108e-05 | 2 |
| 2.952e-05 | 0 |
| 4.132e-05 | 0 |
| 5.785e-05 | 2 |
| 8.099e-05 | 0 |
| 0.0001134 | 4 |
| 0.0001587 | 8 |
| 0.0002222 | 10 |
| 0.0003111 | 10 |
| 0.0004356 | 16 |
| 0.0006098 | 18 |
| 0.0008537 | 36 |
| 0.001195 | 44 |
| 0.001673 | 62 |
| 0.002343 | 44 |
| 0.00328 | 122 |
| 0.004592 | 146 |
| 0.006428 | 152 |
| 0.008999 | 224 |
| 0.0126 | 294 |
| 0.01764 | 508 |
| 0.02469 | 476 |
| 0.03457 | 774 |
| 0.0484 | 1140 |
| 0.06776 | 1474 |
| 0.09487 | 1914 |
| 0.1328 | 2198 |
| 0.1859 | 3372 |
| 0.2603 | 4206 |
| 0.3644 | 3394 |
| 0.5102 | 1906 |
| 0.7143 | 1420 |
| 1 | 2598 |

y: count

> median 0.19s; 4% live under 10 ms (fleeting quotes).

#### Where passive orders are placed

data: [`csv/liq/06-where-passive-orders-are-placed.csv`](csv/liq/06-where-passive-orders-are-placed.csv)

_No data: Every passive order arrived with its own side of the book empty_

> 0 joins the best price, +1 improves it by a tick, −1 sits one tick behind.

## Trades & flow

#### Aggressive orders per second (30 s bins)

data: [`csv/flow/01-aggressive-orders-per-second-30-s-bins.csv`](csv/flow/01-aggressive-orders-per-second-30-s-bins.csv)

| time (s) | Buy | Sell |
| --- | --- | --- |
| 0 | 0.1 | 0.06667 |
| 30 | 0.06667 | 0.2333 |
| 60 | 0.03333 | 0.03333 |
| 90 | 0.03333 | 0.1333 |
| 120 | 0 | 0.03333 |
| 150 | 0.06667 | 0.1333 |
| 180 | 0.2 | 0.1667 |
| 210 | 0.06667 | 0.1 |
| 240 | 0.06667 | 0.06667 |
| 270 | 0.1667 | 0.06667 |
| 300 | 0.1 | 0.1 |
| 330 | 0.03333 | 0.1 |
| 360 | 0.06667 | 0.03333 |
| 390 | 0.03333 | 0.1333 |
| 420 | 0.06667 | 0.06667 |
| 450 | 0.1 | 0.03333 |
| 480 | 0.1 | 0.06667 |
| 510 | 0.03333 | 0.1 |
| 540 | 0 | 0.1333 |
| 570 | 0.06667 | 0.1333 |
| 600 | 0.1 | 0.06667 |
| 630 | 0.1 | 0 |
| 660 | 0.1667 | 0.06667 |
| 690 | 0.2 | 0.06667 |
| 720 | 0.1333 | 0.06667 |
| 750 | 0.06667 | 0.03333 |
| 780 | 0 | 0.1333 |
| 810 | 0.1 | 0 |
| 840 | 0.1 | 0 |
| 870 | 0.06667 | 0 |
| 900 | 0.03333 | 0.1 |
| 930 | 0.1 | 0.1667 |
| 960 | 0.06667 | 0.06667 |
| 990 | 0.1 | 0.1333 |
| 1020 | 0.06667 | 0.03333 |
| 1050 | 0 | 0.1333 |
| 1080 | 0.03333 | 0.1 |
| 1110 | 0.06667 | 0.03333 |
| 1140 | 0.1333 | 0.03333 |
| 1170 | 0.1 | 0.1333 |
| 1200 | 0.1667 | 0.1333 |
| 1230 | 0.1667 | 0.03333 |
| 1260 | 0.1 | 0.03333 |
| 1290 | 0.03333 | 0.03333 |
| 1320 | 0.03333 | 0.1 |
| 1350 | 0.1333 | 0.1333 |
| 1380 | 0.06667 | 0.1 |
| 1410 | 0.1 | 0.06667 |
| 1440 | 0.1333 | 0.1667 |
| 1470 | 0.1 | 0.03333 |
| 1500 | 0 | 0.1 |
| 1530 | 0.06667 | 0.03333 |
| 1560 | 0.03333 | 0.1333 |
| 1590 | 0.03333 | 0.03333 |
| 1620 | 0.06667 | 0.1 |
| 1650 | 0.1 | 0.1667 |
| 1680 | 0.03333 | 0.06667 |
| 1710 | 0.06667 | 0.1 |
| 1740 | 0.1333 | 0.03333 |
| 1770 | 0.1 | 0.1667 |
| 1800 | 0.1 | 0.06667 |
| 1830 | 0.1667 | 0 |
| 1860 | 0.03333 | 0.1667 |
| 1890 | 0.1 | 0 |
| 1920 | 0.2 | 0.1667 |
| 1950 | 0.03333 | 0.1333 |
| 1980 | 0.2 | 0.03333 |
| 2010 | 0.1333 | 0 |
| 2040 | 0 | 0.1333 |
| 2070 | 0.03333 | 0.03333 |
| 2100 | 0.06667 | 0.03333 |
| 2130 | 0.03333 | 0.03333 |
| 2160 | 0.03333 | 0.03333 |
| 2190 | 0.1333 | 0.03333 |
| 2220 | 0.03333 | 0.1 |
| 2250 | 0 | 0.1 |
| 2280 | 0 | 0.1667 |
| 2310 | 0.1333 | 0.06667 |
| 2340 | 0.1333 | 0.03333 |
| 2370 | 0.2 | 0.1333 |
| 2400 | 0.1333 | 0.1333 |
| 2430 | 0.2333 | 0.1667 |
| 2460 | 0.06667 | 0.03333 |
| 2490 | 0 | 0.1333 |
| 2520 | 0.1333 | 0.03333 |
| 2550 | 0.03333 | 0.03333 |
| 2580 | 0.06667 | 0.06667 |
| 2610 | 0.06667 | 0.03333 |
| 2640 | 0.1333 | 0.1 |
| 2670 | 0.06667 | 0.1667 |
| 2700 | 0.03333 | 0.1 |
| 2730 | 0.06667 | 0.1333 |
| 2760 | 0 | 0.1 |
| 2790 | 0.1667 | 0.06667 |
| 2820 | 0.03333 | 0.1 |
| 2850 | 0.03333 | 0.03333 |
| 2880 | 0.06667 | 0.1 |
| 2910 | 0.1 | 0.1 |
| 2940 | 0.3 | 0.1333 |
| 2970 | 0.03333 | 0.1 |
| 3000 | 0.1333 | 0.06667 |
| 3030 | 0.1 | 0.03333 |
| 3060 | 0.06667 | 0.1667 |
| 3090 | 0.1333 | 0.1333 |
| 3120 | 0.1333 | 0.06667 |
| 3150 | 0.1 | 0.06667 |
| 3180 | 0.2333 | 0.03333 |
| 3210 | 0.1667 | 0 |
| 3240 | 0.03333 | 0.03333 |
| 3270 | 0.03333 | 0.1 |
| 3300 | 0.03333 | 0.06667 |
| 3330 | 0.1333 | 0.1667 |
| 3360 | 0.1 | 0.06667 |
| 3390 | 0 | 0.03333 |
| 3420 | 0.1333 | 0.1667 |
| 3450 | 0.03333 | 0.03333 |
| 3480 | 0.06667 | 0.1 |
| 3510 | 0.1 | 0.03333 |
| 3540 | 0.06667 | 0.03333 |
| 3570 | 0.1 | 0.1 |

y: orders / s

> 604 aggressive orders, 7,706 lots. Fills for a passive quote only come from these.

#### Time between aggressive orders

data: [`csv/flow/02-time-between-aggressive-orders.csv`](csv/flow/02-time-between-aggressive-orders.csv)

| seconds | observed | Poisson, same rate |
| --- | --- | --- |
| 0.246 | 36 | 47.04 |
| 0.7291 | 45 | 43.37 |
| 1.212 | 43 | 39.98 |
| 1.695 | 33 | 36.86 |
| 2.178 | 35 | 33.98 |
| 2.661 | 32 | 31.33 |
| 3.144 | 34 | 28.88 |
| 3.627 | 21 | 26.63 |
| 4.11 | 27 | 24.55 |
| 4.593 | 24 | 22.63 |
| 5.076 | 27 | 20.87 |
| 5.559 | 18 | 19.24 |
| 6.042 | 16 | 17.74 |
| 6.525 | 22 | 16.35 |
| 7.008 | 15 | 15.07 |
| 7.491 | 20 | 13.9 |
| 7.974 | 15 | 12.81 |
| 8.457 | 12 | 11.81 |
| 8.94 | 8 | 10.89 |
| 9.423 | 6 | 10.04 |
| 9.906 | 7 | 9.256 |
| 10.39 | 8 | 8.533 |
| 10.87 | 8 | 7.867 |
| 11.36 | 4 | 7.253 |
| 11.84 | 8 | 6.687 |
| 12.32 | 7 | 6.165 |
| 12.8 | 3 | 5.683 |
| 13.29 | 6 | 5.24 |
| 13.77 | 1 | 4.831 |
| 14.25 | 7 | 4.454 |
| 14.74 | 4 | 4.106 |
| 15.22 | 3 | 3.785 |
| 15.7 | 2 | 3.49 |
| 16.19 | 4 | 3.217 |
| 16.67 | 3 | 2.966 |
| 17.15 | 3 | 2.735 |
| 17.63 | 3 | 2.521 |
| 18.12 | 3 | 2.324 |
| 18.6 | 5 | 2.143 |
| 19.08 | 1 | 1.976 |
| 19.57 | 2 | 1.821 |
| 20.05 | 2 | 1.679 |
| 20.53 | 1 | 1.548 |
| 21.02 | 0 | 1.427 |
| 21.5 | 1 | 1.316 |
| 21.98 | 3 | 1.213 |
| 22.46 | 1 | 1.118 |
| 22.95 | 2 | 1.031 |
| 23.43 | 2 | 0.9505 |
| 23.91 | 3 | 0.8763 |

y: count (log)

> Coefficient of variation 0.95 (1 for Poisson; above 1 means clustered arrivals, below 1 means regular ones).

#### Aggressive order size

data: [`csv/flow/03-aggressive-order-size.csv`](csv/flow/03-aggressive-order-size.csv)

| quantity | Buy | Sell |
| --- | --- | --- |
| 1 | 0 | 0 |
| 2 | 0 | 0 |
| 3 | 2 | 2 |
| 4 | 8 | 10 |
| 5 | 14 | 12 |
| 6 | 20 | 18 |
| 7 | 20 | 18 |
| 8 | 17 | 18 |
| 9 | 25 | 17 |
| 10 | 18 | 20 |
| 11 | 25 | 20 |
| 12 | 14 | 19 |
| 13 | 27 | 15 |
| 14 | 12 | 17 |
| 15 | 20 | 9 |
| 16 | 10 | 18 |
| 17 | 11 | 13 |
| 18 | 8 | 16 |
| 19 | 14 | 10 |
| 20 | 8 | 13 |
| 21 | 11 | 7 |
| 22 | 7 | 11 |
| 23 | 5 | 2 |
| 24 | 4 | 4 |
| 25 | 2 | 3 |
| 26 | 3 | 3 |

y: count

> Large aggressive orders are the ones that sweep through our quote and leave us adversely selected.

#### Levels swept past the touch

data: [`csv/flow/04-levels-swept-past-the-touch.csv`](csv/flow/04-levels-swept-past-the-touch.csv)

| ticks past the best opposite price | observed |
| --- | --- |
| 0 | 1 |

y: share of orders

> Mean depth 0 → k = ln(1 + 1/mean) = inf: the estimator avellaneda_stoikov.py uses online.

#### Fill intensity λ(δ) for a quote δ ticks from the mid

data: [`csv/flow/05-fill-intensity-for-a-quote-ticks-from-the-mid.csv`](csv/flow/05-fill-intensity-for-a-quote-ticks-from-the-mid.csv)

| δ (ticks from mid) | Ask side hit by buys | Bid side hit by sells |
| --- | --- | --- |
| 0 | 0.0853 | 0.08252 |
| 0.5 | 0.0853 | 0.08252 |
| 1 | 0.0853 | 0.08252 |

y: hits per second (log)

> The Avellaneda-Stoikov arrival model. A straight line on this log axis means the exponential model fits; a kink means it does not.

#### Cumulative signed aggressive volume / Mid

data: not saved (2,605 points; `--export --all-csv` writes it)

| series | points | x range | y min | mean | median | max | last |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Net buy volume | 604 | 6.863..3590 | -204 | -47.78 | -46 | 130 | 30 |
| Mid | 2,001 | 0..3599 | 9869 | 9962 | 9961 | 10073 | 10037 |

> If price follows net flow closely, aggressive orders carry information.

#### Autocorrelation of aggressive order signs

data: [`csv/flow/07-autocorrelation-of-aggressive-order-signs.csv`](csv/flow/07-autocorrelation-of-aggressive-order-signs.csv)

| lag (orders) | correlation |
| --- | --- |
| 1 | -0.01523 |
| 2 | 0.06917 |
| 3 | 0.01457 |
| 4 | -0.02015 |
| 5 | -0.03837 |
| 6 | -0.003589 |
| 7 | -0.02181 |
| 8 | -0.03014 |
| 9 | -0.0185 |
| 10 | -0.03335 |
| 11 | 0.01468 |
| 12 | 0.02622 |
| 13 | -0.05824 |
| 14 | 0.07916 |
| 15 | -0.08154 |
| 16 | -0.02689 |
| 17 | -0.01198 |
| 18 | 0.04603 |
| 19 | 0.01782 |
| 20 | -0.0003929 |
| 21 | 0.02114 |
| 22 | 0.06916 |
| 23 | 0.02108 |
| 24 | 0.02274 |
| 25 | -0.06173 |
| 26 | -0.03694 |
| 27 | -0.01541 |
| 28 | -0.06349 |
| 29 | 0.02765 |
| 30 | -0.08331 |

y: correlation

Marked: shaded band y ∈ [-0.07975, 0.07975].

> Positive: buys follow buys (split metaorders). Their signs then predict the next order.

## Predictability

#### Future mid move by queue imbalance

data: [`csv/predict/01-future-mid-move-by-queue-imbalance.csv`](csv/predict/01-future-mid-move-by-queue-imbalance.csv)

| imbalance  Qbid / (Qbid + Qask) | 0.1 s | 1 s | 5 s | 30 s |
| --- | --- | --- | --- | --- |
| 0.1621 | -0.2278 ± 0.2676 | -3.57 ± 1.098 | -10.57 ± 3.622 | -10.18 ± 6.862 |
| 0.2623 | -0.01676 ± 0.03738 | -0.1819 ± 0.186 | -0.9158 ± 0.6193 | 0.5021 ± 1.392 |
| 0.353 | -0.0003563 ± 0.02686 | -0.05219 ± 0.1337 | -0.2166 ± 0.4497 | -0.254 ± 1.032 |
| 0.4493 | -0.007111 ± 0.02101 | -0.08421 ± 0.1042 | -0.2804 ± 0.3556 | 0.4765 ± 0.7755 |
| 0.5444 | 0.002698 ± 0.01979 | 0.08924 ± 0.09696 | 0.3369 ± 0.333 | 0.3602 ± 0.7307 |
| 0.6477 | 0.007188 ± 0.02505 | 0.02424 ± 0.1225 | 0.3838 ± 0.4212 | 0.7325 ± 0.9604 |
| 0.7378 | 0.03609 ± 0.04199 | 0.3945 ± 0.2106 | 1.173 ± 0.6704 | 0.5369 ± 1.512 |
| 0.8217 | 0.05882 ± 0.2074 | 1.581 ± 0.8482 | 3.713 ± 3.259 | 1.14 ± 6.161 |

y: mean Δmid (ticks)  (± is the 95% interval)

Marked: horizontal line y = 0.

> Buckets with under 30 samples hidden. A monotone S-curve means imbalance predicts the next move: use the microprice instead of the mid, or lean quotes with it.

#### Is the imbalance edge still there 5 ms later?

data: [`csv/predict/02-is-the-imbalance-edge-still-there-5-ms-later.csv`](csv/predict/02-is-the-imbalance-edge-still-there-5-ms-later.csv)

| imbalance | measured from now | measured from now + 5 ms |
| --- | --- | --- |
| 0.1621 | -0.2278 | -0.2405 |
| 0.2623 | -0.01676 | -0.01462 |
| 0.353 | -0.0003563 | -0.001781 |
| 0.4493 | -0.007111 | -0.006886 |
| 0.5444 | 0.002698 | 0.003897 |
| 0.6477 | 0.007188 | 0.005517 |
| 0.7378 | 0.03609 | 0.03528 |
| 0.8217 | 0.05882 | 0.06618 |

y: mean Δmid over the next 100 ms (ticks)

Marked: horizontal line y = 0.

> Our orders land 5 ms after we decide. If the two curves match, the signal is still worth trading on.

#### After an aggressive order: mid move in its direction

data: [`csv/predict/03-after-an-aggressive-order-mid-move-in-its-direction.csv`](csv/predict/03-after-an-aggressive-order-mid-move-in-its-direction.csv)

| seconds after (log) | price impact (vs pre-trade mid) | taker PnL per lot (vs fill) = −maker PnL |
| --- | --- | --- |
| 0.005 | -0.001656 ± 0.01422 | -1.002 |
| 0.05 | -0.009934 ± 0.05243 | -1.01 |
| 0.1 | -0.008278 ± 0.07498 | -1.008 |
| 0.25 | 0.293 ± 0.148 | -0.707 |
| 0.5 | 1.293 ± 0.2425 | 0.293 |
| 1 | 3.056 ± 0.4205 | 2.056 |
| 2 | 6.652 ± 0.7444 | 5.652 |
| 5 | 10.05 ± 1.414 | 9.053 |
| 10 | 9.149 ± 1.984 | 8.149 |
| 20 | 7.265 ± 2.546 | 6.265 |
| 30 | 6.513 ± 2.919 | 5.513 |
| 60 | 6.152 ± 3.81 | 5.152 |

y: ticks  (± is the 95% interval)

Marked: horizontal line y = 0; vertical line x = 0.005.

> Positive orange line: passive fills lose money at that horizon (adverse selection). The quoted half-spread must cover it.

#### Order flow imbalance → mid move (R²)

data: [`csv/predict/04-order-flow-imbalance-mid-move-r.csv`](csv/predict/04-order-flow-imbalance-mid-move-r.csv)

| bin size (s) | same bin (impact) | next bin (prediction) |
| --- | --- | --- |
| 0.1 | 0.8997 | 0.0002825 |
| 0.5 | 0.9451 | 0.343 |
| 1 | 0.9681 | 0.3407 |
| 5 | 0.9899 | 0.003623 |
| 30 | 0.9913 | 0.03271 |

y: R²

> Same-bin R² is price impact (descriptive). Next-bin R² is real predictability.

#### OFI vs Δmid, 1 s bins

data: not saved (3,599 points; `--export --all-csv` writes it)

| OFI (lots) | fit |
| --- | --- |
| -580 | -11.27 |
| 734 | 14.27 |

y: Δmid (ticks)

| series | points | OFI (lots) range | Δmid (ticks) min | mean | median | max | last |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scatter 1 | 3,597 | -580..734 | -12 | 0.01029 | 0 | 14 | -1 |

> Change the bin size with the sampling selector at the top.

### OFI regression  Δmid = β·OFI

| bin | β (ticks/lot) | R² same | β next | R² next | n |
| --- | --- | --- | --- | --- | --- |
| 0.1 s | 0.0182 | 0.9 | 0.000323 | 0.000282 | 35,989 |
| 0.5 s | 0.019 | 0.945 | 0.0115 | 0.343 | 7,197 |
| 1 s | 0.0194 | 0.968 | 0.0115 | 0.341 | 3,598 |
| 5 s | 0.0197 | 0.99 | 0.00119 | 0.00362 | 718 |
| 30 s | 0.0196 | 0.991 | -0.00356 | 0.0327 | 118 |
