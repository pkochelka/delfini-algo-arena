# Data explorer: sample_v2

Data explorer export of `data/sample_v2.csv`, window 0.0–5,749.9 s (95.8 min), sampling selector 1 s (charts that use it: return histogram and ACF, rolling σ, variance ratio, OFI scatter).
Generated 2026-11-07 12:49 by `uv run data-explorer --data … --export` (see data_explorer/export.py). Numbers are in ticks and seconds. The data behind each chart is in csv/, machine-readable headline stats in summary.json, interactive charts in report.html.

Contents: [Overview](#overview) · [Price & spread](#price--spread) · [Volatility](#volatility) · [Liquidity & book](#liquidity--book) · [Trades & flow](#trades--flow) · [Predictability](#predictability)

## Overview

### Tape fingerprint

| Statistic | Value | Detail |
| --- | --- | --- |
| Window | 95.8 min | 2,000,000 events |
| Events / s | 348 | cancel/insert 0.87 |
| Aggressive orders / min | 352 | 131,778 fills |
| Traded volume | 3,177,589 | 50% buyer-initiated |
| Mid | 10,000 → 10,003 | range 9,966–10,022 |
| Spread (time-weighted) | 1.59 | 41% of time at 1 tick |
| One-sided book | 0.0% | of time a side is empty |
| σ at 100 ms | 0.554 | ticks / √s |
| σ at 1 s | 0.497 | 10 s: 0.517 |
| Touch size | 818 | mean qty at best, per side |
| Order / fill size | 32.5 / 94.2 | mean insert / aggressive order |
| Passive fill rate | 10.9% | resting orders that got any fill |

### Fitted parameters (selected window)

```python
# sample_v2, window 0-5750 s  (python -m analysis.fit)
SIGMA_INIT = 0.5539   # ticks/sqrt(s) at 100 ms; 1 s: 0.4967
K_INIT = 2.934       # ln(1 + 1/mean sweep depth), depth 0.0562
# intensity fit: A = 259.4 /s, k = 4.94 /tick (4 points)
# OFI beta at 1 s = 0.0003069 ticks/lot (R² 0.571), next-bin R² 0.054
# taker markout vs fill: 1 s -0.805, 10 s -0.804 ticks
IMBALANCE_DMID_1S = {"0.05": -0.2788, "0.15": -0.2222, "0.25": -0.1752, "0.35": -0.1324, "0.45": -0.0392, "0.55": 0.0491, "0.65": 0.1249, "0.75": 0.1811, "0.85": 0.1883, "0.95": 0.3004}
```

> Copy into a strategy: submissions cannot read files.

### Activity per 60 s

#### Events / Aggressive orders / Spread (ticks) / σ 1 s (ticks/√s)

data: [`csv/overview/01-events-aggressive-orders-spread-ticks-1-s-ticks-s.csv`](csv/overview/01-events-aggressive-orders-spread-ticks-1-s-ticks-s.csv)

| x | Events | Aggressive orders | Spread (ticks) | σ 1 s (ticks/√s) |
| --- | --- | --- | --- | --- |
| 0 | 20709 | 389 | 1.729 | 0.5043 |
| 60 | 20659 | 355 | 1.567 | 0.4518 |
| 120 | 21027 | 360 | 1.683 | 0.4958 |
| 180 | 20689 | 311 | 1.6 | 0.4282 |
| 240 | 20648 | 357 | 1.6 | 0.4787 |
| 300 | 20956 | 347 | 1.567 | 0.483 |
| 360 | 20844 | 342 | 1.567 | 0.5439 |
| 420 | 21315 | 348 | 1.483 | 0.4564 |
| 480 | 20958 | 344 | 1.7 | 0.4699 |
| 540 | 20493 | 365 | 1.6 | 0.483 |
| 600 | 20915 | 370 | 1.617 | 0.5774 |
| 660 | 20929 | 317 | 1.5 | 0.4378 |
| 720 | 20590 | 358 | 1.667 | 0.5 |
| 780 | 20936 | 351 | 1.65 | 0.4472 |
| 840 | 20738 | 328 | 1.417 | 0.5 |
| 900 | 20590 | 378 | 1.583 | 0.5041 |
| 960 | 20773 | 363 | 1.583 | 0.5041 |
| 1020 | 20661 | 348 | 1.6 | 0.483 |
| 1080 | 20858 | 406 | 1.583 | 0.4655 |
| 1140 | 20963 | 344 | 1.5 | 0.4916 |
| 1200 | 21128 | 349 | 1.567 | 0.5041 |
| 1260 | 21010 | 340 | 1.683 | 0.5439 |
| 1320 | 21239 | 352 | 1.6 | 0.4916 |
| 1380 | 21084 | 341 | 1.517 | 0.5164 |
| 1440 | 20864 | 337 | 1.517 | 0.5401 |
| 1500 | 20893 | 368 | 1.483 | 0.5477 |
| 1560 | 21269 | 332 | 1.55 | 0.4699 |
| 1620 | 20763 | 346 | 1.567 | 0.5439 |
| 1680 | 21267 | 348 | 1.633 | 0.4958 |
| 1740 | 20732 | 354 | 1.633 | 0.4564 |
| 1800 | 20977 | 344 | 1.683 | 0.4425 |
| 1860 | 20731 | 356 | 1.667 | 0.4183 |
| 1920 | 21110 | 343 | 1.683 | 0.5123 |
| 1980 | 20894 | 347 | 1.55 | 0.5244 |
| 2040 | 20312 | 365 | 1.5 | 0.5439 |
| 2100 | 20649 | 348 | 1.7 | 0.5 |
| 2160 | 21316 | 339 | 1.567 | 0.5041 |
| 2220 | 20546 | 342 | 1.65 | 0.5041 |
| 2280 | 20573 | 348 | 1.6 | 0.4916 |
| 2340 | 21187 | 349 | 1.533 | 0.5 |
| 2400 | 20595 | 341 | 1.6 | 0.4699 |
| 2460 | 20873 | 376 | 1.683 | 0.5041 |
| 2520 | 21057 | 319 | 1.6 | 0.5477 |
| 2580 | 20863 | 341 | 1.65 | 0.559 |
| 2640 | 20625 | 351 | 1.55 | 0.4916 |
| 2700 | 20930 | 338 | 1.517 | 0.4873 |
| 2760 | 20645 | 360 | 1.65 | 0.4031 |
| 2820 | 21288 | 320 | 1.617 | 0.4425 |
| 2880 | 21393 | 322 | 1.5 | 0.5164 |
| 2940 | 20795 | 367 | 1.533 | 0.5164 |
| 3000 | 20825 | 382 | 1.55 | 0.5881 |
| 3060 | 20843 | 376 | 1.533 | 0.5401 |
| 3120 | 20911 | 324 | 1.533 | 0.4378 |
| 3180 | 20488 | 331 | 1.667 | 0.5553 |
| 3240 | 20967 | 339 | 1.65 | 0.3926 |
| 3300 | 21073 | 366 | 1.7 | 0.5323 |
| 3360 | 20756 | 363 | 1.567 | 0.6455 |
| 3420 | 21019 | 378 | 1.633 | 0.4031 |
| 3480 | 20996 | 396 | 1.567 | 0.4133 |
| 3540 | 21148 | 363 | 1.533 | 0.4564 |
| 3600 | 21105 | 339 | 1.55 | 0.5083 |
| 3660 | 20759 | 379 | 1.617 | 0.5123 |
| 3720 | 20641 | 345 | 1.583 | 0.4873 |
| 3780 | 21135 | 357 | 1.567 | 0.5553 |
| 3840 | 20871 | 372 | 1.567 | 0.4873 |
| 3900 | 20735 | 348 | 1.567 | 0.5284 |
| 3960 | 20826 | 366 | 1.7 | 0.5362 |
| 4020 | 20890 | 334 | 1.55 | 0.4564 |
| 4080 | 20585 | 342 | 1.6 | 0.3764 |
| 4140 | 20783 | 343 | 1.55 | 0.5439 |
| 4200 | 20885 | 387 | 1.6 | 0.5123 |
| 4260 | 20602 | 351 | 1.717 | 0.4743 |
| 4320 | 20727 | 342 | 1.617 | 0.4743 |
| 4380 | 20419 | 358 | 1.583 | 0.5 |
| 4440 | 20789 | 381 | 1.55 | 0.4787 |
| 4500 | 20712 | 357 | 1.667 | 0.5083 |
| 4560 | 20466 | 359 | 1.633 | 0.5083 |
| 4620 | 21316 | 324 | 1.567 | 0.3979 |
| 4680 | 20615 | 353 | 1.583 | 0.5 |
| 4740 | 21229 | 354 | 1.467 | 0.4699 |
| 4800 | 20905 | 338 | 1.483 | 0.4233 |
| 4860 | 20761 | 376 | 1.567 | 0.4699 |
| 4920 | 20976 | 320 | 1.417 | 0.5284 |
| 4980 | 20876 | 336 | 1.467 | 0.5439 |
| 5040 | 21061 | 342 | 1.617 | 0.5164 |
| 5100 | 21125 | 333 | 1.517 | 0.4699 |
| 5160 | 20932 | 376 | 1.567 | 0.5 |
| 5220 | 20908 | 350 | 1.7 | 0.5041 |
| 5280 | 21168 | 345 | 1.5 | 0.4743 |
| 5340 | 21042 | 313 | 1.617 | 0.5083 |
| 5400 | 20971 | 360 | 1.6 | 0.5737 |
| 5460 | 20468 | 391 | 1.583 | 0.5041 |
| 5520 | 20860 | 353 | 1.55 | 0.5204 |
| 5580 | 20678 | 328 | 1.6 | 0.4958 |
| 5640 | 21312 | 374 | 1.7 | 0.4787 |
| 5700 | 16982 | 310 | 1.6 | 0.5 |

y: y

> Regime changes show up here first: bursts of activity, spread widening, volatility clustering.

## Price & spread

#### Best bid / ask, microprice and aggressive orders

data: not saved (45,000 points; `--export --all-csv` writes it)

| series | points | time (s) range | price (ticks) min | mean | median | max | last |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Best bid | 5,000 | 0..5750 | 9966 | 9992 | 9993 | 10021 | 10003 |
| Best ask | 5,000 | 0..5750 | 9967 | 9994 | 9995 | 10023 | 10004 |
| Microprice | 5,000 | 0..5750 | 9966 | 9993 | 9994 | 1.002e+04 | 1e+04 |
| Aggressive buy | 14,998 | 0.1426..5750 | 9967 | 9994 | 9995 | 10024 | 10004 |
| Aggressive sell | 15,002 | 0.1828..5750 | 9965 | 9992 | 9993 | 10021 | 10003 |

> Zoom the timeline above to see every quote change. Marker size grows with order quantity; it sits at the worst price the order reached.

#### Spread over time

data: not saved (5,000 points; `--export --all-csv` writes it)

| series | points | time (s) range | ticks min | mean | median | max | last |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Spread | 5,000 | 0..5750 | 1 | 1.587 | 2 | 3 | 1 |

#### Spread distribution (share of time)

data: [`csv/price/03-spread-distribution-share-of-time.csv`](csv/price/03-spread-distribution-share-of-time.csv)

| spread (ticks) | share of time |
| --- | --- |
| 1 | 0.4142 |
| 2 | 0.5839 |
| 3 | 0.001932 |
| 4 | 3.408e-06 |

y: share of time

> A market maker quoting inside the spread needs it wider than 1 tick most of the time.

#### Resting depth by price (bids +, asks -)

data: not saved (3,523 points; `--export --all-csv` writes it)

Heatmap, 3,523 non-zero cells (missing cells in the CSV are 0); x (time (s)) 0.07256..5748, y (price (ticks)) 9964..10026; z mean 3.108, max 2496.

> Snapshots every 100 ms. Look for walls (persistent large levels) and how deep the book is.

## Volatility

#### Rolling σ over 100 s

data: not saved (5,750 points; `--export --all-csv` writes it)

| series | points | time (s) range | ticks / √s min | mean | median | max | last |
| --- | --- | --- | --- | --- | --- | --- | --- |
| σ | 5,750 | 0..5749 | 0.3905 | 0.4953 | 0.4975 | 0.6185 | 0.505 |

> Clustering (calm and wild stretches) means a fixed σ under- and over-quotes in turn: estimate it online.

#### Volatility signature

data: [`csv/vol/02-volatility-signature.csv`](csv/vol/02-volatility-signature.csv)

| sampling interval (s) | σ (ticks / √s) |
| --- | --- |
| 0.05 | 0.5799 |
| 0.1 | 0.5539 |
| 0.2 | 0.5274 |
| 0.5 | 0.5028 |
| 1 | 0.4967 |
| 2 | 0.5065 |
| 5 | 0.5065 |
| 10 | 0.5175 |
| 30 | 0.5293 |
| 60 | 0.5564 |

y: σ (ticks / √s)

> Flat: random walk. Rising with the interval: momentum. Falling: bid-ask bounce or mean reversion. Pick σ at the horizon you hold inventory over.

#### 1 s mid returns

data: [`csv/vol/03-1-s-mid-returns.csv`](csv/vol/03-1-s-mid-returns.csv)

| Δmid (ticks) | observed |
| --- | --- |
| -1 | 0.0515 |
| -0.5 | 0.2364 |
| 0 | 0.4182 |
| 0.5 | 0.2286 |
| 1 | 0.0555 |
| 1.5 | 0.004697 |
| 2 | 0.0003479 |

y: share of intervals

| series | points | Δmid (ticks) range | share of intervals min | mean | median | max | last |
| --- | --- | --- | --- | --- | --- | --- | --- |
| normal, same σ | 200 | -1..2 | 0.0001215 | 0.1623 | 0.127 | 0.4016 | 0.0001215 |

> excess kurtosis 0.20, skew 0.02, 42% of intervals unchanged, n = 5,748

#### Autocorrelation of 1 s returns

data: [`csv/vol/04-autocorrelation-of-1-s-returns.csv`](csv/vol/04-autocorrelation-of-1-s-returns.csv)

| lag (intervals) | correlation |
| --- | --- |
| 1 | 0.06171 |
| 2 | -0.01217 |
| 3 | -0.02609 |
| 4 | 0.01957 |
| 5 | 0.008286 |
| 6 | 0.02398 |
| 7 | 0.00141 |
| 8 | 0.01498 |
| 9 | -0.01411 |
| 10 | -0.01358 |
| 11 | -0.005115 |
| 12 | -0.0134 |
| 13 | 0.009694 |
| 14 | 0.01022 |
| 15 | 0.006344 |
| 16 | 0.02715 |
| 17 | 0.01111 |
| 18 | 0.004582 |
| 19 | -0.0164 |
| 20 | 0.0008799 |
| 21 | 0.01675 |
| 22 | -0.001589 |
| 23 | -0.00247 |
| 24 | 0.005816 |
| 25 | -0.008641 |
| 26 | -0.001942 |
| 27 | -0.01129 |
| 28 | -0.02081 |
| 29 | 0.01322 |
| 30 | 0.01639 |

y: correlation

Marked: shaded band y ∈ [-0.02585, 0.02585].

> Bars outside the gray band are significant. Positive at short lags: trends (take); negative: reversion (make).

#### Variance ratio

data: [`csv/vol/05-variance-ratio.csv`](csv/vol/05-variance-ratio.csv)

| horizon (s) | VR(q) |
| --- | --- |
| 2 | 1.062 |
| 5 | 1.071 |
| 10 | 1.111 |
| 30 | 1.159 |
| 60 | 1.186 |

y: VR(q)

Marked: horizontal line y = 1.

> Above 1: moves persist over that horizon. Below 1: they revert.

#### Autocorrelation of 10 ms returns by lag

data: [`csv/vol/06-autocorrelation-of-10-ms-returns-by-lag.csv`](csv/vol/06-autocorrelation-of-10-ms-returns-by-lag.csv)

| lag (s, log) | ACF |
| --- | --- |
| 0.01 | -0.04236 |
| 0.02 | -0.02903 |
| 0.03 | -0.02462 |
| 0.04 | -0.01355 |
| 0.05 | -0.01243 |
| 0.06 | -0.01028 |
| 0.07 | -0.009488 |
| 0.09 | -0.002711 |
| 0.12 | -0.002937 |
| 0.15 | -0.001694 |
| 0.18 | -0.00192 |
| 0.23 | -0.002598 |
| 0.28 | 0.0006777 |
| 0.35 | 0.0002259 |
| 0.44 | -0.002259 |
| 0.55 | 0.002033 |
| 0.69 | -0.001017 |
| 0.87 | 0.003276 |
| 1.08 | 0.002937 |
| 1.35 | -0.00113 |
| 1.69 | -0.001581 |
| 2.11 | -0.001581 |
| 2.64 | 0.0007907 |
| 3.3 | 0.0002259 |
| 4.13 | -0.001581 |
| 5.16 | -0.0004518 |
| 6.45 | -0.0003389 |
| 8.06 | -5.514e-09 |
| 10.07 | 0.0003389 |
| 12.59 | 0.0009036 |
| 15.74 | -0.00113 |
| 19.67 | 0.0006777 |
| 24.58 | 0.001243 |
| 30.73 | 0.0001129 |
| 38.41 | -0.00113 |
| 48 | 0.0003388 |
| 60 | -0.0006778 |

y: correlation

Marked: shaded band y ∈ [-0.002585, 0.002585]; horizontal line y = 0; vertical line x = 0.01.

> One axis from 10 ms to 60 s, so it does not depend on the sampling selector. Negative at the shortest lags: bid-ask bounce or quote flicker. Positive at second-scale lags: momentum. Dashed line: our 5 ms latency. Values look small because most 10 ms returns are zero: compare them with the gray band, not with 1.

#### Autocorrelation of 100 ms returns, every lag to 10 s

data: [`csv/vol/07-autocorrelation-of-100-ms-returns-every-lag-to-10-s.csv`](csv/vol/07-autocorrelation-of-100-ms-returns-every-lag-to-10-s.csv)

| lag (s) | correlation |
| --- | --- |
| 0.1 | -0.08998 |
| 0.2 | -0.0214 |
| 0.3 | -0.008927 |
| 0.4 | -0.002976 |
| 0.5 | 0.00581 |
| 0.6 | 0.01119 |
| 0.7 | 0.005951 |
| 0.8 | 0.01516 |
| 0.9 | 0.007368 |
| 1 | 0.01417 |
| 1.1 | 0.007227 |
| 1.2 | 0.003401 |
| 1.3 | 0.00496 |
| 1.4 | -0.000992 |
| 1.5 | 0.002976 |
| 1.6 | -0.007369 |
| 1.7 | -0.003259 |
| 1.8 | -0.005668 |
| 1.9 | -0.0004252 |
| 2 | -0.0004252 |
| 2.1 | 0.006518 |
| 2.2 | -0.0004252 |
| 2.3 | 0.003826 |
| 2.4 | -0.001701 |
| 2.5 | 0.006377 |
| 2.6 | -0.005101 |
| 2.7 | -0.000992 |
| 2.8 | -0.008644 |
| 2.9 | 0.001984 |
| 3 | -0.005952 |
| 3.1 | 0.0002833 |
| 3.2 | -0.005526 |
| 3.3 | 0.000425 |
| 3.4 | -0.0002835 |
| 3.5 | 0.002125 |
| 3.6 | -0.008219 |
| 3.7 | -0.002551 |
| 3.8 | 0.003543 |
| 3.9 | 0.002551 |
| 4 | 0.008502 |
| 4.1 | -0.001701 |
| 4.2 | -0.009211 |
| 4.3 | -0.001701 |
| 4.4 | 0.007085 |
| 4.5 | 0.005101 |
| 4.6 | -0.0005669 |
| 4.7 | -0.001417 |
| 4.8 | -0.001134 |
| 4.9 | 0.004393 |
| 5 | 0.002834 |
| 5.1 | 0.002976 |
| 5.2 | 0.004818 |
| 5.3 | -0.0005669 |
| 5.4 | 0.001417 |
| 5.5 | 0.0002833 |
| 5.6 | -0.0002835 |
| 5.7 | -0.000992 |
| 5.8 | 0.001134 |
| 5.9 | -5.188e-08 |
| 6 | 0.002267 |
| 6.1 | 0.01233 |
| 6.2 | 0.0017 |
| 6.3 | -0.006377 |
| 6.4 | 0.004393 |
| 6.5 | 0.0007084 |
| 6.6 | 0.001984 |
| 6.7 | -0.001984 |
| 6.8 | 0.00581 |
| 6.9 | -0.003259 |
| 7 | 0.003117 |
| 7.1 | 0.001842 |
| 7.2 | -0.000992 |
| 7.3 | 0.0001416 |
| 7.4 | -0.005668 |
| 7.5 | 0.001275 |
| 7.6 | -0.002834 |
| 7.7 | -0.001134 |
| 7.8 | 0.00581 |
| 7.9 | -0.001701 |
| 8 | 0.0002833 |
| 8.1 | 0.0002833 |
| 8.2 | -0.001275 |
| 8.3 | 0.002125 |
| 8.4 | 0.001984 |
| 8.5 | 0.002692 |
| 8.6 | -0.003968 |
| 8.7 | -0.001417 |
| 8.8 | -0.001842 |
| 8.9 | 0.005526 |
| 9 | -0.003684 |
| 9.1 | -0.00666 |
| 9.2 | -1.727e-07 |
| 9.3 | 0.0007083 |
| 9.4 | 0.002834 |
| 9.5 | 0.0005666 |
| 9.6 | -0.005668 |
| 9.7 | 0.002267 |
| 9.8 | -0.002976 |
| 9.9 | 0.0004249 |
| 10 | 0.003826 |

y: correlation

Marked: shaded band y ∈ [-0.008174, 0.008174].

> 57,497 returns. Between the two charts above: the lag at which bars turn from blue to orange is how long a move keeps going before it starts to revert, which is a natural holding time or signal horizon.

#### Autocorrelation of consecutive mid changes (tick by tick)

data: [`csv/vol/08-autocorrelation-of-consecutive-mid-changes-tick-by-tick.csv`](csv/vol/08-autocorrelation-of-consecutive-mid-changes-tick-by-tick.csv)

| lag (mid changes) | correlation |
| --- | --- |
| 1 | -0.2495 |
| 2 | 0.0974 |
| 3 | -0.03123 |
| 4 | 0.01376 |
| 5 | -0.004341 |
| 6 | 0.004869 |
| 7 | -0.009211 |
| 8 | 0.005928 |
| 9 | 0.01535 |
| 10 | -0.009317 |
| 11 | 0.007304 |
| 12 | -0.00847 |
| 13 | 0.01418 |
| 14 | -0.01588 |
| 15 | 0.0007401 |
| 16 | 0.01228 |
| 17 | -0.01535 |
| 18 | 0.01842 |
| 19 | -0.000107 |
| 20 | 0.005715 |
| 21 | 0.003492 |
| 22 | 0.00254 |
| 23 | 0.002328 |
| 24 | 0.003598 |
| 25 | 0.003281 |
| 26 | -0.006353 |
| 27 | -0.00233 |
| 28 | 0.00254 |
| 29 | 0.001164 |
| 30 | -0.01779 |
| 31 | 0.005398 |
| 32 | -0.004447 |
| 33 | -0.001377 |
| 34 | -0.0002125 |
| 35 | -0.004976 |
| 36 | 0.005716 |
| 37 | 0.01344 |
| 38 | 0.008045 |
| 39 | -0.0003183 |
| 40 | -0.005506 |
| 41 | -0.002436 |
| 42 | 0.01228 |
| 43 | -0.02128 |
| 44 | 0.008256 |
| 45 | -0.01154 |
| 46 | 0.004657 |
| 47 | 0.0109 |
| 48 | 0.004657 |
| 49 | 0.01302 |
| 50 | -0.009529 |

y: correlation

Marked: shaded band y ∈ [-0.0203, 0.0203].

> 9,320 mid changes. Lag 1 strongly negative: moves get undone (fade them). Positive: moves continue (follow them).

#### Time between mid changes

data: [`csv/vol/09-time-between-mid-changes.csv`](csv/vol/09-time-between-mid-changes.csv)

| seconds (log) | mid changes |
| --- | --- |
| 1.4e-05 | 1 |
| 1.947e-05 | 1 |
| 2.707e-05 | 0 |
| 3.765e-05 | 1 |
| 5.235e-05 | 3 |
| 7.28e-05 | 0 |
| 0.0001012 | 3 |
| 0.0001408 | 0 |
| 0.0001958 | 3 |
| 0.0002723 | 6 |
| 0.0003786 | 4 |
| 0.0005265 | 10 |
| 0.0007321 | 14 |
| 0.001018 | 27 |
| 0.001416 | 25 |
| 0.001969 | 51 |
| 0.002738 | 54 |
| 0.003807 | 77 |
| 0.005295 | 83 |
| 0.007363 | 125 |
| 0.01024 | 154 |
| 0.01424 | 164 |
| 0.0198 | 216 |
| 0.02753 | 259 |
| 0.03829 | 306 |
| 0.05324 | 314 |
| 0.07404 | 340 |
| 0.103 | 374 |
| 0.1432 | 422 |
| 0.1991 | 474 |
| 0.2769 | 539 |
| 0.385 | 641 |
| 0.5354 | 808 |
| 0.7446 | 1020 |
| 1.035 | 1030 |
| 1.44 | 764 |
| 2.002 | 501 |
| 2.784 | 284 |
| 3.872 | 152 |
| 5.385 | 60 |
| 7.488 | 10 |

y: count

> 9,320 mid changes. Changes are the events a quoting strategy has to react to.

#### Size of mid changes

data: [`csv/vol/10-size-of-mid-changes.csv`](csv/vol/10-size-of-mid-changes.csv)

| Δmid (ticks) | count |
| --- | --- |
| -1 | 25 |
| -0.5 | 4627 |
| 0.5 | 4651 |
| 1 | 17 |

y: count

> Mostly ±0.5 / ±1 means the price moves tick by tick; large jumps mean gaps in the book.

## Liquidity & book

#### Mean resting quantity by distance from the touch

data: [`csv/liq/01-mean-resting-quantity-by-distance-from-the-touch.csv`](csv/liq/01-mean-resting-quantity-by-distance-from-the-touch.csv)

| ticks behind the best price | Bid | Ask |
| --- | --- | --- |
| 0 | 818.8 | 817 |
| 1 | 987.1 | 981.7 |
| 2 | 120.7 | 122.8 |
| 3 | 8.902 | 9.079 |
| 4 | 0.5556 | 0.5889 |
| 5 | 0.03798 | 0.04791 |
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

data: not saved (5,481 points; `--export --all-csv` writes it)

| series | points | qty range | share of time min | mean | median | max | last |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Bid | 2,687 | 1..2899 | 6.957e-10 | 0.0003722 | 0.000422 | 0.002941 | 1.019e-07 |
| Ask | 2,794 | 1..2999 | 1.739e-10 | 0.0003579 | 0.0004218 | 0.003165 | 2.762e-07 |

> Small, frequently refreshed touch sizes mean our queue position matters less; large ones mean we wait behind them.

#### Queue imbalance  Qbid / (Qbid + Qask)

data: [`csv/liq/03-queue-imbalance-qbid-qbid-qask.csv`](csv/liq/03-queue-imbalance-qbid-qbid-qask.csv)

| imbalance | share of samples |
| --- | --- |
| 0.025 | 0.04157 |
| 0.075 | 0.04279 |
| 0.125 | 0.03548 |
| 0.175 | 0.03288 |
| 0.225 | 0.04001 |
| 0.275 | 0.04244 |
| 0.325 | 0.0494 |
| 0.375 | 0.06627 |
| 0.425 | 0.07271 |
| 0.475 | 0.07636 |
| 0.525 | 0.08332 |
| 0.575 | 0.06993 |
| 0.625 | 0.06262 |
| 0.675 | 0.05758 |
| 0.725 | 0.04592 |
| 0.775 | 0.03827 |
| 0.825 | 0.03757 |
| 0.875 | 0.02992 |
| 0.925 | 0.03705 |
| 0.975 | 0.03792 |

y: share of samples

> Mass near 0 and 1 means one side is often nearly empty: the strongest imbalance signals.

#### Order sizes at insert

data: [`csv/liq/04-order-sizes-at-insert.csv`](csv/liq/04-order-sizes-at-insert.csv)

| quantity | passive (1,033,814) | aggressive (33,718) |
| --- | --- | --- |
| 1.5 | 14 | 784 |
| 3.5 | 26 | 1139 |
| 5.5 | 24 | 1259 |
| 7.5 | 35 | 1231 |
| 9.5 | 27 | 1189 |
| 11.5 | 25 | 1115 |
| 13.5 | 30 | 1097 |
| 15.5 | 19 | 994 |
| 17.5 | 18 | 902 |
| 19.5 | 49350 | 822 |
| 21.5 | 97890 | 761 |
| 23.5 | 98835 | 706 |
| 25.5 | 98089 | 673 |
| 27.5 | 98143 | 662 |
| 29.5 | 98566 | 640 |
| 31.5 | 98447 | 568 |
| 33.5 | 98748 | 568 |
| 35.5 | 98102 | 549 |
| 37.5 | 98231 | 553 |
| 39.5 | 98785 | 509 |
| 41.5 | 13 | 474 |
| 43.5 | 6 | 460 |
| 45.5 | 5 | 437 |
| 47.5 | 10 | 370 |
| 49.5 | 11 | 393 |
| 51.5 | 11 | 335 |
| 53.5 | 8 | 352 |
| 55.5 | 12 | 306 |
| 57.5 | 11 | 301 |
| 59.5 | 8 | 293 |
| 61.5 | 13 | 314 |
| 63.5 | 6 | 272 |
| 65.5 | 4 | 270 |
| 67.5 | 8 | 304 |
| 69.5 | 1 | 290 |
| 71.5 | 6 | 235 |
| 73.5 | 5 | 246 |
| 75.5 | 7 | 248 |
| 77.5 | 3 | 237 |
| 79.5 | 5 | 249 |
| 81.5 | 3 | 210 |
| 83.5 | 8 | 216 |
| 85.5 | 2 | 171 |
| 87.5 | 2 | 177 |
| 89.5 | 4 | 204 |
| 91.5 | 4 | 183 |
| 93.5 | 6 | 189 |
| 95.5 | 5 | 162 |
| 97.5 | 4 | 161 |
| 99.5 | 4 | 170 |
| 101.5 | 4 | 138 |
| 103.5 | 3 | 137 |
| 105.5 | 1 | 141 |
| 107.5 | 2 | 139 |
| 109.5 | 2 | 128 |
| 111.5 | 5 | 137 |
| 113.5 | 1 | 134 |
| 115.5 | 1 | 130 |
| 117.5 | 2 | 130 |
| 119.5 | 5 | 118 |
| 121.5 | 5 | 134 |
| 123.5 | 2 | 137 |
| 125.5 | 4 | 116 |
| 127.5 | 5 | 117 |
| 129.5 | 5 | 137 |
| 131.5 | 3 | 98 |
| 133.5 | 2 | 112 |
| 135.5 | 2 | 97 |
| 137.5 | 2 | 97 |
| 139.5 | 2 | 91 |
| 141.5 | 1 | 89 |
| 143.5 | 3 | 87 |
| 145.5 | 5 | 81 |
| 147.5 | 3 | 97 |
| 149.5 | 2 | 83 |
| 151.5 | 2 | 75 |
| 153.5 | 0 | 76 |
| 155.5 | 3 | 85 |
| 157.5 | 3 | 87 |
| 159.5 | 1 | 63 |
| 161.5 | 3 | 66 |
| 163.5 | 1 | 81 |
| 165.5 | 1 | 68 |
| 167.5 | 1 | 58 |

y: count

> Compare with our ORDER_SIZE: quoting much larger than the crowd sizes is unusual for this tape.

#### Lifetime of resting orders

data: [`csv/liq/05-lifetime-of-resting-orders.csv`](csv/liq/05-lifetime-of-resting-orders.csv)

| seconds (log) | cancelled | filled |
| --- | --- | --- |
| 1.1e-05 | 0 | 0 |
| 1.571e-05 | 0 | 0 |
| 2.244e-05 | 0 | 0 |
| 3.205e-05 | 0 | 0 |
| 4.577e-05 | 0 | 0 |
| 6.537e-05 | 0 | 0 |
| 9.336e-05 | 0 | 1 |
| 0.0001333 | 0 | 3 |
| 0.0001904 | 0 | 5 |
| 0.000272 | 0 | 5 |
| 0.0003884 | 0 | 4 |
| 0.0005547 | 0 | 10 |
| 0.0007923 | 0 | 14 |
| 0.001132 | 4 | 15 |
| 0.001616 | 4 | 21 |
| 0.002308 | 18 | 43 |
| 0.003296 | 38 | 57 |
| 0.004708 | 118 | 79 |
| 0.006724 | 356 | 125 |
| 0.009603 | 693 | 177 |
| 0.01372 | 1568 | 230 |
| 0.01959 | 3069 | 321 |
| 0.02798 | 5760 | 455 |
| 0.03996 | 10572 | 719 |
| 0.05707 | 16893 | 958 |
| 0.0815 | 26341 | 1414 |
| 0.1164 | 38873 | 1987 |
| 0.1662 | 54909 | 2884 |
| 0.2374 | 75231 | 4141 |
| 0.3391 | 98303 | 6404 |
| 0.4843 | 121524 | 9950 |
| 0.6917 | 135632 | 14561 |
| 0.9879 | 129266 | 18916 |
| 1.411 | 101875 | 19203 |
| 2.015 | 63449 | 11971 |
| 2.878 | 31593 | 4480 |
| 4.11 | 12431 | 1042 |
| 5.87 | 3382 | 188 |
| 8.384 | 532 | 29 |
| 11.97 | 34 | 0 |

y: count

> median 0.524s; 0% live under 10 ms (fleeting quotes).

#### Where passive orders are placed

data: [`csv/liq/06-where-passive-orders-are-placed.csv`](csv/liq/06-where-passive-orders-are-placed.csv)

| ticks from the same-side best (+ inside the spread, − behind) | count |
| --- | --- |
| -2 | 35564 |
| -1 | 454600 |
| 0 | 536834 |
| 1 | 5343 |

y: count

> 0 joins the best price, +1 improves it by a tick, −1 sits one tick behind.

## Trades & flow

#### Aggressive orders per second (48 s bins)

data: [`csv/flow/01-aggressive-orders-per-second-48-s-bins.csv`](csv/flow/01-aggressive-orders-per-second-48-s-bins.csv)

| time (s) | Buy | Sell |
| --- | --- | --- |
| 0 | 3.125 | 3.375 |
| 48 | 2.979 | 3.062 |
| 96 | 2.792 | 3.354 |
| 144 | 2.625 | 2.812 |
| 192 | 2.458 | 2.896 |
| 240 | 3.042 | 3.021 |
| 288 | 2.854 | 2.792 |
| 336 | 2.792 | 2.896 |
| 384 | 2.438 | 3.333 |
| 432 | 3 | 2.875 |
| 480 | 2.917 | 2.667 |
| 528 | 3.104 | 3.062 |
| 576 | 3.146 | 2.604 |
| 624 | 3.375 | 3 |
| 672 | 2.833 | 2.375 |
| 720 | 2.792 | 3.146 |
| 768 | 3.083 | 2.854 |
| 816 | 2.958 | 2.875 |
| 864 | 2.479 | 2.917 |
| 912 | 3.125 | 3.25 |
| 960 | 2.771 | 3.229 |
| 1008 | 2.812 | 3.125 |
| 1056 | 2.896 | 3.146 |
| 1104 | 3.333 | 3.417 |
| 1152 | 2.896 | 2.812 |
| 1200 | 2.938 | 3.104 |
| 1248 | 2.396 | 3.021 |
| 1296 | 3 | 2.771 |
| 1344 | 2.917 | 3.021 |
| 1392 | 2.729 | 2.896 |
| 1440 | 2.854 | 2.771 |
| 1488 | 3.229 | 3 |
| 1536 | 2.833 | 2.688 |
| 1584 | 2.812 | 2.708 |
| 1632 | 2.812 | 3.104 |
| 1680 | 2.542 | 3.312 |
| 1728 | 3.042 | 2.833 |
| 1776 | 3 | 2.812 |
| 1824 | 2.979 | 2.667 |
| 1872 | 3.042 | 2.979 |
| 1920 | 2.979 | 2.667 |
| 1968 | 2.854 | 3.042 |
| 2016 | 2.833 | 3.292 |
| 2064 | 2.812 | 2.75 |
| 2112 | 3 | 3 |
| 2160 | 2.646 | 2.833 |
| 2208 | 3.312 | 2.833 |
| 2256 | 2.792 | 2.646 |
| 2304 | 2.896 | 3.062 |
| 2352 | 2.875 | 2.812 |
| 2400 | 3.062 | 2.562 |
| 2448 | 3.417 | 3.083 |
| 2496 | 2.438 | 3.042 |
| 2544 | 2.938 | 2.562 |
| 2592 | 2.708 | 2.875 |
| 2640 | 2.896 | 2.896 |
| 2688 | 2.896 | 2.938 |
| 2736 | 2.896 | 2.583 |
| 2784 | 3.333 | 2.938 |
| 2832 | 2.375 | 2.771 |
| 2880 | 2.729 | 2.562 |
| 2928 | 2.833 | 2.979 |
| 2976 | 3.062 | 3.438 |
| 3024 | 2.958 | 3.042 |
| 3072 | 3.208 | 3.333 |
| 3120 | 2.625 | 3 |
| 3168 | 2.562 | 2.625 |
| 3216 | 2.75 | 3.042 |
| 3264 | 2.875 | 2.854 |
| 3312 | 3.083 | 2.917 |
| 3360 | 3.292 | 2.875 |
| 3408 | 3.333 | 3.062 |
| 3456 | 2.979 | 3.062 |
| 3504 | 3.417 | 3.292 |
| 3552 | 3.083 | 2.854 |
| 3600 | 2.854 | 3.062 |
| 3648 | 3.229 | 2.938 |
| 3696 | 2.688 | 2.896 |
| 3744 | 2.917 | 2.896 |
| 3792 | 3.021 | 3.083 |
| 3840 | 3.25 | 2.958 |
| 3888 | 3.062 | 3.104 |
| 3936 | 2.604 | 2.729 |
| 3984 | 3.229 | 3.042 |
| 4032 | 2.729 | 2.875 |
| 4080 | 2.812 | 2.938 |
| 4128 | 3.271 | 2.417 |
| 4176 | 2.896 | 3.125 |
| 4224 | 3.271 | 3.229 |
| 4272 | 2.708 | 2.979 |
| 4320 | 2.792 | 2.75 |
| 4368 | 3 | 2.688 |
| 4416 | 3.312 | 3.125 |
| 4464 | 3.188 | 3.042 |
| 4512 | 3.208 | 2.854 |
| 4560 | 3.188 | 2.771 |
| 4608 | 2.854 | 2.917 |
| 4656 | 2.854 | 2.604 |
| 4704 | 2.583 | 3.104 |
| 4752 | 3.312 | 2.771 |
| 4800 | 2.854 | 2.771 |
| 4848 | 3.167 | 3 |
| 4896 | 3.021 | 2.667 |
| 4944 | 2.562 | 2.604 |
| 4992 | 3.188 | 2.708 |
| 5040 | 2.833 | 2.542 |
| 5088 | 2.875 | 2.958 |
| 5136 | 3.375 | 2.625 |
| 5184 | 2.625 | 3.146 |
| 5232 | 3.208 | 3 |
| 5280 | 2.771 | 2.917 |
| 5328 | 2.792 | 2.625 |
| 5376 | 2.542 | 3.104 |
| 5424 | 3 | 2.958 |
| 5472 | 3.146 | 3.5 |
| 5520 | 2.875 | 2.958 |
| 5568 | 3.021 | 2.667 |
| 5616 | 2.75 | 3.042 |
| 5664 | 3.25 | 3.062 |
| 5712 | 2.292 | 2.521 |

y: orders / s

> 33,718 aggressive orders, 3,177,589 lots. Fills for a passive quote only come from these.

#### Time between aggressive orders

data: [`csv/flow/02-time-between-aggressive-orders.csv`](csv/flow/02-time-between-aggressive-orders.csv)

| seconds | observed | Poisson, same rate |
| --- | --- | --- |
| 0.007886 | 2961 | 2975 |
| 0.02364 | 2724 | 2713 |
| 0.0394 | 2511 | 2473 |
| 0.05515 | 2250 | 2255 |
| 0.07091 | 2099 | 2056 |
| 0.08666 | 1854 | 1875 |
| 0.1024 | 1655 | 1709 |
| 0.1182 | 1519 | 1558 |
| 0.1339 | 1367 | 1421 |
| 0.1497 | 1283 | 1295 |
| 0.1654 | 1199 | 1181 |
| 0.1812 | 1043 | 1077 |
| 0.1969 | 996 | 981.9 |
| 0.2127 | 897 | 895.2 |
| 0.2285 | 843 | 816.2 |
| 0.2442 | 777 | 744.2 |
| 0.26 | 709 | 678.5 |
| 0.2757 | 664 | 618.6 |
| 0.2915 | 583 | 564 |
| 0.3072 | 491 | 514.3 |
| 0.323 | 475 | 468.9 |
| 0.3387 | 405 | 427.5 |
| 0.3545 | 439 | 389.8 |
| 0.3702 | 355 | 355.4 |
| 0.386 | 292 | 324 |
| 0.4018 | 289 | 295.4 |
| 0.4175 | 251 | 269.4 |
| 0.4333 | 249 | 245.6 |
| 0.449 | 225 | 223.9 |
| 0.4648 | 206 | 204.2 |
| 0.4805 | 183 | 186.1 |
| 0.4963 | 185 | 169.7 |
| 0.512 | 166 | 154.7 |
| 0.5278 | 150 | 141.1 |
| 0.5435 | 114 | 128.6 |
| 0.5593 | 115 | 117.3 |
| 0.5751 | 134 | 106.9 |
| 0.5908 | 96 | 97.49 |
| 0.6066 | 95 | 88.89 |
| 0.6223 | 84 | 81.04 |
| 0.6381 | 59 | 73.89 |
| 0.6538 | 63 | 67.37 |
| 0.6696 | 45 | 61.43 |
| 0.6853 | 57 | 56 |
| 0.7011 | 43 | 51.06 |
| 0.7168 | 41 | 46.56 |
| 0.7326 | 33 | 42.45 |
| 0.7484 | 37 | 38.7 |
| 0.7641 | 39 | 35.29 |
| 0.7799 | 29 | 32.17 |

y: count (log)

> Coefficient of variation 1.00 (1 for Poisson; above 1 means clustered arrivals, below 1 means regular ones).

#### Aggressive order size

data: [`csv/flow/03-aggressive-order-size.csv`](csv/flow/03-aggressive-order-size.csv)

| quantity | Buy | Sell |
| --- | --- | --- |
| 11 | 5484 | 5486 |
| 32 | 3265 | 3259 |
| 53 | 1858 | 1891 |
| 74 | 1309 | 1340 |
| 95 | 939 | 842 |
| 116 | 692 | 687 |
| 137 | 527 | 534 |
| 158 | 385 | 398 |
| 179 | 310 | 322 |
| 200 | 277 | 258 |
| 221 | 212 | 206 |
| 242 | 179 | 164 |
| 263 | 153 | 136 |
| 284 | 119 | 136 |
| 305 | 109 | 127 |
| 326 | 89 | 100 |
| 347 | 94 | 79 |
| 368 | 63 | 85 |
| 389 | 56 | 56 |
| 410 | 73 | 60 |
| 431 | 56 | 42 |
| 452 | 39 | 51 |
| 473 | 51 | 44 |
| 494 | 48 | 37 |
| 515 | 31 | 37 |
| 536 | 34 | 29 |
| 557 | 27 | 42 |
| 578 | 26 | 22 |
| 599 | 23 | 27 |
| 620 | 25 | 12 |
| 641 | 24 | 24 |
| 662 | 22 | 17 |
| 683 | 16 | 22 |
| 704 | 19 | 13 |
| 725 | 16 | 20 |
| 746 | 11 | 16 |
| 767 | 14 | 13 |
| 788 | 9 | 7 |
| 809 | 7 | 16 |
| 830 | 14 | 16 |
| 851 | 17 | 6 |
| 872 | 11 | 9 |
| 893 | 11 | 12 |
| 914 | 4 | 9 |
| 935 | 6 | 12 |
| 956 | 10 | 7 |
| 977 | 8 | 10 |
| 998 | 7 | 10 |
| 1019 | 2 | 3 |
| 1040 | 5 | 2 |
| 1061 | 3 | 5 |
| 1082 | 4 | 3 |

y: count

> Large aggressive orders are the ones that sweep through our quote and leave us adversely selected.

#### Levels swept past the touch

data: [`csv/flow/04-levels-swept-past-the-touch.csv`](csv/flow/04-levels-swept-past-the-touch.csv)

| ticks past the best opposite price | observed | geometric, k = 2.93 |
| --- | --- | --- |
| 0 | 0.9439 | 0.9468 |
| 1 | 0.05605 | 0.05036 |
| 2 | 5.932e-05 | 0.002678 |

y: share of orders

> Mean depth 0.0562 → k = ln(1 + 1/mean) = 2.93: the estimator avellaneda_stoikov.py uses online.

#### Fill intensity λ(δ) for a quote δ ticks from the mid

data: [`csv/flow/05-fill-intensity-for-a-quote-ticks-from-the-mid.csv`](csv/flow/05-fill-intensity-for-a-quote-ticks-from-the-mid.csv)

| δ (ticks from mid) | Ask side hit by buys | Bid side hit by sells | A·e^(−kδ): A=259, k=4.94 |
| --- | --- | --- | --- |
| 0 | 2.934 | 2.93 | 259.4 |
| 0.5 | 2.934 | 2.93 | 21.94 |
| 1 | 1.84 | 1.861 | 1.856 |
| 1.5 | 0.1621 | 0.1722 | 0.157 |
| 2 | 0.004174 | 0.003478 | 0.01328 |
| 2.5 | 0.0001739 | 0.0001739 | 0.001123 |
| 3 | - | - | 9.498e-05 |
| 3.5 | - | - | 8.034e-06 |
| 4 | - | - | 6.796e-07 |
| 4.5 | - | - | 5.748e-08 |
| 5 | - | - | 4.862e-09 |
| 5.5 | - | - | 4.112e-10 |
| 6 | - | - | 3.478e-11 |
| 6.5 | - | - | 2.942e-12 |
| 7 | - | - | 2.489e-13 |
| 7.5 | - | - | 2.105e-14 |
| 8 | - | - | 1.78e-15 |
| 8.5 | - | - | 1.506e-16 |
| 9 | - | - | 1.274e-17 |
| 9.5 | - | - | 1.077e-18 |
| 10 | - | - | 9.113e-20 |
| 10.5 | - | - | 7.708e-21 |
| 11 | - | - | 6.52e-22 |
| 11.5 | - | - | 5.515e-23 |
| 12 | - | - | 4.664e-24 |
| 12.5 | - | - | 3.945e-25 |
| 13 | - | - | 3.337e-26 |
| 13.5 | - | - | 2.823e-27 |
| 14 | - | - | 2.387e-28 |
| 14.5 | - | - | 2.019e-29 |
| 15 | - | - | 1.708e-30 |

y: hits per second (log)

> The Avellaneda-Stoikov arrival model. A straight line on this log axis means the exponential model fits; a kink means it does not.

#### Cumulative signed aggressive volume / Mid

data: not saved (35,719 points; `--export --all-csv` writes it)

| series | points | x range | y min | mean | median | max | last |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Net buy volume | 33,718 | 0.1426..5750 | -38905 | -2.086e+04 | -2.206e+04 | 3043 | -12801 |
| Mid | 2,001 | 0..5750 | 9966 | 9993 | 9994 | 10022 | 1e+04 |

> If price follows net flow closely, aggressive orders carry information.

#### Autocorrelation of aggressive order signs

data: [`csv/flow/07-autocorrelation-of-aggressive-order-signs.csv`](csv/flow/07-autocorrelation-of-aggressive-order-signs.csv)

| lag (orders) | correlation |
| --- | --- |
| 1 | 0.0003257 |
| 2 | 0.003796 |
| 3 | -0.00436 |
| 4 | 0.00172 |
| 5 | -0.007029 |
| 6 | -0.008898 |
| 7 | 0.003469 |
| 8 | -0.002788 |
| 9 | -0.0003861 |
| 10 | -0.01032 |
| 11 | -0.0002674 |
| 12 | 0.001601 |
| 13 | -0.009995 |
| 14 | 0.0002368 |
| 15 | -0.006199 |
| 16 | 0.002491 |
| 17 | 0.00258 |
| 18 | 0.00255 |
| 19 | 0.001038 |
| 20 | 0.001542 |
| 21 | 0.0008003 |
| 22 | -0.0007716 |
| 23 | -0.001335 |
| 24 | -5.982e-05 |
| 25 | -0.01112 |
| 26 | -0.002729 |
| 27 | 0.01237 |
| 28 | -0.001543 |
| 29 | -0.0006825 |
| 30 | 0.007058 |

y: correlation

Marked: shaded band y ∈ [-0.01067, 0.01067].

> Positive: buys follow buys (split metaorders). Their signs then predict the next order.

## Predictability

#### Future mid move by queue imbalance

data: [`csv/predict/01-future-mid-move-by-queue-imbalance.csv`](csv/predict/01-future-mid-move-by-queue-imbalance.csv)

| imbalance  Qbid / (Qbid + Qask) | 0.1 s | 1 s | 5 s | 30 s |
| --- | --- | --- | --- | --- |
| 0.05202 | -0.1267 ± 0.01322 | -0.2788 ± 0.02854 | -0.2773 ± 0.06509 | -0.2322 ± 0.1619 |
| 0.1508 | -0.05877 ± 0.01089 | -0.2222 ± 0.03096 | -0.2086 ± 0.06894 | -0.1563 ± 0.1777 |
| 0.2517 | -0.05068 ± 0.009291 | -0.1752 ± 0.02619 | -0.1747 ± 0.06322 | -0.1062 ± 0.1602 |
| 0.3528 | -0.03376 ± 0.006505 | -0.1324 ± 0.02101 | -0.1558 ± 0.05176 | -0.1096 ± 0.1331 |
| 0.4505 | -0.009636 ± 0.004914 | -0.03918 ± 0.01821 | -0.06646 ± 0.04703 | -0.03591 ± 0.1238 |
| 0.5493 | 0.01012 ± 0.005083 | 0.0491 ± 0.01856 | 0.06662 ± 0.04732 | 0.06916 ± 0.1202 |
| 0.648 | 0.03312 ± 0.00659 | 0.1249 ± 0.02108 | 0.1511 ± 0.05193 | 0.1194 ± 0.1366 |
| 0.7466 | 0.04981 ± 0.009224 | 0.1811 ± 0.02605 | 0.1998 ± 0.06239 | 0.2347 ± 0.1604 |
| 0.849 | 0.06168 ± 0.01096 | 0.1883 ± 0.03116 | 0.2103 ± 0.07148 | 0.269 ± 0.1796 |
| 0.9483 | 0.1272 ± 0.01368 | 0.3004 ± 0.02919 | 0.2902 ± 0.06565 | 0.2746 ± 0.167 |

y: mean Δmid (ticks)  (± is the 95% interval)

Marked: horizontal line y = 0.

> Buckets with under 30 samples hidden. A monotone S-curve means imbalance predicts the next move: use the microprice instead of the mid, or lean quotes with it.

#### Is the imbalance edge still there 5 ms later?

data: [`csv/predict/02-is-the-imbalance-edge-still-there-5-ms-later.csv`](csv/predict/02-is-the-imbalance-edge-still-there-5-ms-later.csv)

| imbalance | measured from now | measured from now + 5 ms |
| --- | --- | --- |
| 0.05202 | -0.1267 | -0.1224 |
| 0.1508 | -0.05877 | -0.05735 |
| 0.2517 | -0.05068 | -0.04867 |
| 0.3528 | -0.03376 | -0.03354 |
| 0.4505 | -0.009636 | -0.0101 |
| 0.5493 | 0.01012 | 0.01035 |
| 0.648 | 0.03312 | 0.0327 |
| 0.7466 | 0.04981 | 0.04828 |
| 0.849 | 0.06168 | 0.06037 |
| 0.9483 | 0.1272 | 0.123 |

y: mean Δmid over the next 100 ms (ticks)

Marked: horizontal line y = 0.

> Our orders land 5 ms after we decide. If the two curves match, the signal is still worth trading on.

#### After an aggressive order: mid move in its direction

data: [`csv/predict/03-after-an-aggressive-order-mid-move-in-its-direction.csv`](csv/predict/03-after-an-aggressive-order-mid-move-in-its-direction.csv)

| seconds after (log) | price impact (vs pre-trade mid) | taker PnL per lot (vs fill) = −maker PnL |
| --- | --- | --- |
| 0.005 | 0.0471 ± 0.003142 | -0.7733 |
| 0.05 | 0.03701 ± 0.003541 | -0.7834 |
| 0.1 | 0.03414 ± 0.004121 | -0.7863 |
| 0.25 | 0.02595 ± 0.005609 | -0.7944 |
| 0.5 | 0.01941 ± 0.007437 | -0.801 |
| 1 | 0.01542 ± 0.01035 | -0.805 |
| 2 | 0.008749 ± 0.01493 | -0.8116 |
| 5 | 0.01876 ± 0.02381 | -0.8016 |
| 10 | 0.01662 ± 0.03462 | -0.8038 |
| 20 | 0.02118 ± 0.0494 | -0.7992 |
| 30 | 0.02393 ± 0.06105 | -0.7965 |
| 60 | 0.02287 ± 0.08707 | -0.7975 |

y: ticks  (± is the 95% interval)

Marked: horizontal line y = 0; vertical line x = 0.005.

> Positive orange line: passive fills lose money at that horizon (adverse selection). The quoted half-spread must cover it.

#### Order flow imbalance → mid move (R²)

data: [`csv/predict/04-order-flow-imbalance-mid-move-r.csv`](csv/predict/04-order-flow-imbalance-mid-move-r.csv)

| bin size (s) | same bin (impact) | next bin (prediction) |
| --- | --- | --- |
| 0.1 | 0.07938 | 0.01945 |
| 0.5 | 0.3494 | 0.08148 |
| 1 | 0.5706 | 0.05405 |
| 5 | 0.8506 | 0.01252 |
| 30 | 0.9305 | 0.01418 |

y: R²

> Same-bin R² is price impact (descriptive). Next-bin R² is real predictability.

#### OFI vs Δmid, 1 s bins

data: not saved (5,002 points; `--export --all-csv` writes it)

| OFI (lots) | fit |
| --- | --- |
| -3618 | -1.11 |
| 3958 | 1.215 |

y: Δmid (ticks)

| series | points | OFI (lots) range | Δmid (ticks) min | mean | median | max | last |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scatter 1 | 5,000 | -3618..3958 | -2 | 0.0022 | 0 | 2 | 0 |

> Change the bin size with the sampling selector at the top.

### OFI regression  Δmid = β·OFI

| bin | β (ticks/lot) | R² same | β next | R² next | n |
| --- | --- | --- | --- | --- | --- |
| 0.1 s | 0.000207 | 0.0794 | 0.000102 | 0.0194 | 57,497 |
| 0.5 s | 0.000272 | 0.349 | 0.000132 | 0.0815 | 11,498 |
| 1 s | 0.000307 | 0.571 | 9.44e-05 | 0.054 | 5,748 |
| 5 s | 0.000358 | 0.851 | 4.34e-05 | 0.0125 | 1,148 |
| 30 s | 0.000367 | 0.93 | 4.57e-05 | 0.0142 | 190 |
