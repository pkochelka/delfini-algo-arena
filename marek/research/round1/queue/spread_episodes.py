"""Spread >= 2 episodes on round1: how long they last, which side closes the gap, and how.

    uv run python research/round1/queue/spread_episodes.py

An episode starts when the market spread goes from 1 to >= 2 tick and ends when it is back to 1.
`opened_by`: the side whose touch moved away (bid down = 'bid broke', ask up = 'ask broke').
`closed_by`: the side whose touch moved in to close the gap. 'continuation' = the side opposite
the one that broke moved in (the mid keeps going the way it went), 'refill' = the broken side came
back. Also: the mid 1 s / 5 s after the close relative to the mid at the open (signed in the
direction of the opening move) and the touch queue sizes at the open.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis import features as F, load_tape  # noqa: E402


def main() -> None:
    tape = load_tape(ROOT / 'data/round1.csv', verbose=False)
    q = tape.quotes.dropna(subset=['bid', 'ask']).reset_index(drop=True)
    sp = q['spread'].to_numpy()
    wide = sp >= 2
    starts = np.flatnonzero(wide[1:] & ~wide[:-1]) + 1
    ends = np.flatnonzero(~wide[1:] & wide[:-1]) + 1
    ends = ends[np.searchsorted(ends, starts[0]):]
    n = min(len(starts), len(ends))
    starts, ends = starts[:n], ends[:n]
    assert (ends > starts).all()
    bid, ask, t = q['bid'].to_numpy(), q['ask'].to_numpy(), q['t_us'].to_numpy()
    pre = starts - 1
    opened_bid = bid[starts] < bid[pre]
    opened_ask = ask[starts] > ask[pre]
    last = ends - 1
    closed_bid = bid[ends] > bid[last]
    closed_ask = ask[ends] < ask[last]
    dur_ms = (t[ends] - t[starts]) / 1e3
    df = pd.DataFrame({'t_us': t[starts], 'dur_ms': dur_ms, 'open': np.where(opened_bid & ~opened_ask, 'bid broke',
                       np.where(opened_ask & ~opened_bid, 'ask broke', 'both/other')),
                       'close': np.where(closed_bid & ~closed_ask, 'bid in', np.where(closed_ask & ~closed_bid, 'ask in', 'both/other')),
                       'width': sp[starts], 'bid_qty': q['bid_qty'].to_numpy()[starts], 'ask_qty': q['ask_qty'].to_numpy()[starts]})
    df['kind'] = np.select([(df['open'] == 'bid broke') & (df['close'] == 'ask in'),
                            (df['open'] == 'ask broke') & (df['close'] == 'bid in'),
                            (df['open'] == 'bid broke') & (df['close'] == 'bid in'),
                            (df['open'] == 'ask broke') & (df['close'] == 'ask in')],
                           ['continuation', 'continuation', 'refill', 'refill'], 'other')
    sgn = np.where(df['open'] == 'ask broke', 1.0, np.where(df['open'] == 'bid broke', -1.0, 0.0))
    mid0 = (bid[pre] + ask[pre]) / 2
    for h in (1, 5, 30):
        m = F.asof(tape, (t[ends] - tape.start_us) / 1e6 + h, 'mid')
        df[f'drift{h}'] = (m - mid0) * sgn
    print('episodes', len(df), 'time share at >=2:', round(float(np.sum(dur_ms)) / 1e3 / tape.duration_s, 4))
    print('duration ms quantiles', np.quantile(dur_ms, [.1, .25, .5, .75, .9, .99]).round(1).tolist())
    print('share shorter than 5 / 10 / 20 ms:', [round(float((dur_ms < x).mean()), 3) for x in (5, 10, 20)])
    print(df.groupby(['open', 'close']).size())
    print(df.groupby('kind').agg(n=('dur_ms', 'size'), dur_med=('dur_ms', 'median'), drift1=('drift1', 'mean'),
                                 drift5=('drift5', 'mean'), drift30=('drift30', 'mean')).round(3))
    # thick vs thin side at the open: which side moves in?
    o = df[df['kind'] != 'other'].copy()
    o['closer_thick'] = np.where(o['close'] == 'bid in', o['bid_qty'] >= o['ask_qty'], o['ask_qty'] >= o['bid_qty'])
    print('gap closed by the side with the thicker touch at the open:', round(float(o['closer_thick'].mean()), 3))
    # closing order: what rests at the new touch at the close (queue we'd have to beat)
    qq = np.where(df['close'] == 'bid in', q['bid_qty'].to_numpy()[ends], q['ask_qty'].to_numpy()[ends])
    print('qty at the new inside level right after the close, quantiles', np.quantile(qq, [.25, .5, .75, .9]).tolist())
    out = Path(__file__).resolve().parent / 'out' / 'spread_episodes.csv'
    df.to_csv(out, index=False)


if __name__ == '__main__':
    main()
